"""
eval/judge.py — LLM-as-judge scoring for answer quality.

Scores each (question, answer, contexts, reference) tuple on a 1–5 rubric:

  faithfulness      — is every claim in the answer supported by the retrieved
                      contexts? (5 = fully grounded, 1 = mostly fabricated)
  correctness       — does the answer agree with the reference answer on the
                      substantive facts? (numbers, thresholds, definitions)
  completeness      — does it cover the material points of the reference?
  citation_support  — could a reviewer verify the answer from the citations
                      provided?

Uses Gemini via google-genai with GEMINI_API_KEY from the project .env
(same key the backend uses; the judge runs from the analyst machine, so
only already-masked eval content is scored). Judge model is overridable
via EVAL_JUDGE_MODEL — by default it differs from the generation default
only by your configuration; for less self-preference bias point it at a
different model family or a stronger tier (e.g. gemini-2.5-pro).

Degrades gracefully: if google-genai or the key is missing, `available`
is False and callers skip judge metrics (their gates auto-skip too).
"""

import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.env import get_env

logger = logging.getLogger(__name__)

_RUBRIC_KEYS = ("faithfulness", "correctness", "completeness", "citation_support")

_JUDGE_PROMPT = """You are a strict evaluator of a credit-risk regulatory question-answering system.
Score the SYSTEM ANSWER on four dimensions, each an integer from 1 (worst) to 5 (best):

1. faithfulness: Is every factual claim in the answer supported by the RETRIEVED CONTEXTS? \
Penalise claims that appear nowhere in the contexts, even if true.
2. correctness: Does the answer agree with the REFERENCE ANSWER on substantive facts \
(numbers, thresholds, definitions, conditions)? Wording differences are fine; factual conflicts are not.
3. completeness: Does the answer cover the material points of the REFERENCE ANSWER? \
Missing a key number or condition caps this at 3.
4. citation_support: Could a reviewer verify the answer's key claims using only the retrieved contexts?

Notes:
- Text may contain anonymisation placeholders like [ORG_1] or [PERSON_2]. Treat them as opaque \
entity names; never penalise their presence.
- If the answer correctly states that the contexts do not contain the information, score \
faithfulness 5 and judge correctness/completeness against that being the right call.

QUESTION:
{question}

REFERENCE ANSWER:
{reference}

RETRIEVED CONTEXTS:
{contexts}

SYSTEM ANSWER:
{answer}

Respond with ONLY a JSON object, no markdown fences, exactly this shape:
{{"faithfulness": <1-5>, "correctness": <1-5>, "completeness": <1-5>, "citation_support": <1-5>, "rationale": "<one sentence>"}}"""


class GeminiJudge:

    def __init__(self, model: Optional[str] = None) -> None:
        self.model = model or get_env("EVAL_JUDGE_MODEL", "gemini-2.5-flash")
        self._client = None
        api_key = get_env("GEMINI_API_KEY")
        if not api_key:
            logger.warning("GEMINI_API_KEY not set — judge disabled.")
            return
        try:
            from google import genai
            self._client = genai.Client(api_key=api_key)
        except ImportError:
            logger.warning("google-genai not installed — judge disabled.")

    @property
    def available(self) -> bool:
        return self._client is not None

    def score(
        self,
        question:  str,
        answer:    str,
        contexts:  List[str],
        reference: str,
    ) -> Optional[Dict]:
        """One scored row, or None on any failure (caller averages over successes)."""
        if not self.available or not answer.strip():
            return None

        ctx_block = "\n---\n".join(c[:2000] for c in contexts[:8]) or "(no contexts returned)"
        prompt = _JUDGE_PROMPT.format(
            question=question,
            reference=reference or "(no reference provided — score correctness on internal consistency)",
            contexts=ctx_block,
            answer=answer[:8000],
        )
        # Free-tier Gemini keys allow ~5 requests/minute — retry 429s with a
        # backoff long enough to ride out the per-minute window.
        for attempt in range(4):
            try:
                resp = self._client.models.generate_content(model=self.model, contents=prompt)
                return self._parse(resp.text or "")
            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    wait = 20 * (attempt + 1)
                    logger.info("Judge rate-limited — retrying in %ss", wait)
                    time.sleep(wait)
                    continue
                logger.warning("Judge call failed: %s", e)
                return None
        logger.warning("Judge gave up after repeated rate limiting.")
        return None

    @staticmethod
    def _parse(text: str) -> Optional[Dict]:
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            return None
        try:
            raw = json.loads(m.group())
        except json.JSONDecodeError:
            return None
        scores: Dict = {}
        for k in _RUBRIC_KEYS:
            v = raw.get(k)
            if not isinstance(v, (int, float)):
                return None
            scores[k] = max(1, min(5, int(v)))
        scores["rationale"] = str(raw.get("rationale", ""))[:400]
        return scores


def aggregate(rows: List[Optional[Dict]]) -> Dict[str, Optional[float]]:
    """Mean per rubric dimension over successfully judged rows (judge_* keys)."""
    ok = [r for r in rows if r]
    out: Dict[str, Optional[float]] = {"judge_rows_scored": len(ok)}
    for k in _RUBRIC_KEYS:
        out[f"judge_{k}"] = (sum(r[k] for r in ok) / len(ok)) if ok else None
    return out
