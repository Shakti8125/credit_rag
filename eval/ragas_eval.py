"""
eval/ragas_eval.py — End-to-end RAG answer quality against the live backend.

Sends the gold question set (eval/golden_set.json — grounded in the CBUAE /
Basel III / IFRS 9 corpus, with reference answers) to the running FastAPI
/query endpoint and scores the answers.

Three scoring layers:
  1. Heuristics (always run, no extra deps):
       - grounding overlap : fraction of answer sentences with strong lexical
                             support in the returned citations (proxy for
                             faithfulness / hallucination rate)
       - citation coverage, expected-term hit rate, must-term hit rate
       - placeholder leaks : masked placeholders like [ORG_1] surviving in
                             regulatory-only answers (must be 0)
       - error/empty rate and p50/p95 latency
  2. LLM-as-judge (--judge, needs GEMINI_API_KEY): faithfulness, correctness
     vs reference answer, completeness, citation support — 1-5 rubric via
     eval/judge.py.
  3. RAGAS (if installed): faithfulness, answer relevancy, context precision.

Results gate against eval/thresholds.json (suite "e2e"), compare against
eval/baseline.json, and write eval/reports/e2e_latest.json.

Prerequisites:
    uvicorn app.main:app --port 8000     (from cloud/backend, with .env keys)
Run:
    python eval/ragas_eval.py [--api http://127.0.0.1:8000] [--judge]
Exit codes: 0 pass, 1 blocking gate failed, 2 backend unreachable.
"""

import argparse
import json
import re
import statistics
import sys
import time
from pathlib import Path

try:
    import requests
except ImportError:
    print("SKIP: this eval needs the 'requests' package (same env as the app).")
    sys.exit(2)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.env import get_env
from shared.hybrid import tokenize
from eval.common import build_gates, print_gates, warn_regressions, write_report, exit_code

GOLDEN_SET = Path(__file__).resolve().parent / "golden_set.json"

# Shared secret for a deployed backend (cloud/backend/app/auth.py). Empty
# against a local uvicorn backend, where the check self-disables — so the same
# command works whether --api points at localhost or the Lambda Function URL.
_API_KEY  = get_env("CLOUD_API_KEY", "") or ""
_HEADERS  = {"X-API-Key": _API_KEY} if _API_KEY else {}

_PLACEHOLDER_RE = re.compile(r"\[(?:ORG|PERSON|BANK|GPE|LOC|FAC|EMAIL|PHONE|SSN|EIN)_\d+\]")


def _sentences(text: str):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if len(s.split()) >= 4]


def _grounding_overlap(answer: str, contexts: list) -> float:
    """
    Fraction of answer sentences whose content tokens are ≥50% covered by
    at least one retrieved context. Crude but dependency-free proxy for
    faithfulness: low values mean the model is answering from priors, not
    the retrieved regulation text.
    """
    ctx_token_sets = [set(tokenize(c)) for c in contexts if c]
    sents = _sentences(answer)
    if not sents or not ctx_token_sets:
        return 0.0
    grounded = 0
    for s in sents:
        toks = set(tokenize(s))
        if not toks:
            continue
        best = max((len(toks & ctx) / len(toks)) for ctx in ctx_token_sets)
        if best >= 0.5:
            grounded += 1
    return grounded / len(sents)


def run(api_base: str, use_judge: bool, sleep_s: float) -> int:
    gold_questions = json.loads(GOLDEN_SET.read_text(encoding="utf-8"))["questions"]

    rows = []
    for gold in gold_questions:
        payload = {"query": gold["query"], "intent": gold.get("intent", "BENCHMARK")}

        # Retry 5xx twice with a long backoff: on free-tier Gemini keys the
        # backend's generation call 429s at ~5 requests/minute and surfaces
        # here as HTTP 500. Pace with --sleep to avoid tripping it at all.
        resp, elapsed = None, 0.0
        for attempt in range(3):
            t0 = time.time()
            try:
                resp = requests.post(f"{api_base}/query", json=payload,
                                     headers=_HEADERS, timeout=120)
                elapsed = time.time() - t0
            except requests.RequestException as e:
                print(f"FATAL: backend unreachable at {api_base} ({e})")
                print("Start it first:  cd cloud/backend && uvicorn app.main:app --port 8000")
                return 2
            if resp.status_code < 500 or attempt == 2:
                break
            wait = 25 * (attempt + 1)
            print(f"  HTTP {resp.status_code} on {gold['id']} — retrying in {wait}s")
            time.sleep(wait)

        if sleep_s:
            time.sleep(sleep_s)

        if resp.status_code != 200:
            rows.append({"id": gold["id"], "query": gold["query"],
                         "error": f"HTTP {resp.status_code}", "latency": elapsed})
            continue

        data      = resp.json()
        answer    = data.get("answer", "") or ""
        citations = [c.get("text", "") for c in data.get("citations", [])]
        must      = gold.get("must_terms", [])

        rows.append({
            "id":            gold["id"],
            "category":      gold.get("category", ""),
            "query":         gold["query"],
            "reference":     gold.get("reference_answer", ""),
            "answer":        answer,
            "contexts":      citations,
            "latency":       elapsed,
            "grounding":     _grounding_overlap(answer, citations),
            "has_citations": bool(citations),
            "placeholder_leaks": len(_PLACEHOLDER_RE.findall(answer)),
            "expected_hit":  any(t.lower() in answer.lower() for t in gold["expect_terms"]),
            "must_hit":      all(t.lower() in answer.lower() for t in must) if must else None,
            "empty":         not answer.strip(),
        })

    scored = [r for r in rows if "error" not in r]
    errors = [r for r in rows if "error" in r]

    print("\n================ RAG E2E EVAL ================")
    print(f"API: {api_base} | {len(scored)} scored, {len(errors)} errors\n")
    for r in scored:
        print(f"  grounding={r['grounding']:.2f}  cits={'Y' if r['has_citations'] else 'N'}  "
              f"expect={'Y' if r['expected_hit'] else 'N'}  {r['latency']:.1f}s  | {r['id']}")
    for r in errors:
        print(f"  ERROR {r['error']} | {r['id']}")

    if not scored:
        print("No scored rows — nothing to evaluate.")
        return 2

    lat = sorted(r["latency"] for r in scored)
    must_rows = [r for r in scored if r["must_hit"] is not None]
    metrics = {
        "questions":          len(rows),
        "grounding_mean":     statistics.mean(r["grounding"] for r in scored),
        "citation_coverage":  sum(r["has_citations"] for r in scored) / len(scored),
        "expected_hit_rate":  sum(r["expected_hit"] for r in scored) / len(scored),
        "must_term_hit_rate": (sum(r["must_hit"] for r in must_rows) / len(must_rows)) if must_rows else None,
        "placeholder_leaks":  sum(r["placeholder_leaks"] for r in scored),
        "empty_answers":      sum(r["empty"] for r in scored),
        "error_rate":         len(errors) / len(rows),
        "latency_p50_s":      lat[len(lat) // 2],
        "latency_p95_s":      lat[max(int(len(lat) * 0.95) - 1, 0)],
    }

    # ── LLM-as-judge (optional) ────────────────────────────────────────
    if use_judge:
        from eval.judge import GeminiJudge, aggregate
        judge = GeminiJudge()
        if not judge.available:
            print("\n(judge requested but unavailable — set GEMINI_API_KEY / install google-genai)")
        else:
            print(f"\nJudging {len(scored)} answers with {judge.model}…")
            verdicts = []
            for r in scored:
                v = judge.score(r["query"], r["answer"], r["contexts"], r["reference"])
                if sleep_s:
                    time.sleep(sleep_s)   # same rate limit as generation
                verdicts.append(v)
                r["judge"] = v
                if v:
                    print(f"  faith={v['faithfulness']} corr={v['correctness']} "
                          f"compl={v['completeness']} cit={v['citation_support']} | {r['id']}")
            metrics.update(aggregate(verdicts))

    # ── Optional RAGAS pass ────────────────────────────────────────────
    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision
        from datasets import Dataset

        ds = Dataset.from_dict({
            "question": [r["query"] for r in scored],
            "answer":   [r["answer"] for r in scored],
            "contexts": [r["contexts"] for r in scored],
        })
        result = evaluate(ds, metrics=[faithfulness, answer_relevancy, context_precision])
        print("\nRAGAS:", result)
        try:
            metrics.update({f"ragas_{k}": float(v) for k, v in result.items()})
        except Exception:
            pass
    except ImportError:
        print("\n(ragas not installed — heuristic/judge scores only. pip install ragas datasets)")

    # ── Summary, gates, report ─────────────────────────────────────────
    print(f"\nMean grounding overlap : {metrics['grounding_mean']:.2f}")
    print(f"Citation coverage      : {metrics['citation_coverage']:.2f}")
    print(f"Expected-term hit rate : {metrics['expected_hit_rate']:.2f}")
    print(f"Placeholder leaks      : {metrics['placeholder_leaks']}")
    print(f"Latency p50 / p95      : {metrics['latency_p50_s']:.1f}s / {metrics['latency_p95_s']:.1f}s\n")

    gates = build_gates("e2e", metrics)
    print_gates(gates)
    warn_regressions("e2e", metrics)
    out = write_report("e2e", metrics, gates, rows=rows)
    print(f"\nDetailed report written to {out}")

    code = exit_code(gates)
    print("RESULT:", "PASS" if code == 0 else "FAIL")
    return code


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://127.0.0.1:8000")
    ap.add_argument("--judge", action="store_true",
                    help="score answers with the Gemini LLM judge (needs GEMINI_API_KEY)")
    ap.add_argument("--sleep", type=float, default=0.0,
                    help="seconds to pause between questions (use ~13 on free-tier Gemini keys: 5 req/min)")
    args = ap.parse_args()
    sys.exit(run(args.api, args.judge, args.sleep))
