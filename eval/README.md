# Evaluation Suite

Industry-grade evaluation harness: every suite gates against configurable
thresholds, writes machine-readable reports, and tracks regressions against a
blessed baseline. One command runs everything:

```
python eval/run_all.py                                   # offline suites
python eval/run_all.py --api http://127.0.0.1:8000 --judge --sleep 13   # + live e2e
```

## Architecture

```
thresholds.json     pass/fail gates per suite — config, not code
golden_set.json     gold questions + reference answers for e2e & judge
baseline.json       blessed metrics; runs warn on regression vs these
common.py           gates, JSON reports, history log, baseline compare
judge.py            Gemini LLM-as-judge (1-5 rubric, retry/backoff)
run_all.py          orchestrator → reports/scorecard.{md,json}
reports/            *_latest.json per suite + history.jsonl  (gitignored)
```

Exit codes everywhere: `0` pass · `1` blocking gate failed (CI fails) · `2`
skipped (missing dependency / backend down).

## Suites, ordered by how often you should run them

### 1. `privacy_eval.py` — masking / leakage (every masking change; CI)

The core guarantee: **zero bank names or PII leave the machine, financial
values survive masking verbatim.** Measures bank-name leak rate, PII leak
rate, financial preservation, egress-firewall catch rate on seeded leaks, and
mask→unmask round-trip fidelity over synthetic credit-document fixtures.
Requires the local tier's env (spaCy + `en_core_web_lg`).

### 2. `adversarial_eval.py` — masking under attack (every masking change; CI)

Attacks the masker + egress firewall with hostile surface forms. A case only
counts as a leak when **both** defense layers miss it. Two severities:

- **must_block** (gate: 0 leaks, blocking) — in-spec forms: case variants,
  bank names inside filenames and markdown tables, standard email/phone/
  SSN/EIN formats.
- **hardening** (catch rate, non-blocking) — out-of-spec obfuscations:
  hyphenated/homoglyph/zero-width bank names, spelled-out emails, spaced
  phone digits, unlisted acronyms. This is the roadmap metric; drive it up
  and ratchet the threshold in `thresholds.json`.

### 3. `retrieval_eval.py` — BM25 vs dense vs hybrid (when tuning retrieval; CI)

Hit@1 / Hit@3 / MRR / nDCG@5 on a labeled mini-corpus, including
keyword-light paraphrases (dense should win) and exact-term queries (BM25
should win). BM25 and RRF run with zero optional dependencies; dense/hybrid
rows appear when sentence-transformers is installed, and their gates
auto-skip when absent.

### 4. `ragas_eval.py` — live-backend answer quality (before release)

Sends `golden_set.json` to the running `/query` endpoint. Three scoring layers:

1. **Heuristics** (always): grounding overlap (lexical faithfulness proxy),
   citation coverage, expected/must-term hit rates, placeholder leaks,
   error/empty rates, p50/p95 latency.
2. **LLM-as-judge** (`--judge`): Gemini scores each answer 1–5 on
   faithfulness, correctness vs the reference answer, completeness, and
   citation support (`judge.py`; model via `EVAL_JUDGE_MODEL` — point it at
   a different family/tier to reduce self-preference bias).
3. **RAGAS** (if `pip install ragas datasets`): faithfulness, answer
   relevancy, context precision.

Free-tier Gemini keys are tightly limited (observed: 5 requests/minute AND
20 requests/day for `gemini-2.5-flash`). Pass `--sleep 13` to pace both
generation and judging; 5xx/429s are retried with backoff regardless. Note a
full run consumes ~14 generation + ~14 judge calls — beyond a 20/day cap, so
on a free key run the e2e suite sparingly (or use a paid-tier key / point
`EVAL_JUDGE_MODEL` at a model with remaining quota).

```
cd cloud/backend && uvicorn app.main:app --port 8000    # terminal 1
python eval/ragas_eval.py --judge --sleep 13            # terminal 2
```

## Thresholds, baselines, regressions

- Gates live in `thresholds.json` (`op` + `value` per metric; add
  `"blocking": false` for informational gates). A suite only gates on
  metrics it actually produced, so optional layers never fail a run by
  being absent.
- After a run you trust: `python eval/run_all.py --update-baseline`
  blesses current metrics. Later runs print `⚠ REGRESSION` when a metric
  degrades beyond tolerance (0.05 absolute for higher-is-better, 25%
  relative for lower-is-better) — informational; gates decide pass/fail.
- Every run appends to `reports/history.jsonl` (timestamp + commit +
  metrics) for trend analysis.

## CI

`.github/workflows/eval.yml` runs the three offline suites (privacy,
adversarial, retrieval) on every PR and push to main, and uploads
`eval/reports/` as an artifact. The e2e/judge suite is deliberately not in
CI — it needs a live backend and API keys; run it locally before release.
