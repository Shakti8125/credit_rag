"""
eval/run_all.py — One-command evaluation harness with a unified scorecard.

Runs every suite as a subprocess (so a missing optional dependency skips
that suite cleanly instead of killing the run), then aggregates the JSON
reports each suite wrote into a single scorecard:

    eval/reports/scorecard.md      human-readable summary
    eval/reports/scorecard.json    machine-readable (CI artifacts, dashboards)

Suites:
    privacy      masking leakage on well-formed docs        (offline, blocking)
    adversarial  masking robustness under attack            (offline, blocking)
    retrieval    BM25 / dense / hybrid ranking quality      (offline, blocking)
    e2e          live-backend answer quality (+ LLM judge)  (needs --api)

Usage:
    python eval/run_all.py                                  # offline suites
    python eval/run_all.py --api http://127.0.0.1:8000      # + e2e
    python eval/run_all.py --api ... --judge                # + LLM-as-judge
    python eval/run_all.py --update-baseline                # bless current metrics

Exit code: 1 if any suite that RAN failed a blocking gate; skipped suites
(exit 2) are reported but don't fail the harness.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

EVAL_DIR     = Path(__file__).resolve().parent
PROJECT_ROOT = EVAL_DIR.parent
REPORTS_DIR  = EVAL_DIR / "reports"

sys.path.insert(0, str(PROJECT_ROOT))
from eval.common import update_baseline  # noqa: E402


def _run_suite(script: str, extra_args=None) -> int:
    print(f"\n{'=' * 70}\n>>> {script} {' '.join(extra_args or [])}\n{'=' * 70}")
    proc = subprocess.run(
        [sys.executable, str(EVAL_DIR / script), *(extra_args or [])],
        cwd=PROJECT_ROOT,
    )
    return proc.returncode


def _load_report(suite: str):
    p = REPORTS_DIR / f"{suite}_latest.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return None


_STATUS = {0: "PASS", 1: "FAIL", 2: "SKIP"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default=None,
                    help="backend base URL — enables the e2e suite")
    ap.add_argument("--judge", action="store_true",
                    help="LLM-as-judge scoring in the e2e suite")
    ap.add_argument("--sleep", type=float, default=0.0,
                    help="pause between e2e questions (use ~13 on free-tier Gemini keys)")
    ap.add_argument("--update-baseline", action="store_true",
                    help="bless the metrics of every passing suite as the new baseline")
    args = ap.parse_args()

    t0 = time.time()
    runs = [
        ("privacy",     _run_suite("privacy_eval.py")),
        ("adversarial", _run_suite("adversarial_eval.py")),
        ("retrieval",   _run_suite("retrieval_eval.py")),
    ]
    if args.api:
        e2e_args = ["--api", args.api] + (["--judge"] if args.judge else [])
        if args.sleep:
            e2e_args += ["--sleep", str(args.sleep)]
        runs.append(("e2e", _run_suite("ragas_eval.py", e2e_args)))

    # ── Scorecard ──────────────────────────────────────────────────────
    lines_md = [
        "# CreditRAG Evaluation Scorecard",
        "",
        f"Run: {time.strftime('%Y-%m-%d %H:%M:%S')}  ·  {time.time() - t0:.0f}s total",
        "",
        "| Suite | Status | Key metrics |",
        "|---|---|---|",
    ]
    summary = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "suites": {}}

    print(f"\n{'=' * 70}\n SCORECARD\n{'=' * 70}")
    for suite, code in runs:
        status = _STATUS.get(code, f"ERR({code})")
        report = _load_report(suite) if code != 2 else None
        key_metrics = ""
        if report:
            shown = [
                f"{g['metric']}={g['value']:.2f}"
                for g in report.get("gates", []) if g.get("blocking", True)
            ][:5]
            key_metrics = ", ".join(shown)
            summary["suites"][suite] = {
                "status": status, "metrics": report["metrics"],
                "commit": report.get("commit"),
            }
        else:
            summary["suites"][suite] = {"status": status}
        print(f"  {suite:<12} {status:<6} {key_metrics}")
        lines_md.append(f"| {suite} | {status} | {key_metrics} |")

        if args.update_baseline and report and code == 0:
            update_baseline(suite, report["metrics"])

    if not args.api:
        note = "e2e suite not run (pass --api http://127.0.0.1:8000 with the backend up)."
        print(f"\n  note: {note}")
        lines_md += ["", f"> {note}"]

    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "scorecard.md").write_text("\n".join(lines_md) + "\n", encoding="utf-8")
    (REPORTS_DIR / "scorecard.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"\nScorecard written to {REPORTS_DIR / 'scorecard.md'}")

    failed = [s for s, c in runs if c == 1]
    if failed:
        print(f"OVERALL: FAIL ({', '.join(failed)})")
        return 1
    print("OVERALL: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
