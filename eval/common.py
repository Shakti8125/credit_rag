"""
eval/common.py — shared evaluation infrastructure.

Gives every suite the same industrial plumbing:

  - Thresholds registry  (eval/thresholds.json — gates live in config, not code)
  - Gate objects         (metric, op, threshold, blocking / non-blocking)
  - Machine-readable reports
        eval/reports/<suite>_latest.json   full run detail (metrics, gates, rows)
        eval/reports/history.jsonl         append-only run log for trend analysis
  - Baseline regression  (eval/baseline.json — warns when a metric degrades
        beyond tolerance vs the last blessed run; update via run_all.py
        --update-baseline)

A suite calls:

    metrics = {"grounding_mean": 0.71, ...}
    gates   = build_gates("e2e", metrics)
    print_gates(gates)
    warn_regressions("e2e", metrics)
    write_report("e2e", metrics, gates, rows=per_case_rows)
    sys.exit(exit_code(gates))

Exit-code convention across all suites:
    0 = all blocking gates passed
    1 = at least one blocking gate failed  (CI must fail)
    2 = suite skipped — missing dependency or unreachable backend
"""

import json
import subprocess
import sys
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# Windows consoles default to cp1252, which chokes on the unicode content the
# adversarial suite deliberately prints (homoglyphs, zero-width chars).
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

EVAL_DIR        = Path(__file__).resolve().parent
PROJECT_ROOT    = EVAL_DIR.parent
REPORTS_DIR     = EVAL_DIR / "reports"
THRESHOLDS_FILE = EVAL_DIR / "thresholds.json"
BASELINE_FILE   = EVAL_DIR / "baseline.json"

_OPS = {
    ">=": lambda v, t: v >= t,
    "<=": lambda v, t: v <= t,
    "==": lambda v, t: v == t,
}


@dataclass
class Gate:
    metric:    str
    value:     float
    op:        str
    threshold: float
    passed:    bool
    blocking:  bool = True

    def line(self) -> str:
        mark = "PASS" if self.passed else ("FAIL" if self.blocking else "warn")
        soft = "" if self.blocking else " (non-blocking)"
        return f"  [{mark}] {self.metric:<24} {self.value:.3f}  target {self.op} {self.threshold}{soft}"


# ---------------------------------------------------------------------------
# Thresholds & gates
# ---------------------------------------------------------------------------

def load_thresholds(suite: str) -> Dict[str, Dict[str, Any]]:
    if not THRESHOLDS_FILE.exists():
        return {}
    all_t = json.loads(THRESHOLDS_FILE.read_text(encoding="utf-8"))
    return all_t.get(suite, {})


def build_gates(suite: str, metrics: Dict[str, float]) -> List[Gate]:
    """
    One gate per thresholded metric that the suite actually produced.
    Optional metrics (judge scores, dense-retrieval rows) simply don't gate
    when absent — no special-casing needed in suite code.
    """
    gates: List[Gate] = []
    for name, spec in load_thresholds(suite).items():
        if name not in metrics or metrics[name] is None:
            continue
        value     = float(metrics[name])
        op        = spec.get("op", ">=")
        threshold = float(spec.get("value", 0))
        gates.append(Gate(
            metric=name,
            value=value,
            op=op,
            threshold=threshold,
            passed=_OPS[op](value, threshold),
            blocking=bool(spec.get("blocking", True)),
        ))
    return gates


def print_gates(gates: List[Gate]) -> None:
    for g in gates:
        print(g.line())


def exit_code(gates: List[Gate]) -> int:
    return 0 if all(g.passed for g in gates if g.blocking) else 1


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=PROJECT_ROOT, timeout=10,
        ).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def write_report(
    suite:   str,
    metrics: Dict[str, Any],
    gates:   List[Gate],
    rows:    Optional[List[Dict[str, Any]]] = None,
) -> Path:
    REPORTS_DIR.mkdir(exist_ok=True)
    passed = exit_code(gates) == 0
    report = {
        "suite":     suite,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "commit":    _git_commit(),
        "passed":    passed,
        "metrics":   metrics,
        "gates":     [asdict(g) for g in gates],
        "rows":      rows or [],
    }
    out = REPORTS_DIR / f"{suite}_latest.json"
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    # Append-only history (metrics only — keeps the file small enough to trend)
    with (REPORTS_DIR / "history.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({
            "suite": suite, "timestamp": report["timestamp"],
            "commit": report["commit"], "passed": passed, "metrics": metrics,
        }, default=str) + "\n")
    return out


# ---------------------------------------------------------------------------
# Baseline regression
# ---------------------------------------------------------------------------

# Absolute drop allowed for higher-is-better metrics; relative rise allowed
# for lower-is-better metrics (latency etc.) before a regression is flagged.
_ABS_TOLERANCE = 0.05
_REL_TOLERANCE = 0.25


def _load_baseline() -> Dict[str, Dict[str, float]]:
    if not BASELINE_FILE.exists():
        return {}
    return json.loads(BASELINE_FILE.read_text(encoding="utf-8"))


def warn_regressions(suite: str, metrics: Dict[str, Any]) -> List[str]:
    """
    Compare against eval/baseline.json. Direction comes from the threshold op:
    '>=' metrics regress when they drop, '<=' metrics regress when they rise.
    Informational — regressions warn, gates decide pass/fail.
    """
    base = _load_baseline().get(suite, {})
    if not base:
        return []
    specs = load_thresholds(suite)
    regressions = []
    for name, old in base.items():
        new = metrics.get(name)
        if new is None or not isinstance(old, (int, float)):
            continue
        op = specs.get(name, {}).get("op", ">=")
        if op == "<=":
            allowed = abs(old) * _REL_TOLERANCE
            if new > old + max(allowed, 1e-9):
                regressions.append(f"{name}: {old:.3f} → {new:.3f} (worse)")
        else:
            if new < old - _ABS_TOLERANCE:
                regressions.append(f"{name}: {old:.3f} → {new:.3f} (worse)")
    if regressions:
        print("\n  ⚠ REGRESSION vs baseline:")
        for r in regressions:
            print(f"    {r}")
    return regressions


def update_baseline(suite: str, metrics: Dict[str, Any]) -> None:
    base = _load_baseline()
    base[suite] = {
        k: v for k, v in metrics.items() if isinstance(v, (int, float))
    }
    BASELINE_FILE.write_text(json.dumps(base, indent=2), encoding="utf-8")
    print(f"  Baseline updated for suite '{suite}'.")
