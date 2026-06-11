"""
local/analysis/policy_checker.py

Policy breach checker — compares a FinancialProfile against a configurable
threshold registry and returns a BreachReport without any user prompt.

Threshold sources:
  - CBUAE Circular 33/2023 (retail and corporate lending limits)
  - Basel III Pillar 1 minimum capital ratios
  - Generic credit underwriting policy (conservative bank norms)
  - Configurable overrides per doc_type

Severity levels:
  BREACH  — value is outside the hard regulatory / policy limit
  WARNING — value is within policy but approaching a threshold (within 10%)
  PASS    — value meets or exceeds requirements
  N/A     — metric was not extracted; cannot assess
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from local.analysis.financial_extractor import FinancialProfile, MetricValue

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class PolicyFinding:
    metric:        str
    extracted:     Optional[float]   # None if N/A
    unit:          str
    threshold:     float
    threshold_dir: str   # "min" (value must be >= threshold) or "max" (value must be <= threshold)
    severity:      str   # BREACH | WARNING | PASS | N/A
    source:        str   # regulatory reference
    message:       str   # plain-English explanation


@dataclass
class BreachReport:
    findings:       List[PolicyFinding] = field(default_factory=list)
    breach_count:   int = 0
    warning_count:  int = 0
    pass_count:     int = 0
    na_count:       int = 0
    doc_type:       str = "Unknown"

    def summary(self) -> str:
        parts = []
        if self.breach_count:
            parts.append(f"🔴 {self.breach_count} BREACH")
        if self.warning_count:
            parts.append(f"🟡 {self.warning_count} WARNING")
        if self.pass_count:
            parts.append(f"🟢 {self.pass_count} PASS")
        return " · ".join(parts) if parts else "No assessable metrics"

    def has_issues(self) -> bool:
        return self.breach_count > 0 or self.warning_count > 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary":       self.summary(),
            "breach_count":  self.breach_count,
            "warning_count": self.warning_count,
            "pass_count":    self.pass_count,
            "na_count":      self.na_count,
            "findings":      [f.__dict__ for f in self.findings],
        }


# ---------------------------------------------------------------------------
# Threshold registry
# ---------------------------------------------------------------------------

# Structure: (metric_key, display_name, threshold, direction, unit, source, warning_buffer_pct)
# warning_buffer_pct: if value is within this % of the threshold, flag as WARNING
_CREDIT_MEMO_THRESHOLDS = [
    # DSCR — minimum 1.25x per CBUAE and standard credit policy
    ("dscr",          "DSCR",                   1.25,  "min", "x",   "CBUAE Circular 33/2023 & Standard Credit Policy",   10),
    # ICR — minimum 1.5x
    ("icr",           "ICR",                    1.50,  "min", "x",   "Standard Credit Policy (ICR ≥ 1.5x)",               10),
    # LTV — maximum 75% for commercial RE per CBUAE
    ("ltv",           "LTV",                    75.0,  "max", "%",   "CBUAE Mortgage Regulations (LTV ≤ 75%)",            10),
    # Leverage — maximum 4x Debt/EBITDA
    ("debt_ebitda",   "Debt / EBITDA",           4.0,   "max", "x",   "Basel III Leveraged Lending Guidance (≤ 4x)",       10),
    # TOL/ATNW — maximum 3x
    ("tol_atnw",      "TOL / ATNW",              3.0,   "max", "x",   "Standard Credit Policy (TOL/ATNW ≤ 3x)",           10),
    # LTC — maximum 80%
    ("ltc",           "Loan-to-Cost (LTC)",      80.0,  "max", "%",   "Project Finance Policy (LTC ≤ 80%)",               10),
    # Current ratio — minimum 1.0x
    ("current_ratio", "Current Ratio",           1.0,   "min", "x",   "Standard Credit Policy (Current Ratio ≥ 1.0x)",    10),
]

_FINANCIAL_STATEMENT_THRESHOLDS = [
    ("current_ratio", "Current Ratio",           1.0,   "min", "x",   "Standard Credit Policy (Current Ratio ≥ 1.0x)",    10),
    ("quick_ratio",   "Quick Ratio",             0.75,  "min", "x",   "Conservative Credit Norm (Quick Ratio ≥ 0.75x)",   10),
    ("roe",           "Return on Equity (ROE)",  8.0,   "min", "%",   "Minimum Acceptable ROE ≥ 8%",                      15),
    ("debt_ebitda",   "Debt / EBITDA",           5.0,   "max", "x",   "Basel III Leveraged Lending (≤ 5x for corporates)",10),
]

_BASEL_THRESHOLDS = [
    ("car",          "Capital Adequacy Ratio",  10.5,  "min", "%",   "Basel III Pillar 1 (Total CAR ≥ 10.5%)",            5),
    ("tier1_ratio",  "Tier 1 Ratio",             8.5,  "min", "%",   "Basel III Pillar 1 (Tier 1 ≥ 8.5%)",               5),
    ("npa_ratio",    "NPA Ratio",                5.0,  "max", "%",   "Regulatory Concern Threshold (NPA > 5%)",           10),
]

_THRESHOLD_MAP = {
    "Internal Credit Proposal (Memo)":         _CREDIT_MEMO_THRESHOLDS + _BASEL_THRESHOLDS,
    "Corporate Financial Statement":           _FINANCIAL_STATEMENT_THRESHOLDS + _BASEL_THRESHOLDS,
    "CBUAE Regulatory Framework / Policy":     _BASEL_THRESHOLDS,
    "General Risk Analytics Dossier":          _CREDIT_MEMO_THRESHOLDS + _FINANCIAL_STATEMENT_THRESHOLDS + _BASEL_THRESHOLDS,
}


# ---------------------------------------------------------------------------
# Checker
# ---------------------------------------------------------------------------

def _get_float(profile: FinancialProfile, key: str) -> Optional[float]:
    """Safely extracts a float value from FinancialProfile regardless of field type."""
    val = getattr(profile, key, None)
    if val is None:
        return None
    if isinstance(val, MetricValue):
        return val.value
    if isinstance(val, (int, float)):
        return float(val)
    return None


class PolicyChecker:
    """
    Compares a FinancialProfile against the regulatory/policy threshold registry.
    Returns a BreachReport proactively — no user query required.
    """

    def check(self, profile: FinancialProfile) -> BreachReport:
        doc_type  = profile.doc_type or "Internal Credit Proposal (Memo)"
        thresholds = _THRESHOLD_MAP.get(doc_type, _CREDIT_MEMO_THRESHOLDS)

        report = BreachReport(doc_type=doc_type)

        for (metric_key, display_name, threshold, direction,
             unit, source, warn_buf_pct) in thresholds:

            extracted = _get_float(profile, metric_key)

            if extracted is None:
                finding = PolicyFinding(
                    metric=display_name,
                    extracted=None,
                    unit=unit,
                    threshold=threshold,
                    threshold_dir=direction,
                    severity="N/A",
                    source=source,
                    message=f"{display_name} not extracted from document — manual verification required.",
                )
                report.na_count += 1
            else:
                # Determine severity
                if direction == "min":
                    if extracted < threshold:
                        severity = "BREACH"
                        msg = (
                            f"{display_name} of {extracted:.2f}{unit} is BELOW the minimum "
                            f"requirement of {threshold}{unit} per {source}."
                        )
                    elif extracted < threshold * (1 + warn_buf_pct / 100):
                        severity = "WARNING"
                        msg = (
                            f"{display_name} of {extracted:.2f}{unit} meets the minimum "
                            f"({threshold}{unit}) but is within {warn_buf_pct}% of the threshold — monitor closely."
                        )
                    else:
                        severity = "PASS"
                        msg = f"{display_name} of {extracted:.2f}{unit} meets the requirement of ≥ {threshold}{unit}."
                else:  # max
                    if extracted > threshold:
                        severity = "BREACH"
                        msg = (
                            f"{display_name} of {extracted:.2f}{unit} EXCEEDS the maximum "
                            f"limit of {threshold}{unit} per {source}."
                        )
                    elif extracted > threshold * (1 - warn_buf_pct / 100):
                        severity = "WARNING"
                        msg = (
                            f"{display_name} of {extracted:.2f}{unit} is within {warn_buf_pct}% of the "
                            f"maximum limit of {threshold}{unit} — elevated risk level."
                        )
                    else:
                        severity = "PASS"
                        msg = f"{display_name} of {extracted:.2f}{unit} is within the limit of ≤ {threshold}{unit}."

                finding = PolicyFinding(
                    metric=display_name,
                    extracted=extracted,
                    unit=unit,
                    threshold=threshold,
                    threshold_dir=direction,
                    severity=severity,
                    source=source,
                    message=msg,
                )

                if severity == "BREACH":
                    report.breach_count += 1
                elif severity == "WARNING":
                    report.warning_count += 1
                else:
                    report.pass_count += 1

            report.findings.append(finding)

        logger.info(
            "PolicyChecker: %s | Breaches=%d Warnings=%d Passes=%d N/A=%d",
            doc_type, report.breach_count, report.warning_count,
            report.pass_count, report.na_count,
        )
        return report
