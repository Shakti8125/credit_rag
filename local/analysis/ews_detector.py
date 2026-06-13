"""
local/analysis/ews_detector.py

Early Warning Signal (EWS) detector for credit documents.

Scans raw (pre-mask) document text for signals that correlate with credit
deterioration across three categories:

  FINANCIAL   — quantitative: declining ratios, margin compression,
                negative cashflow, covenant proximity
  QUALITATIVE — language: going concern, auditor qualifications,
                litigation, management changes, covenant waivers
  STRUCTURAL  — concentration risk, refinancing risk, related-party
                transactions, off-balance-sheet exposures

Each signal carries a verbatim excerpt from the document that triggered it
so every finding is fully auditable.

Output: EWSReport dataclass — stored in session_state["ews_report"]
"""

import re
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class EWSSignal:
    category:  str          # FINANCIAL | QUALITATIVE | STRUCTURAL
    severity:  str          # HIGH | MEDIUM | LOW
    signal:    str          # short label
    detail:    str          # plain-English explanation
    excerpt:   str          # verbatim snippet (≤200 chars)
    source:    str = "Document"


@dataclass
class EWSReport:
    signals:      List[EWSSignal] = field(default_factory=list)
    high_count:   int = 0
    medium_count: int = 0
    low_count:    int = 0
    risk_level:   str = "CLEAR"   # HIGH | MEDIUM | LOW | CLEAR
    doc_type:     str = "Unknown"

    def summary(self) -> str:
        if self.risk_level == "CLEAR":
            return "🟢 No early warning signals detected"
        parts = []
        if self.high_count:
            parts.append(f"🔴 {self.high_count} HIGH")
        if self.medium_count:
            parts.append(f"🟡 {self.medium_count} MEDIUM")
        if self.low_count:
            parts.append(f"🔵 {self.low_count} LOW")
        return " · ".join(parts)

    def has_signals(self) -> bool:
        return bool(self.signals)

    def to_dict(self) -> dict:
        return {
            "risk_level":   self.risk_level,
            "summary":      self.summary(),
            "high_count":   self.high_count,
            "medium_count": self.medium_count,
            "low_count":    self.low_count,
            "signals":      [s.__dict__ for s in self.signals],
        }


# ---------------------------------------------------------------------------
# Qualitative pattern library
# ---------------------------------------------------------------------------

# Each tuple: (signal_label, severity, explanation, [regex_patterns])
_QUALITATIVE_PATTERNS: List[Tuple[str, str, str, List[str]]] = [

    ("Going Concern Language", "HIGH",
     "Document contains going concern language — auditor or management doubts about the entity's ability to continue as a going concern.",
     [r'\bgoing\s+concern\b', r'\bability\s+to\s+continue\b',
      r'\bsubstantial\s+doubt\b', r'\bviability\s+of\s+the\s+(company|entity|business|group)\b']),

    ("Auditor Qualification / Emphasis of Matter", "HIGH",
     "Qualified audit opinion or emphasis of matter detected — signals material uncertainty or departure from accounting standards.",
     [r'\bqualified\s+opinion\b', r'\bemphasis\s+of\s+matter\b',
      r'\bmaterial\s+(uncertainty|misstatement|weakness)\b',
      r'\bmodified\s+opinion\b', r'\badverse\s+opinion\b', r'\bdisclaimer\s+of\s+opinion\b']),

    ("Covenant Waiver Request / Breach", "HIGH",
     "Covenant waiver, breach, or amendment mentioned — borrower has violated or is at risk of violating loan terms.",
     [r'\bcovenant\s+(waiver|breach|violation|default|non[-\s]?compliance)\b',
      r'\bwaiver\s+of\s+(financial\s+)?covenant\b',
      r'\bcovenant\s+(amendment|reset|restructur)\b']),

    ("Material Litigation / Legal Proceedings", "HIGH",
     "Material litigation or legal proceedings identified — potential contingent liability impairing repayment capacity.",
     [r'\bmaterial\s+(litigation|lawsuit|legal\s+proceedings)\b',
      r'\bpending\s+(litigation|lawsuit|arbitration)\b',
      r'\bcontingent\s+liabilit\b',
      r'\bregulatory\s+(enforcement|investigation|sanction)\b']),

    ("Negative Operating Cashflow", "HIGH",
     "Negative operating cashflow — business is not generating sufficient cash from operations.",
     [r'\bnegative\s+(?:operating\s+)?cash\s*flow\b',
      r'\bcash\s*flow\s+(?:from\s+operations?\s+)?(?:is\s+)?negative\b',
      r'\bcash\s+burn\b']),

    ("Key Management / Ownership Change", "MEDIUM",
     "Significant management departure or ownership change — key man risk or strategic uncertainty.",
     [r'\b(CEO|CFO|MD|Managing\s+Director).{0,50}(resign|depart|step\s+down|replac)\b',
      r'\bchange\s+(of|in)\s+(ownership|shareholding|promoter)\b',
      r'\bpromoter.{0,40}(pledg|encumber|sell|divest)\b']),

    ("Significant Related Party Transactions", "MEDIUM",
     "Large or unusual related party transactions — potential fund diversion or conflict of interest.",
     [r'\brelated\s+party\s+transaction.{0,60}(significant|material|large)\b',
      r'\b(loan|advance|guarantee).{0,60}related\s+part\b',
      r'\bfund\s+(diversion|siphon|transfer).{0,40}(group|related|associate)\b']),

    ("Refinancing / Rollover Risk", "MEDIUM",
     "Upcoming debt maturity without clear refinancing plan, or reliance on short-term funding for long-term assets.",
     [r'\brefinanc.{0,60}(risk|concern|challenge|difficult)\b',
      r'\brollover\s+risk\b', r'\bdebt\s+maturit.{0,40}(upcoming|imminent)\b',
      r'\bliquidit.{0,40}(concern|pressure|constraint|stress)\b']),

    ("Revenue Concentration Risk", "MEDIUM",
     "High dependence on a single customer, sector, or geography — sharp revenue decline risk.",
     [r'\b(\d{2,3})\s?%\s+(?:of\s+)?(?:revenue|sales).{0,60}(?:single|one|top)\s+customer\b',
      r'\bcustomer\s+concentration\b', r'\bsector\s+concentration\b']),

    ("Off-Balance Sheet Exposure", "MEDIUM",
     "Material off-balance sheet commitments — may represent undisclosed obligations.",
     [r'\boff[-\s]?balance\s+sheet\b',
      r'\bcontingent\s+(liabilit|obligation).{0,40}(material|significant)\b',
      r'\bspecial\s+purpose\s+(vehicle|entity|SPV)\b']),

    ("Material Asset Impairment / Write-off", "MEDIUM",
     "Material impairment charges or asset write-offs — deteriorating asset quality.",
     [r'\bimpairment\s+(charge|loss|write[-\s]?down).{0,40}(material|significant)\b',
      r'\bgoodwill\s+impairment\b',
      r'\bprovision.{0,40}(material|significant|increas)\b']),

    ("Dividend Restriction / Cash Trap", "LOW",
     "Dividend restriction or cash trap covenant — restricted upstream cash flow.",
     [r'\bdividend\s+(restriction|block|trap|moratorium)\b',
      r'\bcash\s+trap\b', r'\brestricted\s+(payment|distribution)\b']),

    ("Sector / Market Stress Language", "LOW",
     "Document acknowledges sector-level stress or adverse market conditions.",
     [r'\bsector.{0,40}(headwind|stress|downturn|pressure)\b',
      r'\bindustry.{0,40}(headwind|stress|downturn|slow)\b',
      r'\bsupply\s+chain\s+(disruption|constraint)\b']),
]

# ---------------------------------------------------------------------------
# Financial ratio EWS thresholds
# ---------------------------------------------------------------------------

# (metric_key, label, threshold, direction, severity, explanation)
_FINANCIAL_EWS = [
    ("dscr",         "DSCR",         1.10, "min", "HIGH",
     "DSCR below 1.10x — insufficient cashflow to service debt; imminent default risk."),
    ("dscr",         "DSCR",         1.25, "min", "MEDIUM",
     "DSCR below 1.25x policy minimum — debt service coverage is tightening."),
    ("icr",          "ICR",          1.25, "min", "HIGH",
     "ICR below 1.25x — earnings barely cover interest; vulnerable to rate increases."),
    ("ltv",          "LTV",          85.0, "max", "HIGH",
     "LTV exceeds 85% — collateral cover critically thin; high LGD exposure."),
    ("ltv",          "LTV",          75.0, "max", "MEDIUM",
     "LTV above 75% policy maximum — collateral coverage below standard threshold."),
    ("debt_ebitda",  "Debt/EBITDA",   5.0, "max", "HIGH",
     "Debt/EBITDA above 5x — leverage in distressed territory; refinancing risk elevated."),
    ("debt_ebitda",  "Debt/EBITDA",   4.0, "max", "MEDIUM",
     "Debt/EBITDA above 4x — exceeds Basel leveraged lending guidance."),
    ("tol_atnw",     "TOL/ATNW",      4.0, "max", "HIGH",
     "TOL/ATNW above 4x — total outside liabilities critically high relative to net worth."),
    ("current_ratio","Current Ratio", 0.9, "min", "HIGH",
     "Current ratio below 0.9x — short-term liabilities exceed assets; liquidity stress."),
    ("npa_ratio",    "NPA Ratio",     7.0, "max", "HIGH",
     "NPA ratio above 7% — asset quality severely impaired; provisioning may be inadequate."),
    ("car",          "CAR",           9.0, "min", "HIGH",
     "CAR below 9% — capital adequacy below Basel III minimum; regulatory intervention risk."),
]


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _find_excerpt(text: str, pattern: re.Pattern, window: int = 200) -> str:
    m = pattern.search(text)
    if not m:
        return ""
    start = max(0, m.start() - 40)
    end   = min(len(text), m.end() + 160)
    return text[start:end].replace("\n", " ").strip()[:window]


def _get_float(profile, key: str):
    from local.analysis.financial_extractor import MetricValue
    val = getattr(profile, key, None)
    if val is None:
        return None
    if isinstance(val, MetricValue):
        return val.value
    if isinstance(val, (int, float)):
        return float(val)
    return None


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class EarlyWarningDetector:
    """
    Scans a credit document for early warning signals across three categories.
    Runs on raw (pre-mask) text so language patterns are intact.
    """

    def detect(
        self,
        raw_text:          str,
        financial_profile=None,
        doc_type:          str = "Internal Credit Proposal (Memo)",
    ) -> EWSReport:

        report = EWSReport(doc_type=doc_type)

        if not raw_text or not raw_text.strip():
            report.risk_level = "CLEAR"
            return report

        # ── 1. Financial ratio signals ────────────────────────────────
        if financial_profile is not None:
            seen: set = set()
            for (key, label, threshold, direction, severity, explanation) in _FINANCIAL_EWS:
                cache_key = (key, direction)
                if cache_key in seen:
                    continue
                val = _get_float(financial_profile, key)
                if val is None:
                    continue
                triggered = (
                    (direction == "min" and val < threshold) or
                    (direction == "max" and val > threshold)
                )
                if triggered:
                    seen.add(cache_key)
                    report.signals.append(EWSSignal(
                        category = "FINANCIAL",
                        severity = severity,
                        signal   = f"{label} Deterioration",
                        detail   = explanation,
                        excerpt  = f"Extracted value: {val:.2f} (threshold: {threshold})",
                        source   = "Financial Extraction",
                    ))

        # ── 2. Qualitative language signals ───────────────────────────
        for (signal_label, severity, explanation, patterns) in _QUALITATIVE_PATTERNS:
            for pattern_str in patterns:
                pat = re.compile(pattern_str, re.IGNORECASE | re.MULTILINE)
                if pat.search(raw_text):
                    excerpt = _find_excerpt(raw_text, pat)
                    report.signals.append(EWSSignal(
                        category = "QUALITATIVE",
                        severity = severity,
                        signal   = signal_label,
                        detail   = explanation,
                        excerpt  = excerpt,
                        source   = "Document Language",
                    ))
                    break  # one signal per label

        # ── 3. Deduplicate — keep highest severity per label ──────────
        sev_rank = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
        best: dict = {}
        for sig in report.signals:
            if sig.signal not in best or sev_rank[sig.severity] > sev_rank[best[sig.signal].severity]:
                best[sig.signal] = sig

        deduped = sorted(best.values(), key=lambda s: (-sev_rank[s.severity], s.category))
        report.signals      = deduped
        report.high_count   = sum(1 for s in deduped if s.severity == "HIGH")
        report.medium_count = sum(1 for s in deduped if s.severity == "MEDIUM")
        report.low_count    = sum(1 for s in deduped if s.severity == "LOW")

        if report.high_count >= 2:
            report.risk_level = "HIGH"
        elif report.high_count == 1 or report.medium_count >= 3:
            report.risk_level = "MEDIUM"
        elif report.medium_count > 0 or report.low_count > 0:
            report.risk_level = "LOW"
        else:
            report.risk_level = "CLEAR"

        logger.info(
            "EWS: %s | Risk=%s | HIGH=%d MEDIUM=%d LOW=%d | %d signals",
            doc_type, report.risk_level,
            report.high_count, report.medium_count, report.low_count, len(deduped),
        )
        return report
