"""
local/analysis/financial_extractor.py

Structured financial data extractor for credit documents.

Runs on the RAW (unmasked) Docling markdown output — before masking — so
metric values are intact.  The masker's financial_patterns freeze these
values in place; this extractor gives them structured form.

Supports three document types that map to the sidebar classification:
  - "Internal Credit Proposal (Memo)"  → credit ratios, facility terms
  - "Corporate Financial Statement"    → P&L, balance sheet metrics
  - "CBUAE Regulatory Framework / Policy" → not extracted (regulatory text)
  - "General Risk Analytics Dossier"   → best-effort extraction of both

Output: FinancialProfile dataclass — stored in session_state["financial_profile"]
"""

import re
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class MetricValue:
    """A single extracted financial metric with provenance."""
    value:      float
    unit:       str           # "%", "x", "AED M", "USD M", etc.
    raw_text:   str           # original string from document
    context:    str           # surrounding sentence for audit trail


@dataclass
class FinancialProfile:
    """
    Structured container for all extracted financial metrics.
    Fields are Optional — not every document contains every metric.
    """
    # Credit ratios
    dscr:           Optional[MetricValue] = None
    icr:            Optional[MetricValue] = None   # Interest Coverage Ratio
    ltv:            Optional[MetricValue] = None
    leverage:       Optional[MetricValue] = None
    tol_atnw:       Optional[float]       = None   # Total Outside Liabilities / Adjusted TNW
    debt_ebitda:    Optional[float]       = None
    roe:            Optional[MetricValue] = None
    roa:            Optional[MetricValue] = None
    current_ratio:  Optional[float]       = None
    quick_ratio:    Optional[float]       = None

    # Facility terms
    facility_amount:    Optional[MetricValue] = None
    facility_tenor:     Optional[str]         = None   # e.g. "5 years"
    interest_rate:      Optional[MetricValue] = None
    collateral_value:   Optional[MetricValue] = None
    ltc:                Optional[float]       = None   # Loan-to-Cost

    # P&L / Balance sheet
    revenue:        Optional[MetricValue] = None
    ebitda:         Optional[MetricValue] = None
    pat:            Optional[MetricValue] = None   # Profit After Tax
    net_worth:      Optional[MetricValue] = None
    total_assets:   Optional[MetricValue] = None
    total_debt:     Optional[MetricValue] = None

    # Capital adequacy (for bank / NBFC statements)
    car:            Optional[float] = None   # Capital Adequacy Ratio %
    tier1_ratio:    Optional[float] = None
    npa_ratio:      Optional[float] = None

    # Credit rating / scoring
    internal_rating:    Optional[str]   = None
    external_rating:    Optional[str]   = None
    pd_estimate:        Optional[float] = None   # PD %
    lgd_estimate:       Optional[float] = None   # LGD %

    # Extraction metadata
    doc_type:           str = "Unknown"
    extraction_warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Flat dict representation for UI rendering and audit logging."""
        out = {}
        for k, v in self.__dict__.items():
            if k in ("extraction_warnings", "doc_type"):
                out[k] = v
            elif isinstance(v, MetricValue):
                out[k] = {"value": v.value, "unit": v.unit, "raw": v.raw_text}
            elif v is not None:
                out[k] = v
        return out

    def has_data(self) -> bool:
        """True if at least one metric was successfully extracted."""
        return any(
            v is not None
            for k, v in self.__dict__.items()
            if k not in ("doc_type", "extraction_warnings")
        )


# ---------------------------------------------------------------------------
# Extraction patterns
# ---------------------------------------------------------------------------

def _mv(value: float, unit: str, raw: str, context: str = "") -> MetricValue:
    return MetricValue(value=value, unit=unit, raw_text=raw, context=context)


def _extract_ratio(text: str, label_patterns: List[str], unit: str = "x") -> Optional[MetricValue]:
    """
    Generic ratio extractor.  Searches for label followed by a numeric value.
    Handles formats: "DSCR: 1.25x", "DSCR of 1.25", "DSCR = 1.25", "1.25x DSCR"
    """
    for label in label_patterns:
        # label then value
        p1 = re.compile(
            rf'\b{label}\b[\s:=–-]*(?:of\s+)?(\d+(?:\.\d+)?)\s?(?:x|times|%)?',
            re.IGNORECASE
        )
        # value then label
        p2 = re.compile(
            rf'(\d+(?:\.\d+)?)\s?(?:x|times|%)?\s*{label}\b',
            re.IGNORECASE
        )
        for pat in (p1, p2):
            m = pat.search(text)
            if m:
                val = float(m.group(1))
                raw = m.group(0).strip()
                # Extract surrounding sentence for context
                start = max(0, m.start() - 80)
                end   = min(len(text), m.end() + 80)
                ctx   = text[start:end].replace("\n", " ").strip()
                return _mv(val, unit, raw, ctx)
    return None


def _extract_currency(text: str, label_patterns: List[str]) -> Optional[MetricValue]:
    """
    Extracts currency values: "AED 250M", "USD 1.2B", "₹ 500 Cr", "500,000,000"
    """
    multipliers = {"m": 1e6, "million": 1e6, "mn": 1e6,
                   "b": 1e9, "billion": 1e9, "bn": 1e9,
                   "k": 1e3, "thousand": 1e3,
                   "cr": 1e7, "crore": 1e7, "lakh": 1e5, "lac": 1e5}
    currency_re = re.compile(
        r'(?:AED|USD|INR|EUR|GBP|₹|＄|\$)\s?'
        r'(\d+(?:,\d{3})*(?:\.\d+)?)\s?(M|B|K|Mn|Bn|Cr|Lakh|Lac|Million|Billion|Thousand)?',
        re.IGNORECASE
    )
    for label in label_patterns:
        label_re = re.compile(rf'\b{label}\b', re.IGNORECASE)
        m_label  = label_re.search(text)
        if not m_label:
            continue
        # Search in a 200-char window around the label
        window = text[max(0, m_label.start()-50): m_label.end()+150]
        m_val  = currency_re.search(window)
        if m_val:
            raw_num = m_val.group(1).replace(",", "")
            mult_str = (m_val.group(2) or "").lower()
            mult    = multipliers.get(mult_str, 1.0)
            val     = float(raw_num) * mult
            # detect currency symbol
            unit    = "AED" if "AED" in window[:m_val.start()+5] else \
                      "USD" if "USD" in window[:m_val.start()+5] else \
                      "INR" if any(c in window[:m_val.start()+5] for c in ("INR", "₹")) else "CCY"
            raw = m_val.group(0).strip()
            ctx_start = max(0, m_label.start() - 60)
            ctx_end   = min(len(text), m_label.end() + 120)
            ctx = text[ctx_start:ctx_end].replace("\n", " ").strip()
            return _mv(val, unit, raw, ctx)
    return None


def _extract_percentage(text: str, label_patterns: List[str]) -> Optional[MetricValue]:
    for label in label_patterns:
        p = re.compile(
            rf'\b{label}\b[\s:=–-]*(?:of\s+)?(\d+(?:\.\d+)?)\s?%',
            re.IGNORECASE
        )
        m = p.search(text)
        if m:
            val = float(m.group(1))
            raw = m.group(0).strip()
            start = max(0, m.start() - 80)
            end   = min(len(text), m.end() + 80)
            ctx   = text[start:end].replace("\n", " ").strip()
            return _mv(val, "%", raw, ctx)
    return None


def _extract_tenor(text: str) -> Optional[str]:
    """Extracts facility tenor: '5-year', '36 months', '3 years'"""
    p = re.compile(
        r'\b(\d+)\s?[-–]?\s?(year|yr|month|months|mo)\b',
        re.IGNORECASE
    )
    m = p.search(text)
    if m:
        num  = m.group(1)
        unit = m.group(2).lower()
        unit = "year" if unit.startswith("y") else "month"
        return f"{num} {unit}{'s' if int(num) != 1 else ''}"
    return None


def _extract_rating(text: str) -> tuple:
    """Returns (internal_rating, external_rating)."""
    internal, external = None, None
    # Internal: "Risk Grade: B+", "Credit Grade A2", "Internal Rating: 3"
    p_int = re.compile(r'(?:risk|credit|internal)\s+(?:grade|rating|score)[\s:–]*([A-Z][A-Z0-9+\-]{0,3}|\d)', re.IGNORECASE)
    m = p_int.search(text)
    if m:
        internal = m.group(1).strip()
    # External: "Moody's: Baa2", "S&P BB+", "CRISIL AA"
    p_ext = re.compile(r"(?:Moody'?s?|S&P|Fitch|CRISIL|ICRA|CARE|BRICKWORK)[\s:–]+([A-Z][a-z0-9+\-]{0,5})", re.IGNORECASE)
    m = p_ext.search(text)
    if m:
        external = m.group(1).strip()
    return internal, external


# ---------------------------------------------------------------------------
# Main extractor class
# ---------------------------------------------------------------------------

class FinancialExtractor:
    """
    Extracts structured financial metrics from Docling markdown output.
    Document-type-aware: different patterns activated based on doc_type.
    """

    def extract(self, raw_text: str, doc_type: str = "Internal Credit Proposal (Memo)") -> FinancialProfile:
        profile = FinancialProfile(doc_type=doc_type)
        warnings: List[str] = []

        if not raw_text or not raw_text.strip():
            warnings.append("Empty document text — no extraction performed.")
            profile.extraction_warnings = warnings
            return profile

        text = raw_text  # full text; pattern search handles multi-line

        # ── Credit ratios (all doc types) ─────────────────────────────

        profile.dscr = _extract_ratio(
            text, ["DSCR", "Debt Service Coverage Ratio", "Debt Service Coverage"]
        )
        profile.icr = _extract_ratio(
            text, ["ICR", "Interest Coverage Ratio", "Interest Coverage", "ISCR"]
        )
        profile.ltv = _extract_ratio(
            text, ["LTV", "Loan[- ]to[- ]Value", "Loan to Value"], unit="%"
        ) or _extract_percentage(text, ["LTV", "Loan[- ]to[- ]Value"])

        profile.leverage = _extract_ratio(
            text, ["Leverage Ratio", "Leverage", "Gearing Ratio", "Gearing"]
        )
        profile.roe = _extract_percentage(text, ["ROE", "Return on Equity"])
        profile.roa = _extract_percentage(text, ["ROA", "Return on Assets"])

        # TOL/ATNW — usually expressed as a plain ratio
        tol = _extract_ratio(text, [r"TOL\s?/\s?ATNW", r"TOL/TNW", r"Total Outside Liabilities"])
        profile.tol_atnw = tol.value if tol else None

        debt_ebitda = _extract_ratio(text, [r"Debt\s?/\s?EBITDA", r"Net Debt\s?/\s?EBITDA"])
        profile.debt_ebitda = debt_ebitda.value if debt_ebitda else None

        curr = _extract_ratio(text, ["Current Ratio"])
        profile.current_ratio = curr.value if curr else None
        quick = _extract_ratio(text, ["Quick Ratio", "Acid[- ]Test Ratio"])
        profile.quick_ratio = quick.value if quick else None

        # ── Facility terms ────────────────────────────────────────────
        if "Credit Proposal" in doc_type or "Dossier" in doc_type:
            profile.facility_amount = _extract_currency(
                text, ["Facility Amount", "Loan Amount", "Credit Limit",
                        "Proposed Facility", "Facility Size", "Sanctioned Amount"]
            )
            profile.collateral_value = _extract_currency(
                text, ["Collateral Value", "Security Value", "Mortgage Value",
                        "Collateral", "Primary Security"]
            )
            profile.tenor = _extract_tenor(text)
            profile.interest_rate = _extract_percentage(
                text, ["Interest Rate", "Lending Rate", "Spread", "Margin",
                        "Rate of Interest", "ROI"]
            )
            ltc = _extract_ratio(text, ["LTC", "Loan[- ]to[- ]Cost"])
            profile.ltc = ltc.value if ltc else None

        # ── P&L / Balance sheet ───────────────────────────────────────
        if "Financial Statement" in doc_type or "Dossier" in doc_type or "Credit Proposal" in doc_type:
            profile.revenue = _extract_currency(
                text, ["Revenue", "Turnover", "Net Sales", "Total Revenue",
                        "Total Income", "Net Revenue", "Sales"]
            )
            profile.ebitda = _extract_currency(
                text, ["EBITDA", "Operating Profit", "PBDIT", "EBIT"]
            )
            profile.pat = _extract_currency(
                text, ["PAT", "Profit After Tax", "Net Profit", "Net Income",
                        "Profit for the Year", "Net Earnings"]
            )
            profile.net_worth = _extract_currency(
                text, ["Net Worth", "Shareholders.? Equity", "Stockholders.? Equity",
                        "TNW", "Tangible Net Worth", "Equity"]
            )
            profile.total_assets = _extract_currency(
                text, ["Total Assets", "Balance Sheet Size", "Total Balance Sheet"]
            )
            profile.total_debt = _extract_currency(
                text, ["Total Debt", "Total Borrowings", "Total Liabilities",
                        "Outstanding Debt", "Gross Debt"]
            )

        # ── Capital adequacy (banks / NBFCs) ─────────────────────────
        car = _extract_percentage(
            text, ["CAR", "Capital Adequacy Ratio", "CRAR",
                   "Capital to Risk[- ]Weighted Assets Ratio"]
        )
        profile.car = car.value if car else None

        tier1 = _extract_percentage(text, ["Tier[- ]?1 Ratio", "CET1", "Common Equity Tier 1"])
        profile.tier1_ratio = tier1.value if tier1 else None

        npa = _extract_percentage(text, ["NPA Ratio", "GNPA", "NNPA", "Non[- ]Performing Assets"])
        profile.npa_ratio = npa.value if npa else None

        # ── Ratings ───────────────────────────────────────────────────
        profile.internal_rating, profile.external_rating = _extract_rating(text)

        # ── PD / LGD estimates ────────────────────────────────────────
        pd = _extract_percentage(text, ["PD", "Probability of Default", "PD Estimate"])
        profile.pd_estimate = pd.value if pd else None
        lgd = _extract_percentage(text, ["LGD", "Loss Given Default", "LGD Estimate"])
        profile.lgd_estimate = lgd.value if lgd else None

        # ── Warnings for missing critical metrics ─────────────────────
        if "Credit Proposal" in doc_type:
            for metric, name in [
                (profile.dscr,           "DSCR"),
                (profile.facility_amount, "Facility Amount"),
                (profile.ltv,            "LTV"),
            ]:
                if metric is None:
                    warnings.append(f"{name} not found — may need manual input.")

        profile.extraction_warnings = warnings

        found = [k for k, v in profile.to_dict().items()
                 if v is not None and k not in ("doc_type", "extraction_warnings")]
        logger.info(
            "FinancialExtractor: doc_type='%s', extracted %d metrics: %s",
            doc_type, len(found), found
        )
        return profile
