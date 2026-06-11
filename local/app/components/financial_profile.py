"""
local/app/components/financial_profile.py

Renders the structured FinancialProfile extracted from an uploaded document.
Displayed automatically in the right column after document ingestion.
"""

import streamlit as st
from typing import Optional
from local.analysis.financial_extractor import FinancialProfile, MetricValue


def _fmt(val, unit: str = "", decimals: int = 2) -> str:
    if val is None:
        return "—"
    if isinstance(val, MetricValue):
        return f"{val.value:.{decimals}f} {val.unit}".strip()
    if isinstance(val, float):
        return f"{val:.{decimals}f}{(' ' + unit) if unit else ''}".strip()
    return str(val)


def render_financial_profile(profile: Optional[FinancialProfile]):
    """
    Renders a structured financial metrics panel from a FinancialProfile.
    Called in the right sidebar column after document processing.
    """
    if profile is None or not profile.has_data():
        return

    with st.expander("📊 Extracted Financial Profile", expanded=True):
        st.caption(
            f"Auto-extracted from uploaded document · Doc type: **{profile.doc_type}**"
        )

        # ── Credit Ratios ─────────────────────────────────────────────
        ratio_rows = [
            ("DSCR",             _fmt(profile.dscr)),
            ("ICR",              _fmt(profile.icr)),
            ("LTV",              _fmt(profile.ltv)),
            ("Leverage",         _fmt(profile.leverage)),
            ("TOL / ATNW",       _fmt(profile.tol_atnw, "x")),
            ("Debt / EBITDA",    _fmt(profile.debt_ebitda, "x")),
            ("Current Ratio",    _fmt(profile.current_ratio, "x")),
            ("Quick Ratio",      _fmt(profile.quick_ratio, "x")),
        ]
        ratio_rows = [(k, v) for k, v in ratio_rows if v != "—"]

        if ratio_rows:
            st.markdown("**Credit Ratios**")
            for label, value in ratio_rows:
                col_a, col_b = st.columns([2, 1])
                col_a.caption(label)
                col_b.markdown(f"`{value}`")

        # ── Facility Terms ────────────────────────────────────────────
        facility_rows = [
            ("Facility Amount",    _fmt(profile.facility_amount)),
            ("Collateral Value",   _fmt(profile.collateral_value)),
            ("Interest Rate",      _fmt(profile.interest_rate)),
            ("Tenor",              profile.facility_tenor or "—"),
            ("LTC",                _fmt(profile.ltc, "%")),
        ]
        facility_rows = [(k, v) for k, v in facility_rows if v != "—"]

        if facility_rows:
            st.markdown("**Facility Terms**")
            for label, value in facility_rows:
                col_a, col_b = st.columns([2, 1])
                col_a.caption(label)
                col_b.markdown(f"`{value}`")

        # ── P&L / Balance Sheet ───────────────────────────────────────
        pl_rows = [
            ("Revenue",        _fmt(profile.revenue)),
            ("EBITDA",         _fmt(profile.ebitda)),
            ("PAT",            _fmt(profile.pat)),
            ("Net Worth",      _fmt(profile.net_worth)),
            ("Total Assets",   _fmt(profile.total_assets)),
            ("Total Debt",     _fmt(profile.total_debt)),
        ]
        pl_rows = [(k, v) for k, v in pl_rows if v != "—"]

        if pl_rows:
            st.markdown("**Financials**")
            for label, value in pl_rows:
                col_a, col_b = st.columns([2, 1])
                col_a.caption(label)
                col_b.markdown(f"`{value}`")

        # ── Capital Adequacy ──────────────────────────────────────────
        cap_rows = [
            ("CAR",         _fmt(profile.car, "%")),
            ("Tier 1 Ratio", _fmt(profile.tier1_ratio, "%")),
            ("NPA Ratio",   _fmt(profile.npa_ratio, "%")),
        ]
        cap_rows = [(k, v) for k, v in cap_rows if v != "—"]

        if cap_rows:
            st.markdown("**Capital Adequacy**")
            for label, value in cap_rows:
                col_a, col_b = st.columns([2, 1])
                col_a.caption(label)
                col_b.markdown(f"`{value}`")

        # ── Ratings & Model Outputs ───────────────────────────────────
        rating_rows = [
            ("Internal Rating",  profile.internal_rating or "—"),
            ("External Rating",  profile.external_rating or "—"),
            ("PD Estimate",      _fmt(profile.pd_estimate, "%")),
            ("LGD Estimate",     _fmt(profile.lgd_estimate, "%")),
            ("ROE",              _fmt(profile.roe)),
            ("ROA",              _fmt(profile.roa)),
        ]
        rating_rows = [(k, v) for k, v in rating_rows if v != "—"]

        if rating_rows:
            st.markdown("**Ratings & Model Outputs**")
            for label, value in rating_rows:
                col_a, col_b = st.columns([2, 1])
                col_a.caption(label)
                col_b.markdown(f"`{value}`")

        # ── Extraction warnings ───────────────────────────────────────
        if profile.extraction_warnings:
            st.markdown("---")
            for w in profile.extraction_warnings:
                st.caption(f"⚠️ {w}")
