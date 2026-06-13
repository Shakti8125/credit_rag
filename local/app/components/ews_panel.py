"""
local/app/components/ews_panel.py

Renders the Early Warning Signal report panel.
Shows local (regex-detected) signals immediately after upload.
The cloud EWS deep-scan result is rendered when it returns.
"""

import streamlit as st
from typing import Optional
from local.analysis.ews_detector import EWSReport

_SEV_ICON  = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🔵"}
_CAT_ICON  = {"FINANCIAL": "📉", "QUALITATIVE": "📝", "STRUCTURAL": "🏗️"}
_RISK_COLOUR = {"HIGH": "error", "MEDIUM": "warning", "LOW": "info", "CLEAR": "success"}


def render_ews_panel(report: Optional[EWSReport], cloud_result: Optional[str] = None):
    """
    Parameters
    ----------
    report       : EWSReport from local EarlyWarningDetector (runs on upload)
    cloud_result : Gemini EWS narrative string (returned from /ews endpoint)
    """
    if report is None and not cloud_result:
        return

    risk_level = report.risk_level if report else "CLEAR"
    icon_map   = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🔵", "CLEAR": "🟢"}
    icon       = icon_map.get(risk_level, "⚪")

    header_label = (
        f"{icon} Early Warning Signals — {risk_level}"
        if report else "⚡ Early Warning Signals"
    )

    with st.expander(header_label, expanded=(risk_level in ("HIGH", "MEDIUM"))):

        # ── Local scan summary ─────────────────────────────────────
        if report:
            st.caption(
                "Local pattern scan — runs on upload, no API call. "
                "Click **Run Cloud EWS** below for LLM-powered deep analysis."
            )
            if risk_level == "CLEAR":
                st.success("🟢 No early warning signals detected by local scanner.")
            else:
                c1, c2, c3 = st.columns(3)
                c1.metric("🔴 HIGH",   report.high_count)
                c2.metric("🟡 MEDIUM", report.medium_count)
                c3.metric("🔵 LOW",    report.low_count)
                st.markdown("---")

            # Group by category
            for category in ("FINANCIAL", "QUALITATIVE", "STRUCTURAL"):
                sigs = [s for s in report.signals if s.category == category]
                if not sigs:
                    continue
                cat_icon = _CAT_ICON.get(category, "•")
                st.markdown(f"**{cat_icon} {category}**")
                for sig in sigs:
                    sev_icon = _SEV_ICON.get(sig.severity, "•")
                    with st.container():
                        st.markdown(f"{sev_icon} **{sig.signal}** `{sig.severity}`")
                        st.caption(sig.detail)
                        if sig.excerpt and sig.source != "Financial Extraction":
                            st.info(f"*…{sig.excerpt}…*")
                    st.markdown("")

        # ── Cloud EWS narrative ────────────────────────────────────
        if cloud_result:
            st.markdown("---")
            st.markdown("#### 🤖 Cloud EWS Deep Analysis (Gemini)")
            st.markdown(cloud_result)
