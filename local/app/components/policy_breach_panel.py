"""
local/app/components/policy_breach_panel.py

Renders the proactive policy breach report.
Displayed automatically after document ingestion — no user query needed.
"""

import streamlit as st
from typing import Optional
from local.analysis.policy_checker import BreachReport

_SEVERITY_CONFIG = {
    "BREACH":  ("🔴", "error"),
    "WARNING": ("🟡", "warning"),
    "PASS":    ("🟢", "success"),
    "N/A":     ("⚪", "info"),
}


def render_policy_breach_panel(report: Optional[BreachReport]):
    """
    Renders the policy breach report in an expander.
    Auto-expands if there are any breaches.
    """
    if report is None:
        return

    has_issues  = report.has_issues()
    label_parts = []
    if report.breach_count:
        label_parts.append(f"🔴 {report.breach_count} Breach{'es' if report.breach_count > 1 else ''}")
    if report.warning_count:
        label_parts.append(f"🟡 {report.warning_count} Warning{'s' if report.warning_count > 1 else ''}")
    if report.pass_count and not has_issues:
        label_parts.append(f"🟢 {report.pass_count} Pass")

    header = "⚡ Policy Breach Flags — " + (", ".join(label_parts) if label_parts else "No Issues")

    with st.expander(header, expanded=has_issues):
        st.caption(
            "Automatic compliance check against CBUAE, Basel III, and standard credit policy thresholds. "
            "Runs on document upload — no query required."
        )

        if has_issues:
            # Headline metrics row
            c1, c2, c3 = st.columns(3)
            c1.metric("Breaches",  report.breach_count,  delta=None)
            c2.metric("Warnings",  report.warning_count, delta=None)
            c3.metric("Passes",    report.pass_count,    delta=None)
            st.markdown("---")

        # Group by severity for visual hierarchy
        for severity_order in ("BREACH", "WARNING", "PASS", "N/A"):
            group = [f for f in report.findings if f.severity == severity_order]
            if not group:
                continue

            icon, _ = _SEVERITY_CONFIG[severity_order]

            for finding in group:
                extracted_str = (
                    f"{finding.extracted:.2f} {finding.unit}"
                    if finding.extracted is not None else "Not extracted"
                )
                threshold_dir = "≥" if finding.threshold_dir == "min" else "≤"
                threshold_str = f"{threshold_dir} {finding.threshold} {finding.unit}"

                with st.container():
                    col_icon, col_body = st.columns([0.5, 9.5])
                    col_icon.markdown(f"### {icon}")
                    with col_body:
                        st.markdown(f"**{finding.metric}**")
                        st.markdown(
                            f"Extracted: `{extracted_str}` &nbsp;|&nbsp; "
                            f"Threshold: `{threshold_str}`"
                        )
                        if severity_order in ("BREACH", "WARNING"):
                            st.markdown(f"> {finding.message}")
                        st.caption(f"📘 Source: {finding.source}")
                    st.markdown("---")
