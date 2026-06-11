"""
local/app/components/audit_trail.py

Renders the per-session audit trail and provides a PDF export button.
"""

import streamlit as st
from typing import List, Dict, Any


def render_audit_trail(audit_trail: List[Dict[str, Any]], doc_filename: str = "session"):
    """
    Renders the session audit trail in a collapsible expander.
    Provides a one-click PDF download button.
    """
    if not audit_trail:
        return

    with st.expander(f"📋 Session Audit Trail ({len(audit_trail)} entries)", expanded=False):
        st.caption(
            "Full traceable log of every query: intent classification, "
            "retrieved chunks with reranking scores, LLM response, and policy status. "
            "Download as PDF for credit file documentation."
        )

        # ── PDF export button ─────────────────────────────────────────
        try:
            from local.analysis.audit_logger import export_audit_pdf
            pdf_bytes = export_audit_pdf(audit_trail, doc_filename)
            st.download_button(
                label="⬇️ Download Audit Trail (PDF)",
                data=pdf_bytes,
                file_name=f"audit_trail_{doc_filename.replace(' ', '_')}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        except Exception as e:
            # Fallback to plain text
            from local.analysis.audit_logger import _export_plain_text
            txt_bytes = _export_plain_text(audit_trail)
            st.download_button(
                label="⬇️ Download Audit Trail (TXT)",
                data=txt_bytes,
                file_name=f"audit_trail_{doc_filename.replace(' ', '_')}.txt",
                mime="text/plain",
                use_container_width=True,
            )
            st.caption(f"PDF export unavailable ({e}) — downloading as plain text.")

        st.markdown("---")

        # ── Entry list (most recent first) ────────────────────────────
        for entry in reversed(audit_trail):
            ts   = entry.get("timestamp", "—")
            q    = entry.get("query", "")
            intent    = entry.get("intent", "—")
            path      = entry.get("execution_path", "—")
            elapsed   = entry.get("elapsed_sec", 0)
            breach    = entry.get("breach_summary", "—")
            citations = entry.get("citations", [])
            answer    = entry.get("answer", "")

            with st.container():
                col_ts, col_intent, col_path = st.columns([2, 1, 3])
                col_ts.caption(ts)
                col_intent.markdown(f"`{intent}`")
                col_path.caption(path)

                st.markdown(f"**Q:** {q}")

                # Citations summary
                if citations:
                    cit_summary = " · ".join(
                        f"{c.get('source','?')} (score {c.get('score','?')})"
                        for c in citations[:3]
                    )
                    if len(citations) > 3:
                        cit_summary += f" + {len(citations)-3} more"
                    st.caption(f"📚 Chunks: {cit_summary}")

                # Policy status
                if breach and breach != "—":
                    st.caption(f"⚡ Policy: {breach}")

                # Collapsed answer preview
                with st.expander("View answer", expanded=False):
                    st.markdown(answer[:800] + ("…" if len(answer) > 800 else ""))
                    st.caption(f"⏱ {elapsed:.2f}s")

                st.markdown("---")
