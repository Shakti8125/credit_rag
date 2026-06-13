"""
local/app/components/audit_trail.py
"""

import streamlit as st
from typing import List, Dict, Any


def render_audit_trail(audit_trail: List[Dict[str, Any]], doc_filename=None):
    if not audit_trail:
        return

    # ── CRITICAL FIX: doc_filename may be None when no document is uploaded.
    # st.session_state.get(key, default) returns None when the key exists but
    # holds None — the default is only used when the key is absent entirely.
    safe = (doc_filename or "session").replace(" ", "_").replace(".", "_")

    with st.expander(f"📋 Audit Trail ({len(audit_trail)} entries)", expanded=False):
        st.caption(
            "Traceable log of every query — intent, retrieved chunks, "
            "LLM response, policy status. Download for credit file documentation."
        )

        # ── Export ────────────────────────────────────────────────────
        try:
            from local.analysis.audit_logger import export_audit_pdf
            pdf_bytes = export_audit_pdf(audit_trail, safe)
            st.download_button(
                label="⬇️ Download Audit Trail (PDF)",
                data=pdf_bytes,
                file_name=f"audit_trail_{safe}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        except Exception as pdf_err:
            try:
                from local.analysis.audit_logger import _export_plain_text
                txt_bytes = _export_plain_text(audit_trail)
            except Exception:
                txt_bytes = b""
            st.download_button(
                label="⬇️ Download Audit Trail (TXT)",
                data=txt_bytes,
                file_name=f"audit_trail_{safe}.txt",
                mime="text/plain",
                use_container_width=True,
            )
            st.caption(f"PDF unavailable ({pdf_err}) — plain text fallback.")

        st.markdown("---")

        for entry in reversed(audit_trail):
            ts      = entry.get("timestamp", "—")
            q       = entry.get("query", "")
            intent  = entry.get("intent", "—")
            path    = entry.get("execution_path", "—")
            elapsed = entry.get("elapsed_sec", 0)
            breach  = entry.get("breach_summary") or ""
            cits    = entry.get("citations", [])
            answer  = entry.get("answer", "")

            c1, c2, c3 = st.columns([2, 1, 3])
            c1.caption(ts)
            c2.markdown(f"`{intent}`")
            c3.caption(path)

            st.markdown(f"**Q:** {q}")

            if cits:
                preview = " · ".join(
                    f"{c.get('source', c.get('section','?'))} ({c.get('score','?')})"
                    for c in cits[:3]
                )
                if len(cits) > 3:
                    preview += f" +{len(cits)-3}"
                st.caption(f"📚 {preview}")

            if breach and breach not in ("—", ""):
                st.caption(f"⚡ {breach}")

            with st.expander("View answer", expanded=False):
                st.markdown(answer[:800] + ("…" if len(answer) > 800 else ""))
                st.caption(f"⏱ {elapsed:.2f}s")

            st.markdown("---")