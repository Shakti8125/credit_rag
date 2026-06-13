"""
local/app/components/compare_panel.py

Renders the multi-document comparison result panel.
Shows the Gemini comparison narrative + side-by-side citations.
"""

import streamlit as st
from typing import Optional, List, Dict, Any


def render_compare_panel(
    answer:        Optional[str],
    citations_a:   List[Dict[str, Any]],
    citations_b:   List[Dict[str, Any]],
    reg_citations: List[Dict[str, Any]],
    doc_a_label:   str = "Document A",
    doc_b_label:   str = "Document B",
):
    if not answer:
        return

    st.markdown("---")
    st.markdown("### 🔄 Multi-Document Comparison Analysis")
    st.markdown(answer)

    # Side-by-side citation viewer
    if citations_a or citations_b:
        st.markdown("---")
        st.markdown("#### 📚 Retrieved Source Chunks")
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown(f"**{doc_a_label}**")
            if citations_a:
                for i, c in enumerate(citations_a[:5], 1):
                    with st.expander(f"Chunk {i} — {c.get('section','—')} (score {c.get('rerank_score', '?')})"):
                        st.caption(f"Page/Index: {c.get('page','—')}")
                        st.write(c.get("text", ""))
            else:
                st.caption("No chunks retrieved.")

        with col_b:
            st.markdown(f"**{doc_b_label}**")
            if citations_b:
                for i, c in enumerate(citations_b[:5], 1):
                    with st.expander(f"Chunk {i} — {c.get('section','—')} (score {c.get('rerank_score', '?')})"):
                        st.caption(f"Page/Index: {c.get('page','—')}")
                        st.write(c.get("text", ""))
            else:
                st.caption("No chunks retrieved.")

    if reg_citations:
        with st.expander(f"📘 Regulatory Benchmarks ({len(reg_citations)} chunks)", expanded=False):
            for i, c in enumerate(reg_citations[:6], 1):
                st.markdown(f"**[{i}] {c.get('source','—')} · {c.get('section','—')}**")
                st.caption(c.get("text", "")[:300])
                st.markdown("---")
