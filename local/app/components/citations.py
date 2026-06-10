import streamlit as st
from typing import List, Dict

def render_citations(intent: str, citations: List[Dict]):
    """
    Renders an expandable list of the retrieved context chunks.
    Strictly conditionally displayed: Only triggered for BENCHMARK (Path A) 
    and HYBRID (Path C) where external factual grounding is required.
    """
    # 1. Verification Gate: Only render if the routing path justifies it
    if intent not in ["BENCHMARK", "HYBRID"]:
        return

    # 2. Rendering the Audit UI
    if not citations:
        st.warning("⚠️ No regulatory context chunks were retrieved for this query.")
        return

    st.markdown("---")
    st.markdown("#### 📚 Factual Grounding References")
    st.caption("Every factual claim is auditable. Expand the references below to verify the source chunks.")

    # 3. Iterating through the top-K returned chunks
    for idx, chunk in enumerate(citations, 1):
        # Extracting the precise metadata fields matching the project plan
        source_doc = chunk.get("source", "Unknown Regulatory Document")
        section = chunk.get("section", "General Section")
        page = chunk.get("page", "N/A")
        text_preview = chunk.get("text", "")

        # Use an expander to keep the UI clean but fully auditable
        with st.expander(f"Reference [{idx}]: {source_doc} (Page {page})"):
            st.markdown(f"**Section / Article:** `{section}`")
            st.markdown("**Chunk Preview:**")
            st.info(text_preview)