import streamlit as st
from typing import Dict, List

def render_masking_log(masked_entities: Dict[str, str], preserved_financials: List[str]):
    """
    Renders a collapsible expander showing exactly what was redacted and what was preserved.
    This makes the selective redaction design tangible and explainable during a demo.
    """
    with st.expander("🛡️ Masking Audit Log", expanded=False):
        st.caption("Demonstrating local edge privacy: PII is scrubbed before reaching the cloud orchestration layer, while critical risk metrics are safely preserved for the LLM.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🔴 Entities Masked**")
            if not masked_entities:
                st.markdown("*(No entities masked)*")
            else:
                # Matches image spec: {CLIENT_A} -> [redacted]
                for entity_token, original_text in masked_entities.items():
                    st.markdown(f"`{entity_token}` ➔ `[redacted]`")
                    
        with col2:
            st.markdown("**🟢 Financials Preserved**")
            if not preserved_financials:
                st.markdown("*(No financials extracted)*")
            else:
                # Matches image spec: 1.4x DSCR -> preserved
                for metric in preserved_financials:
                    st.markdown(f"`{metric}` ➔ `preserved`")