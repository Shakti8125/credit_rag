import streamlit as st

def render_upload_panel():
    """
    Renders the sidebar control panel for credit document ingestion,
    classification mapping, and local privacy configuration tracking.
    """
    st.sidebar.markdown("### 📥 Document Ingestion Tier")
    st.sidebar.markdown(
        "Upload credit proposals, institutional policy manuals, or financial risk "
        "statements for localized context extraction."
    )

    # Context classification selector to optimize downstream embedding retrieval strategies
    doc_type = st.sidebar.selectbox(
        "Document Type Classification",
        options=[
            "Internal Credit Proposal (Memo)",
            "CBUAE Regulatory Framework / Policy",
            "Corporate Financial Statement",
            "General Risk Analytics Dossier"
        ],
        index=0,
        help="Classifying the document optimizes chunk size thresholds and vector matching pathways."
    )
    st.session_state["document_type"] = doc_type

    # Primary upload widget enforcing production format boundaries
    uploaded_file = st.sidebar.file_uploader(
        "Secure Local Upload", 
        type=["pdf", "docx", "txt"],               # <--- Just change this word
        help="Files are processed and scrubbed locally..."
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🛡️ Local Privacy Guardrails")
    
    # Visual check confirming localized masking execution states
    masking_enabled = st.sidebar.checkbox(
        "Enable Client-Side PII Masking",
        value=True,
        disabled=True,
        help="Mandatory compliance setting. Elements matching counterparty identities, names, and explicit account metrics are scrambled locally."
    )
    
    if masking_enabled:
        st.sidebar.success("🔒 Local Anonymization Active")
        st.sidebar.caption(
            "All sensitive parameters are obfuscated on this terminal "
            "prior to triggering remote cloud orchestration loops."
        )

    # Update session state context cache if a change is registered
    if uploaded_file != st.session_state.get("uploaded_file"):
        st.session_state["uploaded_file"] = uploaded_file
        # Flush downstream token/message states if a new file context is injected
        if "processed_context" in st.session_state:
            del st.session_state["processed_context"]
            
    return uploaded_file