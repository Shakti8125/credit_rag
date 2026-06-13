import streamlit as st


def render_upload_panel():
    """
    Sidebar document ingestion panel.
    Returns (primary_file, secondary_file).
    secondary_file is only shown when Compare mode is active.
    """
    st.sidebar.markdown("### 📥 Document Ingestion")

    doc_type = st.sidebar.selectbox(
        "Document Type",
        options=[
            "Internal Credit Proposal (Memo)",
            "CBUAE Regulatory Framework / Policy",
            "Corporate Financial Statement",
            "General Risk Analytics Dossier",
        ],
        index=0,
        help="Controls extraction patterns and policy thresholds.",
    )
    st.session_state["document_type"] = doc_type

    # Mode selector — drives whether second upload slot appears
    mode = st.sidebar.radio(
        "Analysis Mode",
        options=["Standard", "Compare Two Documents", "Early Warning Scan"],
        index=0,
        horizontal=False,
        help=(
            "Standard: single-doc Q&A and audit.\n"
            "Compare: side-by-side comparison of two documents.\n"
            "EWS: deep early warning signal analysis."
        ),
    )
    st.session_state["analysis_mode"] = mode

    st.sidebar.markdown("---")

    # Primary upload
    primary_label = (
        "Primary Document (Document A)"
        if mode == "Compare Two Documents" else "Upload Document"
    )
    primary_file = st.sidebar.file_uploader(
        primary_label,
        type=["pdf", "docx", "txt"],
        key="primary_uploader",
        help="Processed and anonymised locally before any cloud dispatch.",
    )

    # Secondary upload — Compare mode only
    secondary_file = None
    if mode == "Compare Two Documents":
        secondary_file = st.sidebar.file_uploader(
            "Comparison Document (Document B)",
            type=["pdf", "docx", "txt"],
            key="secondary_uploader",
            help="Second document for side-by-side comparison.",
        )
        if primary_file and secondary_file:
            st.sidebar.info(
                f"📄 **A:** {primary_file.name}\n\n"
                f"📄 **B:** {secondary_file.name}"
            )
        elif primary_file:
            st.sidebar.caption("Waiting for Document B…")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🛡️ Privacy Guardrails")
    st.sidebar.checkbox(
        "Enable Client-Side PII Masking",
        value=True,
        disabled=True,
        help="Mandatory. Entities are anonymised locally before cloud dispatch.",
    )
    st.sidebar.success("🔒 Local Anonymisation Active")
    st.sidebar.caption(
        "All sensitive parameters are obfuscated on this terminal "
        "prior to triggering remote cloud orchestration."
    )

    # Sync to session state
    if primary_file != st.session_state.get("uploaded_file"):
        st.session_state["uploaded_file"] = primary_file
        if "processed_context" in st.session_state:
            del st.session_state["processed_context"]

    return primary_file, secondary_file
