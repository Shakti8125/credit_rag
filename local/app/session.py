"""Streamlit session-state defaults for the CreditRAG frontend."""

import streamlit as st

_DEFAULTS = {
    "last_execution_path":  "Idle",
    "last_citations":       [],
    "last_intent":          "GENERAL",

    # Only masked words and original values
    # No financial values stored
    "mask_dictionary":      {},       # primary doc: {placeholder: original}
    "doc_b_mask_dictionary": {},      # secondary doc: {placeholder: original}

    "doc_text":             None,
    "doc_raw_text":         None,     # pre-mask text — stays on-device; used to re-run analytics on doc-type change
    "doc_type_used":        None,     # document_type the analytics were last computed with
    "doc_faiss_index":      None,
    "registry":             None,
    "last_uploaded_file":   None,     # masked filename — display + API payloads
    "last_uploaded_raw_name": None,   # raw filename — reprocess guard ONLY (never sent to cloud)
    "messages":             [],
    "last_execution_time":  None,
    "financial_profile":    None,
    "breach_report":        None,
    "audit_trail":          [],
    "analysis_mode":        "Standard",
    "doc_b_text":           None,
    "doc_b_label":          None,     # masked filename for doc B
    "doc_b_raw_name":       None,     # raw filename — reprocess guard ONLY
    "doc_b_faiss_index":    None,
    "doc_b_registry":       None,     # secondary doc EntityRegistry
    "ews_report":           None,
    "ews_cloud_result":     None,
}


def init_session_state() -> None:
    for k, v in _DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = v
