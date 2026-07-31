"""EWS intent — cloud-only early warning signal deep scan via /ews."""

import logging
import streamlit as st
import requests

from local.app.config import API_EWS, API_HEADERS
from local.app.masking_utils import unmask_with_merged, mask_outbound_query
from local.app.ui_helpers import stream_response
from components.citations import render_citations

logger = logging.getLogger(__name__)


def handle_ews(prompt: str) -> str:
    doc_text = st.session_state.get("doc_text") or ""
    if not doc_text:
        answer_text = "⚠️ Upload a document before running an EWS scan."
        st.warning(answer_text)
        return answer_text

    st.caption("Running deep early warning scan via Gemini…")
    ews_report = st.session_state.get("ews_report")
    local_sigs = ews_report.to_dict().get("signals", []) if ews_report else []
    # PRIVACY: masked_items (placeholder → original entity) is deliberately
    # NOT sent — the payload leaves this machine fully masked and all
    # unmasking happens locally after the response returns.
    payload = {
        "doc_text":      doc_text,
        "doc_label":     st.session_state.get("last_uploaded_file", "Document"),
        "doc_type":      st.session_state.get("document_type", "Document"),
        "local_signals": local_sigs,
        "query":         mask_outbound_query(prompt),
    }
    resp = requests.post(API_EWS, json=payload, headers=API_HEADERS, timeout=300)
    if resp.status_code != 200:
        answer_text = f"API Error {resp.status_code}: {resp.text}"
        st.error(answer_text)
        return answer_text

    data        = resp.json()
    answer_text = unmask_with_merged(data.get("answer", ""))
    citations   = [
        {**c, "text": unmask_with_merged(c.get("text", ""))}
        for c in data.get("citations", [])
    ]
    st.session_state["ews_cloud_result"]    = answer_text
    st.session_state["last_execution_path"] = data.get("path", "Cloud EWS")
    st.session_state["last_citations"]      = citations
    st.write_stream(stream_response(answer_text))
    render_citations("HYBRID", citations)
    return answer_text
