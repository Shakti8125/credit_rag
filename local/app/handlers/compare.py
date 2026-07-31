"""COMPARE intent — cloud-only multi-document comparison via /compare."""

import logging
import streamlit as st
import requests

from local.app.config import API_COMPARE, API_HEADERS
from local.app.masking_utils import unmask_with_merged, mask_outbound_query
from local.app.ui_helpers import stream_response

logger = logging.getLogger(__name__)


def handle_compare(prompt: str) -> str:
    doc_a_text = st.session_state.get("doc_text") or ""
    doc_b_text = st.session_state.get("doc_b_text") or ""

    if not doc_a_text:
        answer_text = "⚠️ Upload **Document A** first (primary upload slot)."
        st.warning(answer_text)
        return answer_text

    if not doc_b_text:
        answer_text = "⚠️ Upload **Document B** (second slot appears in Compare mode)."
        st.warning(answer_text)
        return answer_text

    doc_a_label = st.session_state.get("last_uploaded_file", "Document A")
    doc_b_label = st.session_state.get("doc_b_label", "Document B")
    st.caption("Sending both documents to Gemini for structured comparison…")
    logger.info(
        "COMPARE | A='%s' %d chars | B='%s' %d chars",
        doc_a_label, len(doc_a_text),
        doc_b_label, len(doc_b_text),
    )
    # PRIVACY: masked_items (placeholder → original entity) is deliberately
    # NOT sent — the payload leaves this machine fully masked and all
    # unmasking happens locally after the response returns.
    payload = {
        "query":             mask_outbound_query(prompt),
        "doc_a_text":        doc_a_text,
        "doc_b_text":        doc_b_text,
        "doc_a_label":       doc_a_label,
        "doc_b_label":       doc_b_label,
        "doc_type":          st.session_state.get("document_type", "Document"),
        "include_regulatory": True,
    }
    resp = requests.post(API_COMPARE, json=payload, headers=API_HEADERS, timeout=300)
    if resp.status_code != 200:
        answer_text = f"API Error {resp.status_code}: {resp.text}"
        st.error(answer_text)
        return answer_text

    data = resp.json()
    # Unmask with BOTH registries — response may contain placeholders from either document
    answer_text = unmask_with_merged(data.get("answer", ""))

    def _unmask_cits(cits):
        return [{**c, "text": unmask_with_merged(c.get("text", ""))} for c in cits]

    ca = _unmask_cits(data.get("citations_a", []))
    cb = _unmask_cits(data.get("citations_b", []))
    st.session_state["last_execution_path"] = data.get("path", "Cloud Compare")
    st.session_state["last_citations"]      = ca + cb
    st.write_stream(stream_response(answer_text))

    if ca or cb:
        st.markdown("---")
        ca_col, cb_col = st.columns(2)
        with ca_col:
            st.markdown(f"**📄 {doc_a_label}**")
            for i, c in enumerate(ca[:4], 1):
                with st.expander(f"Chunk {i} · score {c.get('rerank_score','?')}", expanded=False):
                    st.write(c.get("text", "")[:400])
        with cb_col:
            st.markdown(f"**📄 {doc_b_label}**")
            for i, c in enumerate(cb[:4], 1):
                with st.expander(f"Chunk {i} · score {c.get('rerank_score','?')}", expanded=False):
                    st.write(c.get("text", "")[:400])

    return answer_text
