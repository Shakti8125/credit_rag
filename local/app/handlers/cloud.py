"""EXTRACT/HYBRID/BENCHMARK/GENERAL intents routed to the cloud Gemini path via /query."""

import logging
import streamlit as st
import requests

from local.app.config import API_ENDPOINT, API_HEADERS
from local.app.retrieval import retrieve_doc_chunks
from local.app.masking_utils import unmask_response, mask_outbound_query
from local.app.ui_helpers import stream_response
from components.citations import render_citations

logger = logging.getLogger(__name__)


def handle_cloud(prompt: str, intent: str, document_attached: bool) -> str:
    doc_payload_text, cloud_doc_citations = "", []
    if document_attached and intent in ("EXTRACT", "HYBRID"):
        st.caption("Retrieving relevant chunks from document (FAISS)…")
        doc_payload_text, cloud_doc_citations = retrieve_doc_chunks(prompt, for_cloud=True)
        # Local chunks are masked — unmask for on-screen citation display only
        # (doc_payload_text sent to the cloud stays masked)
        cloud_doc_citations = [
            {**c, "text": unmask_response(c.get("text", ""))}
            for c in cloud_doc_citations
        ]

    # PRIVACY: masked_items (placeholder → original entity) is deliberately
    # NOT sent — it contains the real bank names / PII the masking exists to
    # protect. Everything leaving this machine stays masked; unmasking of the
    # answer and citations happens locally below.
    payload = {
        # Query is masked too — the doc body was always masked, but the typed
        # question itself can name banks/people and previously went verbatim.
        "query":        mask_outbound_query(prompt),
        "intent":       intent,
        "doc_text":     doc_payload_text,
        # Chunks were already hybrid-retrieved + reranked above — tells the
        # backend to use them as-is instead of re-chunking server-side.
        "doc_is_prechunked": bool(doc_payload_text),
        "doc_type":     st.session_state.get("document_type", "Document"),
    }

    logger.info(
        "Cloud payload | intent=%s doc_preview=%s (masked_items withheld)",
        intent,
        (doc_payload_text or "")[:120],
    )

    st.caption("Routing to Gemini via FastAPI…")
    resp = requests.post(API_ENDPOINT, json=payload, headers=API_HEADERS, timeout=120)

    if resp.status_code != 200:
        answer_text = f"API Error {resp.status_code}: {resp.text}"
        st.error(answer_text)
        return answer_text

    data        = resp.json()
    answer_text = unmask_response(data.get("answer", ""))
    # Cloud returns citations masked (it has no unmask dict) — unmask locally
    cloud_citations = [
        {**c, "text": unmask_response(c.get("text", ""))}
        for c in data.get("citations", [])
    ]
    st.session_state["last_execution_path"] = data.get("path", "Cloud")
    st.session_state["last_citations"]      = cloud_doc_citations + cloud_citations
    st.session_state["last_intent"]         = intent
    st.write_stream(stream_response(answer_text))
    render_citations(intent, st.session_state["last_citations"])
    return answer_text
