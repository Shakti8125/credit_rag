"""EXTRACT/HYBRID/GENERAL intents on the Local Edge (Phi-3 GGUF) path."""

import logging
import streamlit as st

from local.app.config import SLM_MODEL_PATH
from local.app.resources import load_inference_engine
from local.app.retrieval import retrieve_doc_chunks, retrieve_regulatory_chunks
from local.app.masking_utils import unmask_response
from local.app.ui_helpers import stream_response
from components.citations import render_citations

logger = logging.getLogger(__name__)


def handle_local_edge(prompt: str, intent: str) -> str:
    engine, local_citations = load_inference_engine(), []

    if engine is None:
        answer_text = (
            "⚠️ Local inference unavailable. "
            f"Ensure the GGUF model exists at `{SLM_MODEL_PATH}`."
        )
    else:
        if intent == "EXTRACT":
            st.caption("Retrieving from document (FAISS + cross-encoder)…")
            doc_context, local_citations = retrieve_doc_chunks(prompt)
            reg_context = ""
        elif intent == "HYBRID":
            st.caption("Retrieving from document and regulatory corpus…")
            doc_context, dc = retrieve_doc_chunks(prompt)
            reg_context, rc = retrieve_regulatory_chunks(prompt)
            local_citations = dc + rc
        else:
            st.caption("Retrieving from regulatory corpus (Pinecone)…")
            doc_context = ""
            reg_context, local_citations = retrieve_regulatory_chunks(prompt)

        # Phi-3 Mini is a small model on a 4k context — prompts stay short
        # and directive, but each intent gets its own task framing.
        _INTENT_INSTRUCTIONS = {
            "EXTRACT": (
                "System: You are a credit risk analyst. Answer the question using "
                "ONLY the document chunks below. Quote metric values exactly as "
                "written. If the chunks do not contain the answer, say so — do not "
                "guess. Be concise: 3-4 sentences.\n\n"
            ),
            "HYBRID": (
                "System: You are a credit risk analyst. Compare the document values "
                "against the regulatory chunks below. For each comparison give the "
                "document value, the regulatory threshold, and whether it complies. "
                "Use only the chunks — never invent thresholds. Be concise: 3-5 "
                "sentences.\n\n"
            ),
            "GENERAL": (
                "System: You are a credit risk analyst. Answer using the regulatory "
                "chunks below if provided; otherwise answer from standard credit "
                "risk and banking regulation knowledge, and say when a precise "
                "threshold would need the source text. Stay within credit risk, "
                "banking regulation, and financial analysis. Be concise: 3-4 "
                "sentences.\n\n"
            ),
        }
        system_prefix = _INTENT_INSTRUCTIONS.get(intent, _INTENT_INSTRUCTIONS["GENERAL"])

        if intent == "EXTRACT" and doc_context:
            ctx = doc_context
        elif intent == "HYBRID":
            parts = []
            if doc_context: parts.append(f"Document:\n{doc_context}")
            if reg_context: parts.append(f"Regulatory:\n{reg_context}")
            ctx = "\n\n".join(parts)
        else:
            ctx = f"Regulatory:\n{reg_context}" if reg_context else ""

        slm_prompt  = f"<|user|>\n{system_prefix}{ctx}Question: {prompt}\n<|end|>\n<|assistant|>"
        st.caption("Generating via Phi-3…")
        raw_answer  = engine.run_inference(slm_prompt, max_tokens=512)
        answer_text = unmask_response(f"*(Local Edge — {intent})* {raw_answer}")

    st.session_state["last_execution_path"] = f"Local Edge · Phi-3 · {intent}"
    st.session_state["last_citations"]      = local_citations
    st.session_state["last_intent"]         = intent
    st.write_stream(stream_response(answer_text))
    render_citations(intent, local_citations)
    return answer_text
