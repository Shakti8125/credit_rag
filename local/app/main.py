import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Project root on sys.path — must happen before any `from local.x` import
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Load .env (Pinecone + Gemini keys) before any service import
load_dotenv(dotenv_path=_PROJECT_ROOT / "cloud" / "backend" / ".env")

import io
import time
import logging
import streamlit as st
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
API_ENDPOINT   = "http://127.0.0.1:8000/query"
SLM_MODEL_PATH = r"C:\Users\Shakti\Desktop\credit-risk-rag_v_0.1\local\slm\models\phi3-basel-q4km.gguf"

# Retrieval knobs
_RERANK_POOL     = 20   # candidates fetched (bi-encoder) before cross-encoder reranking
_PHI3_TOP_K      = 3    # final chunks fed to Phi-3 after reranking (fits 4096-token ctx)
_FAISS_POOL      = 15   # candidates fetched from local FAISS before reranking
_CLOUD_TOP_K     = 5    # final chunks sent to Gemini (larger ctx window)
_CHUNK_CHAR_CAP  = 1200 # max chars per chunk in Phi-3 prompt (hard context guard)

# ---------------------------------------------------------------------------
# CACHED HEAVY RESOURCES — loaded once per Streamlit worker process
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Initialising privacy pipeline (Docling + spaCy)…")
def load_privacy_pipeline():
    from local.privacy.pipeline import PrivacyPipeline
    return PrivacyPipeline(spacy_model="en_core_web_lg")


@st.cache_resource(show_spinner="Loading local Phi-3 GGUF weights…")
def load_inference_engine():
    try:
        from local.slm.inference import LocalModelInference
        return LocalModelInference(model_path=SLM_MODEL_PATH, ctx_size=4096, gpu_layers=0)
    except FileNotFoundError:
        logger.info("Phi-3 GGUF not found at %s — Local Edge path disabled.", SLM_MODEL_PATH)
        return None
    except ImportError:
        logger.info("llama-cpp-python not installed — Local Edge path disabled.")
        return None


@st.cache_resource(show_spinner="Loading sentence-transformer embedding model…")
def load_embedding_model():
    """
    Loads all-MiniLM-L6-v2 once per worker (~80 MB).
    Used to build the ephemeral FAISS index over uploaded document chunks.
    """
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("Embedding model loaded: all-MiniLM-L6-v2")
        return model
    except ImportError:
        logger.warning("sentence-transformers not installed — local FAISS RAG disabled.")
        return None


@st.cache_resource(show_spinner="Connecting to Pinecone (regulatory corpus)…")
def load_pinecone_retriever():
    """
    Connects to the shared Pinecone index that holds the pre-loaded regulatory
    corpus (CBUAE, Basel III, IFRS-9, etc.).
    Used for GENERAL and HYBRID intents on the local Phi-3 path.
    """
    try:
        from local.rag.pinecone_index import PineconeRetriever
        return PineconeRetriever()
    except Exception as e:
        logger.warning("PineconeRetriever unavailable: %s", e)
        return None


@st.cache_resource(show_spinner="Loading cross-encoder reranker…")
def load_reranker():
    """
    Loads cross-encoder/ms-marco-MiniLM-L-6-v2 once per worker.
    Used to rerank both FAISS (doc) and Pinecone (regulatory) recall pools.
    """
    try:
        from local.rag.reranker import CrossEncoderReranker
        return CrossEncoderReranker()
    except Exception as e:
        logger.warning("CrossEncoderReranker unavailable: %s", e)
        return None


# ---------------------------------------------------------------------------
# UI COMPONENTS
# ---------------------------------------------------------------------------
from components.upload_panel   import render_upload_panel
from components.model_toggle   import render_model_toggle
from components.chat           import render_chat_interface
from components.path_indicator import render_path_indicator
from components.masking_log    import render_masking_log
from components.citations      import render_citations

# ---------------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Credit Risk RAG",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------------------------
_DEFAULTS = {
    "last_execution_path":  "Idle",
    "last_citations":       [],
    "last_intent":          "GENERAL",
    "masked_entities":      {},     # {placeholder: original_entity}  — for masking_log UI
    "preserved_financials": [],
    "doc_text":             None,   # masked Markdown — sent to cloud path
    "doc_faiss_index":      None,   # LocalDocumentIndex built from uploaded doc
    "registry":             None,   # EntityRegistry instance — for unmask_text()
    "last_uploaded_file":   None,
    "messages":             [],
    "last_execution_time":  None,
}
for key, default in _DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def stream_response(text: str):
    if not text:
        yield "Error: Empty response."
        return
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.03)


def _extract_preserved_financials(masked_text: str) -> list:
    import re
    patterns = [
        re.compile(r'\b(?:AED|USD|EUR|GBP)?\s?[\$\u20AC\u00A3]?\s?\d+(?:,\d{3})*(?:\.\d+)?\s?(?:M|B|k|Million|Billion)?\b', re.IGNORECASE),
        re.compile(r'\b(?:DSCR|LTV|Leverage|TOL/ATNW|Debt/EBITDA|ROE|ROA)\b\s*(?:of|is)?\s*\d+(?:\.\d+)?\s?%?x?\b', re.IGNORECASE),
        re.compile(r'\b\d+(?:\.\d+)?\s?%\b'),
        re.compile(r'\b\d+(?:\.\d+)?\s?x\b', re.IGNORECASE),
    ]
    found = set()
    for pattern in patterns:
        for match in pattern.finditer(masked_text):
            value = match.group().strip()
            if len(value) > 1:
                found.add(value)
    return sorted(found)


def _unmask_response(text: str) -> str:
    """
    Replaces placeholder tokens ([ORG_1], [PERSON_2] …) with original entity
    names so the user never sees raw masked tokens in the UI.

    Prefers EntityRegistry.unmask_text() which guarantees longest-key-first
    replacement order. Falls back to the flat dict if registry is not available.
    """
    registry = st.session_state.get("registry")
    if registry is not None:
        return registry.unmask_text(text)
    # Flat-dict fallback — sort by key length descending to avoid partial hits
    mapping = st.session_state.get("masked_entities", {})
    for placeholder in sorted(mapping.keys(), key=len, reverse=True):
        text = text.replace(placeholder, mapping[placeholder])
    return text


def process_uploaded_document(uploaded_file) -> bool:
    """
    Full document processing pipeline:
      1. Docling extraction → masked Markdown (PrivacyPipeline)
      2. EntityRegistry stored in session for response unmasking
      3. MarkdownChunker → chunks
      4. LocalDocumentIndex (FAISS) built from chunks — ephemeral, in-memory

    The FAISS index is stored as st.session_state["doc_faiss_index"].
    It is discarded automatically when a new document is uploaded or the
    browser session ends — no disk writes, no network calls.
    """
    pipeline = load_privacy_pipeline()
    if pipeline is None:
        st.error("🚨 Privacy pipeline failed to initialise. Check logs.")
        return False

    try:
        # ── Phase 1: Extract + mask ───────────────────────────────────
        result = pipeline.process_document(
            file_source=io.BytesIO(uploaded_file.getvalue()),
            filename=uploaded_file.name,
        )

        masked_entities = {
            entry["placeholder"]: entry["original_entity"]
            for entry in result["audit_log"]
        }
        preserved_financials = _extract_preserved_financials(result["masked_text"])

        logger.info(
            "Document processed: %d entities masked, %d financials preserved.",
            len(masked_entities), len(preserved_financials),
        )

        st.session_state["doc_text"]             = result["masked_text"]
        st.session_state["masked_entities"]      = masked_entities
        st.session_state["preserved_financials"] = preserved_financials
        st.session_state["last_uploaded_file"]   = uploaded_file.name
        # Store registry so we can call unmask_text() on LLM responses
        st.session_state["registry"]             = result["registry_instance"]

        # ── Phase 2: Chunk masked text ────────────────────────────────
        from local.rag.chunker import MarkdownChunker
        chunker = MarkdownChunker()
        chunks  = chunker.chunk(result["masked_text"])

        if not chunks:
            logger.warning("Chunker produced 0 chunks — FAISS index will not be built.")
            st.session_state["doc_faiss_index"] = None
            return True

        logger.info("Produced %d chunks from uploaded document.", len(chunks))

        # ── Phase 3: Build ephemeral FAISS index ──────────────────────
        embed_model = load_embedding_model()
        if embed_model is None:
            logger.warning("Embedding model unavailable — FAISS index skipped.")
            st.session_state["doc_faiss_index"] = None
            return True

        from local.rag.local_index import LocalDocumentIndex
        faiss_index = LocalDocumentIndex(embed_model)
        faiss_index.build(chunks)
        st.session_state["doc_faiss_index"] = faiss_index

        logger.info(
            "Ephemeral FAISS index ready: %d vectors.", faiss_index.chunk_count
        )
        return True

    except Exception as e:
        from local.privacy.validator import LeakageValidationError
        if isinstance(e, LeakageValidationError):
            st.error(f"🚨 Egress firewall blocked transmission: {e}")
        else:
            st.error(f"🚨 Document processing failed: {e}")
        logger.error("Pipeline error for %s: %s", uploaded_file.name, e, exc_info=True)
        return False


# ---------------------------------------------------------------------------
# RETRIEVAL  — intent-aware, cross-encoder reranked
# ---------------------------------------------------------------------------

def _retrieve_doc_chunks(query: str, top_k: int = _PHI3_TOP_K, for_cloud: bool = False) -> tuple:
    """
    Retrieves top-k chunks from the UPLOADED DOCUMENT via the ephemeral
    FAISS index, reranked by the cross-encoder.

    Parameters
    ----------
    query     : user question
    top_k     : final chunks after reranking
    for_cloud : if True, use _CLOUD_TOP_K and return plain chunk text
                (no Phi-3 system prompt wrapper) suitable for FastAPI payload.

    Returns (context_block_or_raw_chunks: str, citations: list[dict])
    """
    effective_top_k = _CLOUD_TOP_K if for_cloud else top_k

    faiss_index = st.session_state.get("doc_faiss_index")
    reranker    = load_reranker()

    if faiss_index is None or not faiss_index.is_built:
        logger.warning("FAISS index not available — falling back to keyword extraction.")
        return _keyword_fallback(query)

    # Bi-encoder recall
    pool = faiss_index.search(query, top_k=_FAISS_POOL)

    # Cross-encoder rerank
    if reranker is not None and len(pool) > 1:
        ranked = reranker.rerank(query, pool, top_k=effective_top_k)
    else:
        ranked = [(chunk, score) for chunk, score in pool[:effective_top_k]]

    if not ranked:
        return _keyword_fallback(query)

    chunk_texts = []
    citations   = []
    for i, (chunk, score) in enumerate(ranked, 1):
        header = f"[Section: {chunk.section}] " if chunk.section else ""
        # Hard cap per chunk for Phi-3 context budget
        text = chunk.text if for_cloud else chunk.text[:_CHUNK_CHAR_CAP]
        chunk_texts.append(f"Chunk {i} {header}(score {score:.3f}):\n{text}")
        citations.append({
            "section": chunk.section or "Document",
            "text":    chunk.text,   # always store full text in citations
            "score":   round(score, 3),
            "page":    f"Chunk {chunk.index + 1} of {faiss_index.chunk_count}",
        })

    context = "\n\n---\n\n".join(chunk_texts)

    if for_cloud:
        # Raw chunk text — PromptBuilder in query.py will wrap this
        return context, citations

    # Phi-3 formatted context block
    context_block = (
        "System: You are a credit risk analyst. "
        "Answer the question using ONLY the document chunks below. "
        "Be concise — 3-4 sentences. "
        "If the answer is not in the chunks, say so.\n\n"
        f"Document Chunks (top-{effective_top_k}, cross-encoder reranked):\n\n{context}\n\n"
    )
    return context_block, citations


def _retrieve_regulatory_chunks(query: str, top_k: int = _PHI3_TOP_K) -> tuple:
    """
    Retrieves top-k chunks from the REGULATORY CORPUS (Pinecone) reranked
    by the cross-encoder.

    Returns (context_block: str, citations: list[dict])
    Used for GENERAL intent (and the regulatory half of HYBRID).
    """
    pinecone = load_pinecone_retriever()
    reranker = load_reranker()

    if pinecone is None:
        logger.warning("Pinecone unavailable — no regulatory context.")
        return "", []

    # Bi-encoder recall
    pool = pinecone.search(query, top_k=_RERANK_POOL)

    # Cross-encoder rerank
    if reranker is not None and len(pool) > 1:
        ranked = reranker.rerank(query, pool, top_k=top_k)
    else:
        ranked = [(chunk, 0.0) for chunk in pool[:top_k]]

    if not ranked:
        return "", []

    chunk_texts = []
    citations   = []
    for i, (chunk, score) in enumerate(ranked, 1):
        header = f"[{chunk.section}] " if chunk.section else ""
        chunk_texts.append(f"Chunk {i} {header}(score {score:.3f}):\n{chunk.text}")
        citations.append({
            "section": chunk.section or "Regulatory Corpus",
            "text":    chunk.text,
            "score":   round(score, 3),
            "page":    f"Chunk {chunk.index + 1}",
        })

    context = "\n\n---\n\n".join(chunk_texts)
    context_block = (
        f"Regulatory Chunks (top-{top_k}, cross-encoder reranked):\n\n{context}\n\n"
    )
    return context_block, citations


def _keyword_fallback(query: str) -> tuple:
    """
    Keyword-based extraction from doc_text when FAISS is unavailable.
    Returns (context_block, []) — no citations for fallback.
    """
    doc_text   = st.session_state.get("doc_text") or ""
    paragraphs = [p.strip() for p in doc_text.split("\n\n") if p.strip()]
    STOP = {"what", "is", "the", "a", "an", "of", "in", "this", "about",
            "are", "how", "does", "do", "which", "can", "be", "to", "and"}
    keywords = {w.lower() for w in query.split() if w.lower() not in STOP and len(w) > 2}
    scored = sorted(
        paragraphs,
        key=lambda p: len(keywords & {w.lower().strip(".,;:()[]") for w in p.split()}),
        reverse=True,
    )
    excerpt_words: list = []
    for para in scored:
        words = para.split()
        if len(excerpt_words) + len(words) > 500:
            break
        excerpt_words.extend(words)
    excerpt = " ".join(excerpt_words) if excerpt_words else " ".join(doc_text.split()[:500])
    context_block = (
        "System: You are a credit risk analyst. "
        "Answer based on the document excerpt below. Be concise (3-4 sentences).\n\n"
        f"Document Excerpt:\n{excerpt}\n\n"
    )
    return context_block, []


# ---------------------------------------------------------------------------
# INTENT CLASSIFIER
# ---------------------------------------------------------------------------

def classify_intent(prompt: str, document_attached: bool, selected_model: str) -> str:
    """
    Deterministic keyword-based intent classifier.

    Intents:
      EXTRACT   — answer lives entirely in the uploaded document
      HYBRID    — needs uploaded doc + regulatory corpus
      BENCHMARK — no doc; pure regulatory knowledge retrieval
      GENERAL   — no doc; conversational / domain knowledge
    """
    if selected_model == "Phi-3 (Local Edge)":
        if not document_attached:
            return "GENERAL"
        HYBRID_SIGNALS = {
            "compare", "comparison", "versus", "vs", "against", "relative",
            "benchmark", "benchmarking", "exceed", "below", "above", "breach",
            "guideline", "guidelines", "policy", "policies", "requirement",
            "requirements", "comply", "compliant", "compliance", "threshold",
            "limit", "limits", "minimum", "maximum", "meet", "meets", "satisfy",
            "cbuae", "basel", "rbi", "ifrs", "ifrs9", "sr11-7", "eba", "bcbs",
        }
        query_tokens = set(prompt.lower().split())
        return "HYBRID" if query_tokens & HYBRID_SIGNALS else "EXTRACT"

    # Gemini Pro path
    if not document_attached:
        return "BENCHMARK"

    HYBRID_SIGNALS = {
        "compare", "comparison", "versus", "vs", "against", "relative",
        "benchmark", "benchmarking", "exceed", "below", "above", "breach",
        "guideline", "guidelines", "policy", "policies", "requirement",
        "requirements", "comply", "compliant", "compliance", "threshold",
        "limit", "limits", "minimum", "maximum", "meet", "meets", "satisfy",
        "cbuae", "basel", "rbi", "ifrs", "ifrs9", "sr11-7", "eba", "bcbs",
    }
    query_tokens = set(prompt.lower().split())
    intent = "HYBRID" if query_tokens & HYBRID_SIGNALS else "EXTRACT"
    logger.info("Intent='%s' (doc=%s model=%s)", intent, document_attached, selected_model)
    return intent


# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🏦 Credit Risk RAG")
    uploaded_file  = render_upload_panel()
    selected_model = render_model_toggle()

    if uploaded_file and (st.session_state["last_uploaded_file"] != uploaded_file.name):
        with st.spinner("Extracting layout, scrubbing PII, building document index…"):
            success = process_uploaded_document(uploaded_file)
        if success:
            faiss_idx = st.session_state.get("doc_faiss_index")
            if faiss_idx and faiss_idx.is_built:
                st.success(f"✅ Indexed {faiss_idx.chunk_count} chunks from document.")
            st.rerun()

# ---------------------------------------------------------------------------
# MAIN LAYOUT
# ---------------------------------------------------------------------------
col1, col2 = st.columns([2, 1])

with col2:
    render_path_indicator(st.session_state["last_execution_path"])

    if st.session_state["last_execution_time"]:
        st.metric(
            label="⏱️ Inference Latency",
            value=f"{st.session_state['last_execution_time']} sec",
            help="Total time from query submission to rendered response.",
        )

    if st.session_state["doc_text"]:
        render_masking_log(
            st.session_state["masked_entities"],
            st.session_state["preserved_financials"],
        )

with col1:
    prompt = render_chat_interface()

    # ------------------------------------------------------------------
    # EXECUTION PIPELINE
    # ------------------------------------------------------------------
    if prompt:
        document_attached = bool(st.session_state["doc_text"])
        intent = classify_intent(prompt, document_attached, selected_model)

        try:
            with st.chat_message("assistant"):
                with st.spinner("Analysing parameters…"):
                    start_time = time.time()

                    # ── PATH A: LOCAL EDGE (Phi-3) ────────────────────
                    if selected_model == "Phi-3 (Local Edge)":
                        engine          = load_inference_engine()
                        local_citations = []

                        if engine is None:
                            answer_text = (
                                "⚠️ Local inference is unavailable. "
                                "Ensure `llama-cpp-python` is installed and the GGUF "
                                f"model exists at `{SLM_MODEL_PATH}`."
                            )
                        else:
                            if intent == "EXTRACT":
                                st.caption("Retrieving from uploaded document (FAISS + cross-encoder)…")
                                doc_context, local_citations = _retrieve_doc_chunks(prompt)
                                reg_context = ""
                            elif intent == "HYBRID":
                                st.caption("Retrieving from document and regulatory corpus…")
                                doc_context, doc_cit = _retrieve_doc_chunks(prompt)
                                reg_context, reg_cit = _retrieve_regulatory_chunks(prompt)
                                local_citations = doc_cit + reg_cit
                            else:  # GENERAL
                                st.caption("Retrieving from regulatory corpus (Pinecone + cross-encoder)…")
                                doc_context = ""
                                reg_context, local_citations = _retrieve_regulatory_chunks(prompt)

                            system_prefix = (
                                "System: You are a credit risk analyst. "
                                "Answer only within the domain of credit risk, banking regulation, "
                                "financial analysis, and risk management. "
                                "Be concise — 3-4 sentences. "
                                "Use credit risk definitions for domain terms "
                                "(e.g. CAR = Capital Adequacy Ratio, PD = Probability of Default).\n\n"
                            )

                            if intent == "EXTRACT" and doc_context:
                                context_section = doc_context
                            elif intent == "HYBRID":
                                parts = []
                                if doc_context:
                                    parts.append(f"Uploaded Document Context:\n{doc_context}")
                                if reg_context:
                                    parts.append(f"Regulatory Context:\n{reg_context}")
                                context_section = "\n\n".join(parts)
                            elif reg_context:
                                context_section = f"Regulatory Context:\n{reg_context}"
                            else:
                                context_section = ""

                            slm_prompt = (
                                f"<|user|>\n"
                                f"{system_prefix}"
                                f"{context_section}"
                                f"Question: {prompt}\n<|end|>\n<|assistant|>"
                            )

                            st.caption("Generating response offline via Phi-3…")
                            raw_answer  = engine.run_inference(slm_prompt, max_tokens=512)
                            answer_text = f"*(Local Edge — {intent})* {raw_answer}"

                        # Local path: unmask before display
                        answer_text = _unmask_response(answer_text)
                        st.session_state["last_execution_path"] = f"Local Edge · Phi-3 · {intent}"
                        st.session_state["last_citations"]      = local_citations
                        st.session_state["last_intent"]         = intent
                        st.write_stream(stream_response(answer_text))
                        render_citations(intent, local_citations)

                    # ── PATH B: CLOUD ORCHESTRATION (Gemini Pro) ──────
                    else:
                        # Retrieve relevant chunks locally first, send only those
                        # to the cloud — NOT the full doc_text.
                        # for_cloud=True: plain chunk text, no Phi-3 system prefix,
                        # uses _CLOUD_TOP_K, no _CHUNK_CHAR_CAP truncation.
                        doc_payload_text    = ""
                        cloud_doc_citations = []
                        if document_attached and intent in ["EXTRACT", "HYBRID"]:
                            st.caption("Retrieving relevant chunks from uploaded document (FAISS)…")
                            doc_payload_text, cloud_doc_citations = _retrieve_doc_chunks(
                                prompt, for_cloud=True
                            )

                        payload = {
                            "query":        prompt,
                            "intent":       intent,
                            "doc_text":     doc_payload_text,   # masked chunk text only
                            "masked_items": st.session_state["masked_entities"],
                        }

                        # Debug — visible in terminal, never in UI
                        logger.info("--- Outgoing FastAPI Payload ---")
                        logger.info("Query  : %s", payload["query"])
                        logger.info("Intent : %s", payload["intent"])
                        preview = (payload["doc_text"] or "")[:200]
                        logger.info("doc_text preview (200 chars): %s%s",
                                    preview, "…" if len(payload["doc_text"] or "") > 200 else "")
                        logger.info("masked_items count: %d", len(payload["masked_items"] or {}))
                        logger.info("--------------------------------")

                        st.caption("Routing scrubbed chunk payload to FastAPI orchestration tier…")
                        response = requests.post(API_ENDPOINT, json=payload, timeout=120)

                        if response.status_code == 200:
                            data    = response.json()
                            raw_ans = data.get("answer", "")

                            # Unmask LLM response — user sees real entity names
                            answer_text = _unmask_response(raw_ans)

                            st.session_state["last_execution_path"] = data.get(
                                "path", "Cloud Orchestrated"
                            )
                            # Merge local doc citations with regulatory citations from cloud
                            backend_citations = data.get("citations", [])
                            st.session_state["last_citations"] = cloud_doc_citations + backend_citations
                            st.session_state["last_intent"]    = intent

                            st.write_stream(stream_response(answer_text))
                            render_citations(intent, st.session_state["last_citations"])
                        else:
                            answer_text = f"API Error {response.status_code}: {response.text}"
                            st.error(answer_text)

                    # ── TIMING + HISTORY ──────────────────────────────
                    elapsed = round(time.time() - start_time, 2)
                    st.session_state["last_execution_time"] = elapsed
                    st.caption(f"⏱️ **Response generated in {elapsed} seconds.**")
                    st.session_state["messages"].append({
                        "role":      "assistant",
                        "content":   answer_text,
                        "citations": st.session_state["last_citations"],
                        "time":      elapsed,
                    })
                    st.rerun()

        except requests.exceptions.ConnectionError:
            st.error(
                "🚨 Connection Refused: Ensure the FastAPI backend is running on "
                "`http://127.0.0.1:8000`"
            )
