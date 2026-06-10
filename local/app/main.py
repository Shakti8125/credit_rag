import sys
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure the project root (credit-risk-rag/) is on sys.path regardless of
# which directory Streamlit was launched from.  Without this, `from local.x`
# imports fail because Streamlit adds local/app/ to sys.path (the script's
# own directory), not the project root two levels up.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # local/app/main.py → credit-risk-rag/
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import io
import time
import logging
import streamlit as st
import requests

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
API_ENDPOINT  = "http://127.0.0.1:8000/query"
SLM_MODEL_PATH = r"local\slm\models\phi3-basel-q4km.gguf"

# ---------------------------------------------------------------------------
# CACHED HEAVY RESOURCES
# All three loaders run once per Streamlit worker process and are reused
# across every session / rerun — no cold-start penalty after the first load.
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Initialising privacy pipeline (Docling + spaCy)…")
def load_privacy_pipeline():
    """
    Boots PrivacyPipeline once per worker.
    Internally initialises:
      - DocumentExtractor  (Docling DocumentConverter with table-structure pipeline)
      - DocumentMasker     (spaCy en_core_web_lg NER + financial-metric freeze rules)
      - EntityRegistry     (thread-safe token ↔ entity map)
      - DocumentValidator  (regex egress firewall)
    """
    from local.privacy.pipeline import PrivacyPipeline
    return PrivacyPipeline(spacy_model="en_core_web_lg")


@st.cache_resource(show_spinner="Loading local Phi-3 GGUF weights…")
def load_inference_engine():
    """
    Loads LocalModelInference once per worker.
    Used exclusively for LOCAL EDGE GENERATION (Phi-3 path) — NOT for intent
    routing.  Intent classification is handled by classify_intent() using
    deterministic keyword rules which are faster and more reliable.
    Returns None gracefully if the model file is missing or llama-cpp is not
    installed — the Gemini Pro cloud path remains fully operational.
    """
    try:
        from local.slm.inference import LocalModelInference
        return LocalModelInference(model_path=SLM_MODEL_PATH, ctx_size=4096, gpu_layers=0)
    except FileNotFoundError:
        logger.info("Phi-3 GGUF not found at %s — Local Edge path disabled.", SLM_MODEL_PATH)
        return None
    except ImportError:
        logger.info("llama-cpp-python not installed — Local Edge path disabled.")
        return None


@st.cache_resource(show_spinner="Loading embedding model for local RAG…")
def load_embedding_model():
    """
    Loads sentence-transformers all-MiniLM-L6-v2 once per worker (~80MB).
    Used to embed document chunks and queries for the ephemeral FAISS index
    that grounds Phi-3 on the uploaded document.
    Runs entirely locally — no network calls after the initial model download.
    Returns None gracefully if sentence-transformers is not installed.
    """
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("Embedding model loaded: all-MiniLM-L6-v2")
        return model
    except ImportError:
        logger.warning(
            "sentence-transformers not installed — local RAG disabled. "
            "Install with: pip install sentence-transformers faiss-cpu"
        )
        return None


# ---------------------------------------------------------------------------
# UI COMPONENTS (imported after page config to avoid partial-import issues)
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
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------------------------
_DEFAULTS = {
    "last_execution_path": "Idle",
    "last_citations":      [],
    "last_intent":         "GENERAL",
    "masked_entities":     {},        # {placeholder: original_entity} for masking_log
    "preserved_financials": [],
    "doc_text":            None,      # masked Markdown ready for cloud dispatch
    "doc_index":           None,      # Pinecone retriever for shared RAG
    "last_uploaded_file":  None,
    "messages":            [],
    "last_execution_time": None,
}
for key, default in _DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def stream_response(text: str):
    """Yields words one by one to simulate a live token stream."""
    if not text:
        yield "Error: Empty response."
        return
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.03)


def _extract_preserved_financials(masked_text: str) -> list:
    """
    Re-scans the masked text for financial metrics that the DocumentMasker
    deliberately left in place.  Uses the same pattern set as masker.py so the
    two stay in sync.  Returns a deduplicated, sorted list for the masking log.
    """
    import re
    patterns = [
        # Currency with scale: $14.5M, AED 250k, USD 5,000,000
        re.compile(r'\b(?:AED|USD|EUR|GBP)?\s?[\$\u20AC\u00A3]?\s?\d+(?:,\d{3})*(?:\.\d+)?\s?(?:M|B|k|Million|Billion)?\b', re.IGNORECASE),
        # Risk ratios: DSCR 1.25x, LTV 70%, Debt/EBITDA 3.5x
        re.compile(r'\b(?:DSCR|LTV|Leverage|TOL/ATNW|Debt/EBITDA|ROE|ROA)\b\s*(?:of|is)?\s*\d+(?:\.\d+)?\s?%?x?\b', re.IGNORECASE),
        # Percentages: 10.25%
        re.compile(r'\b\d+(?:\.\d+)?\s?%\b'),
        # Multipliers: 2.0x
        re.compile(r'\b\d+(?:\.\d+)?\s?x\b', re.IGNORECASE),
    ]
    found = set()
    for pattern in patterns:
        for match in pattern.finditer(masked_text):
            value = match.group().strip()
            # Filter out bare single digits like "1" or "2" which match \d+x loosely
            if len(value) > 1:
                found.add(value)
    return sorted(found)


def process_uploaded_document(uploaded_file) -> bool:
    """
    Runs the uploaded file through PrivacyPipeline.process_document().
    Stores results in session state.  Returns True on success, False on error.
    """
    pipeline = load_privacy_pipeline()
    if pipeline is None:
        st.error("🚨 Privacy pipeline failed to initialise. Check logs.")
        return False

    try:
        file_bytes = io.BytesIO(uploaded_file.getvalue())
        result = pipeline.process_document(
            file_source=file_bytes,
            filename=uploaded_file.name
        )

        # audit_log is List[{placeholder, original_entity, type}]
        # masking_log.py iterates as: for entity_token, original_text in masked_entities.items()
        # so key = placeholder ([ORG_1]), value = original entity ("RAKBANK")
        masked_entities = {
            entry["placeholder"]: entry["original_entity"]
            for entry in result["audit_log"]
        }

        # Extract preserved financial metrics from the masked text.
        # The pipeline keeps them in-place (doesn't mask them) but doesn't
        # return them separately — so we scan for them here.
        preserved_financials = _extract_preserved_financials(result["masked_text"])

        logger.info(
            "Document processed: %d entities masked, %d financials preserved.",
            len(masked_entities), len(preserved_financials)
        )

        st.session_state["doc_text"]             = result["masked_text"]
        st.session_state["masked_entities"]      = masked_entities
        st.session_state["preserved_financials"] = preserved_financials
        st.session_state["last_uploaded_file"]   = uploaded_file.name

        # Shared Pinecone index is used by both cloud and local routes.
        try:
            from local.rag.pinecone_index import PineconeRetriever
            st.session_state["doc_index"] = PineconeRetriever()
        except Exception as idx_err:
            logger.warning("Pinecone local retriever unavailable: %s", idx_err)
            st.session_state["doc_index"] = None

        return True

    except Exception as e:
        from local.privacy.validator import LeakageValidationError
        if isinstance(e, LeakageValidationError):
            st.error(f"🚨 Egress firewall blocked transmission: {e}")
        else:
            st.error(f"🚨 Document processing failed: {e}")
        logger.error("Pipeline error for %s: %s", uploaded_file.name, e, exc_info=True)
        return False


# Hard token budget for Phi-3 context.
# LangChain chunks are character-based (800 chars ≈ 130 words ≈ 170 tokens).
# 4 chunks × ~170 tokens = ~680 tokens — safe headroom within n_ctx=2048
# after accounting for system prompt (~100 tokens) + answer (512 tokens).
_PHI3_TOP_K = 4


def _retrieve_phi3_context(prompt: str) -> tuple:
    """
    Retrieves the top-k most semantically relevant chunks from the ephemeral
    FAISS index and assembles them into a context block for Phi-3.

    Returns
    -------
    (context_block: str, citations: list[dict])
        context_block — formatted string to inject into the Phi-3 prompt
        citations     — list of {text, section, score} dicts for UI display
                        Empty list when falling back to keyword extraction.
    """
    index    = st.session_state.get("doc_index")
    doc_text = st.session_state.get("doc_text") or ""

    # ── Path A: FAISS semantic retrieval ─────────────────────────────
    if index is not None and index.is_built:
        results = index.search(prompt, top_k=20)

        if results:
            try:
                from local.rag.reranker import CrossEncoderReranker
                reranked = CrossEncoderReranker().rerank(prompt, results, _PHI3_TOP_K)
                results = [(c, float(sc)) for c, sc in reranked]
            except Exception:
                results = results[:_PHI3_TOP_K]
            chunk_texts = []
            citations   = []
            for i, (chunk, score) in enumerate(results, 1):
                header = f"[Section: {chunk.section}] " if chunk.section else ""
                chunk_texts.append(f"Chunk {i} {header}(relevance {score:.2f}):\n{chunk.text}")
                citations.append({
                    "section": chunk.section or "Document",
                    "text":    chunk.text,
                    "score":   round(score, 3),
                    "page":    f"Chunk {chunk.index + 1} of {index.chunk_count}",
                })

            context = "\n\n---\n\n".join(chunk_texts)
            total_words = len(doc_text.split())
            context_block = (
                "System: You are a credit risk analyst. "
                "Answer the question using ONLY the document chunks below. "
                "Be concise — complete your answer within 3-4 sentences. "
                "If the answer is not in the chunks, say so.\n\n"
                f"Retrieved Document Chunks ({_PHI3_TOP_K} of {index.chunk_count} total, "
                f"document is {total_words:,} words):\n\n"
                f"{context}\n\n"
            )
            return context_block, citations

    # ── Path B: Keyword fallback (no index available) ─────────────────
    logger.warning("FAISS index unavailable — falling back to keyword extraction for Phi-3 context.")
    paragraphs = [p.strip() for p in doc_text.split("\n\n") if p.strip()]
    STOP = {"what", "is", "the", "a", "an", "of", "in", "this", "about",
            "are", "how", "does", "do", "which", "can", "be", "to", "and"}
    keywords = {w.lower() for w in prompt.split() if w.lower() not in STOP and len(w) > 2}
    scored = sorted(
        paragraphs,
        key=lambda p: len(keywords & {w.lower().strip(".,;:()[]") for w in p.split()}),
        reverse=True
    )
    excerpt_words: list = []
    for para in scored:
        words = para.split()
        if len(excerpt_words) + len(words) > 400:
            break
        excerpt_words.extend(words)

    excerpt = " ".join(excerpt_words) if excerpt_words else " ".join(doc_text.split()[:400])
    context_block = (
        "System: You are a credit risk analyst. "
        "Answer based on the document excerpt below. "
        "Be concise — complete your answer within 3-4 sentences.\n\n"
        f"Document Excerpt:\n{excerpt}\n\n"
    )
    return context_block, []


def classify_intent(prompt: str, document_attached: bool, selected_model: str) -> str:
    """
    Deterministic keyword-based intent classifier.

    The intent space is fully determined by two binary signals — document
    attached and model selected — with one genuine ambiguity: when a document
    IS attached and Gemini Pro is selected, we need to distinguish EXTRACT
    (answer from the document) vs HYBRID (answer requires document + regulatory
    vector store).  A keyword scan resolves this in O(n) with zero latency and
    zero failure modes.

    An SLM router was evaluated but rejected: the input dimensionality is too
    low to justify inference latency (~300ms/query) and the additional failure
    mode of mis-classification.  The Phi-3 GGUF is reserved for generation on
    the Local Edge path only.
    """
    # Phi-3 Local Edge: always document-grounded or pure conversational
    if selected_model == "Phi-3 (Local Edge)":
        return "EXTRACT" if document_attached else "GENERAL"

    # Gemini Pro, no document: pure regulatory knowledge retrieval
    if not document_attached:
        return "BENCHMARK"

    # Gemini Pro + document attached: only real decision point
    # HYBRID = needs both the uploaded doc AND the Pinecone regulatory archive
    # EXTRACT = answer lives entirely within the uploaded document
    HYBRID_SIGNALS = {
        # Comparison language
        "compare", "comparison", "versus", "vs", "against", "relative",
        "benchmark", "benchmarking", "exceed", "below", "above", "breach",
        # Compliance / regulatory language
        "guideline", "guidelines", "policy", "policies", "requirement",
        "requirements", "comply", "compliant", "compliance", "threshold",
        "limit", "limits", "minimum", "maximum", "meet", "meets", "satisfy",
        # Specific regulatory frameworks in the Pinecone index
        "cbuae", "basel", "rbi", "ifrs", "ifrs9", "sr11-7", "eba", "bcbs",
    }
    query_tokens = set(prompt.lower().split())
    intent = "HYBRID" if query_tokens & HYBRID_SIGNALS else "EXTRACT"

    logger.info("Intent classified as '%s' (document_attached=%s, model=%s)",
                intent, document_attached, selected_model)
    return intent



def _unmask_response(text: str) -> str:
    """Restore original entities before showing output."""
    try:
        mapping = st.session_state.get("masked_entities", {})
        for placeholder, original in mapping.items():
            text = text.replace(placeholder, original)
        return text
    except Exception:
        return text


# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🏦 Credit Risk RAG")
    uploaded_file  = render_upload_panel()
    selected_model = render_model_toggle()

    # Process document only when a *new* file is uploaded
    if uploaded_file and (st.session_state["last_uploaded_file"] != uploaded_file.name):
        with st.spinner("Extracting layout with Docling and scrubbing PII locally…"):
            success = process_uploaded_document(uploaded_file)
        if success:
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
            help="Total time from query submission to rendered response."
        )

    if st.session_state["doc_text"]:
        render_masking_log(
            st.session_state["masked_entities"],
            st.session_state["preserved_financials"]
        )

with col1:
    prompt = render_chat_interface()

    # ------------------------------------------------------------------
    # EXECUTION PIPELINE
    # ------------------------------------------------------------------
    if prompt:
        document_attached = bool(st.session_state["doc_text"])

        # ── Intent classification ──────────────────────────────────────
        intent = classify_intent(prompt, document_attached, selected_model)

        payload = {
            "query":        prompt,
            "intent":       intent,
            "doc_text":     st.session_state["doc_text"],
            "masked_items": st.session_state["masked_entities"],
        }

        # NOTE: user message is already appended + rendered by render_chat_interface().
        # Do NOT append again here — that causes the duplicate message visible in chat.

        try:
            with st.chat_message("assistant"):
                with st.spinner("Analysing parameters…"):
                    start_time = time.time()

                    # ── PATH A: LOCAL EDGE (Phi-3 GGUF) ──────────────
                    if selected_model == "Phi-3 (Local Edge)":
                        engine        = load_inference_engine()
                        local_citations = []   # initialise here; populated below if index available

                        if engine is None:
                            answer_text = (
                                "⚠️ Local inference is unavailable. "
                                "Ensure `llama-cpp-python` is installed and the GGUF "
                                f"model exists at `{SLM_MODEL_PATH}`."
                            )
                        else:
                            raw_context = st.session_state.get("doc_text") or ""
                            has_doc = bool(raw_context.strip())

                            if has_doc:
                                context_block, local_citations = _retrieve_phi3_context(prompt)
                            else:
                                local_citations = []
                                # GENERAL path — no document attached.
                                # Inject a credit risk domain system prompt so the model
                                # answers within the credit risk domain rather than
                                # drawing on unrelated general knowledge.
                                context_block = (
                                    "System: You are a credit risk analyst assistant. "
                                    "Answer only within the domain of credit risk, banking regulation, "
                                    "financial analysis, and risk management. "
                                    "Be concise — complete your answer within 3-4 sentences. "
                                    "If a term has a specific credit risk meaning (e.g. CAR = Capital "
                                    "Adequacy Ratio, PD = Probability of Default, LGD = Loss Given "
                                    "Default), always use the credit risk definition. "
                                    "Do not answer questions outside this domain.\n\n"
                                )

                            slm_prompt = (
                                f"<|user|>\n"
                                f"{context_block}"
                                f"Question: {prompt}\n<|end|>\n<|assistant|>"
                            )
                            st.caption("Generating response offline via Phi-3…")
                            # 512 tokens gives enough room for a complete answer.
                            # The context block is already bounded by _PHI3_TOP_K chunks
                            # so total prompt + answer stays within n_ctx=2048.
                            raw_answer  = engine.run_inference(slm_prompt, max_tokens=512)
                            answer_text = f"*(Local Edge Inference)* {raw_answer}"

                        st.session_state["last_execution_path"] = "Local Edge (Phi-3)"
                        st.session_state["last_citations"]      = local_citations
                        st.session_state["last_intent"]         = intent
                        answer_text = _unmask_response(answer_text)
                        st.write_stream(stream_response(answer_text))
                        render_citations(intent, local_citations)

                    # ── PATH B: CLOUD ORCHESTRATION (Gemini Pro) ──────
# ---------- PATH B: CLOUD ORCHESTRATION (Gemini Pro) ----------
                    else:
                        st.caption("Routing scrubbed payload to FastAPI orchestration tier...")

                        response = requests.post(
                            API_ENDPOINT,
                            json=payload,
                            timeout=120
                        )

                        if response.status_code == 200:

                            data = response.json()

                            answer_text = data.get("answer", "")

                            st.session_state["last_execution_path"] = data.get(
                                "path",
                                "Cloud Hybrid Orchestrated"
                            )

                            st.session_state["last_citations"] = data.get(
                                "citations",
                                []
                            )

                            st.session_state["last_intent"] = intent


                            # Restore original entities before showing user
                            answer_text = _unmask_response(answer_text)


                            st.write_stream(
                                stream_response(answer_text)
                            )


                            render_citations(
                                intent,
                                st.session_state["last_citations"]
                            )


                        else:

                            answer_text = (
                                f"API Error {response.status_code}: "
                                f"{response.text}"
                            )

                            st.error(answer_text)

        except requests.exceptions.ConnectionError:
            st.error(
                "🚨 Connection Refused: Ensure the FastAPI backend is running on "
                "`http://127.0.0.1:8000`"
            )