import sys
import os
from pathlib import Path
from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

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
API_COMPARE    = "http://127.0.0.1:8000/compare"
API_EWS        = "http://127.0.0.1:8000/ews"
SLM_MODEL_PATH = r"C:\Users\Shakti\Desktop\credit-risk-rag_v_0.1\local\slm\models\phi3-basel-q4km.gguf"

_RERANK_POOL    = 20
_PHI3_TOP_K     = 3
_FAISS_POOL     = 15
_CLOUD_TOP_K    = 5
_CHUNK_CHAR_CAP = 1200

# ---------------------------------------------------------------------------
# PAGE CONFIG — must be first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="CreditRAG · Risk Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# GLOBAL CSS — futuristic dark risk management theme
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* ── Base & fonts ─────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #070d19;
    color: #c8d8e8;
}

/* ── Sidebar ─────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a1628 0%, #0d1f3c 100%);
    border-right: 1px solid #1a3a5c;
}
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: #4fc3f7 !important;
}
[data-testid="stSidebar"] label { color: #8bafc8 !important; font-size: 0.78rem !important; letter-spacing: 0.04em; text-transform: uppercase; }
[data-testid="stSidebar"] .stSelectbox > div > div { background: #0f1e35 !important; border: 1px solid #1e3a5c !important; color: #c8d8e8 !important; border-radius: 6px; }
[data-testid="stSidebar"] .stRadio > div { gap: 6px; }
[data-testid="stSidebar"] .stRadio label { color: #a0c0d8 !important; font-size: 0.85rem !important; text-transform: none !important; letter-spacing: 0 !important; }

/* ── Main background ─────────────────────────────────────────── */
.main .block-container {
    background-color: #070d19;
    padding-top: 1.5rem;
    max-width: 1400px;
}

/* ── Headings ────────────────────────────────────────────────── */
h1, h2, h3 { color: #4fc3f7 !important; font-weight: 600 !important; letter-spacing: -0.01em; }
h1 { font-size: 1.6rem !important; }
h2 { font-size: 1.25rem !important; }
h3 { font-size: 1.05rem !important; }

/* ── Metrics ─────────────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #0d1f3c 0%, #0f2540 100%);
    border: 1px solid #1e3a5c;
    border-radius: 10px;
    padding: 14px 18px;
}
[data-testid="stMetricLabel"] { color: #7ca8c8 !important; font-size: 0.72rem !important; text-transform: uppercase; letter-spacing: 0.08em; }
[data-testid="stMetricValue"] { color: #4fc3f7 !important; font-size: 1.5rem !important; font-weight: 700 !important; font-family: 'JetBrains Mono', monospace !important; }

/* ── Expanders ───────────────────────────────────────────────── */
[data-testid="stExpander"] {
    background: #0a1628 !important;
    border: 1px solid #1a3355 !important;
    border-radius: 10px !important;
    margin-bottom: 8px !important;
}
[data-testid="stExpander"] summary {
    color: #90caf9 !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
}

/* ── Alerts ──────────────────────────────────────────────────── */
[data-testid="stAlert"] { border-radius: 8px !important; border-left-width: 4px !important; }
div[data-baseweb="notification"][kind="error"]   { background: #1a0a0a !important; border-color: #ef5350 !important; }
div[data-baseweb="notification"][kind="warning"] { background: #1a1400 !important; border-color: #ffb74d !important; }
div[data-baseweb="notification"][kind="success"] { background: #071a0e !important; border-color: #66bb6a !important; }
div[data-baseweb="notification"][kind="info"]    { background: #051428 !important; border-color: #4fc3f7 !important; }

/* ── Code / inline code ──────────────────────────────────────── */
code {
    background: #0f2236 !important;
    color: #80d8ff !important;
    border: 1px solid #1a3a5c !important;
    border-radius: 4px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.82em !important;
    padding: 1px 5px !important;
}

/* ── Buttons ─────────────────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #0d47a1 0%, #1565c0 100%) !important;
    color: #e3f2fd !important;
    border: 1px solid #1976d2 !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    letter-spacing: 0.02em !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #1565c0 0%, #1976d2 100%) !important;
    box-shadow: 0 0 16px rgba(79,195,247,0.25) !important;
    transform: translateY(-1px) !important;
}
.stDownloadButton > button {
    background: linear-gradient(135deg, #004d40 0%, #00695c 100%) !important;
    color: #e0f2f1 !important;
    border: 1px solid #00897b !important;
    border-radius: 8px !important;
}

/* ── Chat messages ───────────────────────────────────────────── */
[data-testid="stChatMessage"] {
    background: #0a1628 !important;
    border: 1px solid #1a3355 !important;
    border-radius: 12px !important;
    margin-bottom: 8px !important;
}
[data-testid="stChatMessageContent"] { color: #c8d8e8 !important; }

/* ── Chat input ──────────────────────────────────────────────── */
[data-testid="stChatInput"] > div {
    background: #0d1f3c !important;
    border: 1px solid #1e3a5c !important;
    border-radius: 12px !important;
}
[data-testid="stChatInput"] textarea {
    color: #c8d8e8 !important;
    background: transparent !important;
}
[data-testid="stChatInput"] textarea::placeholder { color: #4a6a88 !important; }

/* ── File uploader ───────────────────────────────────────────── */
[data-testid="stFileUploader"] {
    background: #0a1628 !important;
    border: 1px dashed #1e3a5c !important;
    border-radius: 10px !important;
    padding: 8px !important;
}
[data-testid="stFileUploader"] label { color: #7ca8c8 !important; }

/* ── Dividers ────────────────────────────────────────────────── */
hr { border-color: #1a3355 !important; }

/* ── Spinner ─────────────────────────────────────────────────── */
[data-testid="stSpinner"] > div { border-top-color: #4fc3f7 !important; }

/* ── Caption ─────────────────────────────────────────────────── */
.stCaption, [data-testid="stCaptionContainer"] { color: #5a8aaa !important; font-size: 0.78rem !important; }

/* ── Scrollbar ───────────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #070d19; }
::-webkit-scrollbar-thumb { background: #1e3a5c; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #2a5080; }

/* ── Status badge pill ───────────────────────────────────────── */
.status-pill {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}
.pill-active { background: #0a3320; color: #66bb6a; border: 1px solid #2e7d32; }
.pill-idle   { background: #1a1a2e; color: #7986cb; border: 1px solid #3949ab; }
.pill-warn   { background: #1a0f00; color: #ffb74d; border: 1px solid #e65100; }
.pill-breach { background: #1a0505; color: #ef5350; border: 1px solid #b71c1c; }

/* ── Header banner ───────────────────────────────────────────── */
.header-banner {
    background: linear-gradient(90deg, #071428 0%, #0d2040 50%, #071428 100%);
    border: 1px solid #1a3a5c;
    border-radius: 12px;
    padding: 16px 24px;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.header-title {
    font-size: 1.3rem;
    font-weight: 700;
    color: #4fc3f7;
    letter-spacing: -0.02em;
}
.header-sub {
    font-size: 0.75rem;
    color: #4a7a9b;
    margin-top: 2px;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}
.header-badge {
    font-size: 0.7rem;
    color: #4fc3f7;
    border: 1px solid #1a5276;
    border-radius: 6px;
    padding: 4px 10px;
    font-family: 'JetBrains Mono', monospace;
}

/* ── Intent badge ────────────────────────────────────────────── */
.intent-chip {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 20px;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-left: 8px;
    vertical-align: middle;
}
.chip-extract   { background: #0a2040; color: #64b5f6; border: 1px solid #1565c0; }
.chip-hybrid    { background: #1a0a30; color: #ce93d8; border: 1px solid #7b1fa2; }
.chip-general   { background: #0a1a10; color: #81c784; border: 1px solid #2e7d32; }
.chip-benchmark { background: #1a1000; color: #ffcc02; border: 1px solid #f57f17; }
.chip-compare   { background: #001a1a; color: #4dd0e1; border: 1px solid #00838f; }
.chip-ews       { background: #1a0a0a; color: #ef9a9a; border: 1px solid #c62828; }

/* ── Panel section header ────────────────────────────────────── */
.panel-section {
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #3a6a8a;
    margin: 16px 0 6px 0;
    padding-bottom: 4px;
    border-bottom: 1px solid #0f2236;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# CACHED RESOURCES
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Initialising privacy pipeline…")
def load_privacy_pipeline():
    from local.privacy.pipeline import PrivacyPipeline
    return PrivacyPipeline(spacy_model="en_core_web_lg")


@st.cache_resource(show_spinner="Loading Phi-3 GGUF…")
def load_inference_engine():
    try:
        from local.slm.inference import LocalModelInference
        return LocalModelInference(model_path=SLM_MODEL_PATH, ctx_size=4096, gpu_layers=0)
    except (FileNotFoundError, ImportError) as e:
        logger.info("Local inference unavailable: %s", e)
        return None


@st.cache_resource(show_spinner="Loading embedding model…")
def load_embedding_model():
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer("all-MiniLM-L6-v2")
    except ImportError:
        return None


@st.cache_resource(show_spinner="Connecting to Pinecone…")
def load_pinecone_retriever():
    try:
        from local.rag.pinecone_index import PineconeRetriever
        return PineconeRetriever()
    except Exception as e:
        logger.warning("PineconeRetriever: %s", e)
        return None


@st.cache_resource(show_spinner="Loading cross-encoder…")
def load_reranker():
    try:
        from local.rag.reranker import CrossEncoderReranker
        return CrossEncoderReranker()
    except Exception as e:
        logger.warning("CrossEncoderReranker: %s", e)
        return None


# ---------------------------------------------------------------------------
# COMPONENT IMPORTS
# ---------------------------------------------------------------------------
from components.upload_panel        import render_upload_panel
from components.model_toggle        import render_model_toggle
from components.chat                import render_chat_interface
from components.masking_log         import render_masking_log
from components.citations           import render_citations
from components.financial_profile   import render_financial_profile
from components.policy_breach_panel import render_policy_breach_panel
from components.audit_trail         import render_audit_trail

# ---------------------------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------------------------
_DEFAULTS = {
    "last_execution_path":  "Idle",
    "last_citations":       [],
    "last_intent":          "GENERAL",
    "masked_entities":      {},
    "preserved_financials": [],
    "doc_text":             None,
    "doc_faiss_index":      None,
    "registry":             None,
    "last_uploaded_file":   None,   # NOTE: initialised to None, not "session"
    "messages":             [],
    "last_execution_time":  None,
    "financial_profile":    None,
    "breach_report":        None,
    "audit_trail":          [],
    "analysis_mode":        "Standard",
    "doc_b_text":           None,
    "doc_b_label":          None,
    "doc_b_faiss_index":    None,
    "ews_report":           None,
    "ews_cloud_result":     None,
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


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
            val = match.group().strip()
            if len(val) > 1:
                found.add(val)
    return sorted(found)


def _unmask_response(text: str) -> str:
    """Unmask STRICTLY after LLM generation — never before."""
    registry = st.session_state.get("registry")
    if registry is not None:
        return registry.unmask_text(text)
    mapping = st.session_state.get("masked_entities", {})
    for ph in sorted(mapping.keys(), key=len, reverse=True):
        text = text.replace(ph, mapping[ph])
    return text


def _intent_chip(intent: str) -> str:
    classes = {
        "EXTRACT": "chip-extract", "HYBRID": "chip-hybrid",
        "GENERAL": "chip-general", "BENCHMARK": "chip-benchmark",
        "COMPARE": "chip-compare", "EWS": "chip-ews",
    }
    cls = classes.get(intent, "chip-general")
    return f'<span class="intent-chip {cls}">{intent}</span>'


def process_uploaded_document(uploaded_file, slot: str = "primary") -> bool:
    pipeline = load_privacy_pipeline()
    if pipeline is None:
        st.error("🚨 Privacy pipeline failed to initialise.")
        return False

    doc_type = st.session_state.get("document_type", "Internal Credit Proposal (Memo)")

    try:
        result = pipeline.process_document(
            file_source=io.BytesIO(uploaded_file.getvalue()),
            filename=uploaded_file.name,
            doc_type=doc_type,
        )

        masked_entities      = {e["placeholder"]: e["original_entity"] for e in result["audit_log"]}
        preserved_financials = _extract_preserved_financials(result["masked_text"])

        if slot == "primary":
            st.session_state["doc_text"]             = result["masked_text"]
            st.session_state["masked_entities"]      = masked_entities
            st.session_state["preserved_financials"] = preserved_financials
            st.session_state["last_uploaded_file"]   = uploaded_file.name
            st.session_state["registry"]             = result["registry_instance"]
            st.session_state["financial_profile"]    = result.get("financial_profile")
            st.session_state["breach_report"]        = result.get("breach_report")
            st.session_state["ews_report"]           = result.get("ews_report")
            st.session_state["ews_cloud_result"]     = None

            embed_model = load_embedding_model()
            if embed_model:
                from local.rag.chunker     import MarkdownChunker
                from local.rag.local_index import LocalDocumentIndex
                chunks = MarkdownChunker().chunk(result["masked_text"])
                if chunks:
                    idx = LocalDocumentIndex(embed_model)
                    idx.build(chunks)
                    st.session_state["doc_faiss_index"] = idx
        else:
            st.session_state["doc_b_text"]  = result["masked_text"]
            st.session_state["doc_b_label"] = uploaded_file.name
            embed_model = load_embedding_model()
            if embed_model:
                from local.rag.chunker     import MarkdownChunker
                from local.rag.local_index import LocalDocumentIndex
                chunks = MarkdownChunker().chunk(result["masked_text"])
                if chunks:
                    idx = LocalDocumentIndex(embed_model)
                    idx.build(chunks)
                    st.session_state["doc_b_faiss_index"] = idx

        logger.info(
            "Doc processed [%s]: %d entities masked, %d financials",
            slot, len(masked_entities), len(preserved_financials),
        )
        return True

    except Exception as e:
        from local.privacy.validator import LeakageValidationError
        if isinstance(e, LeakageValidationError):
            st.error(f"🚨 Egress firewall blocked: {e}")
        else:
            st.error(f"🚨 Document processing failed: {e}")
        logger.error("Pipeline error [%s] %s: %s", slot, uploaded_file.name, e, exc_info=True)
        return False


# ---------------------------------------------------------------------------
# RETRIEVAL
# ---------------------------------------------------------------------------

def _retrieve_doc_chunks(query: str, top_k: int = _PHI3_TOP_K,
                          for_cloud: bool = False,
                          faiss_key: str = "doc_faiss_index") -> tuple:
    effective_top_k = _CLOUD_TOP_K if for_cloud else top_k
    faiss_index     = st.session_state.get(faiss_key)
    reranker        = load_reranker()

    if faiss_index is None or not faiss_index.is_built:
        return _keyword_fallback(query)

    pool = faiss_index.search(query, top_k=_FAISS_POOL)

    if reranker is not None and len(pool) > 1:
        ranked = reranker.rerank(query, pool, top_k=effective_top_k)
    else:
        ranked = [(c, s) for c, s in pool[:effective_top_k]]

    if not ranked:
        return _keyword_fallback(query)

    chunk_texts, citations = [], []
    for i, (chunk, score) in enumerate(ranked, 1):
        header = f"[Section: {chunk.section}] " if chunk.section else ""
        text   = chunk.text if for_cloud else chunk.text[:_CHUNK_CHAR_CAP]
        chunk_texts.append(f"Chunk {i} {header}(score {score:.3f}):\n{text}")
        citations.append({
            "section": chunk.section or "Document",
            "text":    chunk.text,
            "score":   round(score, 3),
            "page":    f"Chunk {chunk.index + 1} of {faiss_index.chunk_count}",
        })

    context = "\n\n---\n\n".join(chunk_texts)
    if for_cloud:
        return context, citations

    return (
        "System: You are a credit risk analyst. "
        "Answer using ONLY the chunks below. Be concise — 3-4 sentences.\n\n"
        f"Document Chunks (top-{effective_top_k}, reranked):\n\n{context}\n\n"
    ), citations


def _retrieve_regulatory_chunks(query: str, top_k: int = _PHI3_TOP_K) -> tuple:
    pinecone = load_pinecone_retriever()
    reranker = load_reranker()

    if pinecone is None:
        return "", []

    pool = pinecone.search(query, top_k=_RERANK_POOL)

    if reranker is not None and len(pool) > 1:
        ranked = reranker.rerank(query, pool, top_k=top_k)
    else:
        ranked = [(c, 0.0) for c in pool[:top_k]]

    if not ranked:
        return "", []

    chunk_texts, citations = [], []
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
    return f"Regulatory Chunks (top-{top_k}, reranked):\n\n{context}\n\n", citations


def _keyword_fallback(query: str) -> tuple:
    doc_text   = st.session_state.get("doc_text") or ""
    paragraphs = [p.strip() for p in doc_text.split("\n\n") if p.strip()]
    STOP       = {"what","is","the","a","an","of","in","this","about","are",
                  "how","does","do","which","can","be","to","and"}
    keywords   = {w.lower() for w in query.split() if w.lower() not in STOP and len(w) > 2}
    scored     = sorted(
        paragraphs,
        key=lambda p: len(keywords & {w.lower().strip(".,;:()[]") for w in p.split()}),
        reverse=True,
    )
    words: list = []
    for para in scored:
        ws = para.split()
        if len(words) + len(ws) > 500:
            break
        words.extend(ws)
    excerpt = " ".join(words) if words else " ".join(doc_text.split()[:500])
    return (
        "System: You are a credit risk analyst. Answer based on the excerpt below. "
        "Be concise.\n\nDocument Excerpt:\n" + excerpt + "\n\n"
    ), []


# ---------------------------------------------------------------------------
# INTENT CLASSIFIER
# ---------------------------------------------------------------------------

def classify_intent(prompt: str, document_attached: bool, selected_model: str) -> str:
    mode = st.session_state.get("analysis_mode", "Standard")
    if mode == "Compare Two Documents":
        return "COMPARE"
    if mode == "Early Warning Scan":
        return "EWS"

    if selected_model == "Phi-3 (Local Edge)":
        if not document_attached:
            return "GENERAL"
        HYBRID_SIGNALS = {
            "compare","versus","vs","benchmark","exceed","below","above","breach",
            "guideline","policy","requirement","comply","compliance","threshold",
            "limit","minimum","maximum","cbuae","basel","rbi","ifrs","eba","bcbs",
        }
        return "HYBRID" if set(prompt.lower().split()) & HYBRID_SIGNALS else "EXTRACT"

    if not document_attached:
        return "BENCHMARK"

    HYBRID_SIGNALS = {
        "compare","versus","vs","benchmark","exceed","below","above","breach",
        "guideline","policy","requirement","comply","compliance","threshold",
        "limit","minimum","maximum","cbuae","basel","rbi","ifrs","eba","bcbs",
    }
    intent = "HYBRID" if set(prompt.lower().split()) & HYBRID_SIGNALS else "EXTRACT"
    logger.info("Intent='%s' doc=%s model=%s", intent, document_attached, selected_model)
    return intent


# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
with st.sidebar:
    # Brand mark
    st.markdown("""
    <div style="padding: 12px 0 20px 0;">
        <div style="font-size:1.2rem;font-weight:700;color:#4fc3f7;letter-spacing:-0.02em;">
            🛡️ CreditRAG
        </div>
        <div style="font-size:0.65rem;color:#3a6a8a;text-transform:uppercase;letter-spacing:0.12em;margin-top:2px;">
            Risk Intelligence Platform
        </div>
    </div>
    """, unsafe_allow_html=True)

    primary_file, secondary_file = render_upload_panel()
    selected_model               = render_model_toggle()

    # Privacy status
    st.markdown('<div class="panel-section">Security Status</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="background:#071a0e;border:1px solid #1b5e20;border-radius:8px;padding:10px 14px;">
        <div style="font-size:0.72rem;font-weight:600;color:#66bb6a;text-transform:uppercase;letter-spacing:0.08em;">
            🔒 Local Anonymisation Active
        </div>
        <div style="font-size:0.7rem;color:#388e3c;margin-top:4px;line-height:1.5;">
            All PII masked on-device before cloud dispatch.
        </div>
    </div>
    """, unsafe_allow_html=True)

    mode = st.session_state.get("analysis_mode", "Standard")

    # Process primary document
    if primary_file and st.session_state["last_uploaded_file"] != primary_file.name:
        with st.spinner(f"Processing {primary_file.name}…"):
            ok = process_uploaded_document(primary_file, slot="primary")
        if ok:
            idx    = st.session_state.get("doc_faiss_index")
            breach = st.session_state.get("breach_report")
            if idx and idx.is_built:
                st.success(f"✅ {idx.chunk_count} chunks indexed")
            if breach:
                if breach.breach_count:
                    st.error(f"🔴 {breach.breach_count} policy breach(es)")
                elif breach.warning_count:
                    st.warning(f"🟡 {breach.warning_count} warning(s)")
                else:
                    st.success("🟢 All policy checks passed")
            st.rerun()

    # Process secondary document (Compare mode)
    if (
        mode == "Compare Two Documents"
        and secondary_file
        and st.session_state.get("doc_b_label") != secondary_file.name
    ):
        with st.spinner(f"Processing {secondary_file.name}…"):
            ok = process_uploaded_document(secondary_file, slot="secondary")
        if ok:
            idx = st.session_state.get("doc_b_faiss_index")
            if idx and idx.is_built:
                st.success(f"✅ Doc B: {idx.chunk_count} chunks")
            st.rerun()


# ---------------------------------------------------------------------------
# MAIN LAYOUT
# ---------------------------------------------------------------------------

# Header banner
doc_name  = st.session_state.get("last_uploaded_file") or "No document loaded"
mode_disp = st.session_state.get("analysis_mode", "Standard")
st.markdown(f"""
<div class="header-banner">
    <div>
        <div class="header-title">Risk Intelligence Terminal</div>
        <div class="header-sub">Credit Risk RAG · Regulatory Intelligence · Policy Compliance</div>
    </div>
    <div style="text-align:right;">
        <div class="header-badge">📄 {doc_name}</div>
        <div style="font-size:0.65rem;color:#3a6a8a;margin-top:4px;text-transform:uppercase;letter-spacing:0.08em;">
            Mode: {mode_disp}
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([2, 1], gap="medium")

# ── RIGHT PANEL ───────────────────────────────────────────────────────────
with col2:

    # Execution status
    path = st.session_state["last_execution_path"]
    if path == "Idle":
        pill_cls, pill_lbl = "pill-idle", "● IDLE"
    elif "Local" in path:
        pill_cls, pill_lbl = "pill-active", "● LOCAL EDGE"
    else:
        pill_cls, pill_lbl = "pill-active", "● CLOUD ACTIVE"

    col_path, col_lat = st.columns([2, 1])
    with col_path:
        st.markdown(
            f'<div class="panel-section">Execution Path</div>'
            f'<span class="status-pill {pill_cls}">{pill_lbl}</span>'
            f'<div style="font-size:0.72rem;color:#4a7a9b;margin-top:6px;">{path}</div>',
            unsafe_allow_html=True,
        )
    with col_lat:
        if st.session_state["last_execution_time"]:
            st.metric("⏱ Latency", f"{st.session_state['last_execution_time']}s")

    st.markdown("---")

    if st.session_state["doc_text"]:
        st.markdown('<div class="panel-section">Policy & Risk Signals</div>', unsafe_allow_html=True)
        render_policy_breach_panel(st.session_state.get("breach_report"))
        render_financial_profile(st.session_state.get("financial_profile"))

        st.markdown('<div class="panel-section">Privacy Audit</div>', unsafe_allow_html=True)
        render_masking_log(
            st.session_state["masked_entities"],
            st.session_state["preserved_financials"],
        )

    st.markdown('<div class="panel-section">Session Audit Trail</div>', unsafe_allow_html=True)
    # FIX: use `or "session"` to guard None value — .get() default only applies
    # when the key is absent, not when it holds None
    safe_doc_name = st.session_state.get("last_uploaded_file") or "session"
    render_audit_trail(
        st.session_state.get("audit_trail", []),
        doc_filename=safe_doc_name,
    )

# ── LEFT PANEL (main chat) ────────────────────────────────────────────────
with col1:

    # Mode-specific info banners
    if mode_disp == "Compare Two Documents":
        doc_a = st.session_state.get("last_uploaded_file") or "—"
        doc_b = st.session_state.get("doc_b_label") or "not loaded"
        st.markdown(f"""
        <div style="background:#001a1a;border:1px solid #006064;border-radius:10px;padding:12px 16px;margin-bottom:12px;">
            <div style="font-size:0.72rem;font-weight:700;color:#4dd0e1;text-transform:uppercase;letter-spacing:0.08em;">
                🔄 Comparison Mode Active
            </div>
            <div style="font-size:0.8rem;color:#80deea;margin-top:6px;">
                <b>A:</b> {doc_a} &nbsp;·&nbsp; <b>B:</b> {doc_b}
            </div>
        </div>
        """, unsafe_allow_html=True)

    elif mode_disp == "Early Warning Scan":
        st.markdown("""
        <div style="background:#1a0505;border:1px solid #7f0000;border-radius:10px;padding:12px 16px;margin-bottom:12px;">
            <div style="font-size:0.72rem;font-weight:700;color:#ef9a9a;text-transform:uppercase;letter-spacing:0.08em;">
                ⚡ Early Warning Scan Mode
            </div>
            <div style="font-size:0.8rem;color:#ef9a9a;margin-top:4px;opacity:0.8;">
                Queries trigger deep Gemini EWS analysis. Upload a document first.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Last intent indicator
    last_intent = st.session_state.get("last_intent", "")
    if last_intent:
        st.markdown(
            f'<div style="margin-bottom:8px;font-size:0.75rem;color:#4a7a9b;">Last intent: '
            f'{_intent_chip(last_intent)}</div>',
            unsafe_allow_html=True,
        )

    prompt = render_chat_interface()

    if prompt:
        document_attached = bool(st.session_state["doc_text"])
        intent = classify_intent(prompt, document_attached, selected_model)

        # Block Phi-3 for Tier 2 intents
        if intent in ("COMPARE", "EWS") and selected_model == "Phi-3 (Local Edge)":
            st.warning(
                "⚠️ **Compare** and **EWS** modes are cloud-only. "
                "Switch to **Gemini Pro (Cloud)** in the sidebar."
            )
            st.stop()

        try:
            with st.chat_message("assistant"):
                with st.spinner("Analysing…"):
                    start_time = time.time()

                    # ====================================================
                    # COMPARE
                    # ====================================================
                    if intent == "COMPARE":
                        doc_a_text = st.session_state.get("doc_text") or ""
                        doc_b_text = st.session_state.get("doc_b_text") or ""

                        if not doc_a_text:
                            answer_text = "⚠️ Upload **Document A** first (primary upload slot)."
                            st.warning(answer_text)
                        elif not doc_b_text:
                            answer_text = "⚠️ Upload **Document B** (second slot appears in Compare mode)."
                            st.warning(answer_text)
                        else:
                            doc_a_label = st.session_state.get("last_uploaded_file", "Document A")
                            doc_b_label = st.session_state.get("doc_b_label", "Document B")
                            st.caption(f"Sending both documents to Gemini for structured comparison…")
                            logger.info(
                                "COMPARE | A='%s' %d chars | B='%s' %d chars",
                                doc_a_label, len(doc_a_text),
                                doc_b_label, len(doc_b_text),
                            )
                            payload = {
                                "query":             prompt,
                                "doc_a_text":        doc_a_text,
                                "doc_b_text":        doc_b_text,
                                "doc_a_label":       doc_a_label,
                                "doc_b_label":       doc_b_label,
                                "doc_type":          st.session_state.get("document_type", "Document"),
                                "masked_items":      st.session_state["masked_entities"],
                                "include_regulatory": True,
                            }
                            resp = requests.post(API_COMPARE, json=payload, timeout=300)
                            if resp.status_code == 200:
                                data        = resp.json()
                                answer_text = _unmask_response(data.get("answer", ""))
                                st.session_state["last_execution_path"] = data.get("path", "Cloud Compare")
                                st.session_state["last_citations"]      = data.get("citations_a", []) + data.get("citations_b", [])
                                st.write_stream(stream_response(answer_text))
                                # Citation viewer
                                ca = data.get("citations_a", [])
                                cb = data.get("citations_b", [])
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
                            else:
                                answer_text = f"API Error {resp.status_code}: {resp.text}"
                                st.error(answer_text)

                    # ====================================================
                    # EWS
                    # ====================================================
                    elif intent == "EWS":
                        doc_text = st.session_state.get("doc_text") or ""
                        if not doc_text:
                            answer_text = "⚠️ Upload a document before running an EWS scan."
                            st.warning(answer_text)
                        else:
                            st.caption("Running deep early warning scan via Gemini…")
                            ews_report = st.session_state.get("ews_report")
                            local_sigs = ews_report.to_dict().get("signals", []) if ews_report else []
                            payload = {
                                "doc_text":     doc_text,
                                "doc_label":    st.session_state.get("last_uploaded_file", "Document"),
                                "doc_type":     st.session_state.get("document_type", "Document"),
                                "masked_items": st.session_state["masked_entities"],
                                "local_signals": local_sigs,
                                "query":        prompt,
                            }
                            resp = requests.post(API_EWS, json=payload, timeout=300)
                            if resp.status_code == 200:
                                data        = resp.json()
                                answer_text = _unmask_response(data.get("answer", ""))
                                st.session_state["ews_cloud_result"]    = answer_text
                                st.session_state["last_execution_path"] = data.get("path", "Cloud EWS")
                                st.session_state["last_citations"]      = data.get("citations", [])
                                st.write_stream(stream_response(answer_text))
                                render_citations("HYBRID", data.get("citations", []))
                            else:
                                answer_text = f"API Error {resp.status_code}: {resp.text}"
                                st.error(answer_text)

                    # ====================================================
                    # LOCAL EDGE — Phi-3
                    # ====================================================
                    elif selected_model == "Phi-3 (Local Edge)":
                        engine, local_citations = load_inference_engine(), []

                        if engine is None:
                            answer_text = (
                                "⚠️ Local inference unavailable. "
                                f"Ensure the GGUF model exists at `{SLM_MODEL_PATH}`."
                            )
                        else:
                            if intent == "EXTRACT":
                                st.caption("Retrieving from document (FAISS + cross-encoder)…")
                                doc_context, local_citations = _retrieve_doc_chunks(prompt)
                                reg_context = ""
                            elif intent == "HYBRID":
                                st.caption("Retrieving from document and regulatory corpus…")
                                doc_context, dc = _retrieve_doc_chunks(prompt)
                                reg_context, rc = _retrieve_regulatory_chunks(prompt)
                                local_citations = dc + rc
                            else:
                                st.caption("Retrieving from regulatory corpus (Pinecone)…")
                                doc_context = ""
                                reg_context, local_citations = _retrieve_regulatory_chunks(prompt)

                            system_prefix = (
                                "System: You are a credit risk analyst. "
                                "Answer only within credit risk, banking regulation, and financial analysis. "
                                "Be concise — 3-4 sentences.\n\n"
                            )
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
                            answer_text = _unmask_response(f"*(Local Edge — {intent})* {raw_answer}")

                        st.session_state["last_execution_path"] = f"Local Edge · Phi-3 · {intent}"
                        st.session_state["last_citations"]      = local_citations
                        st.session_state["last_intent"]         = intent
                        st.write_stream(stream_response(answer_text))
                        render_citations(intent, local_citations)

                    # ====================================================
                    # CLOUD — Gemini Pro
                    # ====================================================
                    else:
                        doc_payload_text, cloud_doc_citations = "", []
                        if document_attached and intent in ("EXTRACT", "HYBRID"):
                            st.caption("Retrieving relevant chunks from document (FAISS)…")
                            doc_payload_text, cloud_doc_citations = _retrieve_doc_chunks(
                                prompt, for_cloud=True
                            )

                        payload = {
                            "query":        prompt,
                            "intent":       intent,
                            "doc_text":     doc_payload_text,
                            "masked_items": st.session_state["masked_entities"],
                            "doc_type":     st.session_state.get("document_type", "Document"),
                        }

                        logger.info(
                            "Cloud payload | intent=%s doc_preview=%s masked=%d",
                            intent,
                            (doc_payload_text or "")[:120],
                            len(st.session_state["masked_entities"]),
                        )

                        st.caption("Routing to Gemini via FastAPI…")
                        resp = requests.post(API_ENDPOINT, json=payload, timeout=120)

                        if resp.status_code == 200:
                            data        = resp.json()
                            answer_text = _unmask_response(data.get("answer", ""))
                            st.session_state["last_execution_path"] = data.get("path", "Cloud")
                            st.session_state["last_citations"]      = cloud_doc_citations + data.get("citations", [])
                            st.session_state["last_intent"]         = intent
                            st.write_stream(stream_response(answer_text))
                            render_citations(intent, st.session_state["last_citations"])
                        else:
                            answer_text = f"API Error {resp.status_code}: {resp.text}"
                            st.error(answer_text)

                    # ── TIMING + AUDIT TRAIL ──────────────────────────
                    elapsed = round(time.time() - start_time, 2)
                    st.session_state["last_execution_time"] = elapsed
                    st.caption(f"⏱️ **{elapsed}s** · Intent: {intent}")

                    try:
                        from local.analysis.audit_logger import build_entry
                        breach = st.session_state.get("breach_report")
                        ews    = st.session_state.get("ews_report")
                        breach_sum = " | ".join(filter(None, [
                            breach.summary() if breach else None,
                            ews.summary()    if ews    else None,
                        ])) or None

                        st.session_state["audit_trail"].append(build_entry(
                            query          = prompt,
                            intent         = intent,
                            answer         = answer_text,
                            citations      = st.session_state["last_citations"],
                            execution_path = st.session_state["last_execution_path"],
                            elapsed_sec    = elapsed,
                            masked_count   = len(st.session_state.get("masked_entities", {})),
                            doc_filename   = st.session_state.get("last_uploaded_file"),
                            breach_summary = breach_sum,
                        ))
                    except Exception as ae:
                        logger.warning("Audit trail: %s", ae)

                    st.session_state["messages"].append({
                        "role":      "assistant",
                        "content":   answer_text,
                        "citations": st.session_state["last_citations"],
                        "time":      elapsed,
                    })
                    st.rerun()

        except requests.exceptions.ConnectionError:
            st.error(
                "🚨 Connection refused — ensure FastAPI is running: "
                "`uvicorn app.main:app --reload --port 8000`"
            )