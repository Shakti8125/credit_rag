"""
cloud/backend/app/routes/compare.py  — Tier 2: Multi-document comparison.
Cloud-only. No local.* imports — fully self-contained.

All chunking and FAISS logic is inlined so this module works purely within
the cloud/backend package scope.

POST /compare
"""

import re
import logging
import numpy as np
from typing import Optional, List, Dict, Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from app.services.generation import GenerationService
from app.services.retrieval  import PineconeRetrievalService, Citation

logger = logging.getLogger(__name__)
router = APIRouter()

# ---------------------------------------------------------------------------
# Singletons
# ---------------------------------------------------------------------------
try:
    _llm = GenerationService()
except Exception as e:
    logger.warning("GenerationService unavailable: %s", e)
    _llm = None

try:
    _retrieval = PineconeRetrievalService()
except Exception as e:
    logger.warning("PineconeRetrievalService unavailable: %s", e)
    _retrieval = None

# Tier 2 knobs — larger than standard route
_TOP_K       = 7
_RERANK_POOL = 25
_CHUNK_SIZE  = 3600   # chars — larger for Gemini's 1M token window
_CHUNK_OVER  = 600


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class CompareRequest(BaseModel):
    query:              str
    doc_a_text:         str
    doc_b_text:         str
    doc_a_label:        Optional[str]            = "Document A"
    doc_b_label:        Optional[str]            = "Document B"
    doc_type:           Optional[str]            = "Credit Document"
    masked_items:       Optional[Dict[str, str]] = None
    include_regulatory: bool                     = True


class CompareResponse(BaseModel):
    answer:        str
    path:          str
    citations_a:   List[Dict[str, Any]] = []
    citations_b:   List[Dict[str, Any]] = []
    reg_citations: List[Dict[str, Any]] = []


# ---------------------------------------------------------------------------
# Self-contained chunker (no local.* dependency)
# ---------------------------------------------------------------------------

def _chunk_text(text: str) -> List[str]:
    """
    Simple recursive character splitter with overlap.
    Splits on paragraph breaks first, then sentences, then words.
    Returns list of chunk strings.
    """
    if not text or not text.strip():
        return []

    separators = ["\n\n", "\n", ". ", " ", ""]
    chunks: List[str] = []

    def _split(s: str, sep_idx: int):
        if len(s) <= _CHUNK_SIZE:
            if s.strip():
                chunks.append(s.strip())
            return
        sep = separators[sep_idx] if sep_idx < len(separators) else ""
        parts = s.split(sep) if sep else list(s)
        current = ""
        for part in parts:
            add = part + (sep if sep else "")
            if len(current) + len(add) <= _CHUNK_SIZE:
                current += add
            else:
                if current.strip():
                    chunks.append(current.strip())
                # overlap: carry last _CHUNK_OVER chars forward
                overlap = current[-_CHUNK_OVER:] if len(current) > _CHUNK_OVER else current
                current = overlap + add
        if current.strip():
            chunks.append(current.strip())

    _split(text, 0)
    return [c for c in chunks if len(c.split()) >= 8]  # drop stub fragments


# ---------------------------------------------------------------------------
# Self-contained FAISS index builder (no local.* dependency)
# ---------------------------------------------------------------------------

def _build_faiss_index(chunks: List[str], label: str):
    """
    Embeds chunks with all-MiniLM-L6-v2 and returns a (faiss_index, embeddings)
    tuple. Returns (None, None) on failure.
    """
    try:
        import faiss
        from sentence_transformers import SentenceTransformer

        model      = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(
            chunks,
            batch_size=32,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)

        dim   = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        logger.info("FAISS built for %s: %d chunks, dim=%d", label, len(chunks), dim)
        return index, model

    except ImportError as e:
        logger.error("faiss-cpu or sentence-transformers missing: %s", e)
        return None, None
    except Exception as e:
        logger.error("FAISS build failed for %s: %s", label, e, exc_info=True)
        return None, None


def _search_faiss(
    query: str,
    chunks: List[str],
    index,
    model,
    top_k: int = _TOP_K,
    rerank_pool: int = _RERANK_POOL,
) -> List[Dict[str, Any]]:
    """
    Searches the FAISS index for the query, optionally reranks with cross-encoder.
    Returns list of {"text": str, "score": float, "chunk_idx": int}.
    """
    if index is None or not chunks:
        return []

    fetch_k = min(rerank_pool, len(chunks))

    q_vec = model.encode(
        [query], normalize_embeddings=True, convert_to_numpy=True
    ).astype(np.float32)

    scores, indices = index.search(q_vec, fetch_k)

    candidates = [
        {"text": chunks[i], "score": float(s), "chunk_idx": int(i)}
        for s, i in zip(scores[0], indices[0])
        if i >= 0
    ]

    # Cross-encoder rerank if available
    if _retrieval is not None and _retrieval._cross_encoder is not None and len(candidates) > 1:
        try:
            pairs  = [(query, c["text"]) for c in candidates]
            ce_scores = _retrieval._cross_encoder.predict(pairs)
            for cand, ce_s in zip(candidates, ce_scores):
                cand["score"] = float(ce_s)
            candidates.sort(key=lambda c: c["score"], reverse=True)
            logger.info("Cross-encoder reranked %d→top-%d", len(candidates), top_k)
        except Exception as e:
            logger.warning("Cross-encoder reranking failed: %s", e)

    return candidates[:top_k]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unmask(text: str, masked_items: Optional[Dict[str, str]]) -> str:
    if not masked_items or not text:
        return text
    for ph in sorted(masked_items.keys(), key=len, reverse=True):
        text = text.replace(ph, masked_items[ph])
    return text


def _format_for_prompt(results: List[Dict], label: str) -> str:
    if not results:
        return f"[No content retrieved for {label} — document may be too short to chunk]"
    blocks = []
    for i, r in enumerate(results, 1):
        blocks.append(f"Chunk {i} (score {r['score']:.3f}):\n{r['text']}")
    return "\n\n---\n\n".join(blocks)


def _build_compare_prompt(
    query: str,
    doc_a_label: str, doc_a_ctx: str,
    doc_b_label: str, doc_b_ctx: str,
    reg_ctx: str,
    doc_type: str,
) -> str:
    reg_section = f"\n\nREGULATORY BENCHMARKS:\n{reg_ctx}" if reg_ctx else ""
    return (
        f"You are an elite Banking Risk Committee Analyst specialising in {doc_type} review.\n\n"
        f"Perform a structured side-by-side comparison of the two documents below.\n"
        f"Your response MUST cover all five sections:\n\n"
        f"### 1. Key Metric Deltas\n"
        f"Compare every financial ratio/metric present in both documents "
        f"(DSCR, LTV, leverage, ICR, revenue, PAT, NPA, CAR, etc.). "
        f"If a metric appears in only one document, state that explicitly.\n\n"
        f"### 2. Risk Profile Shift\n"
        f"Which document represents a stronger/weaker credit risk and why? "
        f"Support with specific data points.\n\n"
        f"### 3. Regulatory Compliance\n"
        f"Benchmark both against CBUAE / Basel III thresholds. "
        f"Flag any breaches or near-breaches in either document.\n\n"
        f"### 4. Notable Qualitative Differences\n"
        f"Covenants, collateral, tenor, concentration risk, management changes, "
        f"methodology differences, model assumptions.\n\n"
        f"### 5. Credit Committee Recommendation\n"
        f"One-paragraph recommendation addressing both documents.\n\n"
        f"Be precise. Where data is absent in one document, state that explicitly "
        f"rather than inferring.\n\n"
        f"{'='*60}\n"
        f"DOCUMENT A — {doc_a_label}:\n{doc_a_ctx}\n\n"
        f"{'='*60}\n"
        f"DOCUMENT B — {doc_b_label}:\n{doc_b_ctx}"
        f"{reg_section}\n\n"
        f"{'='*60}\n"
        f"ANALYST QUERY: {query}"
    )


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------

@router.post("/compare", response_model=CompareResponse, status_code=status.HTTP_200_OK)
async def handle_comparison(payload: CompareRequest):
    """
    Multi-document comparison.

    1. Chunk both documents server-side with self-contained chunker (3600 chars)
    2. Build ephemeral FAISS indices for each
    3. Retrieve + cross-encoder rerank from each
    4. Optionally pull Pinecone regulatory benchmarks
    5. Build structured 5-section comparison prompt
    6. Gemini generation
    7. Unmask STRICTLY after generation
    """
    logger.info(
        "Compare: '%s' (%d chars) vs '%s' (%d chars)",
        payload.doc_a_label, len(payload.doc_a_text),
        payload.doc_b_label, len(payload.doc_b_text),
    )

    if not _llm:
        raise HTTPException(status_code=503, detail="Cloud LLM engine offline.")

    if not payload.doc_a_text.strip():
        raise HTTPException(status_code=400, detail="doc_a_text is empty.")
    if not payload.doc_b_text.strip():
        raise HTTPException(status_code=400, detail="doc_b_text is empty.")

    try:
        # ── Chunk ─────────────────────────────────────────────────────
        chunks_a = _chunk_text(payload.doc_a_text)
        chunks_b = _chunk_text(payload.doc_b_text)
        logger.info(
            "Chunks: A=%d B=%d (size=%d overlap=%d)",
            len(chunks_a), len(chunks_b), _CHUNK_SIZE, _CHUNK_OVER,
        )

        # ── Build FAISS ───────────────────────────────────────────────
        idx_a, model_a = _build_faiss_index(chunks_a, payload.doc_a_label)
        idx_b, model_b = _build_faiss_index(chunks_b, payload.doc_b_label)

        # ── Retrieve ──────────────────────────────────────────────────
        results_a = _search_faiss(payload.query, chunks_a, idx_a, model_a)
        results_b = _search_faiss(payload.query, chunks_b, idx_b, model_b)

        reg_citations: List[Citation] = []
        if payload.include_regulatory and _retrieval:
            try:
                reg_citations = _retrieval.retrieve_context(
                    query=payload.query,
                    namespace="cbuae-manuals",
                    top_k=_TOP_K,
                    rerank_pool=_RERANK_POOL,
                )
            except Exception as e:
                logger.warning("Regulatory retrieval failed: %s", e)

        # ── Build prompt ──────────────────────────────────────────────
        reg_ctx = "\n\n---\n\n".join(
            f"Source: {c.source} | {c.section}\n{c.text}" for c in reg_citations
        ) if reg_citations else ""

        prompt = _build_compare_prompt(
            query       = payload.query,
            doc_a_label = payload.doc_a_label,
            doc_a_ctx   = _format_for_prompt(results_a, payload.doc_a_label),
            doc_b_label = payload.doc_b_label,
            doc_b_ctx   = _format_for_prompt(results_b, payload.doc_b_label),
            reg_ctx     = reg_ctx,
            doc_type    = payload.doc_type or "Credit Document",
        )

        # ── Generate ──────────────────────────────────────────────────
        raw_answer = _llm.generate_text(prompt=prompt, max_tokens=8192)

        # ── Unmask STRICTLY after generation ──────────────────────────
        answer = _unmask(raw_answer, payload.masked_items)

        # Build citation dicts (unmask text field too)
        def _to_cit_dicts(results, label):
            return [
                {
                    "source":       label,
                    "section":      f"Chunk {r['chunk_idx'] + 1}",
                    "page":         r["chunk_idx"],
                    "text":         _unmask(r["text"], payload.masked_items),
                    "rerank_score": r["score"],
                }
                for r in results
            ]

        return CompareResponse(
            answer        = answer,
            path          = (
                f"Cloud Compare · {payload.doc_a_label} vs "
                f"{payload.doc_b_label} · Gemini"
            ),
            citations_a   = _to_cit_dicts(results_a, payload.doc_a_label),
            citations_b   = _to_cit_dicts(results_b, payload.doc_b_label),
            reg_citations = [
                {
                    "source":       c.source,
                    "section":      c.section,
                    "page":         c.page,
                    "text":         _unmask(c.text, payload.masked_items),
                    "rerank_score": c.rerank_score,
                }
                for c in reg_citations
            ],
        )

    except Exception as e:
        logger.error("Compare route error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")
