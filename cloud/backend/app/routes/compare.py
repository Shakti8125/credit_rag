"""
cloud/backend/app/routes/compare.py  — Tier 2: Multi-document comparison.
Cloud-only. No local.* imports.

Chunking + hybrid retrieval delegated to app.services.doc_search
(shared with /ews).

POST /compare
"""

import logging
import time
from typing import Optional, List, Dict, Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from app.services.retrieval import Citation
from app.services.engines  import llm_engine as _llm, retrieval_engine as _retrieval, record_telemetry
from app.services.doc_search import chunk_text, hybrid_search

logger = logging.getLogger(__name__)
router = APIRouter()

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
    include_regulatory: bool                     = True


class CompareResponse(BaseModel):
    answer:        str
    path:          str
    citations_a:   List[Dict[str, Any]] = []
    citations_b:   List[Dict[str, Any]] = []
    reg_citations: List[Dict[str, Any]] = []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------



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

    _t0 = time.perf_counter()

    try:
        # ── Chunk ─────────────────────────────────────────────────────
        chunks_a = chunk_text(payload.doc_a_text, _CHUNK_SIZE, _CHUNK_OVER)
        chunks_b = chunk_text(payload.doc_b_text, _CHUNK_SIZE, _CHUNK_OVER)
        logger.info(
            "Chunks: A=%d B=%d (size=%d overlap=%d)",
            len(chunks_a), len(chunks_b), _CHUNK_SIZE, _CHUNK_OVER,
        )

        # ── Hybrid retrieve (dense + BM25 fused, cross-encoder reranked) ──
        _ce = _retrieval._cross_encoder if _retrieval is not None else None
        results_a = hybrid_search(payload.query, chunks_a, _TOP_K, _RERANK_POOL,
                                  cross_encoder=_ce, label=payload.doc_a_label)
        results_b = hybrid_search(payload.query, chunks_b, _TOP_K, _RERANK_POOL,
                                  cross_encoder=_ce, label=payload.doc_b_label)

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

        # Response returned MASKED — the frontend unmasks locally; this
        # service never holds the placeholder→original map.
        def _to_cit_dicts(results, label):
            return [
                {
                    "source":       label,
                    "section":      f"Chunk {r['chunk_idx'] + 1}",
                    "page":         r["chunk_idx"],
                    "text":         r["text"],
                    "rerank_score": r["score"],
                }
                for r in results
            ]

        # Telemetry records a fixed path string, NOT the returned one — the
        # doc labels are client-supplied filenames, which the adversarial eval
        # already treats as a PII carrier. Nothing user-derived leaves here.
        record_telemetry("COMPARE", "Cloud Compare · Gemini",
                         int((time.perf_counter() - _t0) * 1000))

        return CompareResponse(
            answer        = raw_answer,
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
                    "text":         c.text,
                    "rerank_score": c.rerank_score,
                }
                for c in reg_citations
            ],
        )

    except Exception as e:
        logger.error("Compare route error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")
