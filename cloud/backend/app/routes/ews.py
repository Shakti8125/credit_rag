"""
cloud/backend/app/routes/ews.py  — Tier 2: Early Warning Signal deep analysis.
Cloud-only. No local.* imports — fully self-contained.

POST /ews
"""

import logging
import time
from typing import Optional, List, Dict, Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from app.services.engines  import llm_engine as _llm, retrieval_engine as _retrieval, record_telemetry
from app.services.doc_search import chunk_text, hybrid_search

logger = logging.getLogger(__name__)
router = APIRouter()

_TOP_K       = 8
_RERANK_POOL = 25
_CHUNK_SIZE  = 3600
_CHUNK_OVER  = 600

# EWS-focused retrieval query used when user doesn't provide one
_EWS_DEFAULT_QUERY = (
    "risk signals deterioration covenant compliance cashflow impairment "
    "litigation management concentration going concern negative cashflow "
    "covenant waiver refinancing related party transactions"
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class EWSRequest(BaseModel):
    doc_text:      str
    doc_label:     Optional[str]            = "Uploaded Document"
    doc_type:      Optional[str]            = "Internal Credit Proposal (Memo)"
    local_signals: Optional[List[Dict]]    = None
    query:         Optional[str]            = ""


class EWSResponse(BaseModel):
    answer:    str
    path:      str
    citations: List[Dict[str, Any]] = []


# ---------------------------------------------------------------------------
# Retrieval — chunking + hybrid search delegated to app.services.doc_search
# ---------------------------------------------------------------------------

def _build_and_search(
    doc_text: str,
    query:    str,
    label:    str,
) -> List[Dict[str, Any]]:
    """
    Chunks doc_text and runs hybrid (dense + BM25, RRF-fused, cross-encoder
    reranked) retrieval. Returns [{"text", "score", "chunk_idx"}].
    """
    chunks = chunk_text(doc_text, _CHUNK_SIZE, _CHUNK_OVER)
    if not chunks:
        logger.warning("EWS: chunker produced 0 chunks for '%s'.", label)
        return []

    _ce = _retrieval._cross_encoder if _retrieval is not None else None
    return hybrid_search(query, chunks, _TOP_K, _RERANK_POOL,
                         cross_encoder=_ce, label=label)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------



def _format_local_signals(signals: Optional[List[Dict]]) -> str:
    if not signals:
        return "None detected by the local pattern scanner."
    lines = []
    for s in signals:
        sev  = s.get("severity", "?")
        name = s.get("signal",   "Unknown")
        cat  = s.get("category", "")
        excp = s.get("excerpt",  "")
        line = f"  [{sev}] {name} ({cat})"
        if excp:
            line += f'\n    Excerpt: "{excp[:120]}"'
        lines.append(line)
    return "\n".join(lines)


def _build_ews_prompt(
    doc_chunks_str:    str,
    doc_label:         str,
    doc_type:          str,
    local_signals_str: str,
    analyst_query:     str,
) -> str:
    focus = f"\n\nANALYST FOCUS: {analyst_query}" if analyst_query else ""
    return (
        f"You are a Senior Credit Risk Officer performing an Early Warning Signal (EWS) "
        f"review of a {doc_type}.\n\n"
        f"The local risk scanner has already identified the following signals:\n"
        f"{local_signals_str}\n\n"
        f"Your task: perform a DEEP EWS analysis of the document chunks below. "
        f"Go beyond what a regex scanner can find — interpret tone, trends, "
        f"implicit admissions, and forward-looking risk language.\n\n"
        f"Your response MUST follow this exact structure:\n\n"
        f"### Overall Risk Assessment: [HIGH / MEDIUM / LOW / CLEAR]\n"
        f"One sentence justification.\n\n"
        f"### Financial Signals\n"
        f"For each signal: **[SEVERITY] Signal Name** — explanation with data point if available.\n\n"
        f"### Qualitative Signals\n"
        f"Going concern, auditor qualifications, covenant waivers, management changes, "
        f"litigation, negative cashflow language.\n\n"
        f"### Structural Signals\n"
        f"Revenue concentration, refinancing risk, related-party transactions, "
        f"off-balance-sheet exposures, sector stress.\n\n"
        f"### Risk Narrative\n"
        f"2-3 paragraph synthesis of the overall credit risk picture. "
        f"Be specific about what the document reveals and what it conspicuously omits.\n\n"
        f"### Recommended Actions\n"
        f"Bullet list of immediate steps for the credit committee.\n\n"
        f"{'='*60}\n"
        f"DOCUMENT CHUNKS — {doc_label} (cross-encoder reranked, 3600-char chunks):\n"
        f"{doc_chunks_str}"
        f"{focus}"
    )


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------

@router.post("/ews", response_model=EWSResponse, status_code=status.HTTP_200_OK)
async def handle_ews_analysis(payload: EWSRequest):
    """
    EWS deep analysis.

    1. Self-contained chunk + FAISS + cross-encoder rerank on doc_text
    2. Build structured EWS prompt (includes local pre-detected signals)
    3. Gemini generation
    4. Unmask STRICTLY after generation
    """
    logger.info(
        "EWS: doc='%s' type='%s' size=%d chars local_signals=%d",
        payload.doc_label, payload.doc_type,
        len(payload.doc_text),
        len(payload.local_signals or []),
    )

    if not _llm:
        raise HTTPException(status_code=503, detail="Cloud LLM engine offline.")
    if not payload.doc_text.strip():
        raise HTTPException(status_code=400, detail="doc_text is empty.")

    _t0 = time.perf_counter()

    try:
        ews_query = payload.query.strip() if payload.query else _EWS_DEFAULT_QUERY

        results = _build_and_search(payload.doc_text, ews_query, payload.doc_label or "Document")

        # Fallback: if FAISS pipeline failed (missing deps), use truncated raw text
        if results:
            doc_ctx = "\n\n---\n\n".join(
                f"Chunk {r['chunk_idx']+1} (score {r['score']:.3f}):\n{r['text']}"
                for r in results
            )
        else:
            logger.warning("EWS: FAISS unavailable — using raw doc_text (12k chars).")
            doc_ctx = payload.doc_text[:12000]

        local_signals_str = _format_local_signals(payload.local_signals)

        prompt = _build_ews_prompt(
            doc_chunks_str    = doc_ctx,
            doc_label         = payload.doc_label  or "Uploaded Document",
            doc_type          = payload.doc_type   or "Credit Document",
            local_signals_str = local_signals_str,
            analyst_query     = payload.query or "",
        )

        # ── Generate ──────────────────────────────────────────────────
        raw_answer = _llm.generate_text(prompt=prompt, max_tokens=8192)

        # Response returned MASKED — the frontend unmasks locally; this
        # service never holds the placeholder→original map.
        citations = [
            {
                "source":       payload.doc_label or "Document",
                "section":      f"Chunk {r['chunk_idx']+1}",
                "page":         r["chunk_idx"],
                "text":         r["text"],
                "rerank_score": r["score"],
            }
            for r in results
        ]

        # Fixed path string for telemetry, not the returned one — doc_label is
        # a client-supplied filename and must not reach the telemetry ledger.
        record_telemetry("EWS", "Cloud EWS · Gemini",
                         int((time.perf_counter() - _t0) * 1000))

        return EWSResponse(
            answer    = raw_answer,
            path      = f"Cloud EWS · {payload.doc_label} · Gemini",
            citations = citations,
        )

    except Exception as e:
        logger.error("EWS route error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"EWS analysis failed: {str(e)}")
