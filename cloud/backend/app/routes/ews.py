"""
cloud/backend/app/routes/ews.py  — Tier 2: Early Warning Signal deep analysis.
Cloud-only. No local.* imports — fully self-contained.

POST /ews
"""

import logging
import numpy as np
from typing import Optional, List, Dict, Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from app.services.generation import GenerationService
from app.services.retrieval  import PineconeRetrievalService, Citation

logger = logging.getLogger(__name__)
router = APIRouter()

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
    masked_items:  Optional[Dict[str, str]] = None
    local_signals: Optional[List[Dict]]    = None
    query:         Optional[str]            = ""


class EWSResponse(BaseModel):
    answer:    str
    path:      str
    citations: List[Dict[str, Any]] = []


# ---------------------------------------------------------------------------
# Self-contained chunker (mirrors compare.py — no local.* dependency)
# ---------------------------------------------------------------------------

def _chunk_text(text: str) -> List[str]:
    if not text or not text.strip():
        return []

    separators = ["\n\n", "\n", ". ", " ", ""]
    chunks: List[str] = []

    def _split(s: str, sep_idx: int):
        if len(s) <= _CHUNK_SIZE:
            if s.strip():
                chunks.append(s.strip())
            return
        sep   = separators[sep_idx] if sep_idx < len(separators) else ""
        parts = s.split(sep) if sep else list(s)
        current = ""
        for part in parts:
            add = part + (sep if sep else "")
            if len(current) + len(add) <= _CHUNK_SIZE:
                current += add
            else:
                if current.strip():
                    chunks.append(current.strip())
                overlap = current[-_CHUNK_OVER:] if len(current) > _CHUNK_OVER else current
                current = overlap + add
        if current.strip():
            chunks.append(current.strip())

    _split(text, 0)
    return [c for c in chunks if len(c.split()) >= 8]


# ---------------------------------------------------------------------------
# Self-contained FAISS (no local.* dependency)
# ---------------------------------------------------------------------------

def _build_and_search(
    doc_text: str,
    query:    str,
    label:    str,
) -> List[Dict[str, Any]]:
    """
    Chunks doc_text, builds FAISS index, retrieves top-k, cross-encoder reranks.
    Returns list of {"text": str, "score": float, "chunk_idx": int}.
    """
    chunks = _chunk_text(doc_text)
    if not chunks:
        logger.warning("EWS: chunker produced 0 chunks for '%s'.", label)
        return []

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

        fetch_k = min(_RERANK_POOL, len(chunks))
        q_vec   = model.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True
        ).astype(np.float32)

        scores, indices = index.search(q_vec, fetch_k)

        candidates = [
            {"text": chunks[i], "score": float(s), "chunk_idx": int(i)}
            for s, i in zip(scores[0], indices[0])
            if i >= 0
        ]

        # Cross-encoder rerank
        if _retrieval is not None and _retrieval._cross_encoder is not None and len(candidates) > 1:
            try:
                pairs     = [(query, c["text"]) for c in candidates]
                ce_scores = _retrieval._cross_encoder.predict(pairs)
                for cand, ce_s in zip(candidates, ce_scores):
                    cand["score"] = float(ce_s)
                candidates.sort(key=lambda c: c["score"], reverse=True)
            except Exception as e:
                logger.warning("EWS cross-encoder reranking failed: %s", e)

        logger.info("EWS FAISS: %d chunks indexed, %d retrieved for '%s'.", len(chunks), len(candidates[:_TOP_K]), label)
        return candidates[:_TOP_K]

    except ImportError as e:
        logger.error("faiss-cpu or sentence-transformers missing: %s", e)
        return []
    except Exception as e:
        logger.error("EWS FAISS error for '%s': %s", label, e, exc_info=True)
        return []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unmask(text: str, masked_items: Optional[Dict[str, str]]) -> str:
    if not masked_items or not text:
        return text
    for ph in sorted(masked_items.keys(), key=len, reverse=True):
        text = text.replace(ph, masked_items[ph])
    return text


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

        # ── Unmask STRICTLY after generation ──────────────────────────
        answer = _unmask(raw_answer, payload.masked_items)

        citations = [
            {
                "source":       payload.doc_label or "Document",
                "section":      f"Chunk {r['chunk_idx']+1}",
                "page":         r["chunk_idx"],
                "text":         _unmask(r["text"], payload.masked_items),
                "rerank_score": r["score"],
            }
            for r in results
        ]

        return EWSResponse(
            answer    = answer,
            path      = f"Cloud EWS · {payload.doc_label} · Gemini",
            citations = citations,
        )

    except Exception as e:
        logger.error("EWS route error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"EWS analysis failed: {str(e)}")
