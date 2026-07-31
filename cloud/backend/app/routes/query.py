"""
cloud/backend/app/routes/query.py

Cloud RAG inference endpoint.

Flow per request:
  1. Retrieve relevant chunks from Pinecone (BENCHMARK / HYBRID / GENERAL)
     with cross-encoder reranking — via PineconeRetrievalService.
  2. For EXTRACT / HYBRID: retrieve relevant chunks from the uploaded
     document via an ephemeral FAISS index built server-side from doc_text
     (if full text is provided), or leverage pre-compiled client chunks.
  3. Build prompt using PromptBuilder templates.
  4. Generate answer with Gemini via GenerationService.
  5. Unmask the LLM response and citation metadata cards before returning.
"""

import logging
import time
from typing import Optional, List, Dict, Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from app.services.retrieval  import Citation
from app.services.engines    import llm_engine, retrieval_engine, record_telemetry
from app.services.prompt_builder import PromptBuilder
from app.services.doc_injector   import DocumentInjector
from shared.chunking import chunk_by_words
from shared.hybrid import BM25Okapi

logger = logging.getLogger(__name__)
router = APIRouter()

doc_injector = DocumentInjector()   # token-length gatekeeper for doc payloads

# Retrieval knobs
_CLOUD_TOP_K       = 6    # final chunks after reranking
_CLOUD_RERANK_POOL = 20  # bi-encoder recall pool before cross-encoder


# ---------------------------------------------------------------------------
# Inline schemas
# ---------------------------------------------------------------------------
class QueryRequest(BaseModel):
    # PRIVACY CONTRACT: everything in this payload is already masked by the
    # local tier. There is deliberately NO masked_items field — the
    # placeholder→original dictionary must never leave the analyst's machine;
    # responses are returned masked and the frontend unmasks locally.
    query:        str
    intent:       str
    doc_text:     Optional[str]            = None
    doc_type:     Optional[str]            = "Document" # Added dynamically assigned doc_type
    # True when doc_text holds chunks already retrieved+reranked client-side.
    # Explicit flag — the old substring sniffing ("Document Chunks"/"System:")
    # silently failed for normal frontend payloads, so the server re-chunked
    # and re-indexed pre-retrieved text on every request.
    doc_is_prechunked: bool = False


class QueryResponse(BaseModel):
    answer:    str
    path:      str
    citations: List[Dict[str, Any]] = []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------



def _retrieve_doc_chunks_cloud(query: str, doc_text: str) -> List[Citation]:
    """
    Selects relevant chunks from raw uploaded-document text using BM25
    (shared.hybrid) + optional cross-encoder reranking.

    This path only fires when the frontend sends RAW doc text instead of
    pre-retrieved chunks (the normal flow — the Streamlit tier runs hybrid
    FAISS retrieval client-side and sends preformatted context). BM25 keeps
    this fallback deployment-safe: no embedding model download/load per
    request, no local/* imports, works cold in a Lambda container.
    """
    # Safeguard: if the frontend already passed pre-compiled, formatted
    # context blocks, bypass server-side selection entirely.
    if "Document Chunks" in doc_text or "System:" in doc_text:
        return []

    chunks = chunk_by_words(doc_text, max_words=400, overlap=60)
    if not chunks:
        return []

    bm25 = BM25Okapi(chunks)
    pool = bm25.top_n(query, n=min(_CLOUD_RERANK_POOL, len(chunks)))
    if not pool:
        # Query shares no vocabulary with the doc — fall back to leading chunks
        pool = [(i, 0.0) for i in range(min(_CLOUD_TOP_K, len(chunks)))]

    # Cross-encoder rerank if retrieval_engine has one loaded
    if (
        retrieval_engine is not None
        and retrieval_engine._cross_encoder is not None
        and len(pool) > 1
    ):
        pairs  = [(query, chunks[i]) for i, _ in pool]
        scores = retrieval_engine._cross_encoder.predict(pairs)
        ranked = sorted(
            zip([i for i, _ in pool], scores),
            key=lambda x: x[1],
            reverse=True,
        )[:_CLOUD_TOP_K]
    else:
        ranked = pool[:_CLOUD_TOP_K]

    return [
        Citation(
            source       = "Uploaded Document",
            section      = "Document",
            text         = chunks[i],
            page         = i,
            rerank_score = float(score),
        )
        for i, score in ranked
    ]


# ---------------------------------------------------------------------------
# Route handler
# ---------------------------------------------------------------------------

@router.post("/query", response_model=QueryResponse, status_code=status.HTTP_200_OK)
async def handle_rag_inference(payload: QueryRequest):
    """
    Cloud RAG inference pipeline routing execution payloads dynamically based on intent.
    """
    logger.info("Cloud pipeline request incoming: intent=%s", payload.intent)
    _t0 = time.perf_counter()

    try:
        if not llm_engine:
            raise ValueError("Cloud LLM generation engine instance is offline.")

        reg_citations: List[Citation]  = []
        doc_citations: List[Citation]  = []
        citations_output: List[Dict]   = []

        # =================================================================
        # 1. RETRIEVAL PHASE
        # =================================================================

        # Regulatory corpus (Pinecone) — BENCHMARK, GENERAL, HYBRID
        if payload.intent in ("BENCHMARK", "GENERAL", "HYBRID") and retrieval_engine:
            try:
                reg_citations = retrieval_engine.retrieve_context(
                    query=payload.query,
                    namespace="cbuae-manuals",
                    top_k=_CLOUD_TOP_K,
                    rerank_pool=_CLOUD_RERANK_POOL,
                )
            except Exception as e:
                logger.warning("Pinecone infrastructure retrieval failed: %s", e)

        # Explicit flag from the frontend, plus legacy substring sniffing for
        # any older client that doesn't send doc_is_prechunked yet.
        is_client_preformatted = bool(payload.doc_text) and (
            payload.doc_is_prechunked
            or "Document Chunks" in payload.doc_text
            or "System:" in payload.doc_text
        )

        # Uploaded document — EXTRACT, HYBRID. Only when the frontend sent RAW
        # text; pre-retrieved chunks must never be re-chunked server-side.
        if (
            payload.intent in ("EXTRACT", "HYBRID")
            and payload.doc_text
            and not is_client_preformatted
        ):
            try:
                doc_citations = _retrieve_doc_chunks_cloud(payload.query, payload.doc_text)
            except Exception as e:
                logger.warning("Doc-level chunk selection failed: %s", e)

        # =================================================================
        # 2. PROMPT CONSTRUCTION
        # =================================================================
        execution_path = ""
        prompt_string = ""

        if is_client_preformatted:
            # Clean optimization pass: use pre-compiled chunk layout directly
            execution_path = f"Cloud Orchestrated · Gemini Pro · {payload.intent} (Client Edge Chunks Optimized)"

            reg_context = ""
            if payload.intent == "HYBRID" and reg_citations:
                reg_context = "\n\n".join(
                    f"Regulatory Chunk (score {c.rerank_score:.3f}):\n{c.text}"
                    for c in reg_citations
                )
            prompt_string = PromptBuilder.preformatted_prompt(
                query=payload.query,
                doc_context=payload.doc_text,
                reg_context=reg_context,
                doc_type=payload.doc_type,
            )
        else:
            # Full fallback framework processing path if raw full string text is injected
            if payload.intent == "HYBRID":
                execution_path = "Cloud Hybrid (Doc Chunks + Pinecone RAG)"
                prompt_string  = PromptBuilder.hybrid_prompt(
                    query=payload.query,
                    citations=reg_citations,
                    doc_citations=doc_citations,
                    doc_type=payload.doc_type # Connected doc_type here
                )

            elif payload.intent == "EXTRACT":
                execution_path = "Cloud Extraction (Doc Chunks — FAISS reranked)"
                if doc_citations:
                    prompt_string = PromptBuilder.audit_prompt(
                        query=payload.query,
                        doc_citations=doc_citations,
                        doc_type=payload.doc_type # Connected doc_type here
                    )
                else:
                    execution_path = "Cloud Extraction (Full Doc Injection — FAISS unavailable)"
                    safe_doc       = doc_injector.prepare_payload(payload.doc_text or "")
                    prompt_string  = PromptBuilder.audit_prompt_full_doc(
                        query=payload.query,
                        doc_text=safe_doc,
                        doc_type=payload.doc_type # Connected doc_type here
                    )

            else:  # BENCHMARK / GENERAL
                execution_path = "Cloud RAG (Pinecone — cross-encoder reranked)"
                prompt_string  = PromptBuilder.grounding_prompt(
                    query=payload.query,
                    citations=reg_citations,
                )

        if not prompt_string.strip():
            prompt_string = payload.query

        # =================================================================
        # 3. GENERATION
        # =================================================================
        raw_answer = llm_engine.generate_text(prompt=prompt_string)

        # =================================================================
        # 4. RESPONSE — returned MASKED. Unmasking happens on the analyst's
        # machine; this service never holds the placeholder→original map.
        # =================================================================
        for c in doc_citations + reg_citations:
            citations_output.append({
                "source":       c.source,
                "section":      c.section,
                "page":         c.page,
                "text":         c.text,
                "rerank_score": c.rerank_score,
            })

        record_telemetry(
            payload.intent, execution_path, int((time.perf_counter() - _t0) * 1000)
        )

        return QueryResponse(
            answer=raw_answer,
            path=execution_path,
            citations=citations_output,
        )

    except Exception as e:
        logger.error("Cloud inference engine runtime pipeline error: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Cloud execution error: {str(e)}",
        )