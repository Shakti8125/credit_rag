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
from typing import Optional, List, Dict, Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from app.services.generation import GenerationService
from app.services.retrieval  import PineconeRetrievalService, Citation
from app.services.prompt_builder import PromptBuilder
from app.services.doc_injector   import DocumentInjector

logger = logging.getLogger(__name__)
router = APIRouter()

# ---------------------------------------------------------------------------
# Service singletons (initialised once at import time)
# ---------------------------------------------------------------------------
try:
    llm_engine = GenerationService()
except Exception as e:
    logger.warning("GenerationService unavailable: %s", e)
    llm_engine = None

try:
    retrieval_engine = PineconeRetrievalService()
except Exception as e:
    logger.warning("PineconeRetrievalService unavailable: %s", e)
    retrieval_engine = None

doc_injector = DocumentInjector()   # token-length gatekeeper for doc payloads

# Retrieval knobs
_CLOUD_TOP_K       = 6    # final chunks after reranking
_CLOUD_RERANK_POOL = 20  # bi-encoder recall pool before cross-encoder


# ---------------------------------------------------------------------------
# Inline schemas
# ---------------------------------------------------------------------------
class QueryRequest(BaseModel):
    query:        str
    intent:       str
    doc_text:     Optional[str]            = None
    masked_items: Optional[Dict[str, str]] = None   # {placeholder: original_entity}
    doc_type:     Optional[str]            = "Document" # Added dynamically assigned doc_type


class QueryResponse(BaseModel):
    answer:    str
    path:      str
    citations: List[Dict[str, Any]] = []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unmask_text(text: str, masked_items: Optional[Dict[str, str]]) -> str:
    """
    Replaces placeholder tokens in text structures with original entity names.
    Replacements applied longest-key-first to prevent partial substitution
    (e.g. [ORG_10] matched before [ORG_1]).
    """
    if not masked_items or not text:
        return text
    result = text
    for placeholder in sorted(masked_items.keys(), key=len, reverse=True):
        result = result.replace(placeholder, masked_items[placeholder])
    return result


def _build_faiss_index_from_doc(doc_text: str):
    """
    Builds an ephemeral in-memory FAISS index from the masked doc_text.
    """
    try:
        from sentence_transformers import SentenceTransformer
        from local.rag.chunker     import MarkdownChunker
        from local.rag.local_index import LocalDocumentIndex

        chunker = MarkdownChunker()
        chunks  = chunker.chunk(doc_text)

        if not chunks:
            logger.warning("Cloud doc chunker produced 0 chunks.")
            return None

        embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        index       = LocalDocumentIndex(embed_model)
        index.build(chunks)
        logger.info("Cloud ephemeral FAISS: %d chunks indexed.", index.chunk_count)
        return index

    except ImportError as e:
        logger.warning("FAISS/sentence-transformers unavailable on cloud env: %s", e)
        return None
    except Exception as e:
        logger.error("Cloud FAISS build failed: %s", e, exc_info=True)
        return None


def _retrieve_doc_chunks_cloud(query: str, doc_text: str) -> List[Citation]:
    """
    Retrieves relevant chunks from the uploaded document using FAISS +
    cross-encoder reranking. Protects against re-chunking if chunks are 
    already optimized by the client-side engine.
    """
    # Safeguard: If the frontend already passed pre-compiled, formatted context blocks, bypass indexing
    if "Document Chunks" in doc_text or "System:" in doc_text:
        return []

    faiss_index = _build_faiss_index_from_doc(doc_text)
    if faiss_index is None or not faiss_index.is_built:
        return []

    # Bi-encoder recall
    pool = faiss_index.search(query, top_k=_CLOUD_RERANK_POOL)

    # Cross-encoder rerank if retrieval_engine has one loaded
    if (
        retrieval_engine is not None
        and retrieval_engine._cross_encoder is not None
        and len(pool) > 1
    ):
        pairs  = [(query, chunk.text) for chunk, _ in pool]
        scores = retrieval_engine._cross_encoder.predict(pairs)
        ranked = sorted(
            zip([c for c, _ in pool], scores),
            key=lambda x: x[1],
            reverse=True,
        )[:_CLOUD_TOP_K]
    else:
        ranked = [(chunk, score) for chunk, score in pool[:_CLOUD_TOP_K]]

    citations = []
    for chunk, score in ranked:
        citations.append(Citation(
            source       = "Uploaded Document",
            section      = chunk.section or "Document",
            text         = chunk.text,
            page         = chunk.index,
            rerank_score = float(score),
        ))
    return citations


# ---------------------------------------------------------------------------
# Route handler
# ---------------------------------------------------------------------------

@router.post("/query", response_model=QueryResponse, status_code=status.HTTP_200_OK)
async def handle_rag_inference(payload: QueryRequest):
    """
    Cloud RAG inference pipeline routing execution payloads dynamically based on intent.
    """
    logger.info("Cloud pipeline request incoming: intent=%s", payload.intent)

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

        # Uploaded document (FAISS) — EXTRACT, HYBRID
        if payload.intent in ("EXTRACT", "HYBRID") and payload.doc_text:
            try:
                doc_citations = _retrieve_doc_chunks_cloud(payload.query, payload.doc_text)
            except Exception as e:
                logger.warning("Doc-level local FAISS extraction failed: %s", e)

        # =================================================================
        # 2. PROMPT CONSTRUCTION
        # =================================================================
        execution_path = ""
        prompt_string = ""

        # Check if the incoming payload text contains pre-formatted client-side RAG blocks
        is_client_preformatted = payload.doc_text and ("Document Chunks" in payload.doc_text or "System:" in payload.doc_text)

        if is_client_preformatted:
            # Clean optimization pass: use pre-compiled chunk layout directly
            execution_path = f"Cloud Orchestrated · Gemini Pro · {payload.intent} (Client Edge Chunks Optimized)"
            
            if payload.intent == "HYBRID" and reg_citations:
                # Splice in backend regulatory contexts natively next to pre-formatted local chunks
                reg_context = "\n\n".join([f"Regulatory Chunk (score {c.rerank_score:.3f}):\n{c.text}" for c in reg_citations])
                prompt_string = (
                    f"{payload.doc_text}\n\n"
                    f"Regulatory Context Additions:\n{reg_context}\n\n"
                    f"Question: {payload.query}"
                )
            else:
                prompt_string = f"{payload.doc_text}\n\nQuestion: {payload.query}"
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
        # 4. UNMASK RESPONSE & CITATIONS
        # Handles both text generation blocks and citation card arrays 
        # so the client UI completely bypasses any missing replacement fields.
        # =================================================================
        answer_text = _unmask_text(raw_answer, payload.masked_items)

        # Unmask text fields inside the citation mappings to hide raw placeholders
        for c in doc_citations + reg_citations:
            unmasked_citation_text = _unmask_text(c.text, payload.masked_items)
            citations_output.append({
                "source":       c.source,
                "section":      c.section,
                "page":         c.page,
                "text":         unmasked_citation_text,  # ✅ Clear text for expanding UI source components
                "rerank_score": c.rerank_score,
            })

        return QueryResponse(
            answer=answer_text,
            path=execution_path,
            citations=citations_output,
        )

    except Exception as e:
        logger.error("Cloud inference engine runtime pipeline error: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Cloud execution error: {str(e)}",
        )