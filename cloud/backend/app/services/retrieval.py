"""
cloud/backend/app/services/retrieval.py

Pinecone bi-encoder recall + cross-encoder reranking for the cloud FastAPI route.

retrieve_context() fetches a larger pool (rerank_pool) from Pinecone, then
reranks with cross-encoder/ms-marco-MiniLM-L-6-v2 before returning top_k.
"""

import logging
from typing import List, Optional

from pinecone import Pinecone
from pydantic import BaseModel

from app.services.secrets import get_secret
from shared.env import get_env
from shared.hybrid import BM25Okapi, rrf_fuse

logger = logging.getLogger(__name__)


class Citation(BaseModel):
    source:       str
    section:      Optional[str]   = "General Context"
    text:         str
    page:         Optional[int]   = 0
    rerank_score: Optional[float] = None


class PineconeRetrievalService:

    def __init__(self):
        # get_secret resolves from the environment first and falls back to SSM
        # Parameter Store, so under Lambda the key never has to sit in a
        # plaintext function environment variable. Local dev is unchanged.
        try:
            api_key = get_secret("PINECONE_API_KEY")
        except Exception as e:
            api_key = None
            logger.error("PINECONE_API_KEY could not be resolved from env or SSM: %s", e)

        index_name = get_env("PINECONE_INDEX_NAME", "creditrag")

        self.pc              = Pinecone(api_key=api_key)
        self.index           = self.pc.Index(index_name)
        self.embedding_model = "multilingual-e5-large"

        # Cross-encoder — loaded eagerly so errors surface at startup
        self._cross_encoder = None
        self._load_cross_encoder()

        logger.info(
            "PineconeRetrievalService ready (embed=%s reranker=%s)",
            self.embedding_model,
            "ms-marco-MiniLM-L-6-v2" if self._cross_encoder else "disabled",
        )

    def _load_cross_encoder(self):
        try:
            from sentence_transformers import CrossEncoder
            self._cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            logger.info("Cross-encoder loaded: ms-marco-MiniLM-L-6-v2")
        except ImportError:
            logger.warning(
                "sentence-transformers not installed — cross-encoder reranking disabled. "
                "Install with: pip install sentence-transformers"
            )
        except Exception as e:
            logger.warning("Cross-encoder load failed (%s) — reranking disabled.", e)

    def retrieve_context(
        self,
        query:       str,
        namespace:   str,
        top_k:       int = 6,
        rerank_pool: int = 20,
    ) -> List[Citation]:
        """
        Retrieve top_k most relevant chunks with cross-encoder reranking.

        Parameters
        ----------
        query       : user question (may be masked — masking doesn't hurt embedding quality)
        namespace   : Pinecone namespace to search
        top_k       : final number of chunks returned after reranking
        rerank_pool : candidates fetched from Pinecone before reranking
        """
        logger.info(
            "Retrieving top %d (pool=%d) from namespace '%s'",
            top_k, rerank_pool, namespace,
        )

        # ── Embed ──────────────────────────────────────────────────────
        try:
            embed_response = self.pc.inference.embed(
                model=self.embedding_model,
                inputs=[query],
                parameters={"input_type": "query"},
            )
            query_embedding = embed_response.data[0].values
        except Exception as e:
            logger.error("Query embedding failed: %s", e)
            return []

        # ── Recall ─────────────────────────────────────────────────────
        fetch_k = max(rerank_pool, top_k)
        results = self.index.query(
            vector=query_embedding,
            top_k=fetch_k,
            namespace=namespace,
            include_metadata=True,
        )

        matches = results.get("matches", [])
        if not matches:
            logger.warning("Pinecone returned 0 matches for namespace '%s'.", namespace)
            return []

        candidates: List[Citation] = []
        for match in matches:
            meta = match.get("metadata", {})
            candidates.append(Citation(
                source       = meta.get("source",  "Unknown Document"),
                section      = meta.get("section", "General Context"),
                text         = meta.get("text",    ""),
                page         = int(meta.get("page", 0)) if meta.get("page") else 0,
                rerank_score = None,
            ))

        # ── Hybrid fusion: dense order (Pinecone) + BM25 over the pool ──
        # RRF is rank-based so the two score scales never need calibration.
        # The fused list is also truncated before the cross-encoder, halving
        # the CPU reranking cost versus scoring the full recall pool.
        ce_pool = min(len(candidates), max(top_k * 2, 10))
        try:
            bm25       = BM25Okapi([c.text for c in candidates])
            dense_rank = list(range(len(candidates)))         # Pinecone already sorted
            bm25_rank  = [i for i, _ in bm25.top_n(query, len(candidates))]
            fused      = rrf_fuse([dense_rank, bm25_rank])[:ce_pool]
            candidates = [candidates[i] for i, _ in fused]
            logger.info("Hybrid fusion: pool %d → %d for cross-encoder.", len(matches), len(candidates))
        except Exception as e:
            logger.warning("BM25 fusion failed (%s) — dense order only.", e)
            candidates = candidates[:ce_pool]

        # ── Cross-encoder rerank ───────────────────────────────────────
        if self._cross_encoder is not None and len(candidates) > 1:
            pairs = [(query, c.text) for c in candidates]
            try:
                scores = self._cross_encoder.predict(pairs)
                for citation, score in zip(candidates, scores):
                    citation.rerank_score = float(score)
                candidates.sort(key=lambda c: (c.rerank_score or 0.0), reverse=True)
                logger.info(
                    "Reranked %d → top-%d. Best=%.3f",
                    len(candidates), top_k, candidates[0].rerank_score,
                )
            except Exception as e:
                logger.warning("Cross-encoder reranking failed (%s) — using hybrid order.", e)

        return candidates[:top_k]
