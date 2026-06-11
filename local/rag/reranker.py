"""
local/rag/reranker.py

Cross-encoder reranker shared by both the local Phi-3 path and the cloud
FastAPI path (imported in retrieval.py).

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
  - Fast (6-layer MiniLM), strong on passage re-ranking tasks
  - Returns a raw logit score (higher = more relevant); NOT a probability

Usage:
    reranker = CrossEncoderReranker()           # load once, cache_resource
    ranked   = reranker.rerank(query, chunks, top_k=5)
    # ranked: List[Tuple[TextChunk | str, float]]  sorted descending
"""

import logging
from typing import List, Tuple, Union

from local.rag.chunker import TextChunk

logger = logging.getLogger(__name__)


class CrossEncoderReranker:

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        try:
            from sentence_transformers import CrossEncoder
            self.model      = CrossEncoder(model_name)
            self.model_name = model_name
            logger.info("CrossEncoderReranker loaded: %s", model_name)
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is required for cross-encoder reranking. "
                "Install with: pip install sentence-transformers"
            ) from e

    def rerank(
        self,
        query:  str,
        chunks: List[Union[TextChunk, Tuple[TextChunk, float]]],
        top_k:  int = 5,
    ) -> List[Tuple[TextChunk, float]]:
        """
        Rerank *chunks* against *query* using the cross-encoder.

        Accepts either:
          - List[TextChunk]                — raw chunk objects
          - List[Tuple[TextChunk, float]]  — (chunk, bi_encoder_score) pairs
            from LocalDocumentIndex.search()

        Returns List[Tuple[TextChunk, float]] sorted by cross-encoder score
        descending, truncated to top_k.
        """
        if not chunks:
            return []

        # Normalise input to List[TextChunk]
        raw_chunks: List[TextChunk] = []
        for item in chunks:
            if isinstance(item, tuple):
                raw_chunks.append(item[0])
            else:
                raw_chunks.append(item)

        pairs  = [(query, c.text if hasattr(c, "text") else str(c)) for c in raw_chunks]
        scores = self.model.predict(pairs)

        ranked = sorted(
            zip(raw_chunks, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        logger.info(
            "Cross-encoder reranked %d → top-%d. Best score: %.3f",
            len(ranked), top_k, ranked[0][1] if ranked else 0.0,
        )

        return [(chunk, float(score)) for chunk, score in ranked[:top_k]]
