"""
local/rag/local_index.py

Ephemeral in-memory FAISS index for uploaded-document RAG.

Lifecycle:
  - Built once per uploaded document from the masked Markdown text.
  - Stored in st.session_state["doc_faiss_index"] — discarded when a new
    document is uploaded or the browser session ends.
  - No data written to disk. No network calls.

The sentence-transformer model (all-MiniLM-L6-v2, ~80 MB) is loaded once via
st.cache_resource in main.py and injected at construction time so it is
shared across all index rebuilds without reloading.
"""

import logging
import numpy as np
from typing import List, Tuple

from local.rag.chunker import TextChunk

logger = logging.getLogger(__name__)


class LocalDocumentIndex:
    """
    FAISS FlatIP index (cosine similarity via L2-normalised inner product)
    over sentence-transformer embeddings of TextChunk objects.

    Only used for the uploaded document. Regulatory corpus retrieval uses
    PineconeRetriever (pinecone_index.py).

    Usage:
        index = LocalDocumentIndex(embedding_model)
        index.build(chunks)                          # call once after chunking
        results = index.search("What is the DSCR?", top_k=10)
        # returns List[Tuple[TextChunk, float]]
    """

    def __init__(self, embedding_model):
        self._model  = embedding_model
        self._index  = None          # faiss.IndexFlatIP
        self._chunks: List[TextChunk] = []
        self._dim:    int = 0

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, chunks: List[TextChunk]) -> None:
        """
        Embeds all chunks and populates the FAISS index.
        Must be called once after MarkdownChunker.chunk() returns.
        """
        try:
            import faiss
        except ImportError as e:
            raise ImportError(
                "faiss-cpu is required for local document indexing. "
                "Install with: pip install faiss-cpu"
            ) from e

        if not chunks:
            raise ValueError("Cannot build index from an empty chunk list.")

        self._chunks = chunks
        texts = [c.text for c in chunks]

        logger.info("Embedding %d document chunks…", len(texts))

        embeddings: np.ndarray = self._model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,   # L2 norm → inner-product == cosine similarity
        )

        self._dim   = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(self._dim)
        self._index.add(embeddings.astype(np.float32))

        logger.info(
            "FAISS index built: %d vectors, dim=%d.",
            self._index.ntotal, self._dim,
        )

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 10) -> List[Tuple[TextChunk, float]]:
        """
        Embeds the query and returns the top-k most similar chunks as
        (TextChunk, cosine_score) tuples sorted descending.

        Returns a larger pool than the final top-k so the cross-encoder
        in main.py can rerank before truncating to the display limit.
        """
        if self._index is None:
            raise RuntimeError("Index has not been built yet. Call build() first.")
        if not query.strip():
            return []

        top_k = min(top_k, len(self._chunks))

        query_vec: np.ndarray = self._model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)

        scores, indices = self._index.search(query_vec, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append((self._chunks[idx], float(score)))

        logger.info(
            "FAISS search '%s…': top-%d scores %s",
            query[:40], top_k, [f"{s:.3f}" for _, s in results],
        )
        return results

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_built(self) -> bool:
        return self._index is not None and self._index.ntotal > 0

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    def __repr__(self) -> str:
        status = (
            f"{self._index.ntotal} vectors, dim={self._dim}"
            if self.is_built else "not built"
        )
        return f"LocalDocumentIndex({status})"
