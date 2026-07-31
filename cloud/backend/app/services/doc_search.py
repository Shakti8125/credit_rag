"""
Ephemeral uploaded-document search for cloud routes (/compare, /ews).

Consolidates the chunker + FAISS + rerank logic previously duplicated in
compare.py and ews.py. Two key fixes over the old inline versions:

  1. The sentence-transformer embedder is a lazy process-wide singleton —
     previously each request constructed a new SentenceTransformer
     (~1-3s each; /compare paid it twice, once per document).
  2. Retrieval is hybrid: dense (FAISS cosine) and lexical (BM25) rankings
     fused with Reciprocal Rank Fusion before cross-encoder reranking.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from shared.hybrid import BM25Okapi, rrf_fuse

logger = logging.getLogger(__name__)

_embedder = None
_embedder_failed = False


def get_embedder():
    """Lazy singleton for all-MiniLM-L6-v2 — loaded once per process."""
    global _embedder, _embedder_failed
    if _embedder is None and not _embedder_failed:
        try:
            from sentence_transformers import SentenceTransformer
            _embedder = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("Doc-search embedder loaded (all-MiniLM-L6-v2).")
        except Exception as e:
            logger.error("Embedder unavailable: %s", e)
            _embedder_failed = True
    return _embedder


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Recursive character splitter with overlap.
    Splits on paragraph breaks first, then sentences, then words.
    """
    if not text or not text.strip():
        return []

    separators = ["\n\n", "\n", ". ", " ", ""]
    chunks: List[str] = []

    def _split(s: str, sep_idx: int):
        if len(s) <= chunk_size:
            if s.strip():
                chunks.append(s.strip())
            return
        sep   = separators[sep_idx] if sep_idx < len(separators) else ""
        parts = s.split(sep) if sep else list(s)
        current = ""
        for part in parts:
            add = part + (sep if sep else "")
            if len(current) + len(add) <= chunk_size:
                current += add
            else:
                if current.strip():
                    chunks.append(current.strip())
                carried = current[-overlap:] if len(current) > overlap else current
                current = carried + add
        if current.strip():
            chunks.append(current.strip())

    _split(text, 0)
    return [c for c in chunks if len(c.split()) >= 8]  # drop stub fragments


def hybrid_search(
    query:         str,
    chunks:        List[str],
    top_k:         int,
    rerank_pool:   int,
    cross_encoder: Optional[Any] = None,
    label:         str = "document",
) -> List[Dict[str, Any]]:
    """
    Hybrid retrieval over an ephemeral chunk list.
    Dense (FAISS) + BM25 rankings fused via RRF, then cross-encoder reranked.
    Falls back to BM25-only when FAISS/embedder is unavailable.
    Returns list of {"text": str, "score": float, "chunk_idx": int}.
    """
    if not chunks:
        return []

    fetch_k = min(rerank_pool, len(chunks))

    # ── Lexical ranking (always available) ─────────────────────────────
    bm25      = BM25Okapi(chunks)
    bm25_rank = [i for i, _ in bm25.top_n(query, fetch_k)]

    # ── Dense ranking ──────────────────────────────────────────────────
    dense_rank: List[int] = []
    model = get_embedder()
    if model is not None:
        try:
            import faiss

            embeddings = model.encode(
                chunks,
                batch_size=32,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            ).astype(np.float32)

            index = faiss.IndexFlatIP(embeddings.shape[1])
            index.add(embeddings)

            q_vec = model.encode(
                [query], normalize_embeddings=True, convert_to_numpy=True
            ).astype(np.float32)
            _, indices = index.search(q_vec, fetch_k)
            dense_rank = [int(i) for i in indices[0] if i >= 0]
        except Exception as e:
            logger.warning("Dense retrieval failed for %s (%s) — BM25 only.", label, e)

    # ── Fusion ─────────────────────────────────────────────────────────
    rankings = [r for r in (dense_rank, bm25_rank) if r]
    if not rankings:
        # Query shares no vocabulary and dense failed — leading chunks
        fused_ids = list(range(min(top_k, len(chunks))))
    else:
        fused_ids = [i for i, _ in rrf_fuse(rankings)[:fetch_k]]

    candidates = [
        {"text": chunks[i], "score": 0.0, "chunk_idx": i}
        for i in fused_ids
    ]

    # ── Cross-encoder rerank ───────────────────────────────────────────
    if cross_encoder is not None and len(candidates) > 1:
        try:
            pairs     = [(query, c["text"]) for c in candidates]
            ce_scores = cross_encoder.predict(pairs)
            for cand, s in zip(candidates, ce_scores):
                cand["score"] = float(s)
            candidates.sort(key=lambda c: c["score"], reverse=True)
        except Exception as e:
            logger.warning("Cross-encoder reranking failed for %s: %s", label, e)

    logger.info(
        "Hybrid doc search '%s': %d chunks → dense=%d bm25=%d → top-%d",
        label, len(chunks), len(dense_rank), len(bm25_rank), top_k,
    )
    return candidates[:top_k]
