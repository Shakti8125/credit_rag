"""
Hybrid (lexical + semantic) retrieval primitives shared by both tiers.

BM25Okapi is a dependency-free BM25 implementation used for keyword scoring.
rrf_fuse() combines two ranked lists via Reciprocal Rank Fusion — rank-based,
so dense cosine scores and BM25 scores never need to be calibrated against
each other.

Used by:
  - local/rag/local_index.py       (FAISS dense + BM25 over uploaded doc chunks)
  - cloud/.../services/retrieval.py (Pinecone dense pool + BM25 re-scoring)
  - cloud/.../routes/query.py       (pure-BM25 server-side chunk selection —
                                     avoids loading an embedding model per request)
"""

import math
import re
from collections import Counter
from typing import Dict, List, Sequence, Tuple

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[./%-][a-z0-9]+)*")

# Minimal stopword list — keeps domain terms (e.g. "above", "below" matter in
# credit policy queries less than metric names; keep the list conservative)
_STOPWORDS = {
    "the", "a", "an", "of", "in", "on", "to", "and", "or", "is", "are",
    "was", "were", "be", "been", "it", "its", "this", "that", "these",
    "those", "for", "with", "as", "at", "by", "from", "into", "about",
    "what", "which", "who", "how", "does", "do", "did", "can", "could",
    "will", "would", "shall", "should", "has", "have", "had", "not",
}


def tokenize(text: str) -> List[str]:
    """Lowercase alphanumeric tokenizer preserving ratio/percent joins (e.g. 'debt/ebitda', '4.5%')."""
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


class BM25Okapi:
    """
    Standard Okapi BM25 over a fixed corpus of texts.
    Pure Python — no external dependency; fine for the corpus sizes here
    (uploaded-doc chunks and Pinecone recall pools, i.e. tens to a few
    thousand entries).
    """

    def __init__(self, corpus: Sequence[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b  = b
        self._doc_tokens: List[List[str]] = [tokenize(t) for t in corpus]
        self._doc_lens = [len(toks) for toks in self._doc_tokens]
        self._avgdl = (sum(self._doc_lens) / len(self._doc_lens)) if self._doc_lens else 0.0
        self._tfs: List[Counter] = [Counter(toks) for toks in self._doc_tokens]

        df: Counter = Counter()
        for toks in self._doc_tokens:
            df.update(set(toks))
        n = len(self._doc_tokens)
        # BM25+-style floor at 0 to avoid negative IDF for very common terms
        self._idf: Dict[str, float] = {
            term: max(0.0, math.log((n - f + 0.5) / (f + 0.5) + 1.0))
            for term, f in df.items()
        }

    def get_scores(self, query: str) -> List[float]:
        q_tokens = tokenize(query)
        scores = [0.0] * len(self._doc_tokens)
        if not q_tokens or self._avgdl == 0:
            return scores
        for i, (tf, dl) in enumerate(zip(self._tfs, self._doc_lens)):
            s = 0.0
            for term in q_tokens:
                if term not in tf:
                    continue
                idf  = self._idf.get(term, 0.0)
                freq = tf[term]
                denom = freq + self.k1 * (1 - self.b + self.b * dl / self._avgdl)
                s += idf * freq * (self.k1 + 1) / denom
            scores[i] = s
        return scores

    def top_n(self, query: str, n: int) -> List[Tuple[int, float]]:
        """Returns [(corpus_index, score)] for the n best-scoring entries, descending."""
        scores = self.get_scores(query)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return [(i, s) for i, s in ranked[:n] if s > 0.0]


def rrf_fuse(
    rankings: Sequence[Sequence[int]],
    k: int = 60,
) -> List[Tuple[int, float]]:
    """
    Reciprocal Rank Fusion over multiple ranked lists of item ids.

    Each ranking is a sequence of item ids ordered best-first. Returns
    [(item_id, fused_score)] sorted descending. k=60 is the standard
    constant from the original RRF paper.
    """
    fused: Dict[int, float] = {}
    for ranking in rankings:
        for rank, item_id in enumerate(ranking):
            fused[item_id] = fused.get(item_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(fused.items(), key=lambda x: x[1], reverse=True)
