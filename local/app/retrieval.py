"""Document/regulatory chunk retrieval for the Streamlit app (FAISS + Pinecone + cross-encoder)."""

import streamlit as st

from local.app.config import PHI3_TOP_K, FAISS_POOL, CLOUD_TOP_K, CHUNK_CHAR_CAP, RERANK_POOL
from local.app.resources import load_reranker, load_pinecone_retriever


def retrieve_doc_chunks(query: str, top_k: int = PHI3_TOP_K,
                         for_cloud: bool = False,
                         faiss_key: str = "doc_faiss_index") -> tuple:
    effective_top_k = CLOUD_TOP_K if for_cloud else top_k
    faiss_index     = st.session_state.get(faiss_key)
    reranker        = load_reranker()

    if faiss_index is None or not faiss_index.is_built:
        return keyword_fallback(query)

    pool = faiss_index.search(query, top_k=FAISS_POOL)

    if reranker is not None and len(pool) > 1:
        ranked = reranker.rerank(query, pool, top_k=effective_top_k)
    else:
        ranked = [(c, s) for c, s in pool[:effective_top_k]]

    if not ranked:
        return keyword_fallback(query)

    chunk_texts, citations = [], []
    for i, (chunk, score) in enumerate(ranked, 1):
        header = f"[Section: {chunk.section}] " if chunk.section else ""
        text   = chunk.text if for_cloud else chunk.text[:CHUNK_CHAR_CAP]
        chunk_texts.append(f"Chunk {i} {header}(score {score:.3f}):\n{text}")
        citations.append({
            "section": chunk.section or "Document",
            "text":    chunk.text,
            "score":   round(score, 3),
            "page":    f"Chunk {chunk.index + 1} of {faiss_index.chunk_count}",
        })

    context = "\n\n---\n\n".join(chunk_texts)
    if for_cloud:
        return context, citations

    # Context only — the system instruction is owned by the handler
    # (handlers/local_edge.py) so it isn't duplicated in the prompt.
    return (
        f"Document Chunks (top-{effective_top_k}, reranked):\n\n{context}\n\n"
    ), citations


def retrieve_regulatory_chunks(query: str, top_k: int = PHI3_TOP_K) -> tuple:
    pinecone = load_pinecone_retriever()
    reranker = load_reranker()

    if pinecone is None:
        return "", []

    pool = pinecone.search(query, top_k=RERANK_POOL)

    if reranker is not None and len(pool) > 1:
        ranked = reranker.rerank(query, pool, top_k=top_k)
    else:
        ranked = [(c, 0.0) for c in pool[:top_k]]

    if not ranked:
        return "", []

    chunk_texts, citations = [], []
    for i, (chunk, score) in enumerate(ranked, 1):
        header = f"[{chunk.section}] " if chunk.section else ""
        chunk_texts.append(f"Chunk {i} {header}(score {score:.3f}):\n{chunk.text}")
        citations.append({
            "section": chunk.section or "Regulatory Corpus",
            "text":    chunk.text,
            "score":   round(score, 3),
            "page":    f"Chunk {chunk.index + 1}",
        })

    context = "\n\n---\n\n".join(chunk_texts)
    return f"Regulatory Chunks (top-{top_k}, reranked):\n\n{context}\n\n", citations


def keyword_fallback(query: str) -> tuple:
    doc_text   = st.session_state.get("doc_text") or ""
    paragraphs = [p.strip() for p in doc_text.split("\n\n") if p.strip()]
    STOP       = {"what","is","the","a","an","of","in","this","about","are",
                  "how","does","do","which","can","be","to","and"}
    keywords   = {w.lower() for w in query.split() if w.lower() not in STOP and len(w) > 2}
    scored     = sorted(
        paragraphs,
        key=lambda p: len(keywords & {w.lower().strip(".,;:()[]") for w in p.split()}),
        reverse=True,
    )
    words: list = []
    for para in scored:
        ws = para.split()
        if len(words) + len(ws) > 500:
            break
        words.extend(ws)
    excerpt = " ".join(words) if words else " ".join(doc_text.split()[:500])
    return ("Document Excerpt:\n" + excerpt + "\n\n"), []
