import logging
from typing import List, Tuple

logger=logging.getLogger(__name__)

class CrossEncoderReranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder
        self.model=CrossEncoder(model_name)

    def rerank(self, query, chunks, top_k=4):
        pairs=[(query, c.text if hasattr(c,"text") else c) for c in chunks]
        scores=self.model.predict(pairs)
        ranked=sorted(zip(chunks,scores), key=lambda x:x[1], reverse=True)
        return ranked[:top_k]
