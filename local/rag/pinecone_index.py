"""
local/rag/pinecone_index.py

Pinecone retriever for the LOCAL Phi-3 path — queries the pre-loaded
regulatory corpus (CBUAE, Basel, IFRS-9, etc.) stored in Pinecone.

Used ONLY for GENERAL and HYBRID intents where the query needs regulatory
grounding. EXTRACT intent queries the uploaded document via LocalDocumentIndex
(local_index.py) instead.

Environment variables (loaded from cloud/backend/.env):
    PINECONE_API_KEY
    PINECONE_INDEX_NAME   (default: "creditrag")
    PINECONE_NAMESPACE    (default: "cbuae-manuals")
"""

import os
import logging
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from local.rag.chunker import TextChunk

# Load .env from project root / cloud / backend
_current_dir  = Path(__file__).resolve().parent
_project_root = _current_dir.parent.parent
load_dotenv(dotenv_path=_project_root / "cloud" / "backend" / ".env")

logger = logging.getLogger(__name__)


class PineconeRetriever:
    """
    Bi-encoder recall via Pinecone Inference API (multilingual-e5-large).
    Returns List[TextChunk] — scoring / reranking is handled by
    CrossEncoderReranker in main.py after this call.
    """

    def __init__(self):
        from pinecone import Pinecone

        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise ValueError(
                f"PINECONE_API_KEY is missing. "
                f"Check .env at: {_project_root / 'cloud' / 'backend' / '.env'}"
            )

        self.pc        = Pinecone(api_key=api_key)
        self.index     = self.pc.Index(os.getenv("PINECONE_INDEX_NAME", "creditrag"))
        self.namespace = os.getenv("PINECONE_NAMESPACE", "cbuae-manuals")
        logger.info(
            "PineconeRetriever connected: index=%s namespace=%s",
            os.getenv("PINECONE_INDEX_NAME", "creditrag"), self.namespace,
        )

    def search(self, query: str, top_k: int = 20) -> List[TextChunk]:
        """
        Returns up to top_k TextChunk objects ranked by bi-encoder score.
        Call with a larger pool (20) so the cross-encoder in main.py has
        enough candidates to rerank before truncating to the display limit.
        """
        embed_response = self.pc.inference.embed(
            model="multilingual-e5-large",
            inputs=[query],
            parameters={"input_type": "query"},
        )

        result = self.index.query(
            vector=embed_response.data[0].values,
            top_k=top_k,
            namespace=self.namespace,
            include_metadata=True,
        )

        chunks = []
        for m in result.get("matches", []):
            md = m.get("metadata", {})
            chunks.append(TextChunk(
                text    = md.get("text",    ""),
                index   = len(chunks),
                section = md.get("section", md.get("source", "")),
            ))

        logger.info(
            "Pinecone returned %d candidates for query '%s…'",
            len(chunks), query[:50],
        )
        return chunks
