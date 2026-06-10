import os
import logging
from typing import List, Optional
from pinecone import Pinecone
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class Citation(BaseModel):
    source: str
    section: Optional[str] = "General Context"
    text: str
    page: Optional[int] = 0

class PineconeRetrievalService:
    def __init__(self):
        api_key = os.getenv("PINECONE_API_KEY")
        index_name = os.getenv("PINECONE_INDEX_NAME", "creditrag")
        
        if not api_key:
            logger.error("PINECONE_API_KEY environment variable is missing!")
            
        self.pc = Pinecone(api_key=api_key)
        self.index = self.pc.Index(index_name)
        self.embedding_model = "multilingual-e5-large"
        
        logger.info(f"Initialized Pinecone Retrieval via Inference API ({self.embedding_model})")

    def retrieve_context(self, query: str, namespace: str, top_k: int = 5) -> List[Citation]:
        logger.info(f"Retrieving top {top_k} chunks from namespace: '{namespace}'")
        
        try:
            embedding_response = self.pc.inference.embed(
                model=self.embedding_model,
                inputs=[query],
                parameters={"input_type": "query"} 
            )
            query_embedding = embedding_response.data[0].values
        except Exception as e:
            logger.error(f"Failed to embed query via Pinecone: {e}")
            return []
        
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            namespace=namespace,
            include_metadata=True
        )
        
        citations = []
        for match in results.get("matches", []):
            meta = match.get("metadata", {})
            citations.append(Citation(
                source=meta.get("source", "Unknown Document"),
                section=meta.get("section", "General Context"),
                text=meta.get("text", "No text content returned."),
                page=int(meta.get("page", 0)) if meta.get("page") else 0
            ))
            
        return citations