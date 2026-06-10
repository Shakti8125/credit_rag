import os
from typing import List, Tuple
from local.rag.chunker import TextChunk

class PineconeRetriever:
    def __init__(self):
        from pinecone import Pinecone
        self.pc=Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index=self.pc.Index(os.getenv("PINECONE_INDEX_NAME","creditrag"))
        self.namespace=os.getenv("PINECONE_NAMESPACE","cbuae-manuals")

    def search(self, query, top_k=20):
        emb=self.pc.inference.embed(model="multilingual-e5-large", inputs=[query], parameters={"input_type":"query"})
        res=self.index.query(vector=emb.data[0].values, top_k=top_k, namespace=self.namespace, include_metadata=True)
        out=[]
        for m in res.get("matches",[]):
            md=m.get("metadata",{})
            out.append(TextChunk(text=md.get("text",""), index=len(out), section=md.get("section","")))
        return out
