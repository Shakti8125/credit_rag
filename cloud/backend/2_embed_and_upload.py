import os
import json
import logging
from dotenv import load_dotenv
from pinecone import Pinecone

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()

STAGING_FILE = "./chunks_staging.json"
NAMESPACE = "cbuae-manuals"
PINECONE_MODEL = "multilingual-e5-large" 

def embed_and_upload_cloud():
    if not os.path.exists(STAGING_FILE):
        logger.error(f"Cannot find {STAGING_FILE}.")
        return

    with open(STAGING_FILE, "r", encoding="utf-8") as f:
        documents = json.load(f)

    logger.info(f"Loaded {len(documents)} chunks. Connecting to Pinecone...")

    api_key = os.getenv("PINECONE_API_KEY")
    pc = Pinecone(api_key=api_key)
    
    # IMPORTANT: Update this Host URL to your new 1024-dimension index URL
    index = pc.Index(
        name="creditrag", 
        host="https://creditrag-b8sfzgi.svc.aped-4627-b74a.pinecone.io"
    )

    batch_size = 20
    logger.info(f"Using Pinecone's '{PINECONE_MODEL}' to embed and upload...")
    
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        texts_to_embed = [doc["text"] for doc in batch]
        
        embedding_response = pc.inference.embed(
            model=PINECONE_MODEL,
            inputs=texts_to_embed,
            parameters={"input_type": "passage", "truncate": "END"}
        )
        
        vectors_payload = []
        for doc, embedding_data in zip(batch, embedding_response.data):
            vectors_payload.append({
                "id": doc["id"],
                "values": embedding_data.values,
                "metadata": {
                    "text": doc["text"],
                    "source": doc["source"],
                    "section": doc["section"]
                }
            })
            
        index.upsert(vectors=vectors_payload, namespace=NAMESPACE)
        logger.info(f" -> Upserted batch {i // batch_size + 1}")
        
    logger.info("Cloud ingestion completely finished!")

if __name__ == "__main__":
    embed_and_upload_cloud()