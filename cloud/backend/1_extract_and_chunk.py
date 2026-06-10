import os
import glob
import json
import logging
from docling.document_converter import DocumentConverter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RAW_DIR = "./base_documents"
STAGING_FILE = "./chunks_staging.json"
CHUNK_SIZE_WORDS = 200
CHUNK_OVERLAP_WORDS = 40

def chunk_text(text: str, max_words: int, overlap: int):
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words - overlap):
        chunks.append(" ".join(words[i:i + max_words]))
        if i + max_words >= len(words):
            break
    return chunks

def extract_to_json():
    converter = DocumentConverter()
    pdf_files = glob.glob(os.path.join(RAW_DIR, "*.pdf"))
    all_chunks = []
    
    for file_path in pdf_files:
        filename = os.path.basename(file_path)
        logger.info(f"Extracting {filename}...")
        try:
            result = converter.convert(file_path)
            raw_text = result.document.export_to_markdown()
            chunks = chunk_text(raw_text, CHUNK_SIZE_WORDS, CHUNK_OVERLAP_WORDS)
            
            for i, chunk_text_data in enumerate(chunks):
                all_chunks.append({
                    "id": f"{filename}_chunk_{i}",
                    "text": chunk_text_data,
                    "source": filename,
                    "section": f"Chunk {i}"
                })
        except Exception as e:
            logger.error(f"Failed to process {filename}: {e}")

    with open(STAGING_FILE, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=4)
    logger.info(f"Successfully saved {len(all_chunks)} chunks to {STAGING_FILE}")

if __name__ == "__main__":
    extract_to_json()