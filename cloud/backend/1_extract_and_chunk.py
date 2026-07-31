import os
import sys
import glob
import json
from pathlib import Path
from docling.document_converter import DocumentConverter

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from shared.chunking import chunk_by_words
from shared.logging_config import get_logger

logger = get_logger(__name__)

RAW_DIR = "./base_documents"
STAGING_FILE = "./chunks_staging.json"
CHUNK_SIZE_WORDS = 200
CHUNK_OVERLAP_WORDS = 40

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
            chunks = chunk_by_words(raw_text, CHUNK_SIZE_WORDS, CHUNK_OVERLAP_WORDS)
            
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