"""
Shared chunking primitives.

TextChunk is the common chunk representation used by both tiers' retrieval
code. chunk_by_words() is the plain word-window splitter used by the cloud
ingestion script (cloud/backend/1_extract_and_chunk.py) for the base
regulatory corpus.

local/rag/chunker.py's MarkdownChunker (markdown-header-aware, langchain
recursive splitting) is intentionally NOT unified with chunk_by_words() —
it produces materially different, higher-quality chunks for the interactively
uploaded documents and depends on langchain-text-splitters, a local-only
dependency. It reuses TextChunk from here for a consistent chunk shape.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class TextChunk:
    text:       str
    index:      int
    section:    str = ""
    word_count: int = 0

    def __post_init__(self):
        self.word_count = len(self.text.split())


def chunk_by_words(text: str, max_words: int, overlap: int) -> List[str]:
    """Plain word-window split — no markdown/semantic awareness."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words - overlap):
        chunks.append(" ".join(words[i:i + max_words]))
        if i + max_words >= len(words):
            break
    return chunks
