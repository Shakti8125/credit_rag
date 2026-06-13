"""
local/rag/large_chunker.py

LargeContextChunker — extends MarkdownChunker with larger chunk size and
overlap tuned for Tier 2 tasks (multi-doc comparison, EWS) where broader
context windows improve Gemini's synthesis quality.

Used ONLY on the cloud path. Phi-3 still uses the standard MarkdownChunker.

Chunk size: 3600 chars (~600 words) vs 2400 for standard tasks
Overlap:     600 chars (~17%) vs 400 for standard tasks
"""

from local.rag.chunker import MarkdownChunker

LARGE_CHUNK_SIZE    = 3600
LARGE_CHUNK_OVERLAP = 600


class LargeContextChunker(MarkdownChunker):
    """
    Drop-in replacement for MarkdownChunker with larger chunk/overlap
    settings for Tier 2 cloud-only routes (/compare, /ews).
    """
    def __init__(self):
        super().__init__(
            chunk_size=LARGE_CHUNK_SIZE,
            chunk_overlap=LARGE_CHUNK_OVERLAP,
        )
