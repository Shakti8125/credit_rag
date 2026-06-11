"""
local/rag/chunker.py

RAG chunking for financial/regulatory documents.
- Markdown header-aware splitting
- Larger semantic chunks suited to dense regulatory prose
- Sentence boundary preservation
- Placeholder sanitisation for masked entity tokens
"""

import re
import logging
from dataclasses import dataclass
from typing import List

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration  — increased for dense regulatory/credit-memo documents
# ---------------------------------------------------------------------------

DEFAULT_CHUNK_SIZE    = 2400   # chars  (~400 words — captures full regulatory clauses)
DEFAULT_CHUNK_OVERLAP = 400    # chars  (~17% overlap preserves cross-sentence context)
MIN_CHUNK_WORDS       = 10     # discard stub fragments

HEADERS_TO_SPLIT = [
    ("#",    "h1"),
    ("##",   "h2"),
    ("###",  "h3"),
    ("####", "h4"),
]


@dataclass
class TextChunk:
    text:       str
    index:      int
    section:    str = ""
    word_count: int = 0

    def __post_init__(self):
        self.word_count = len(self.text.split())


# ---------------------------------------------------------------------------
# Placeholder sanitiser
# Converts registry tokens like [ORG_1] → Organisation_1 so the SLM doesn't
# hit RLHF refusal behaviour on bracket-wrapped unknown tokens.
# ---------------------------------------------------------------------------

_PLACEHOLDER_MAP = {
    "ORG":    "Organisation",
    "PERSON": "Person",
    "GPE":    "Location",
    "LOC":    "Location",
    "FAC":    "Facility",
    "DATE":   "Date",
    "EMAIL":  "email-redacted",
    "PHONE":  "phone-redacted",
    "SSN":    "id-redacted",
    "EIN":    "ein-redacted",
}

_PLACEHOLDER_RE = re.compile(r"\[([A-Z]+)_(\d+)\]")


def _sanitise_placeholders(text: str) -> str:
    def _replace(match: re.Match) -> str:
        label   = match.group(1)
        index   = match.group(2)
        readable = _PLACEHOLDER_MAP.get(label, label.capitalize())
        return f"{readable}_{index}"
    return _PLACEHOLDER_RE.sub(_replace, text)


# ---------------------------------------------------------------------------
# Chunker
# ---------------------------------------------------------------------------

class MarkdownChunker:

    def __init__(
        self,
        chunk_size:    int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ):
        self.chunk_size    = chunk_size
        self.chunk_overlap = chunk_overlap
        self._build_splitters()

    def _build_splitters(self):
        try:
            from langchain_text_splitters import (
                MarkdownHeaderTextSplitter,
                RecursiveCharacterTextSplitter,
            )
        except ImportError:
            raise ImportError("Install: pip install langchain-text-splitters")

        # Stage 1: split on markdown structure; headers stored in metadata
        self._header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=HEADERS_TO_SPLIT,
            strip_headers=True,
        )

        # Stage 2: recursive semantic splitting
        # Comma deliberately excluded — it breaks financial statements
        self._recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
            is_separator_regex=False,
        )

    def chunk(self, text: str) -> List[TextChunk]:
        if not text or not text.strip():
            return []

        header_docs = self._header_splitter.split_text(text)
        all_docs    = self._recursive_splitter.split_documents(header_docs)

        chunks = []
        for doc in all_docs:
            raw_text = doc.page_content.strip()
            if len(raw_text.split()) < MIN_CHUNK_WORDS:
                continue

            meta    = doc.metadata
            section = " > ".join(str(v) for k, v in sorted(meta.items()) if v)

            clean_text = _sanitise_placeholders(raw_text)
            if section:
                clean_text = f"Section: {section}\n\n{clean_text}"

            chunks.append(TextChunk(
                text=clean_text,
                index=len(chunks),
                section=section,
            ))

        logger.info(
            "Chunked %d chars into %d chunks (size=%d overlap=%d)",
            len(text), len(chunks), self.chunk_size, self.chunk_overlap,
        )
        return chunks
