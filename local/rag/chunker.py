"""
local/rag/chunker.py

Improved RAG chunking:
- Markdown header aware splitting
- Larger semantic chunks
- Sentence boundary preservation
- Better metadata handling
"""

import re
import logging
from dataclasses import dataclass
from typing import List

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Increased for financial/regulatory documents
DEFAULT_CHUNK_SIZE = 1800
DEFAULT_CHUNK_OVERLAP = 250

# Allow short but meaningful financial definitions
MIN_CHUNK_WORDS = 10


HEADERS_TO_SPLIT = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
    ("####", "h4"),
]


@dataclass
class TextChunk:
    text: str
    index: int
    section: str = ""
    word_count: int = 0

    def __post_init__(self):
        self.word_count = len(self.text.split())


# ---------------------------------------------------------------------------
# Placeholder sanitiser
# ---------------------------------------------------------------------------

_PLACEHOLDER_MAP = {
    "ORG": "Organisation",
    "PERSON": "Person",
    "GPE": "Location",
    "LOC": "Location",
    "FAC": "Facility",
    "DATE": "Date",
    "EMAIL": "email@redacted",
    "PHONE": "phone-redacted",
    "SSN": "id-redacted",
    "EIN": "ein-redacted",
}


_PLACEHOLDER_RE = re.compile(r"\[([A-Z]+)_(\d+)\]")


def _sanitise_placeholders(text: str) -> str:

    def _replace(match: re.Match) -> str:
        label = match.group(1)
        index = match.group(2)

        readable = _PLACEHOLDER_MAP.get(
            label,
            label.capitalize()
        )

        return f"{readable}_{index}"

    return _PLACEHOLDER_RE.sub(_replace, text)



# ---------------------------------------------------------------------------
# Chunker
# ---------------------------------------------------------------------------


class MarkdownChunker:

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ):

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self._build_splitters()


    def _build_splitters(self):

        try:
            from langchain_text_splitters import (
                MarkdownHeaderTextSplitter,
                RecursiveCharacterTextSplitter,
            )

        except ImportError:
            raise ImportError(
                "Install: pip install langchain-text-splitters"
            )


        # ---------------------------------------------------------------
        # Stage 1:
        # Split by markdown structure
        #
        # Headers removed from content because they are stored in metadata.
        # This prevents repeated headers from dominating embeddings.
        # ---------------------------------------------------------------

        self._header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=HEADERS_TO_SPLIT,
            strip_headers=True,
        )


        # ---------------------------------------------------------------
        # Stage 2:
        # Recursive semantic splitting
        #
        # Removed comma splitting because it breaks financial statements.
        # ---------------------------------------------------------------

        self._recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,

            separators=[
                "\n\n",
                "\n",
                ". ",
                " ",
                ""
            ],

            length_function=len,

            is_separator_regex=False,
        )



    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk(self, text: str) -> List[TextChunk]:

        if not text or not text.strip():
            return []


        # Stage 1
        header_docs = self._header_splitter.split_text(text)


        # Stage 2
        all_docs = self._recursive_splitter.split_documents(header_docs)


        chunks = []


        for doc in all_docs:

            raw_text = doc.page_content.strip()


            # Skip useless fragments
            if len(raw_text.split()) < MIN_CHUNK_WORDS:
                continue



            meta = doc.metadata


            section = " > ".join(
                str(v)
                for k, v in sorted(meta.items())
                if v
            )


            clean_text = _sanitise_placeholders(raw_text)


            # Put section back as context without embedding noise
            if section:

                clean_text = (
                    f"Section: {section}\n\n"
                    f"{clean_text}"
                )


            chunks.append(
                TextChunk(
                    text=clean_text,
                    index=len(chunks),
                    section=section,
                )
            )


        logger.info(
            "Chunked %d chars into %d chunks "
            "(size=%d overlap=%d)",

            len(text),
            len(chunks),
            self.chunk_size,
            self.chunk_overlap,
        )


        return chunks