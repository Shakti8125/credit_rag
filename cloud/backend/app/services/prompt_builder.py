"""
cloud/backend/app/services/prompt_builder.py

Centralised prompt templates for the cloud FastAPI route.
All methods are static — no instance required.

Design rules applied to every intent prompt:
  - Explicit role + task, stated once.
  - Grounding contract: answer only from supplied context; say what's missing.
  - Citation contract: name the source document/section inline.
  - Output-shape guidance so answers are scannable (direct answer first,
    tables only for multi-metric comparisons) without forcing structure on
    trivial questions.
  - Placeholder awareness: text is anonymised; tokens like [ORG_1]/[PERSON_2]
    must be preserved verbatim, never guessed at or expanded.
"""

from typing import List
from app.services.retrieval import Citation


_PLACEHOLDER_RULE = (
    "The text is anonymised: tokens like [ORG_1], [PERSON_2], [BANK_1] stand for "
    "masked entities. Reproduce them verbatim wherever you reference them — never "
    "guess, expand, or invent the underlying names.\n"
)

_GROUNDING_RULE = (
    "Ground every claim in the supplied context. If the context does not contain "
    "the answer, say exactly what is missing instead of inferring. Never fabricate "
    "numbers, thresholds, or article references.\n"
)


def _format_citations(citations: List[Citation]) -> str:
    blocks = []
    for c in citations:
        header = f"--- Source: {c.source} | Section: {c.section} | Page: {c.page} ---"
        blocks.append(f"{header}\n{c.text}")
    return "\n\n".join(blocks)


class PromptBuilder:

    @staticmethod
    def grounding_prompt(query: str, citations: List[Citation]) -> str:
        """
        BENCHMARK / GENERAL: Factual answer grounded strictly in regulatory chunks.
        """
        context_str = _format_citations(citations)
        return (
            "ROLE: Regulatory compliance analyst for a UAE bank (CBUAE, Basel III, IFRS 9).\n\n"
            "TASK: Answer the question using ONLY the regulatory extracts below.\n"
            + _GROUNDING_RULE +
            "Cite inline as [<document> §<section>] immediately after each claim.\n"
            "Where the extracts give a specific threshold, ratio, or article number, "
            "quote it exactly — do not round or paraphrase numeric limits.\n\n"
            "FORMAT: Lead with the direct answer in 1-2 sentences. Follow with "
            "supporting detail as short bullets only if the question genuinely has "
            "multiple parts. Keep the whole answer under ~250 words unless the "
            "question demands enumeration.\n\n"
            f"REGULATORY EXTRACTS:\n{context_str}\n\n"
            f"QUESTION:\n{query}"
        )

    @staticmethod
    def audit_prompt(query: str, doc_citations: List[Citation], doc_type: str = "Document") -> str:
        """
        EXTRACT: Answer from retrieved document chunks (hybrid-retrieved, reranked).
        """
        context_str = _format_citations(doc_citations)
        return (
            f"ROLE: Senior credit risk analyst reviewing an anonymised {doc_type}.\n\n"
            "TASK: Answer the question strictly from the document extracts below.\n"
            + _GROUNDING_RULE + _PLACEHOLDER_RULE +
            "Quote metric values exactly as written (units, currency, multiples). "
            "If the same metric appears with different values in different extracts, "
            "surface the discrepancy explicitly.\n\n"
            "FORMAT: Direct answer first. Use a compact table only when comparing "
            "3+ metrics; otherwise prose. Reference extracts inline as "
            "[Chunk <n> §<section>].\n\n"
            f"DOCUMENT EXTRACTS ({doc_type.upper()}, ANONYMISED):\n{context_str}\n\n"
            f"QUESTION:\n{query}"
        )

    @staticmethod
    def audit_prompt_full_doc(query: str, doc_text: str, doc_type: str = "Document") -> str:
        """
        EXTRACT fallback: Full document injection when chunk retrieval is unavailable.
        DocumentInjector has already token-gated doc_text before this is called.
        """
        return (
            f"ROLE: Senior credit risk analyst reviewing an anonymised {doc_type}.\n\n"
            "TASK: Answer the question strictly from the full document below.\n"
            + _GROUNDING_RULE + _PLACEHOLDER_RULE +
            "Quote metric values exactly as written (units, currency, multiples).\n\n"
            "FORMAT: Direct answer first, then supporting evidence with the section "
            "or heading it came from. Keep it concise — do not summarise the whole "
            "document unless asked to.\n\n"
            f"FULL DOCUMENT ({doc_type.upper()}, ANONYMISED):\n{doc_text}\n\n"
            f"QUESTION:\n{query}"
        )

    @staticmethod
    def preformatted_prompt(
        query:       str,
        doc_context: str,
        reg_context: str = "",
        doc_type:    str = "Document",
    ) -> str:
        """
        EXTRACT / HYBRID with client-side pre-retrieved chunks: the frontend
        already ran hybrid retrieval + reranking, so doc_context is the final
        chunk block. Wraps it with the same role/grounding contract as the
        server-retrieved paths (previously this path sent bare text + question
        with no instructions at all).
        """
        reg_section = f"\n\nREGULATORY EXTRACTS:\n{reg_context}" if reg_context else ""
        task = (
            "Compare the document extracts against the regulatory extracts and "
            "answer the question. For each comparison state the document value, "
            "the regulatory threshold, and the verdict (COMPLIANT / BREACH / "
            "NEAR-BREACH)."
            if reg_context else
            "Answer the question strictly from the document extracts."
        )
        return (
            f"ROLE: Senior credit risk analyst reviewing an anonymised {doc_type}.\n\n"
            f"TASK: {task}\n"
            + _GROUNDING_RULE + _PLACEHOLDER_RULE +
            "Quote metric values exactly as written.\n\n"
            "FORMAT: Direct answer first; table only for 3+ metric comparisons.\n\n"
            f"DOCUMENT EXTRACTS ({doc_type.upper()}, ANONYMISED, PRE-RANKED):\n{doc_context}"
            f"{reg_section}\n\n"
            f"QUESTION:\n{query}"
        )

    @staticmethod
    def hybrid_prompt(
        query:         str,
        citations:     List[Citation],
        doc_citations: List[Citation],
        doc_type:      str = "Document"
    ) -> str:
        """
        HYBRID: Synthesise retrieved document chunks with regulatory benchmarks.
        """
        reg_str = _format_citations(citations)
        doc_str = _format_citations(doc_citations)
        return (
            f"ROLE: Credit committee analyst benchmarking an anonymised {doc_type} "
            "against regulatory requirements (CBUAE, Basel III, IFRS 9).\n\n"
            "TASK: Compare the document's figures against the regulatory extracts "
            "and answer the question.\n"
            + _GROUNDING_RULE + _PLACEHOLDER_RULE +
            "For every comparison state three things: the document's value, the "
            "regulatory threshold (with its source cited as [<document> §<section>]), "
            "and the verdict — COMPLIANT, BREACH, or NEAR-BREACH (within 10% of the "
            "limit). If either side of a comparison is missing from the context, say "
            "so rather than assuming a value.\n\n"
            "FORMAT: Direct answer first. Present multi-metric comparisons as a "
            "table: Metric | Document Value | Regulatory Threshold | Verdict. "
            "End with a one-sentence overall assessment.\n\n"
            f"REGULATORY EXTRACTS:\n{reg_str}\n\n"
            f"DOCUMENT EXTRACTS ({doc_type.upper()}, ANONYMISED):\n{doc_str}\n\n"
            f"QUESTION:\n{query}"
        )
