"""
cloud/backend/app/services/prompt_builder.py

Centralised prompt templates for the cloud FastAPI route.
All methods are static — no instance required.
"""

from typing import List
from app.services.retrieval import Citation


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
            "You are a strict Regulatory Compliance Analyst.\n"
            "Answer the user's question FACTUALLY based STRICTLY on the provided "
            "regulatory chunks. Cite the document name and section in your response. "
            "Do not make claims not supported by the chunks. "
            "If the context is insufficient, state that clearly.\n\n"
            f"REGULATORY CONTEXT:\n{context_str}\n\n"
            f"USER QUERY:\n{query}"
        )

    @staticmethod
    def audit_prompt(query: str, doc_citations: List[Citation], doc_type: str = "Document") -> str:
        """
        EXTRACT: Answer from retrieved document chunks (FAISS reranked).
        """
        context_str = _format_citations(doc_citations)
        return (
            "You are an expert Credit Risk Underwriting Auditor.\n"
            f"Evaluate the provided {doc_type} and answer the user's query.\n"
            "Identify compliance gaps, flag potential credit risks, and evaluate "
            "the financial metrics presented. Be analytical and rigorous.\n\n"
            f"DOCUMENT CHUNKS ({doc_type.upper()} - ANONYMISED, RERANKED):\n{context_str}\n\n"
            f"USER QUERY:\n{query}"
        )

    @staticmethod
    def audit_prompt_full_doc(query: str, doc_text: str, doc_type: str = "Document") -> str:
        """
        EXTRACT fallback: Full document injection when FAISS is unavailable.
        DocumentInjector has already token-gated doc_text before this is called.
        """
        return (
            "You are an expert Credit Risk Underwriting Auditor.\n"
            f"Evaluate the provided {doc_type} and answer the user's query.\n"
            "Identify compliance gaps, flag potential credit risks, and evaluate "
            "the financial metrics presented. Be analytical and rigorous.\n\n"
            f"UPLOADED DOCUMENT ({doc_type.upper()} - ANONYMISED):\n{doc_text}\n\n"
            f"USER QUERY:\n{query}"
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
            "You are an elite Banking Risk Committee Analyst.\n"
            f"Synthesise the financial data from the {doc_type} with the "
            "external regulatory benchmarks. Compare the applicant's profile directly "
            "against the regulatory frameworks. Identify any deviations or violations.\n\n"
            f"EXTERNAL REGULATORY BENCHMARKS:\n{reg_str}\n\n"
            f"DOCUMENT CHUNKS ({doc_type.upper()} - ANONYMISED):\n{doc_str}\n\n"
            f"USER QUERY:\n{query}"
        )