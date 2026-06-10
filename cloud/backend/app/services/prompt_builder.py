from typing import List
from app.schemas.response import Citation

class PromptBuilder:
    """
    Centralizes all prompt templates. Controls the stark tonal shift between 
    factual regulatory retrieval (Path A) and critical internal auditing (Path B).
    """

    @staticmethod
    def grounding_prompt(query: str, citations: List[Citation]) -> str:
        """
        Path A (BENCHMARK): Highly factual, grounded strictly in provided regulatory chunks.
        """
        context_blocks = [
            f"--- Source: {c.source} | Section: {c.section} | Page: {c.page} ---\n{c.text}"
            for c in citations
        ]
        context_str = "\n\n".join(context_blocks)
        
        return f"""You are a strict Regulatory Compliance Analyst.
Answer the user's question FACTUALLY based STRICTLY on the provided regulatory chunks. 
You MUST explicitly cite the document name and section/article in your response text.
Do not make claims not supported by the provided chunks. If the context is insufficient, state that clearly.

REGULATORY CONTEXT:
{context_str}

USER QUERY:
{query}"""

    @staticmethod
    def audit_prompt(query: str, doc_text: str) -> str:
        """
        Path B (EXTRACT): Evaluative, identifying compliance gaps and flagging risks.
        """
        return f"""You are an expert Credit Risk Underwriting Auditor.
Evaluate the provided internal credit memo and answer the user's query.
Your objective is to identify compliance gaps, flag potential credit risks, and evaluate the financial metrics presented.
Adopt an analytical, rigorous, and highly evaluative tone.

INTERNAL CREDIT MEMO (ANONYMIZED):
{doc_text}

USER QUERY:
{query}"""

    @staticmethod
    def hybrid_prompt(query: str, citations: List[Citation], doc_text: str) -> str:
        """
        Path C (HYBRID): Synthesizes extracted value from the internal document with 
        external benchmarks from the regulatory corpus.
        """
        context_blocks = [
            f"--- Source: {c.source} | Section: {c.section} ---\n{c.text}"
            for c in citations
        ]
        context_str = "\n\n".join(context_blocks)

        return f"""You are an elite Banking Risk Committee Analyst.
Synthesize the financial data extracted from the internal credit memo with the external regulatory benchmarks provided.
Compare the applicant's profile directly against the historical baselines and regulatory frameworks.
Clearly identify any deviations or violations.

EXTERNAL REGULATORY BENCHMARKS:
{context_str}

INTERNAL CREDIT MEMO (ANONYMIZED):
{doc_text}

USER QUERY:
{query}"""