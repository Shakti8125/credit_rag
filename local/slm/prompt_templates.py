"""
Defines the strict instruction-following structures and few-shot templates 
required to drive intent classification using the local Phi-3 Mini SLM.
"""

# Phi-3 specific structural control tokens
PHI3_USER_START = "<|user|>"
PHI3_END = "<|end|>"
PHI3_ASSISTANT_START = "<|assistant|>"

# Elite system instruction focusing the model strictly on multi-class categorization
ROUTING_SYSTEM_INSTRUCTION = (
    "You are an elite Credit Risk Intent Router. Your sole task is to classify the user's query "
    "into exactly one of four processing pathways based on the definitions below.\n\n"
    "Pathways:\n"
    "1. EXTRACT: Use when the query is primarily about the UPLOADED DOCUMENT. This includes: "
    "summarising what the document is about, explaining its contents, pulling specific metrics, "
    "numbers, formulas, covenants, definitions, or legal clauses from it, or answering any "
    "open-ended question whose answer lives inside the uploaded file.\n"
    "2. BENCHMARK: Use when the query asks to cross-reference or compare asset performance, risk criteria, "
    "or financial numbers against external historical baselines, market indexes, peer groups, or credit "
    "policy guidelines — WITHOUT needing the uploaded document.\n"
    "3. HYBRID: Use when the query requires BOTH pulling numbers/data from the newly uploaded text AND "
    "benchmarking or comparing those numbers against historical vector archives or credit portfolio baselines.\n"
    "4. GENERAL: Use for standard conversational greetings, meta-questions about system health, or abstract "
    "macroeconomic definitions that do not interact with a specific portfolio or document context.\n\n"
    "Key rule: If a document has been uploaded and the query is asking about IT (even broadly), "
    "default to EXTRACT unless a comparison to external benchmarks is explicitly requested.\n\n"
    "Constraint: Output ONLY the single uppercase classification token (EXTRACT, BENCHMARK, HYBRID, or GENERAL). "
    "Do not include explanations, intro text, wrapping quotes, or punctuation."
)

# High-fidelity few-shot training vectors reflecting practical domain scenarios
FEW_SHOT_EXAMPLES = [
    {
        "query": "What is the uploaded document about?",
        "route": "EXTRACT"
    },
    {
        "query": "Summarise the key findings in this credit memo.",
        "route": "EXTRACT"
    },
    {
        "query": "What is the borrower's calculated DSCR and Debt/EBITDA ratio for FY2025 in this report?",
        "route": "EXTRACT"
    },
    {
        "query": "What are the main risk factors mentioned in this document?",
        "route": "EXTRACT"
    },
    {
        "query": "How does an underwriting LTV of 78% compare to our internal risk tier guidelines for commercial real estate?",
        "route": "BENCHMARK"
    },
    {
        "query": "What are the Basel III minimum capital requirements for market risk?",
        "route": "BENCHMARK"
    },
    {
        "query": "Pull the current year liquidity profile from this memo and verify if it violates our historical portfolio ceilings.",
        "route": "HYBRID"
    },
    {
        "query": "Does the DSCR in this report meet CBUAE guidelines?",
        "route": "HYBRID"
    },
    {
        "query": "Good morning assistant, can you explain what a trailing twelve months calculation represents in standard accounting?",
        "route": "GENERAL"
    },
    {
        "query": "What is the difference between PD and LGD?",
        "route": "GENERAL"
    },
]

def get_formatted_routing_prompt(user_query: str) -> str:
    """
    Assembles system instructions, few-shot credit profiles, and the target 
    query into a compliant Phi-3 instruction-tuned string payload.
    """
    prompt_buffer = [
        f"{PHI3_USER_START}\n{ROUTING_SYSTEM_INSTRUCTION}\n",
        "Here are examples of correct routing classifications:"
    ]
    
    # Append localized examples
    for example in FEW_SHOT_EXAMPLES:
        prompt_buffer.append(f"Query: {example['query']}\nOutput: {example['route']}")
        
    # Append active runtime context block
    prompt_buffer.append("\nNow classify the target query:")
    prompt_buffer.append(f"Query: {user_query}\nOutput:{PHI3_END}")
    prompt_buffer.append(f"{PHI3_ASSISTANT_START}")
    
    return "\n".join(prompt_buffer)