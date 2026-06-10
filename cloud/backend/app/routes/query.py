import logging
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

# Import both the Generation and Retrieval services
from app.services.generation import GenerationService
try:
    from app.services.retrieval import PineconeRetrievalService
except ImportError:
    PineconeRetrievalService = None

logger = logging.getLogger(__name__)

# Primary routing instance
router = APIRouter()

# --- INITIALIZE SERVICES ---
try:
    llm_engine = GenerationService()
except Exception as e:
    logger.warning(f"Could not initialize GenerationService (API key might be missing): {e}")
    llm_engine = None

try:
    if PineconeRetrievalService:
        retrieval_engine = PineconeRetrievalService()
    else:
        retrieval_engine = None
except Exception as e:
    logger.warning(f"Could not initialize PineconeRetrievalService: {e}")
    retrieval_engine = None


# --- INLINE SCHEMAS ---
class QueryRequest(BaseModel):
    query: str
    intent: str
    doc_text: Optional[str] = None
    masked_items: Optional[Dict[str, str]] = None

class QueryResponse(BaseModel):
    answer: str
    path: str
    citations: List[Dict[str, Any]] = []

# --- ROUTE HANDLER ---
@router.post(
    "/query",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK
)
async def handle_rag_inference(payload: QueryRequest):
    """
    HTTP POST route processing operational client traffic.
    Now equipped with dynamic Pinecone Vector Retrieval and PII Masking.
    """
    logger.info(f"Routing cloud execution pipeline for received intent: {payload.intent}")
    
    try:
        if not llm_engine:
            raise ValueError("Cloud LLM Engine is offline.")

        citations_output = []
        retrieved_context = ""

        # =====================================================================
        # 1. RETRIEVAL PHASE (Triggered if intent requires base regulatory documents)
        # =====================================================================
        if payload.intent in ["BENCHMARK", "GENERAL", "HYBRID"] and retrieval_engine:
            try:
                # Query Pinecone for the top 3 most relevant chunks
                raw_citations = retrieval_engine.retrieve_context(
                    query=payload.query, 
                    namespace="cbuae-manuals", # Change this if your Pinecone namespace is different
                    top_k=3 
                )
                
                # Format context string and prepare citations for the Streamlit UI
                context_blocks = []
                for c in raw_citations:
                    context_blocks.append(f"--- Document: {c.source} (Page {c.page}) ---\n{c.text}")
                    citations_output.append({
                        "source": c.source,
                        "page": c.page,
                        "text": c.text
                    })
                
                retrieved_context = "\n\n".join(context_blocks)
            except Exception as retrieve_err:
                logger.warning(f"Pinecone retrieval failed: {retrieve_err}")

        # =====================================================================
        # 2. MASKING & DYNAMIC PROMPT CONSTRUCTION PHASE
        # =====================================================================
        
        # Start with the raw document text from the payload
        scrubbed_doc_text = payload.doc_text if payload.doc_text else ""
        
        # Apply the masking dictionary if items are present
        if payload.masked_items and scrubbed_doc_text:
            logger.info("Applying PII masking/scrubbing to uploaded document text...")
            for placeholder, original_value in payload.masked_items.items():
                # Replace the sensitive original value with its safe placeholder (e.g., 'COMPANY_A')
                scrubbed_doc_text = scrubbed_doc_text.replace(original_value, placeholder)

        prompt_string = ""
        execution_path_name = ""

        if payload.intent == "HYBRID" and scrubbed_doc_text:
            execution_path_name = "Cloud Hybrid (Uploaded Memo + Pinecone RAG)"
            prompt_string = (
                f"You are a credit risk analyst. Use the two sources below to answer the query.\n\n"
                f"SOURCE 1: Uploaded Client Memo (Scrubbed)\n{scrubbed_doc_text}\n\n"
                f"SOURCE 2: Institutional Base Documents / Guidelines\n{retrieved_context}\n\n"
                f"Question: {payload.query}"
            )
        
        elif payload.intent in ["BENCHMARK", "GENERAL"]:
            execution_path_name = "Cloud RAG (Pinecone Base Docs)"
            prompt_string = (
                f"You are a credit risk analyst. Use the regulatory guidelines provided below to answer the query.\n\n"
                f"Base Regulatory Context:\n{retrieved_context}\n\n"
                f"Question: {payload.query}"
            )
            
        else: # EXTRACT intent
            execution_path_name = "Cloud Extraction (Uploaded Memo Only)"
            prompt_string = (
                f"Context:\n{scrubbed_doc_text}\n\n"
                f"Question:\n{payload.query}"
            )

        # Safety fallback
        if not prompt_string.strip():
            prompt_string = payload.query

        # 🔍 DEBUG PRINT: This will print the exact prompt text directly to your terminal screen
        print("\n" + "="*50)
        print("🚀 [DEBUG] EXACT PROMPT BEING SENT TO GEMINI:")
        print("="*50)
        print(prompt_string)
        print("="*50 + "\n")

        # =====================================================================
        # 3. GENERATION PHASE
        # =====================================================================
        answer_text = llm_engine.generate_text(prompt=prompt_string)

        # Return the comprehensive JSON structure
        return QueryResponse(
            answer=answer_text,
            path=execution_path_name,
            citations=citations_output
        )
        
    except Exception as e:
        logger.error(f"Pipeline breakdown within route controller: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Cloud execution error: {str(e)}"
        )