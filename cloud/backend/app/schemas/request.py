from pydantic import BaseModel, Field
from typing import Dict, Optional

class QueryRequest(BaseModel):
    """
    Strict data contract for incoming requests from the local Streamlit UI.
    Malformed requests will automatically return a 422 Unprocessable Entity.
    """
    query: str = Field(
        ..., 
        description="The masked user query string."
    )
    intent: str = Field(
        ..., 
        description="The routing intent computed locally: EXTRACT, BENCHMARK, HYBRID, or GENERAL."
    )
    doc_text: Optional[str] = Field(
        None, 
        description="The fully masked markdown text of the uploaded document (used for Path B/Hybrid)."
    )
    masked_items: Optional[Dict[str, str]] = Field(
        default_factory=dict, 
        description="Dictionary of masked entities (e.g., {'{CLIENT_A}': 'John Doe'}) for tracking state."
    )