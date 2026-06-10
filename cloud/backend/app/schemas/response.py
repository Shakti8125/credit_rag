from pydantic import BaseModel, Field
from typing import List, Dict, Optional

class Citation(BaseModel):
    """
    Represents a single vector database chunk retrieved during Path A or Hybrid queries.
    """
    source: str = Field(..., description="The name of the regulatory document.")
    section: str = Field(..., description="The specific article or header section.")
    page: int = Field(..., description="The page number from the original PDF.")
    text: str = Field(..., description="The raw chunk text used to ground the LLM.")

class QueryResponse(BaseModel):
    """
    Strict data contract for outgoing responses sent back to the local UI.
    """
    answer: str = Field(
        ..., 
        description="The LLM generated response (simulated streaming on the frontend)."
    )
    path: str = Field(
        ..., 
        description="The execution path taken by the cloud backend (e.g., 'Path A - Retrieval', 'Path B - Injection')."
    )
    citations: Optional[List[Citation]] = Field(
        default_factory=list, 
        description="List of document chunks used for grounding the answer."
    )
    masked_items: Optional[Dict[str, str]] = Field(
        default_factory=dict, 
        description="Pass-through of masked items back to the UI."
    )