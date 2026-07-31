"""Query intent classification for routing between EXTRACT/HYBRID/BENCHMARK/GENERAL/COMPARE/EWS."""

import logging
import streamlit as st

logger = logging.getLogger(__name__)

_HYBRID_SIGNALS = {
    "compare","versus","vs","benchmark","exceed","below","above","breach",
    "guideline","policy","requirement","comply","compliance","threshold",
    "limit","minimum","maximum","cbuae","basel","rbi","ifrs","eba","bcbs",
}


def classify_intent(prompt: str, document_attached: bool, selected_model: str) -> str:
    mode = st.session_state.get("analysis_mode", "Standard")
    if mode == "Compare Two Documents":
        return "COMPARE"
    if mode == "Early Warning Scan":
        return "EWS"

    if selected_model == "local":
        if not document_attached:
            return "GENERAL"
        return "HYBRID" if set(prompt.lower().split()) & _HYBRID_SIGNALS else "EXTRACT"

    if not document_attached:
        return "BENCHMARK"

    intent = "HYBRID" if set(prompt.lower().split()) & _HYBRID_SIGNALS else "EXTRACT"
    logger.info("Intent='%s' doc=%s model=%s", intent, document_attached, selected_model)
    return intent
