import logging
from typing import Dict, Any
from local.slm.inference import LocalModelInference
from local.slm.prompt_templates import get_formatted_routing_prompt

logger = logging.getLogger(__name__)

class CreditRiskRouter:
    """
    High-level intent classifier that determines whether incoming requests route 
    to Path A (Vector Rag Search) or Path B (Full Doc Processing Context Pipeline).
    """
    def __init__(self, inference_engine: LocalModelInference) -> None:
        self.engine = inference_engine
        self.valid_routes = {"EXTRACT", "BENCHMARK", "HYBRID", "GENERAL"}

    def route_query(self, user_query: str, document_attached: bool = True) -> Dict[str, Any]:
        """
        Evaluates user question strings to compute the ideal cloud processing pathway destination.

        Args:
            user_query: The raw or masked question string submitted by the analyst.
            document_attached: True if a processed document is present in the session state.

        Returns:
            A routing layout profile containing the destination token and strategy metadata.
        """
        logger.debug(f"Evaluating routing classification sequence for text: {user_query[:50]}...")

        # Rule-Based Guardrail: Route queries directly to GENERAL if no context document is attached
        if not document_attached:
            logger.info("Active session lacks an attached context document. Enforcing GENERAL deterministic routing.")
            return {
                "route": "GENERAL",
                "strategy": "rule_forced_fallback",
                "reason": "no_active_context_document"
            }

        try:
            # 1. Fetch the formatted instruction structure 
            formatted_prompt = get_formatted_routing_prompt(user_query)
            
            # 2. Run local token generation
            raw_prediction = self.engine.run_inference(formatted_prompt, max_tokens=12)
            
            # 3. Clean up the output string, keeping only alphanumeric characters
            sanitized_route = "".join(char for char in raw_prediction if char.isalnum()).upper()
            
            # 4. Route matching verification
            if sanitized_route in self.valid_routes:
                logger.info(f"SLM Router intent assignment completed successfully. Selected Path: {sanitized_route}")
                return {
                    "route": sanitized_route,
                    "strategy": "slm_prediction",
                    "reason": "model_classification"
                }
            
            # Catch out-of-bounds tokens or text anomalies gracefully
            logger.warning(f"SLM hallucinated an invalid routing token layout: '{sanitized_route}'. Triggering fallback.")
            return {
                "route": "HYBRID",
                "strategy": "validation_fallback",
                "reason": f"invalid_token_returned_{sanitized_route}"
            }

        except Exception as e:
            logger.error(f"Routing orchestrator thread encountered an unhandled error: {e}", exc_info=True)
            # High-availability defense: Fall back to HYBRID to keep the pipeline moving
            return {
                "route": "HYBRID",
                "strategy": "emergency_safeguard_fallback",
                "reason": f"unhandled_system_exception: {str(e)}"
            }