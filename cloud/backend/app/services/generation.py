import logging
from google import genai
from google.genai import types
from app.services.secrets import get_secret

logger = logging.getLogger(__name__)

class GenerationService:
    """
    Dedicated generation wrapper for the Google Gemini 1.5 Pro engine.
    Utilizes the modern google.genai SDK.
    """
    def __init__(self):
        logger.info("Initializing Generation Service and setting up Gemini SDK context...")
        
        # Pull the API key dynamically from the secure SSM/Env parameter manager
        api_key = get_secret("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Critical configuration failure: GEMINI_API_KEY is unresolved.")
            
        # Initialize the modern GenAI client
        self.client = genai.Client(api_key=api_key)
        
        # Using Gemini 1.5 Pro for multi-source synthesis and deep text token windows
        self.model_id = "gemini-3.1-flash-lite"

    def generate_text(
        self, 
        prompt: str, 
        temperature: float = 0.1, 
        top_p: float = 0.95, 
        max_tokens: int = 8192
    ) -> str:
        """
        Dispatches structured prompts to Gemini. Enforces tight deterministic 
        bounds (low temperature) to avoid hallucinations in compliance reporting.
        """
        if not prompt:
            raise ValueError("Inference failed: prompt text payload cannot be empty.")

        # Establish generation constraints using the new types config
        config = types.GenerateContentConfig(
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_tokens
        )

        try:
            logger.info(f"Dispatching prompt token payload to {self.model_id} gateway...")
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=prompt,
                config=config
            )
            
            # Verify string output presence to catch safety block triggers early
            if not response or not response.text:
                logger.error("The upstream model gateway returned an empty value or triggered a content filter block.")
                raise RuntimeError("Empty response received from foundation model processing loop.")
                
            return response.text

        except Exception as e:
            logger.error(f"API interaction exception inside Gemini foundation layer: {str(e)}", exc_info=True)
            raise RuntimeError(f"Cloud LLM inference engine failure: {str(e)}")