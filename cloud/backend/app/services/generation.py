import logging
from google import genai
from google.genai import types
from app.services.secrets import get_secret

logger = logging.getLogger(__name__)


class GenerationService:
    """
    Wrapper for Google Gemini generation via the google-genai SDK.
    Model ID is read from GEMINI_MODEL env var so you can change it
    without redeploying — defaults to gemini-1.5-flash.
    """

    def __init__(self):
        logger.info("Initialising GenerationService…")
        api_key = get_secret("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is unresolved.")
        self.client   = genai.Client(api_key=api_key)
        # Read model from env so you can override without code change.
        # gemini-1.5-flash: separate free-tier quota from gemini-2.0-flash.
        import os
        self.model_id = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        logger.info("GenerationService ready: model=%s", self.model_id)

    def generate_text(
        self,
        prompt:      str,
        temperature: float = 0.1,
        top_p:       float = 0.95,
        max_tokens:  int   = 8192,
    ) -> str:
        if not prompt:
            raise ValueError("Prompt cannot be empty.")

        config = types.GenerateContentConfig(
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_tokens,
        )

        try:
            logger.info("Dispatching to %s…", self.model_id)
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=prompt,
                config=config,
            )
            if not response or not response.text:
                raise RuntimeError("Empty response from Gemini — possible content filter.")
            return response.text
        except Exception as e:
            logger.error("Gemini error: %s", e, exc_info=True)
            raise RuntimeError(f"Gemini generation failed: {str(e)}")
