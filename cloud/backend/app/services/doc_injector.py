import logging
import tiktoken

logger = logging.getLogger(__name__)

class DocumentInjector:
    """
    Path B Full Document Injector. Passes the entire masked credit memo into the LLM.
    Acts as a safety gatekeeper to protect the 1M token context window.
    """
    def __init__(self, max_tokens: int = 900_000):
        self.max_tokens = max_tokens
        # We use cl100k_base as it is a highly reliable proxy for LLM token boundaries
        self.encoding = tiktoken.get_encoding("cl100k_base")

    def prepare_payload(self, doc_text: str) -> str:
        """
        Validates the token length. If it breaches the 900k threshold, it truncates 
        the document to ensure the cloud API call does not fail.
        """
        if not doc_text:
            return ""
            
        # Execute pre-flight token check
        tokens = self.encoding.encode(doc_text)
        token_count = len(tokens)
        
        if token_count > self.max_tokens:
            logger.warning(f"Document token count ({token_count}) exceeds safe limit ({self.max_tokens}). Truncating payload.")
            # Truncate safely at the token level, then decode back to text
            truncated_tokens = tokens[:self.max_tokens]
            return self.encoding.decode(truncated_tokens)
            
        logger.info(f"Document pre-flight check passed. Total token count: {token_count}")
        return doc_text