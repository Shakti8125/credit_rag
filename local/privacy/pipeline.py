import io
import logging
from pathlib import Path
from typing import Dict, Any, Union

# Import components from the local privacy package layout
from local.privacy.extractor import DocumentExtractor
from local.privacy.entity_registry import EntityRegistry
from local.privacy.masker import DocumentMasker
from local.privacy.validator import DocumentValidator

# Configure structured logging for pipeline lifecycle auditing
logger = logging.getLogger(__name__)

class PrivacyPipeline:
    """
    Facade controller that orchestrates the local document ingestion flow:
    Extracts Markdown layouts, masks PII while protecting metrics, and 
    runs a pre-flight firewall audit before data leaves the local boundary.
    """
    
    def __init__(self, spacy_model: str = "en_core_web_trf") -> None:
        """
        Initializes the privacy pipeline and boots up internal underlying dependencies.
        
        Args:
            spacy_model: Name of the spaCy NLP pipeline to use for entity extraction.
                         Defaults to 'en_core_web_trf' (Transformer-based accuracy).
        """
        logger.info("Initializing Local Privacy Orchestration Pipeline Components...")
        
        # Instantiate stateless or persistent workers
        self.extractor = DocumentExtractor()
        self.registry = EntityRegistry()
        self.masker = DocumentMasker(registry=self.registry, spacy_model=spacy_model)
        self.validator = DocumentValidator()
        
        logger.info("All local privacy pipeline sub-systems fully initialized.")

    def process_document(self, file_source: Union[str, Path, io.BytesIO], filename: str) -> Dict[str, Any]:
        """
        Executes the end-to-end local data-cleaning processing loop.

        Args:
            file_source: Either a system path string, a Path object, or an in-memory 
                         BytesIO buffer uploaded directly via the UI tier.
            filename: The baseline name of the document, utilized for generating metadata logs.

        Returns:
            A structured execution summary containing the anonymized payload text,
            structural metadata parameters, and the complete session token audit logs.
            
        Raises:
            Exception: Propagates extraction errors, masking faults, or 
                       Pre-flight LeakageValidationErrors to the UI layer.
        """
        logger.info(f"Pipeline execution started for target context payload: '{filename}'")
        
        # Ensure fresh registry isolation when beginning a completely new document tracking loop
        self.registry.clear()
        
        # Phase 1: Structural Analysis & Layout Extraction (Docling)
        logger.info("Pipeline Phase 1/3: Initiating structural text extraction...")
        extraction_result = self.extractor.extract(file_source, filename)
        raw_markdown = extraction_result["text"]
        document_metadata = extraction_result["metadata"]
        logger.info(f"Phase 1 Complete. Extracted {len(raw_markdown)} characters from raw document source.")
        
        # Phase 2: Contextual PII Anonymization & Metric Freezing (spaCy + Regex Guards)
        logger.info("Pipeline Phase 2/3: Passing text payload to context masking matrix...")
        masked_markdown = self.masker.mask(raw_markdown)
        logger.info("Phase 2 Complete. Privacy preservation entity mapping sequence completed.")
        
        # Phase 3: Pre-Flight Safety Firewall Verification (Regex Leak Check)
        logger.info("Pipeline Phase 3/3: Running pre-flight egress security checks...")
        # Throws a LeakageValidationError if unmasked sensitive tracking strings match patterns
        validation_status = self.validator.validate(masked_markdown)
        logger.info("Phase 3 Complete. Egress security verification passed.")
        
        # Compile complete processing summary object
        logger.info(f"Successfully processed and cleared document execution path for: '{filename}'")
        return {
            "masked_text": masked_markdown,
            "metadata": document_metadata,
            "validation_report": validation_status,
            "audit_log": self.registry.get_audit_log(),
            "registry_instance": self.registry
        }