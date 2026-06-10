import io
import logging
from pathlib import Path
from typing import Dict, Any, Union

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.document import DocumentStream

# Configure module-level logger
logger = logging.getLogger(__name__)

class ExtractionError(Exception):
    """Custom exception raised when document text extraction fails."""
    pass

class DocumentExtractor:
    """
    Handles raw text and structure extraction from credit documents using IBM Docling.
    Outputs rich Markdown preserving tables, headers, and semantic hierarchy.
    """

    def __init__(self) -> None:
        """
        Initializes the Docling DocumentConverter.
        This is a heavy operation; instantiate this class once per worker/session 
        to avoid cold-start latency on every extraction request.
        """
        logger.info("Initializing Docling DocumentConverter...")
        try:
            # PdfPipelineOptions is the PDF-specific subclass that carries
            # do_table_structure, do_ocr, etc.  The base PipelineOptions does not.
            # It must be passed through PdfFormatOption into format_options — not
            # directly to DocumentConverter — as of Docling v1.16+.
            pdf_options = PdfPipelineOptions(do_table_structure=True)
            pdf_options.table_structure_options.do_cell_matching = True

            self.converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_options)
                }
            )
            logger.info("Docling DocumentConverter initialized successfully.")
        except Exception as e:
            logger.critical(f"Failed to initialize Docling models: {e}", exc_info=True)
            raise RuntimeError(f"Docling initialization failed: {e}")

    def extract(self, file_source: Union[str, Path, io.BytesIO], filename: str = "uploaded_document.pdf") -> Dict[str, Any]:
        """
        Extracts semantic text (Markdown) from the provided document source.

        Args:
            file_source: The document to process. Can be a file path (str or Path) 
                         or an in-memory byte stream (io.BytesIO).
            filename: The logical name of the file (critical for Streamlit byte streams).

        Returns:
            A dictionary containing the 'text' (Markdown) and structural 'metadata'.

        Raises:
            ExtractionError: If the document cannot be parsed or read.
        """
        logger.debug(f"Starting extraction for document: {filename}")
        
        try:
            # Route inputs based on type (File path vs Memory Stream)
            if isinstance(file_source, io.BytesIO):
                source = DocumentStream(name=filename, stream=file_source)
                resolved_name = filename
            else:
                source_path = Path(file_source)
                if not source_path.exists():
                    raise FileNotFoundError(f"Document path does not exist: {source_path}")
                source = source_path
                resolved_name = source_path.name

            # Execute the Docling conversion pipeline
            conversion_result = self.converter.convert(source)
            document = conversion_result.document
            
            # Export to clean Markdown to preserve tabular financial data natively
            markdown_text = document.export_to_markdown()
            
            # Calculate metadata payload
            metadata = {
                "source": resolved_name,
                "component_count": len(list(document.iterate_items())),
                "extraction_engine": "docling_granite"
            }
            
            logger.info(f"Successfully extracted Markdown from {resolved_name}. Length: {len(markdown_text)} chars.")
            
            return {
                "text": markdown_text,
                "metadata": metadata
            }
            
        except Exception as e:
            error_msg = f"Extraction failed for document '{filename}': {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise ExtractionError(error_msg) from e