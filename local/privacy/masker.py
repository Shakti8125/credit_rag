import re
import logging
from typing import List, Tuple, Set
import spacy
from spacy.tokens import Doc

from local.privacy.entity_registry import EntityRegistry

logger = logging.getLogger(__name__)

class DocumentMasker:
    """
    Protects data privacy by identifying and masking PII using spaCy NER, 
    while preserving structural financial metrics via explicit freezing rules.
    """
    def __init__(self, registry: EntityRegistry, spacy_model: str = "en_core_web_trf") -> None:
        self.registry = registry
        
        logger.info(f"Loading spaCy NER model: {spacy_model}...")
        try:
            # Primary choice: Transformer pipeline for high recall
            self.nlp = spacy.load(spacy_model)
        except OSError:
            logger.warning(f"Target model '{spacy_model}' unavailable. Falling back to 'en_core_web_sm'.")
            self.nlp = spacy.load("en_core_web_sm")
            
        # Compiled financial metric match rules to guarantee preservation of risk parameters
        self.financial_patterns = [
            # Currency expressions with scale variants: $14.5M, £250k, USD 5,000,000, 45 Billion
            re.compile(r'\b(?:AED|USD|EUR|GBP)?\s?[\$\u20AC\u00A3]?\s?\d+(?:\.\d+)?\s?(?:M|B|k|Million|Billion)?\b', re.IGNORECASE),
            # Risk ratio formulas: DSCR 1.25x, LTV 70%, Debt/EBITDA of 3.5x
            re.compile(r'\b(?:DSCR|LTV|Leverage|TOL/ATNW|Debt/EBITDA|ROE|ROA)\b\s*(?:of|is)?\s*\d+(?:\.\d+)?\s?%?x?\b', re.IGNORECASE),
            # Isolated standard metrics and sizing multipliers: 10.25%, 2.0x
            re.compile(r'\b\d+(?:\.\d+)?\s?%\b'),
            re.compile(r'\b\d+(?:\.\d+)?\s?x\b', re.IGNORECASE)
        ]
        
        # Target labels to match against high-risk categories
        self.target_ner_labels: Set[str] = {"PERSON", "ORG", "GPE", "DATE", "FAC", "LOC"}

        # Regex patterns for structured PII that spaCy NER misses.
        # Each entry: (label, compiled_pattern)
        # These are applied as a pre-pass in mask() BEFORE spaCy runs, so the
        # validator's egress check never sees raw PII in the output.
        self.pii_patterns: List[Tuple[str, re.Pattern]] = [
            # Email addresses
            ("EMAIL", re.compile(
                r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}",
                re.IGNORECASE
            )),
            # International / local phone numbers: +971-4-123-4567, (04) 123 4567, 04-1234567
            ("PHONE", re.compile(
                r"\b(?:\+?[\d\s\-\.]{1,4})?(?:\(?\d{2,4}\)?[\s\-\.]?)(?:\d{3,4}[\s\-\.]?){1,3}\d{3,4}\b"
            )),
            # US SSN: 123-45-6789
            ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
            # US EIN / corporate tax ID: 12-3456789
            ("EIN", re.compile(r"\b\d{2}-\d{7}\b")),
        ]

    def _apply_regex_masks(self, text: str) -> str:
        """
        Pre-pass: replaces structured PII (phones, emails, SSNs, EINs) with
        registry tokens using compiled regex patterns.  Runs before spaCy so
        the NER model never sees raw PII and the validator's egress check passes.
        Matches are applied in reverse offset order to preserve string positions.
        """
        modifications: List[Tuple[int, int, str]] = []

        for label, pattern in self.pii_patterns:
            for match in pattern.finditer(text):
                value = match.group().strip()
                if not value:
                    continue
                placeholder = self.registry.register_entity(value, label)
                modifications.append((match.start(), match.end(), placeholder))

        if not modifications:
            return text

        # Sort descending by start offset so replacements don't shift indices
        modifications.sort(key=lambda x: x[0], reverse=True)
        text_buffer = list(text)
        for start, end, replacement in modifications:
            text_buffer[start:end] = list(replacement)

        logger.debug("Regex pre-pass masked %d structured PII items.", len(modifications))
        return "".join(text_buffer)

    def _get_protected_spans(self, text: str) -> List[Tuple[int, int]]:
        """Scans the text string to register boundaries of critical financial numbers."""
        protected_spans = []
        for pattern in self.financial_patterns:
            for match in pattern.finditer(text):
                protected_spans.append(match.span())
        return protected_spans

    def _is_overlapping(self, entity_span: Tuple[int, int], protected_spans: List[Tuple[int, int]]) -> bool:
        """Determines if a named entity span intersects with a frozen financial metric boundary."""
        ent_start, ent_end = entity_span
        for p_start, p_end in protected_spans:
            # Overlap condition check
            if not (ent_end <= p_start or ent_start >= p_end):
                return True
        return False

    def mask(self, text: str) -> str:
        """
        Transforms sensitive raw Markdown text into an anonymized payload
        safe for remote API context parsing.

        Masking order:
          1. Regex pre-pass  — structured PII (phones, emails, SSNs, EINs)
          2. Financial freeze — record spans that must NOT be touched
          3. spaCy NER       — named entities (ORG, PERSON, GPE, DATE, FAC, LOC)
        """
        if not text.strip():
            return text

        # Step 1: Regex pre-pass for structured PII spaCy would miss
        text = self._apply_regex_masks(text)

        # Step 2: Record regions containing vital financial information
        protected_spans = self._get_protected_spans(text)

        # Step 3: Run spaCy named entity recognition
        doc: Doc = self.nlp(text)
        
        valid_modifications: List[Tuple[int, int, str]] = []
        
        for ent in doc.ents:
            if ent.label_ in self.target_ner_labels:
                ent_span = (ent.start_char, ent.end_char)
                
                # Verify entity doesn't conflict with financial metrics
                if not self._is_overlapping(ent_span, protected_spans):
                    placeholder = self.registry.register_entity(ent.text, ent.label_)
                    valid_modifications.append((ent.start_char, ent.end_char, placeholder))
                else:
                    logger.debug(f"Preserved frozen metric overlap window for text: '{ent.text}'")

        # Step 4: Apply token changes in reverse string index order
        # This keeps character array positions steady during text transformations
        sorted_modifications = sorted(valid_modifications, key=lambda x: x[0], reverse=True)
        
        text_buffer = list(text)
        for start, end, replacement in sorted_modifications:
            text_buffer[start:end] = list(replacement)
            
        return "".join(text_buffer)