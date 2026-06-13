"""
local/privacy/pipeline.py

Four-phase local document processing pipeline:
  Phase 1 — Docling extraction → raw Markdown
  Phase 2 — PII masking (regex + spaCy NER)
  Phase 3 — Egress firewall
  Phase 4 — Financial extraction + policy check + EWS detection  ← Tier 1+2

Phase 4 runs on RAW text from Phase 1 (before masking) so numeric values
are intact. The masker preserves metric values in place after this runs.
"""

import io
import logging
from pathlib import Path
from typing import Dict, Any, Union

from local.privacy.extractor       import DocumentExtractor
from local.privacy.entity_registry import EntityRegistry
from local.privacy.masker          import DocumentMasker
from local.privacy.validator       import DocumentValidator

logger = logging.getLogger(__name__)


class PrivacyPipeline:

    def __init__(self, spacy_model: str = "en_core_web_trf") -> None:
        logger.info("Initialising Privacy Pipeline…")
        self.extractor = DocumentExtractor()
        self.registry  = EntityRegistry()
        self.masker    = DocumentMasker(registry=self.registry, spacy_model=spacy_model)
        self.validator = DocumentValidator()
        logger.info("Privacy Pipeline ready.")

    def process_document(
        self,
        file_source: Union[str, Path, io.BytesIO],
        filename:    str,
        doc_type:    str = "Internal Credit Proposal (Memo)",
    ) -> Dict[str, Any]:
        """
        End-to-end document processing.

        Returns
        -------
        {
            "masked_text"        : str
            "raw_text"           : str            — pre-mask (for EWS / financial extraction)
            "metadata"           : dict
            "validation_report"  : dict
            "audit_log"          : list
            "registry_instance"  : EntityRegistry
            "financial_profile"  : FinancialProfile | None
            "breach_report"      : BreachReport   | None
            "ews_report"         : EWSReport      | None
        }
        """
        logger.info("Pipeline start: '%s' (doc_type=%s)", filename, doc_type)
        self.registry.clear()

        # ── Phase 1: Extraction ────────────────────────────────────────
        logger.info("Phase 1/4: Docling extraction…")
        extraction = self.extractor.extract(file_source, filename)
        raw_text   = extraction["text"]
        metadata   = extraction["metadata"]
        logger.info("Phase 1 done: %d chars.", len(raw_text))

        # ── Phase 4 (runs on raw text BEFORE masking) ─────────────────
        financial_profile = None
        breach_report     = None
        ews_report        = None
        try:
            from local.analysis.financial_extractor import FinancialExtractor
            from local.analysis.policy_checker      import PolicyChecker
            from local.analysis.ews_detector        import EarlyWarningDetector

            logger.info("Phase 4a: Financial extraction…")
            financial_profile = FinancialExtractor().extract(raw_text, doc_type=doc_type)

            logger.info("Phase 4b: Policy breach check…")
            breach_report = PolicyChecker().check(financial_profile)

            logger.info("Phase 4c: Early warning signal detection…")
            ews_report = EarlyWarningDetector().detect(
                raw_text=raw_text,
                financial_profile=financial_profile,
                doc_type=doc_type,
            )

            logger.info(
                "Phase 4 done: metrics=%d breaches=%d warnings=%d ews_signals=%d ews_risk=%s",
                len([k for k, v in financial_profile.to_dict().items()
                     if v and k not in ("doc_type", "extraction_warnings")]),
                breach_report.breach_count,
                breach_report.warning_count,
                len(ews_report.signals),
                ews_report.risk_level,
            )
        except ImportError:
            logger.warning("analysis package not available — skipping Phase 4.")
        except Exception as e:
            logger.warning("Phase 4 failed (%s) — continuing without analytics.", e, exc_info=True)

        # ── Phase 2: Masking ───────────────────────────────────────────
        logger.info("Phase 2/4: PII masking…")
        masked_text = self.masker.mask(raw_text)
        logger.info("Phase 2 done.")

        # ── Phase 3: Egress firewall ───────────────────────────────────
        logger.info("Phase 3/4: Egress validation…")
        validation_status = self.validator.validate(masked_text)
        logger.info("Phase 3 done.")

        logger.info("Pipeline complete: '%s'", filename)
        return {
            "masked_text":       masked_text,
            "raw_text":          raw_text,
            "metadata":          metadata,
            "validation_report": validation_status,
            "audit_log":         self.registry.get_audit_log(),
            "registry_instance": self.registry,
            "financial_profile": financial_profile,
            "breach_report":     breach_report,
            "ews_report":        ews_report,
        }
