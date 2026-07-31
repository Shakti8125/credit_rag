"""Document upload → privacy pipeline → FAISS index wiring for the Streamlit app."""

import io
import logging
import streamlit as st

from local.app.resources import load_privacy_pipeline, load_embedding_model

logger = logging.getLogger(__name__)


def process_uploaded_document(uploaded_file, slot: str = "primary") -> bool:
    pipeline = load_privacy_pipeline()
    if pipeline is None:
        st.error("🚨 Privacy pipeline failed to initialise.")
        return False

    doc_type = st.session_state.get("document_type", "Internal Credit Proposal (Memo)")

    try:
        result = pipeline.process_document(
            file_source=io.BytesIO(uploaded_file.getvalue()),
            filename=uploaded_file.name,
            doc_type=doc_type,
        )

        registry = result.get("registry_instance")

        if registry:
            mask_dictionary = registry.get_mask_dictionary()
        else:
            mask_dictionary = {
                e["placeholder"]: e["original_entity"]
                for e in result.get("audit_log", [])
            }

        # ── Option 1: mask bank names embedded in the filename ────────
        # The pipeline masks document body text, but the filename is stored
        # separately and sent as doc_label/doc_a_label/doc_b_label in API
        # payloads.  A filename like "RAK Bank Credit Card Model.docx"
        # leaks the bank name to Gemini even when the body is fully masked.
        #
        # pipeline.masker.mask_filename() applies only the bank dictionary
        # pass (no NER, no regex PII — filenames don't need those) using the
        # same EntityRegistry so the placeholder is consistent with the body.
        raw_filename    = uploaded_file.name
        masked_filename = pipeline.masker.mask_filename(raw_filename)
        if masked_filename != raw_filename:
            logger.info(
                "Filename masked: '%s' → '%s'",
                raw_filename, masked_filename,
            )

        if slot == "primary":
            st.session_state["doc_text"]          = result["masked_text"]
            # Raw text stays on-device only — enables re-running analytics
            # when the analyst changes Document Type after upload.
            st.session_state["doc_raw_text"]      = result["raw_text"]
            st.session_state["doc_type_used"]     = doc_type
            st.session_state["mask_dictionary"]   = mask_dictionary
            # Masked filename for display/API payloads; RAW filename for the
            # sidebar reprocess guard — comparing the masked name against
            # uploaded_file.name never matches, causing an infinite
            # reprocessing loop on every Streamlit rerun.
            st.session_state["last_uploaded_file"]     = masked_filename
            st.session_state["last_uploaded_raw_name"] = raw_filename
            st.session_state["registry"]           = result["registry_instance"]
            st.session_state["financial_profile"]  = result.get("financial_profile")
            st.session_state["breach_report"]      = result.get("breach_report")
            st.session_state["ews_report"]         = result.get("ews_report")
            st.session_state["ews_cloud_result"]   = None

            embed_model = load_embedding_model()
            if embed_model:
                from local.rag.chunker     import MarkdownChunker
                from local.rag.local_index import LocalDocumentIndex
                chunks = MarkdownChunker().chunk(result["masked_text"])
                if chunks:
                    idx = LocalDocumentIndex(embed_model)
                    idx.build(chunks)
                    st.session_state["doc_faiss_index"] = idx
        else:
            st.session_state["doc_b_text"]            = result["masked_text"]
            # Masked filename for display; raw name for the reprocess guard
            st.session_state["doc_b_label"]           = masked_filename
            st.session_state["doc_b_raw_name"]        = raw_filename
            st.session_state["doc_b_registry"]        = result["registry_instance"]
            st.session_state["doc_b_mask_dictionary"] = mask_dictionary
            embed_model = load_embedding_model()
            if embed_model:
                from local.rag.chunker     import MarkdownChunker
                from local.rag.local_index import LocalDocumentIndex
                chunks = MarkdownChunker().chunk(result["masked_text"])
                if chunks:
                    idx = LocalDocumentIndex(embed_model)
                    idx.build(chunks)
                    st.session_state["doc_b_faiss_index"] = idx

        logger.info(
            "Doc processed [%s]: %d entities masked",
            slot, len(mask_dictionary),
        )
        return True

    except Exception as e:
        from local.privacy.validator import LeakageValidationError
        if isinstance(e, LeakageValidationError):
            st.error(f"🚨 Egress firewall blocked: {e}")
        else:
            st.error(f"🚨 Document processing failed: {e}")
        logger.error("Pipeline error [%s] %s: %s", slot, uploaded_file.name, e, exc_info=True)
        return False


def recompute_analytics(doc_type: str) -> bool:
    """
    Re-runs Phase 4 analytics (financial extraction, policy check, EWS) on the
    stored raw text with a new document type — without re-extracting,
    re-masking, or re-indexing. Called when the analyst changes Document Type
    after a document is already loaded (the selector previously only took
    effect at upload time, which made it look unused).
    """
    raw_text = st.session_state.get("doc_raw_text")
    if not raw_text:
        return False

    try:
        from local.analysis.financial_extractor import FinancialExtractor
        from local.analysis.policy_checker      import PolicyChecker
        from local.analysis.ews_detector        import EarlyWarningDetector

        profile = FinancialExtractor().extract(raw_text, doc_type=doc_type)
        breach  = PolicyChecker().check(profile)
        ews     = EarlyWarningDetector().detect(
            raw_text=raw_text,
            financial_profile=profile,
            doc_type=doc_type,
        )

        st.session_state["financial_profile"] = profile
        st.session_state["breach_report"]     = breach
        st.session_state["ews_report"]        = ews
        st.session_state["doc_type_used"]     = doc_type
        logger.info(
            "Analytics recomputed for doc_type='%s': breaches=%d warnings=%d",
            doc_type, breach.breach_count, breach.warning_count,
        )
        return True
    except Exception as e:
        logger.warning("Analytics recompute failed: %s", e, exc_info=True)
        return False
