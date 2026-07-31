"""
Session-state-aware masking/unmasking helpers, built on shared.masking.

Unmasking is DICT-BASED (snapshot dictionaries captured at processing time),
not registry-object-based: PrivacyPipeline is a cached singleton whose
registry.clear() runs on every process_document, so the registry object
stored in session state for Document A is silently wiped and refilled when
Document B is processed (both session keys alias the same object). The
snapshot dicts are immune to that.
"""

import logging
import streamlit as st

from shared.masking import unmask_text as _unmask_text

logger = logging.getLogger(__name__)


def _merged_dict() -> dict:
    """
    Combined placeholder→original mapping across both documents plus any
    entities registered while masking outbound queries.

    Doc A and Doc B use independent placeholder counters, so both can emit
    e.g. [ORG_1] with different originals. Primary-doc entries win collisions
    (inserted last).
    """
    merged = {}
    merged.update(st.session_state.get("doc_b_mask_dictionary", {}))
    merged.update(st.session_state.get("mask_dictionary", {}))
    return merged


def unmask_response(text: str) -> str:
    """Unmask STRICTLY after LLM generation — never before."""
    return _unmask_text(text, _merged_dict())


def unmask_with_merged(text: str) -> str:
    """Alias kept for handler readability — same merged-dict unmasking."""
    return _unmask_text(text, _merged_dict())


def mask_outbound_query(prompt: str) -> str:
    """
    Masks the analyst's query before it is sent to the cloud.

    The document body is masked by the pipeline, but the typed query was
    previously sent verbatim — "What is RAKBANK's exposure?" leaked the bank
    name. Runs the full DocumentMasker pass (bank dictionary, regex PII,
    NER) using the shared pipeline registry so query placeholders are
    consistent with document placeholders, then refreshes the session mask
    dictionary so responses containing those placeholders unmask locally.

    Fails open with a warning if the privacy pipeline is unavailable —
    matching the pipeline's own behaviour for analytics — but that only
    happens when no document processing is possible at all.
    """
    if not prompt or not prompt.strip():
        return prompt

    try:
        from local.app.resources import load_privacy_pipeline
        pipeline = load_privacy_pipeline()
        masked = pipeline.masker.mask(prompt)

        # Query masking may have registered new entities — fold them into the
        # session snapshot (existing primary-doc entries win collisions).
        registry_dump = pipeline.registry.get_mask_dictionary()
        st.session_state["mask_dictionary"] = {
            **registry_dump,
            **st.session_state.get("mask_dictionary", {}),
        }

        if masked != prompt:
            logger.info("Outbound query masked before cloud dispatch.")
        return masked
    except Exception as e:
        logger.warning("Query masking unavailable (%s) — sending query as typed.", e)
        return prompt
