"""
Shared masking infrastructure used by both tiers.

EntityRegistry is framework-agnostic bookkeeping (no spaCy/heavy deps), so it
lives here and is imported by local/privacy (which does the actual heavy
NER-based masking) as well as by the cloud routes (which only need to unmask
placeholders using the dict handed back by the local tier).

unmask_text() is the single dict-based placeholder-replacement routine —
previously duplicated across cloud/backend/app/routes/{query,compare,ews}.py.
"""

import threading
from typing import Dict, List, Any, Optional


class EntityRegistry:
    """
    Thread-safe registry that maps real-world sensitive entities to deterministic
    placeholders and maintains an audit log for downstream UI visibility.
    """
    def __init__(self) -> None:
        # Maps raw text to a token: {"Alpha Corp": "[ORG_1]"}
        self._forward_registry: Dict[str, str] = {}
        # Maps token back to raw text: {"[ORG_1]": "Alpha Corp"}
        self._reverse_registry: Dict[str, str] = {}
        # Tracks sequential IDs per label class: {"PERSON": 1, "ORG": 2}
        self._counters: Dict[str, int] = {}
        # Reentrant lock ensuring safe execution within concurrent Streamlit sessions
        self._lock = threading.RLock()

    def register_entity(self, raw_text: str, entity_label: str) -> str:
        """
        Registers a sensitive text string and returns a persistent token.
        If the text was previously registered, returns the existing token.
        """
        cleaned_text = raw_text.strip()
        if not cleaned_text:
            return raw_text

        with self._lock:
            # Maintain context uniformity across multiple occurrences
            if cleaned_text in self._forward_registry:
                return self._forward_registry[cleaned_text]

            # Generate new sequential index token
            current_counter = self._counters.get(entity_label, 1)
            placeholder = f"[{entity_label}_{current_counter}]"

            # Update tracking states
            self._counters[entity_label] = current_counter + 1
            self._forward_registry[cleaned_text] = placeholder
            self._reverse_registry[placeholder] = cleaned_text

            return placeholder

    def unmask_text(self, masked_text: str) -> str:
        """
        Reverses the masking process by replacing tokens with their original text strings.
        Used when translating responses returning from cloud inference.
        """
        with self._lock:
            return unmask_text(masked_text, self._reverse_registry)

    def get_audit_log(self) -> List[Dict[str, Any]]:
        """
        Backward compatible audit log.
        Used by privacy pipeline.
        Does not include financial values.
        """
        with self._lock:
            return [
                {
                    "placeholder": placeholder,
                    "original_entity": original
                }
                for placeholder, original
                in self._reverse_registry.items()
            ]

    def get_mask_dictionary(self) -> Dict[str, str]:
        with self._lock:
            return dict(self._reverse_registry)

    def clear(self) -> None:
        """Resets the state registry for a new document session execution loop."""
        with self._lock:
            self._forward_registry.clear()
            self._reverse_registry.clear()
            self._counters.clear()


def unmask_text(text: str, masked_items: Optional[Dict[str, str]]) -> str:
    """
    Replaces placeholder tokens (e.g. "[ORG_1]") with their original values.

    Sorted by placeholder length descending so "[ORG_11]" is replaced before
    "[ORG_1]" — otherwise the shorter token would partially match inside the
    longer one and corrupt the substitution.
    """
    if not masked_items or not text:
        return text
    for placeholder in sorted(masked_items.keys(), key=len, reverse=True):
        text = text.replace(placeholder, masked_items[placeholder])
    return text
