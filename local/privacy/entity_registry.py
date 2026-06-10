import threading
from typing import Dict, List, Any

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
        unmasked = masked_text
        with self._lock:
            # Sorting by length descending eliminates partial substitution bugs 
            # (e.g., matching [ORG_1] before [ORG_11])
            sorted_tokens = sorted(self._reverse_registry.items(), key=lambda x: len(x[0]), reverse=True)
            for placeholder, real_value in sorted_tokens:
                unmasked = unmasked.replace(placeholder, real_value)
        return unmasked

    def get_audit_log(self) -> List[Dict[str, Any]]:
        """
        Formats registry entries into an exportable layout for masking_log.py.
        """
        with self._lock:
            return [
                {
                    "placeholder": placeholder,
                    "original_entity": real_value,
                    "type": placeholder.strip("[]").split("_")[0]
                }
                for placeholder, real_value in self._reverse_registry.items()
            ]

    def clear(self) -> None:
        """Resets the state registry for a new document session execution loop."""
        with self._lock:
            self._forward_registry.clear()
            self._reverse_registry.clear()
            self._counters.clear()