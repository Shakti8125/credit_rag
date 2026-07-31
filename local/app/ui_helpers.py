"""Small UI-rendering helpers shared across the sidebar/layout/handlers."""

import time


def stream_response(text: str):
    """
    Streams the answer in small word-groups for the typing effect.

    Yields 8 words per tick at 10ms — ~800 words/s. The previous version
    slept 30ms per word (~33 words/s), which added ~9s of artificial
    latency to a 300-word answer on top of the real API time.
    """
    if not text:
        yield "Error: Empty response."
        return
    words = text.split(" ")
    group = 8
    for i in range(0, len(words), group):
        yield " ".join(words[i:i + group]) + " "
        time.sleep(0.01)


def intent_chip(intent: str) -> str:
    classes = {
        "EXTRACT": "chip-extract", "HYBRID": "chip-hybrid",
        "GENERAL": "chip-general", "BENCHMARK": "chip-benchmark",
        "COMPARE": "chip-compare", "EWS": "chip-ews",
    }
    cls = classes.get(intent, "chip-general")
    return f'<span class="intent-chip {cls}">{intent}</span>'
