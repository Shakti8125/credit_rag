"""
Process-wide singletons for GenerationService and PineconeRetrievalService.

Both are expensive to construct (Gemini client init, Pinecone connection,
cross-encoder model load — several seconds). They must be created exactly
once per process. Previously query.py, compare.py, and ews.py each built
their own copy at import time, tripling startup cost and reconnecting to
Pinecone/loading the cross-encoder three times over.
"""

import logging

from app.services.dynamo     import TelemetryLogger
from app.services.generation import GenerationService
from app.services.retrieval  import PineconeRetrievalService
from shared.env import get_env

logger = logging.getLogger(__name__)

try:
    llm_engine = GenerationService()
except Exception as e:
    logger.warning("GenerationService unavailable: %s", e)
    llm_engine = None

try:
    retrieval_engine = PineconeRetrievalService()
except Exception as e:
    logger.warning("PineconeRetrievalService unavailable: %s", e)
    retrieval_engine = None

# Telemetry is opt-in via TELEMETRY_TABLE, set in the deployed Lambda's
# environment. Unset — the local-dev default — means no DynamoDB client is
# built and no per-request write is attempted at all, so developer machines
# never pay a doomed boto3 round trip on every query.
_telemetry_table = get_env("TELEMETRY_TABLE")
telemetry = TelemetryLogger(table_name=_telemetry_table) if _telemetry_table else None
if telemetry is None:
    logger.info("Telemetry disabled (TELEMETRY_TABLE unset).")


def record_telemetry(intent: str, path: str, latency_ms: int) -> None:
    """
    Record one request's operational metadata — intent, execution path, latency.
    Never payload text. No-ops when telemetry is disabled, and log_transaction
    swallows its own errors, so this can never affect a response.
    """
    if telemetry is not None:
        telemetry.log_transaction(intent, path, latency_ms)
