import uuid
import time
import logging
import boto3

from shared.env import get_env

logger = logging.getLogger(__name__)

class TelemetryLogger:
    """
    Telemetry logger recording system operational metadata to DynamoDB.
    Strictly excludes query payloads, texts, or tokens to prevent data leakage.

    The target table's partition key must be `LogId` (String) — see the item
    written in log_transaction below. Creating it with any other key name makes
    every write fail with ValidationException.
    """
    def __init__(self, table_name: str = "CreditRAG_Telemetry", region: str | None = None):
        # Lambda injects AWS_REGION automatically; ap-south-1 is the deployment
        # default for anything running outside it.
        region = region or get_env("AWS_REGION", "ap-south-1")
        try:
            self.dynamodb = boto3.resource("dynamodb", region_name=region)
            self.table = self.dynamodb.Table(table_name)
        except Exception as e:
            logger.warning(f"DynamoDB binding bypassed (Local initialization mode): {str(e)}")
            self.table = None

    def log_transaction(self, intent: str, path: str, latency_ms: int) -> None:
        """
        Commits performance metrics safely to the remote AWS DynamoDB ledger.
        Catches and swallows database connection errors to protect the user path.
        """
        if not self.table:
            logger.info("Telemetry logging skipped: No active database connection.")
            return

        try:
            self.table.put_item(
                Item={
                    "LogId": str(uuid.uuid4()),
                    "Timestamp": int(time.time()),
                    "Intent": intent,
                    "ExecutionPath": path,
                    "LatencyMs": latency_ms
                }
            )
        except Exception as e:
            logger.error(f"Telemetry persistence write failed silently: {str(e)}")