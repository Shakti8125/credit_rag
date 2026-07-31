"""
Shared-secret authentication for the deployed cloud tier.

The Lambda Function URL is created with --auth-type NONE: IAM (SigV4) auth
would require the analyst machine to carry AWS credentials and sign every
request, which it deliberately does not. Authentication therefore lives here,
in the application — /query, /compare and /ews each require an X-API-Key
header matching the CREDITRAG_API_KEY secret held in SSM Parameter Store.

/health is deliberately left unauthenticated so load-balancer probes and
warmup pings need no credentials.

Enforcement is a no-op when the secret is unset, so `uvicorn app.main:app` on
a developer machine and the offline eval suites keep working unconfigured.

Scope note: this is quota protection, not a privacy control. The privacy
contract is upheld by the local tier — payloads reaching these routes are
already masked — so a leaked key costs Gemini/Pinecone spend, not customer
data.
"""

import hmac
import logging

from fastapi import Header, HTTPException, status

from app.services.secrets import get_secret

logger = logging.getLogger(__name__)

# None = not yet resolved; "" = resolved and absent, i.e. enforcement disabled.
_expected_key: str | None = None


def _load_expected_key() -> str:
    """
    Resolve CREDITRAG_API_KEY once per process (get_secret has its own warm
    cache, but this also caches the *absent* case, which get_secret raises on).
    """
    global _expected_key
    if _expected_key is None:
        try:
            _expected_key = get_secret("CREDITRAG_API_KEY") or ""
        except Exception:
            _expected_key = ""

        if not _expected_key:
            logger.warning(
                "CREDITRAG_API_KEY is not configured — API key enforcement is DISABLED. "
                "Expected for local development; never for a public endpoint."
            )
    return _expected_key


async def require_api_key(x_api_key: str = Header(default="")) -> None:
    """FastAPI dependency. Raises 401 unless the request carries the shared secret."""
    expected = _load_expected_key()
    if not expected:
        return

    # Compare as bytes: compare_digest is constant-time (no early-exit timing
    # oracle on the key) and rejects non-ASCII str inputs outright.
    if not hmac.compare_digest(x_api_key.encode("utf-8"), expected.encode("utf-8")):
        logger.warning("Rejected request with missing or invalid X-API-Key.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-API-Key header.",
        )
