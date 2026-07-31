"""
Endpoint URLs, model path, and retrieval tuning constants for the Streamlit frontend.

The cloud API base is overridable via CLOUD_API_BASE so this tier never hardcodes
a deployment target; the model path is resolved via shared/paths.py instead of a
hardcoded absolute path.
"""

from shared.env import get_env
from shared.paths import SLM_MODEL_PATH as _SLM_MODEL_PATH

_API_BASE = get_env("CLOUD_API_BASE", "http://127.0.0.1:8000")

API_ENDPOINT = f"{_API_BASE}/query"
API_COMPARE  = f"{_API_BASE}/compare"
API_EWS      = f"{_API_BASE}/ews"

# Shared secret for the deployed backend's X-API-Key check (see
# cloud/backend/app/auth.py). Left empty against a local uvicorn backend, where
# enforcement self-disables — so the same code path works in both places.
CLOUD_API_KEY = get_env("CLOUD_API_KEY", "") or ""
API_HEADERS   = {"X-API-Key": CLOUD_API_KEY} if CLOUD_API_KEY else {}

SLM_MODEL_PATH = str(_SLM_MODEL_PATH)

RERANK_POOL    = 20
PHI3_TOP_K     = 3
FAISS_POOL     = 15
CLOUD_TOP_K    = 5
CHUNK_CHAR_CAP = 1200
