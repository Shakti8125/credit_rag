import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from shared.env import load_env, get_env
from shared.logging_config import setup_logging, get_logger

load_env()
setup_logging()

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from mangum import Mangum

from app.auth import require_api_key
from app.routes.query   import router as query_router
from app.routes.health  import router as health_router
from app.routes.compare import router as compare_router
from app.routes.ews     import router as ews_router

logger = get_logger(__name__)

app = FastAPI(
    title="Credit Risk RAG Backend",
    description="Orchestration layer for localised edge masking and cloud generative synthesis.",
    version="2.0.0",
)

# CORS_ALLOW_ORIGINS: comma-separated origin list for deployed environments
# (e.g. "https://app.example.com,https://staging.example.com").
# Defaults to "*" for local development. Note: allow_credentials must be
# False with a wildcard origin — browsers reject that combination.
_origins = [o.strip() for o in get_env("CORS_ALLOW_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=_origins != ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inference routes require the X-API-Key shared secret (see app/auth.py); the
# check self-disables when CREDITRAG_API_KEY is unset, so local dev is
# unaffected. /health stays open so probes and warmup pings need no credential.
_protected = [Depends(require_api_key)]

app.include_router(query_router,   dependencies=_protected)
app.include_router(health_router)
app.include_router(compare_router, dependencies=_protected)   # Tier 2 — multi-doc comparison
app.include_router(ews_router,     dependencies=_protected)   # Tier 2 — early warning signals

handler = Mangum(app)
