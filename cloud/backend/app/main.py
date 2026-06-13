import os
import logging
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from mangum import Mangum

from app.routes.query   import router as query_router
from app.routes.health  import router as health_router
from app.routes.compare import router as compare_router
from app.routes.ews     import router as ews_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Credit Risk RAG Backend",
    description="Orchestration layer for localised edge masking and cloud generative synthesis.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(query_router)
app.include_router(health_router)
app.include_router(compare_router)   # Tier 2 — multi-doc comparison
app.include_router(ews_router)        # Tier 2 — early warning signals

handler = Mangum(app)
