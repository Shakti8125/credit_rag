import os
import logging
from dotenv import load_dotenv

# CRITICAL: Load environment variables BEFORE importing any application routes or services.
# This ensures that when query_router initializes Gemini and Pinecone, the API keys are already in memory.
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from mangum import Mangum

# Import the routing nodes we built earlier
from app.routes.query import router as query_router
from app.routes.health import router as health_router

# Configure standard logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize the core FastAPI application
app = FastAPI(
    title="Credit Risk RAG Backend",
    description="Orchestration layer for localized edge masking and cloud generative synthesis.",
    version="1.0.0"
)

# Configure CORS so your local Streamlit frontend can communicate with this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the operational routes
app.include_router(query_router)
app.include_router(health_router)

# Wrap the FastAPI app with Mangum to enable AWS Lambda serverless execution later
handler = Mangum(app)