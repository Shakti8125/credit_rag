import logging
from fastapi import APIRouter, status

logger = logging.getLogger(__name__)

# Initialize the router for health tracking
router = APIRouter()

@router.get(
    "/health",
    status_code=status.HTTP_200_OK,
    summary="Infrastructure liveness probe",
    description="Monitors API availability for AWS Gateway target groups and EventBridge warmup rules."
)
async def check_health():
    """
    Returns a fast, static JSON response to verify the container runtime is active.
    """
    logger.debug("Health check endpoint pinged.")
    return {
        "status": "healthy",
        "service": "credit-risk-rag-cloud",
        "version": "1.0.0"
    }