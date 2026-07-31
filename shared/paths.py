"""Central path constants, resolved relative to this file so nothing hardcodes an absolute path."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

LOCAL_DIR  = PROJECT_ROOT / "local"
CLOUD_DIR  = PROJECT_ROOT / "cloud"
DATA_DIR   = PROJECT_ROOT / "data"

MODELS_DIR      = LOCAL_DIR / "slm" / "models"
SLM_MODEL_PATH  = MODELS_DIR / "phi3-basel-q4km.gguf"

ENV_FILE = PROJECT_ROOT / ".env"
