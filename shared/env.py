"""Single .env resolution point, used by both tiers so neither reaches into the other's folder."""

import os
from typing import Optional

from dotenv import load_dotenv

from shared.paths import ENV_FILE

_loaded = False


def load_env() -> None:
    global _loaded
    if _loaded:
        return
    load_dotenv(dotenv_path=ENV_FILE)
    _loaded = True


def get_env(key: str, default: Optional[str] = None) -> Optional[str]:
    load_env()
    return os.getenv(key, default)
