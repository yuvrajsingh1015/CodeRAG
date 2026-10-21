import logging
from functools import lru_cache
from typing import Optional, Tuple

from openai import OpenAI

from coderag.config import AI_BACKEND, OPENAI_API_KEY, OPENAI_BASE_URL

logger = logging.getLogger(__name__)

LOCAL_NAMES = {"local", "ollama"}
REMOTE_NAMES = {"remote", "openai"}


def create_ai_client(
    backend: str = AI_BACKEND,
    api_key: str = OPENAI_API_KEY,
    base_url: str = OPENAI_BASE_URL,
) -> Tuple[Optional[OpenAI], str]:
    """
    Create a generic OpenAI-compatible client.

    Backends:
    - "local" / "ollama"  → uses BASE_URL (Ollama, LM Studio, ...)
    - "remote" / "openai" → uses API key (hosted service)
    """

    try:
        backend = backend.strip().lower()

        # --- Local backend (uses base URL) ---
        if backend in LOCAL_NAMES:
            base_url = base_url or "http://localhost:11434/v1/"
            # Ollama ignores the key, but the SDK wants one
            client = OpenAI(api_key=api_key or "local-key", base_url=base_url)

            logger.info("Using local backend at %s", base_url)
            return client, f"Local backend ({base_url})"

        # --- Remote backend (uses API key) ---
        if backend in REMOTE_NAMES:
            if not api_key:
                logger.error("Missing API key for remote backend")
                return None, "Unavailable"

            client = OpenAI(api_key=api_key)
            logger.info("Using remote backend")
            return client, "OpenAI"

        # --- Unknown backend ---
        logger.error("Unsupported AI_BACKEND: %s", backend)
        return None, "Unavailable"

    except Exception as e:
        logger.error("Failed to initialize AI client: %s", e)
        return None, "Unavailable"


@lru_cache(maxsize=1)
def get_ai_client() -> Tuple[Optional[OpenAI], str]:
    """Shared client for the configured backend, created once."""
    return create_ai_client()
