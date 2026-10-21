from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _get_env(*names: str, default: str = "") -> str:
    for name in names:
        value = os.getenv(name)
        if value and value.strip():
            return value.strip()
    return default


# "local" (Ollama or any OpenAI-compatible server), "remote" (OpenAI) or "gemini" (Google AI Studio)
AI_BACKEND: str = _get_env("AI_BACKEND", default="local").lower()

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

# default models per backend, used when CHAT_MODEL / EMBEDDING_MODEL are not set
DEFAULT_MODELS: dict[str, tuple[str, str]] = {
    "local": ("gemma3:latest", "nomic-embed-text"),
    "remote": ("gpt-4o-mini", "text-embedding-3-small"),
    "gemini": ("gemini-flash-latest", "gemini-embedding-001"),
}
_BACKEND_ALIASES = {"ollama": "local", "openai": "remote", "google": "gemini"}
_default_chat, _default_embed = DEFAULT_MODELS.get(
    _BACKEND_ALIASES.get(AI_BACKEND, AI_BACKEND), DEFAULT_MODELS["local"]
)

OPENAI_API_KEY: str = _get_env("API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", default="")
OPENAI_BASE_URL: str = _get_env("BASE_URL", default="http://localhost:11434/v1/")
CHAT_MODEL: str = _get_env("CHAT_MODEL", default=_default_chat)
EMBEDDING_MODEL: str = _get_env("EMBEDDING_MODEL", default=_default_embed)

WATCHED_DIR: Path = Path(_get_env("WATCHED_DIR", default=".")).expanduser().resolve()
# Where indexes are stored. Each indexed folder gets its own subfolder in here.
CACHE_DIR: Path = Path(_get_env("CODERAG_CACHE_DIR", default="~/.cache/coderag")).expanduser().resolve()

# 0 means "use whatever the embedding model returns"
EMBEDDING_DIM: int = int(_get_env("EMBEDDING_DIM", default="0"))

CHUNK_LINES: int = int(_get_env("CHUNK_LINES", default="60"))
CHUNK_OVERLAP: int = int(_get_env("CHUNK_OVERLAP", default="10"))
TOP_K: int = int(_get_env("TOP_K", default="5"))
MAX_FILE_BYTES: int = int(_get_env("MAX_FILE_BYTES", default="200000"))


# This creates a set of folder names to skip during indexing.
IGNORE_DIRS: set[str] = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    "node_modules",
    "build",
    "dist",
    "target",
    ".idea",
    ".vscode",
    ".coderag",
    ".gradle",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
}

ALLOWED_EXTENSIONS: set[str] = {
    ".py",
    ".java",
    ".kt",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".rs",
    ".go",
    ".c",
    ".h",
    ".cpp",
    ".hpp",
    ".cs",
    ".rb",
    ".php",
    ".swift",
    ".scala",
    ".ml",
    ".sh",
    ".sql",
    ".md",
    ".toml",
    ".yaml",
    ".yml",
}
