import logging
from typing import List, Optional

import numpy as np

from coderag.ai_client import get_ai_client
from coderag.config import EMBEDDING_DIM, EMBEDDING_MODEL
from coderag.retry import api_retry

logger = logging.getLogger(__name__)

BATCH_SIZE = 32


def _chunk_text(text: str, max_chars: int = 4000, overlap: int = 400) -> List[str]:
    """Chunk text with small overlap to preserve boundary context."""
    text = text.strip()
    if not text:
        return []

    if len(text) <= max_chars:
        return [text]

    # [0:4000], [3600:7600], [7200:11200], ...
    step = max_chars - overlap
    return [text[i : i + max_chars] for i in range(0, len(text), step)]


def normalize(vecs: np.ndarray) -> np.ndarray:
    """L2-normalize rows so inner product == cosine similarity."""
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (vecs / norms).astype(np.float32)


# indexing can wait through a few rate limit windows (Gemini free tier: 100 embeddings per minute)
@api_retry(attempts=6)
def _embed_batch(inputs: List[str]) -> np.ndarray:
    """Call embeddings API with retry/backoff. Returns shape (n, d)."""
    client, _provider = get_ai_client()
    if client is None:
        raise RuntimeError("AI client not initialized")

    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=inputs,
        timeout=30,
    )
    # float32 for faiss
    return np.array([d.embedding for d in response.data], dtype=np.float32)


def embed_texts(texts: List[str]) -> np.ndarray:
    """Embed many texts in batches. Returns normalized vectors shaped (n, d)."""
    if not texts:
        return np.zeros((0, max(EMBEDDING_DIM, 1)), dtype=np.float32)

    parts = [_embed_batch(texts[i : i + BATCH_SIZE]) for i in range(0, len(texts), BATCH_SIZE)]
    vecs = np.vstack(parts)

    if EMBEDDING_DIM and vecs.shape[1] != EMBEDDING_DIM:
        raise ValueError(
            f"Configured EMBEDDING_DIM={EMBEDDING_DIM}, but model '{EMBEDDING_MODEL}' returned {vecs.shape[1]}"
        )
    return normalize(vecs)


def generate_embeddings(text: str) -> Optional[np.ndarray]:
    """Generate one averaged embedding vector shaped (1, d). Used for queries."""
    if not text or not text.strip():
        logger.warning("Empty text provided for embedding generation")
        return None

    try:
        chunks = _chunk_text(text)
        if not chunks:
            return None

        vecs = embed_texts(chunks)

        # axis 0 means average across rows, reshape (d,) -> (1, d)
        avg = np.mean(vecs, axis=0, dtype=np.float32).reshape(1, -1)
        return normalize(avg)

    except Exception as e:
        logger.error("Failed to generate embeddings: %s", e)
        return None
