from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np

from coderag import config
from coderag.embeddings import generate_embeddings
from coderag.index import Chunk, CodeIndex

logger = logging.getLogger(__name__)

QueryEmbedFn = Callable[[str], Optional[np.ndarray]]


@dataclass
class SearchResult:
    chunk: Chunk
    score: float

    @property
    def location(self) -> str:
        return f"{self.chunk.path}:{self.chunk.start_line}-{self.chunk.end_line}"


def search_code(
    index: CodeIndex,
    query: str,
    k: int = config.TOP_K,
    embed_query: QueryEmbedFn = generate_embeddings,
    min_score: float = 0.0,
) -> List[SearchResult]:
    """Embed the query and return the k most similar chunks, best first."""
    if not query.strip() or len(index) == 0:
        return []

    vector = embed_query(query)
    if vector is None:
        logger.error("Could not embed query")
        return []

    hits = index.query(vector, k)
    return [SearchResult(chunk, score) for chunk, score in hits if score >= min_score]
