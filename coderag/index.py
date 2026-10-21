from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional

import faiss
import numpy as np

from coderag import config

logger = logging.getLogger(__name__)

EmbedFn = Callable[[List[str]], np.ndarray]


@dataclass
class Chunk:
    path: str  # relative to the indexed root
    start_line: int  # 1-based, inclusive
    end_line: int
    text: str


@dataclass
class IndexStats:
    files: int = 0
    reused: int = 0
    embedded: int = 0
    removed: int = 0
    chunks: int = 0


def iter_source_files(root: Path) -> Iterator[Path]:
    """Walk root and yield files with an allowed extension, skipping ignored folders."""
    for dirpath, dirnames, filenames in os.walk(root):
        # editing dirnames in place stops os.walk from going into those folders
        dirnames[:] = sorted(d for d in dirnames if d not in config.IGNORE_DIRS and not d.startswith("."))
        for name in sorted(filenames):
            path = Path(dirpath) / name
            if path.suffix.lower() not in config.ALLOWED_EXTENSIONS:
                continue
            try:
                if path.stat().st_size > config.MAX_FILE_BYTES:
                    logger.info("Skipping large file %s", path)
                    continue
            except OSError:
                continue
            yield path


def chunk_lines(
    text: str,
    path: str,
    max_lines: int = config.CHUNK_LINES,
    overlap: int = config.CHUNK_OVERLAP,
) -> List[Chunk]:
    """Split a file into line windows that overlap a little, so a function cut at a boundary is still found."""
    lines = text.splitlines()
    if not any(line.strip() for line in lines):
        return []

    step = max(max_lines - overlap, 1)
    chunks = []
    for start in range(0, len(lines), step):
        window = lines[start : start + max_lines]
        if any(line.strip() for line in window):
            chunks.append(Chunk(path, start + 1, start + len(window), "\n".join(window)))
        if start + max_lines >= len(lines):
            break
    return chunks


def _embedding_input(chunk: Chunk) -> str:
    # the file path carries a lot of meaning (e.g. "auth/login.py"), so embed it with the code
    return f"File: {chunk.path} (lines {chunk.start_line}-{chunk.end_line})\n{chunk.text}"


def _file_hash(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


def index_paths(root: Path, cache_dir: Path = config.CACHE_DIR) -> tuple:
    """Index and metadata file for a folder, e.g. ~/.cache/coderag/myrepo-1a2b3c4d/index.faiss."""
    root = Path(root).resolve()
    folder = cache_dir / f"{root.name}-{_file_hash(str(root).encode())[:8]}"
    return folder / "index.faiss", folder / "meta.json"


class CodeIndex:
    """FAISS inner-product index over code chunks, plus the metadata to map hits back to files."""

    def __init__(self, root: Path, dim: int = 0, model: str = config.EMBEDDING_MODEL):
        self.root = Path(root).resolve()
        self.model = model
        self.dim = dim
        self.chunks: List[Chunk] = []
        self.file_hashes: Dict[str, str] = {}
        self.faiss_index: Optional[faiss.Index] = faiss.IndexFlatIP(dim) if dim else None

    def __len__(self) -> int:
        return len(self.chunks)

    # ---------- building ----------

    def build(self, embed_fn: EmbedFn, progress: Optional[Callable[[int, int], None]] = None) -> IndexStats:
        """
        (Re)index self.root. Files whose content hash did not change keep their old vectors,
        so only new or edited files are sent to the embedding model.
        """
        stats = IndexStats()
        old_vectors = self._vectors_by_file()

        new_chunks: List[Chunk] = []
        new_hashes: Dict[str, str] = {}
        parts: List[np.ndarray] = []
        to_embed: List[Chunk] = []

        files = list(iter_source_files(self.root))
        stats.files = len(files)

        for i, path in enumerate(files):
            rel = path.relative_to(self.root).as_posix()
            try:
                data = path.read_bytes()
                text = data.decode("utf-8")
            except (OSError, UnicodeDecodeError):
                logger.info("Skipping unreadable file %s", rel)
                continue

            digest = _file_hash(data)
            new_hashes[rel] = digest

            if self.file_hashes.get(rel) == digest:
                # unchanged file (empty files have no vectors at all)
                if rel in old_vectors:
                    chunks, vecs = old_vectors[rel]
                    new_chunks.extend(chunks)
                    parts.append(vecs)
                stats.reused += 1
            else:
                chunks = chunk_lines(text, rel)
                new_chunks.extend(chunks)
                to_embed.extend(chunks)
                # placeholder, filled after the batch call below
                parts.append(np.empty((len(chunks), 0), dtype=np.float32))
                stats.embedded += 1

            if progress:
                progress(i + 1, len(files))

        stats.removed = len(set(self.file_hashes) - set(new_hashes))

        fresh = embed_fn([_embedding_input(c) for c in to_embed]) if to_embed else None
        vectors = self._merge(parts, fresh)

        self.chunks = new_chunks
        self.file_hashes = new_hashes
        if vectors is not None and len(vectors):
            self.dim = vectors.shape[1]
            self.faiss_index = faiss.IndexFlatIP(self.dim)
            self.faiss_index.add(vectors)
        else:
            self.faiss_index = faiss.IndexFlatIP(self.dim) if self.dim else None

        stats.chunks = len(self.chunks)
        return stats

    def _merge(self, parts: List[np.ndarray], fresh: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Put reused and freshly embedded vectors back together in chunk order."""
        out, offset = [], 0
        for part in parts:
            if part.shape[1] == 0:
                n = part.shape[0]
                if n:
                    out.append(fresh[offset : offset + n])
                offset += n
            else:
                out.append(part)
        if not out:
            return None
        return np.ascontiguousarray(np.vstack(out), dtype=np.float32)

    def _vectors_by_file(self) -> Dict[str, tuple]:
        """Group the current chunks and their stored vectors by file."""
        if self.faiss_index is None or self.faiss_index.ntotal == 0:
            return {}
        all_vecs = self.faiss_index.reconstruct_n(0, self.faiss_index.ntotal)
        grouped: Dict[str, tuple] = {}
        for row, chunk in enumerate(self.chunks):
            chunks, rows = grouped.setdefault(chunk.path, ([], []))
            chunks.append(chunk)
            rows.append(row)
        return {path: (chunks, all_vecs[rows]) for path, (chunks, rows) in grouped.items()}

    # ---------- search ----------

    def query(self, vector: np.ndarray, k: int) -> List[tuple]:
        """Return [(chunk, score)] for the k nearest chunks to an already normalized query vector."""
        if self.faiss_index is None or self.faiss_index.ntotal == 0:
            return []
        vector = np.ascontiguousarray(vector.reshape(1, -1), dtype=np.float32)
        if vector.shape[1] != self.dim:
            raise ValueError(f"Query has dimension {vector.shape[1]}, index has {self.dim}")
        scores, ids = self.faiss_index.search(vector, min(k, self.faiss_index.ntotal))
        return [(self.chunks[i], float(s)) for s, i in zip(scores[0], ids[0], strict=True) if i != -1]

    # ---------- persistence ----------

    def save(self, cache_dir: Path = config.CACHE_DIR) -> None:
        index_file, meta_file = index_paths(self.root, cache_dir)
        index_file.parent.mkdir(parents=True, exist_ok=True)
        if self.faiss_index is not None:
            faiss.write_index(self.faiss_index, str(index_file))
        meta = {
            "root": str(self.root),
            "model": self.model,
            "dim": self.dim,
            "file_hashes": self.file_hashes,
            "chunks": [asdict(c) for c in self.chunks],
        }
        meta_file.write_text(json.dumps(meta), encoding="utf-8")

    @classmethod
    def load(
        cls,
        root: Path,
        model: str = config.EMBEDDING_MODEL,
        cache_dir: Path = config.CACHE_DIR,
    ) -> CodeIndex:
        """Load a saved index. Returns an empty one if nothing usable is on disk."""
        index = cls(root, model=model)
        index_file, meta_file = index_paths(index.root, cache_dir)
        if not (index_file.exists() and meta_file.exists()):
            return index

        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.warning("Could not read %s, starting fresh", meta_file)
            return index

        # vectors from another embedding model are not comparable
        if meta.get("root") != str(index.root) or meta.get("model") != model:
            logger.info("Saved index was built with another model, starting fresh")
            return index

        index.dim = meta["dim"]
        index.file_hashes = meta["file_hashes"]
        index.chunks = [Chunk(**c) for c in meta["chunks"]]
        index.faiss_index = faiss.read_index(str(index_file))

        if index.faiss_index.ntotal != len(index.chunks):
            logger.warning("Index and metadata are out of sync, starting fresh")
            return cls(root, model=model)
        return index
