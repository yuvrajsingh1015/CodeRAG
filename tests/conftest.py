import hashlib
import re

import numpy as np
import pytest

from coderag.embeddings import normalize

DIM = 64


def fake_embed(texts):
    """Bag-of-words hashed into DIM buckets. Similar words give similar vectors, no API needed."""
    out = np.zeros((len(texts), DIM), dtype=np.float32)
    for row, text in enumerate(texts):
        for word in re.findall(r"[a-z]+", text.lower()):
            bucket = int(hashlib.md5(word.encode()).hexdigest(), 16) % DIM
            out[row, bucket] += 1.0
    return normalize(out)


def fake_query(text):
    return fake_embed([text])


class CountingEmbed:
    """Wraps fake_embed and records how many texts were embedded."""

    def __init__(self):
        self.calls = 0
        self.texts = 0

    def __call__(self, texts):
        self.calls += 1
        self.texts += len(texts)
        return fake_embed(texts)


@pytest.fixture
def embed():
    return CountingEmbed()


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    (root / "auth").mkdir(parents=True)
    (root / "db").mkdir()
    (root / "node_modules" / "lib").mkdir(parents=True)
    (root / "auth" / "login.py").write_text(
        "def login(user, password):\n    token = create_session_token(user)\n    return token\n"
    )
    (root / "db" / "connection.py").write_text(
        "import sqlite3\n\ndef connect_database(path):\n    return sqlite3.connect(path)\n"
    )
    (root / "README.md").write_text("# Demo\nA tiny demo project.\n")
    (root / "image.png").write_bytes(b"\x89PNG not code")
    (root / "node_modules" / "lib" / "index.js").write_text("module.exports = 1\n")
    (root / ".github" / "workflows").mkdir(parents=True)
    (root / ".github" / "workflows" / "deploy.yml").write_text(
        "on: push\njobs:\n  deploy:\n    runs-on: ubuntu-latest\n"
    )
    (root / ".cache").mkdir()
    (root / ".cache" / "junk.py").write_text("x = 1\n")
    return root


@pytest.fixture
def cache(tmp_path):
    return tmp_path / "cache"
