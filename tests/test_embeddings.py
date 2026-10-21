from types import SimpleNamespace

import numpy as np
from openai import RateLimitError

from coderag import embeddings


def _rate_limit(message):
    # skip the SDK constructor, it needs a real HTTP response object
    error = RateLimitError.__new__(RateLimitError)
    Exception.__init__(error, message)
    return error


def _state(error):
    return SimpleNamespace(outcome=SimpleNamespace(exception=lambda: error), attempt_number=1)


def test_wait_uses_retry_delay_from_error():
    assert embeddings._wait(_state(_rate_limit("Quota exceeded. Please retry in 17.69s."))) == 18.69


def test_wait_without_delay_falls_back_to_30s():
    assert embeddings._wait(_state(_rate_limit("Too many requests"))) == 30.0


def test_wait_is_short_for_other_errors():
    assert embeddings._wait(_state(TimeoutError())) <= 8


def test_normalize_rows():
    vecs = embeddings.normalize(np.array([[3.0, 4.0], [0.0, 0.0]]))
    assert np.allclose(vecs[0], [0.6, 0.8])
    assert np.allclose(vecs[1], [0.0, 0.0])


def test_chunk_text_overlap():
    chunks = embeddings._chunk_text("a" * 9000, max_chars=4000, overlap=400)
    assert [len(c) for c in chunks] == [4000, 4000, 1800]
