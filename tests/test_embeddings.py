import numpy as np

from coderag import embeddings


def test_normalize_rows():
    vecs = embeddings.normalize(np.array([[3.0, 4.0], [0.0, 0.0]]))
    assert np.allclose(vecs[0], [0.6, 0.8])
    assert np.allclose(vecs[1], [0.0, 0.0])


def test_chunk_text_overlap():
    chunks = embeddings._chunk_text("a" * 9000, max_chars=4000, overlap=400)
    assert [len(c) for c in chunks] == [4000, 4000, 1800]
