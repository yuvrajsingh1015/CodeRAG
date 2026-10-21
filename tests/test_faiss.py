import numpy as np

from coderag.index import CodeIndex, index_paths
from tests.conftest import fake_query


def test_save_and_load_roundtrip(repo, embed, cache):
    index = CodeIndex(repo, model="fake")
    index.build(embed)
    index.save(cache)

    loaded = CodeIndex.load(repo, model="fake", cache_dir=cache)

    assert len(loaded) == len(index)
    assert loaded.file_hashes == index.file_hashes
    q = fake_query("connect database")
    assert [c.path for c, _ in loaded.query(q, 2)] == [c.path for c, _ in index.query(q, 2)]


def test_loaded_index_is_reused_incrementally(repo, embed, cache):
    index = CodeIndex(repo, model="fake")
    index.build(embed)
    index.save(cache)
    calls = embed.calls

    loaded = CodeIndex.load(repo, model="fake", cache_dir=cache)
    stats = loaded.build(embed)

    assert stats.reused == 4
    assert embed.calls == calls


def test_other_model_starts_fresh(repo, embed, cache):
    index = CodeIndex(repo, model="fake")
    index.build(embed)
    index.save(cache)

    assert len(CodeIndex.load(repo, model="other", cache_dir=cache)) == 0


def test_missing_index_is_empty(repo, cache):
    assert len(CodeIndex.load(repo, cache_dir=cache)) == 0


def test_each_folder_gets_its_own_files(tmp_path, cache):
    a, _ = index_paths(tmp_path / "a", cache)
    b, _ = index_paths(tmp_path / "b", cache)
    assert a != b


def test_scores_are_cosine(repo, embed):
    index = CodeIndex(repo)
    index.build(embed)
    vecs = index.faiss_index.reconstruct_n(0, index.faiss_index.ntotal)
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-5)
