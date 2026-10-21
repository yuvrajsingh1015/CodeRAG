from coderag.index import CodeIndex, chunk_lines, iter_source_files


def test_iter_source_files_skips_ignored_and_binary(repo):
    files = sorted(p.relative_to(repo).as_posix() for p in iter_source_files(repo))
    assert files == ["README.md", "auth/login.py", "db/connection.py"]


def test_chunk_lines_overlap_and_line_numbers():
    text = "\n".join(f"line {i}" for i in range(1, 26))
    chunks = chunk_lines(text, "a.py", max_lines=10, overlap=2)
    assert [(c.start_line, c.end_line) for c in chunks] == [(1, 10), (9, 18), (17, 25)]
    assert chunks[1].text.splitlines()[0] == "line 9"


def test_chunk_lines_empty_file():
    assert chunk_lines("\n\n  \n", "a.py") == []


def test_build_indexes_all_chunks(repo, embed):
    index = CodeIndex(repo)
    stats = index.build(embed)
    assert stats.files == 3
    assert stats.embedded == 3
    assert len(index) == index.faiss_index.ntotal == 3


def test_rebuild_only_embeds_changed_files(repo, embed):
    index = CodeIndex(repo)
    index.build(embed)
    first = embed.texts

    (repo / "auth" / "login.py").write_text("def logout(user):\n    pass\n")
    stats = index.build(embed)

    assert stats.reused == 2
    assert stats.embedded == 1
    assert embed.texts - first == 1
    assert {c.path for c in index.chunks} == {"README.md", "auth/login.py", "db/connection.py"}


def test_rebuild_drops_deleted_files(repo, embed):
    index = CodeIndex(repo)
    index.build(embed)
    (repo / "db" / "connection.py").unlink()

    stats = index.build(embed)

    assert stats.removed == 1
    assert "db/connection.py" not in {c.path for c in index.chunks}
    assert index.faiss_index.ntotal == len(index)


def test_unchanged_rebuild_makes_no_api_call(repo, embed):
    index = CodeIndex(repo)
    index.build(embed)
    calls = embed.calls
    index.build(embed)
    assert embed.calls == calls


def test_empty_files_count_as_unchanged(repo, embed):
    (repo / "auth" / "__init__.py").write_text("")
    index = CodeIndex(repo)
    index.build(embed)
    stats = index.build(embed)
    assert stats.embedded == 0
    assert stats.reused == 4
