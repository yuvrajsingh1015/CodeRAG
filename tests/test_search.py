from coderag.index import CodeIndex
from coderag.search import search_code
from prompt_flow import _by_line, build_messages, format_context, search_query
from tests.conftest import fake_query


def built(repo, embed):
    index = CodeIndex(repo)
    index.build(embed)
    return index


def test_finds_the_right_file(repo, embed):
    results = search_code(
        built(repo, embed), "where is the session token created on login", k=1, embed_query=fake_query
    )
    assert results[0].chunk.path == "auth/login.py"


def test_results_are_sorted_and_limited(repo, embed):
    results = search_code(built(repo, embed), "connect sqlite database", k=2, embed_query=fake_query)
    assert len(results) == 2
    assert results[0].chunk.path == "db/connection.py"
    assert results[0].score >= results[1].score


def test_location_format(repo, embed):
    result = search_code(built(repo, embed), "login", k=1, embed_query=fake_query)[0]
    assert result.location == "auth/login.py:1-3"


def test_empty_query_or_index(repo, embed):
    assert search_code(built(repo, embed), "   ", embed_query=fake_query) == []
    assert search_code(CodeIndex(repo), "login", embed_query=fake_query) == []


def test_prompt_contains_cited_context(repo, embed):
    results = search_code(built(repo, embed), "login token", k=1, embed_query=fake_query)
    messages = build_messages("How does login work?", results, history=[{"role": "user", "content": "hi"}])

    assert messages[0]["role"] == "system"
    assert messages[1] == {"role": "user", "content": "hi"}
    assert "### auth/login.py:1-3" in messages[-1]["content"]
    assert "```py" in format_context(results)
    assert "   1 | def login(user, password):" in format_context(results)
    assert messages[-1]["content"].endswith("Question: How does login work?")


def test_prompt_without_results():
    messages = build_messages("anything?", [])
    assert "(no matching code found)" in messages[-1]["content"]


def test_file_list_goes_into_system_prompt():
    messages = build_messages("where?", [], files=["src/b.py", "src/a.py"])
    assert messages[0]["content"].endswith("Files in the project:\nsrc/a.py\nsrc/b.py")


def test_file_list_left_out_for_big_repos():
    messages = build_messages("where?", [], files=[f"f{i}.py" for i in range(1000)])
    assert "Files in the project" not in messages[0]["content"]


def test_line_numbers_are_removed_from_streamed_answers():
    pieces = ["See `a.py:3-4`:\n```py\n   3 | def f", "():\n   4 |     return 1\n", "```\nDone, 2 | 3 stays."]
    assert "".join(_by_line(iter(pieces))) == "See `a.py:3-4`:\n```py\ndef f():\n    return 1\n```\nDone, 2 | 3 stays."


def test_follow_up_is_searched_with_previous_question():
    history = [
        {"role": "user", "content": "How does login work?"},
        {"role": "assistant", "content": "It creates a session token."},
    ]
    assert search_query("And where is it stored?", history) == "How does login work?\nAnd where is it stored?"
    assert search_query("How does login work?") == "How does login work?"
