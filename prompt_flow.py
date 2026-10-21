from __future__ import annotations

import logging
import re
from typing import Dict, Iterator, List, Optional

from coderag import config
from coderag.ai_client import get_ai_client
from coderag.index import CodeIndex
from coderag.retry import api_retry, describe_error
from coderag.search import SearchResult, search_code

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a senior developer helping someone understand a codebase.
Answer using only the code snippets provided as context.
Each snippet line starts with its line number. Cite the exact lines you rely on as `path:start-end`.
When you quote code, leave out the line numbers.
Only mention files, functions and values that appear in the context. Never guess file names or extensions.
If the context does not contain the answer, say so plainly and suggest where to look instead of guessing.
Keep answers short and concrete, and show small code excerpts when they help."""

# how many earlier messages from the chat are sent along for follow-up questions
HISTORY_TURNS = 6
# the file list helps with "where is X?" questions, but is left out for big repos
MAX_LISTED_FILES = 300


def format_context(results: List[SearchResult]) -> str:
    """Turn search results into a context block the model can cite."""
    blocks = []
    for r in results:
        lang = r.chunk.path.rsplit(".", 1)[-1] if "." in r.chunk.path else ""
        # number every line so the model can cite exact lines, not just the chunk
        lines = r.chunk.text.split("\n")
        numbered = "\n".join(f"{r.chunk.start_line + i:>4} | {line}" for i, line in enumerate(lines))
        blocks.append(f"### {r.location}\n```{lang}\n{numbered}\n```")
    return "\n\n".join(blocks)


def build_messages(
    question: str,
    results: List[SearchResult],
    history: Optional[List[Dict[str, str]]] = None,
    files: Optional[List[str]] = None,
) -> List[Dict[str, str]]:
    system = SYSTEM_PROMPT
    if files and len(files) <= MAX_LISTED_FILES:
        system += "\n\nFiles in the project:\n" + "\n".join(sorted(files))
    messages = [{"role": "system", "content": system}]
    messages.extend((history or [])[-HISTORY_TURNS:])

    context = format_context(results) if results else "(no matching code found)"
    messages.append({"role": "user", "content": f"Context:\n\n{context}\n\nQuestion: {question}"})
    return messages


# "  12 | code" -> "code", the prefix format_context adds to every line
_LINE_NUMBER = re.compile(r"^\s*\d+ \| ?", re.MULTILINE)


def strip_line_numbers(text: str) -> str:
    return _LINE_NUMBER.sub("", text)


def _by_line(pieces: Iterator[str]) -> Iterator[str]:
    """Re-chunk streamed text into whole lines so line-number prefixes can be removed reliably."""
    buffer = ""
    for piece in pieces:
        buffer += piece
        *lines, buffer = buffer.split("\n")
        for line in lines:
            yield strip_line_numbers(line) + "\n"
    if buffer:
        yield strip_line_numbers(buffer)


# a chat should not hang for minutes, so give up sooner than indexing does
@api_retry(attempts=3)
def _open_stream(client, model: str, messages: List[Dict[str, str]]):
    return client.chat.completions.create(model=model, messages=messages, stream=True, temperature=0.2)


def stream_answer(messages: List[Dict[str, str]], model: str = config.CHAT_MODEL) -> Iterator[str]:
    """Yield the answer piece by piece as the model writes it."""
    client, _provider = get_ai_client()
    if client is None:
        yield "AI backend is not available. Check AI_BACKEND, BASE_URL and API_KEY in your .env."
        return

    try:
        stream = _open_stream(client, model, messages)
        pieces = (e.choices[0].delta.content for e in stream if e.choices and e.choices[0].delta.content)
        # models sometimes copy the context line numbers into quoted code
        yield from _by_line(pieces)
    except Exception as e:
        logger.debug("Chat request failed: %s", e)
        yield f"\n\nError: {describe_error(e)}"


def search_query(question: str, history: Optional[List[Dict[str, str]]] = None) -> str:
    """
    Follow-ups like "and what is the default?" say nothing about the topic on their own,
    so the previous question is searched together with the new one.
    """
    previous = [m["content"] for m in (history or []) if m["role"] == "user"]
    return f"{previous[-1]}\n{question}" if previous else question


def ask(
    index: CodeIndex,
    question: str,
    history: Optional[List[Dict[str, str]]] = None,
    k: int = config.TOP_K,
) -> tuple:
    """Retrieve context for a question. Returns (results, answer stream)."""
    results = search_code(index, search_query(question, history), k=k)
    return results, stream_answer(build_messages(question, results, history, files=list(index.file_hashes)))
