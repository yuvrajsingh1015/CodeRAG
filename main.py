"""Command line interface: index a folder, then ask questions about it."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from coderag import config
from coderag.embeddings import embed_texts
from coderag.index import CodeIndex
from prompt_flow import ask


def build_index(root: Path) -> CodeIndex:
    index = CodeIndex.load(root)
    stats = index.build(embed_texts)
    index.save()
    print(
        f"Indexed {stats.files} files into {stats.chunks} chunks "
        f"({stats.embedded} embedded, {stats.reused} unchanged, {stats.removed} removed)"
    )
    return index


def answer(index: CodeIndex, question: str, history: list, show_sources: bool) -> str:
    results, stream = ask(index, question, history)
    if show_sources and results:
        print("Sources: " + ", ".join(r.location for r in results) + "\n")

    parts = []
    for piece in stream:
        print(piece, end="", flush=True)
        parts.append(piece)
    print()
    return "".join(parts)


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(prog="coderag", description="Ask questions about a local codebase.")
    parser.add_argument("-v", "--verbose", action="store_true", help="show debug logs")
    sub = parser.add_subparsers(dest="command", required=True)

    p_index = sub.add_parser("index", help="build or update the index for a folder")
    p_index.add_argument("path", nargs="?", default=str(config.WATCHED_DIR))

    p_ask = sub.add_parser("ask", help="ask one question")
    p_ask.add_argument("question")
    p_ask.add_argument("--path", default=str(config.WATCHED_DIR))
    p_ask.add_argument("--sources", action="store_true", help="print the matched files first")

    p_chat = sub.add_parser("chat", help="interactive chat in the terminal")
    p_chat.add_argument("--path", default=str(config.WATCHED_DIR))

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING)

    if args.command == "index":
        build_index(Path(args.path))
        return 0

    # ask and chat update the index first, which is fast when nothing changed
    index = build_index(Path(args.path))

    if args.command == "ask":
        answer(index, args.question, [], args.sources)
        return 0

    history: list = []
    print("Type a question, or 'exit' to quit.")
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if question.lower() in {"exit", "quit"}:
            return 0
        if not question:
            continue
        reply = answer(index, question, history, show_sources=True)
        history += [{"role": "user", "content": question}, {"role": "assistant", "content": reply}]


if __name__ == "__main__":
    sys.exit(main())
