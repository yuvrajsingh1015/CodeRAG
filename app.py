"""Streamlit chat UI. Run with: streamlit run app.py"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from coderag import config
from coderag.ai_client import get_ai_client
from coderag.embeddings import embed_texts
from coderag.index import CodeIndex
from prompt_flow import ask

st.set_page_config(page_title="CodeRAG", page_icon="🔎", layout="wide")


def load_index(root: str) -> CodeIndex:
    """Keep one index per folder in the session so reruns don't reload it from disk."""
    key = f"index::{root}"
    if key not in st.session_state:
        st.session_state[key] = CodeIndex.load(Path(root))
    return st.session_state[key]


def reindex(index: CodeIndex) -> None:
    bar = st.sidebar.progress(0.0, text="Scanning files...")

    def progress(done: int, total: int) -> None:
        bar.progress(done / total, text=f"Scanning {done}/{total} files")

    try:
        with st.spinner("Embedding changed files..."):
            stats = index.build(embed_texts, progress=progress)
        index.save()
        st.sidebar.success(
            f"{stats.chunks} chunks from {stats.files} files ({stats.embedded} embedded, {stats.reused} unchanged)"
        )
    except Exception as e:
        st.sidebar.error(f"Indexing failed: {e}")
    finally:
        bar.empty()


# ---------- sidebar ----------

st.sidebar.title("CodeRAG")
root = st.sidebar.text_input("Folder to index", value=str(config.WATCHED_DIR))

if not Path(root).expanduser().is_dir():
    st.sidebar.error("That folder does not exist.")
    st.stop()

root = str(Path(root).expanduser().resolve())
index = load_index(root)

if st.sidebar.button("Index / update", type="primary", use_container_width=True):
    reindex(index)

_client, provider = get_ai_client()
st.sidebar.caption(f"Backend: {provider}")
st.sidebar.caption(f"Chat model: {config.CHAT_MODEL} · Embeddings: {config.EMBEDDING_MODEL}")
st.sidebar.caption(f"{len(index)} chunks in index")

k = st.sidebar.slider("Snippets per question", 1, 15, config.TOP_K)

if st.sidebar.button("Clear chat", use_container_width=True):
    st.session_state.messages = []

# ---------- chat ----------

st.title("Ask your codebase")
st.caption(root)

if "messages" not in st.session_state:
    st.session_state.messages = []


def show_sources(sources: list) -> None:
    with st.expander(f"Sources ({len(sources)})"):
        for s in sources:
            st.markdown(f"**`{s['location']}`** · score {s['score']:.2f}")
            st.code(s["text"], language=s["lang"])


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            show_sources(msg["sources"])

if len(index) == 0:
    st.info("Click **Index / update** in the sidebar to index this folder first.")

question = st.chat_input("e.g. Where is the config loaded?", disabled=len(index) == 0)

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # only role/content go to the model, not the sources
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[:-1]]

    with st.chat_message("assistant"):
        with st.spinner("Searching..."):
            results, stream = ask(index, question, history, k=k)
        reply = st.write_stream(stream)
        sources = [
            {
                "location": r.location,
                "score": r.score,
                "text": r.chunk.text,
                "lang": r.chunk.path.rsplit(".", 1)[-1],
            }
            for r in results
        ]
        if sources:
            show_sources(sources)

    st.session_state.messages.append({"role": "assistant", "content": reply, "sources": sources})
