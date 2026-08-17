"""Helix — conversational RAG UI (Streamlit).

Run from the project root:

    uv run streamlit run ui/app.py

Flow per question: retrieve top-k chunks from Qdrant -> generate an answer with
conversation history via OpenRouter -> show the answer with cited sources.
Document ingestion is enqueued to the Celery worker (non-blocking).
"""

from __future__ import annotations

import os
import sys

# Make the `app` package importable regardless of the working directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

import streamlit as st

from app.generation import generate_answer
from app.retrieval import retrieve

st.set_page_config(page_title="Helix", page_icon="🧬", layout="wide")

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")


@st.cache_resource
def _celery():
    """Lightweight Celery client to enqueue ingestion (no worker code imported)."""
    from celery import Celery

    return Celery("helix-ui", broker=REDIS_URL, backend=REDIS_URL)


def _render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"📚 {len(sources)} source(s)"):
        for i, s in enumerate(sources, 1):
            trail = " › ".join(s.get("headings") or []) or "—"
            pages = s.get("pages") or []
            page_str = f" · p.{','.join(map(str, pages))}" if pages else ""
            score = s.get("score")
            score_str = f" · score {score:.3f}" if isinstance(score, (int, float)) else ""
            st.markdown(f"**[{i}] {s.get('source', 'unknown')}**{page_str}{score_str}")
            st.caption(trail)
            st.markdown(f"> {(s.get('text') or '')[:400]}…")


# --- Sidebar: settings + ingestion ---------------------------------------

with st.sidebar:
    st.title("🧬 Helix")
    st.caption("Conversational product RAG")

    top_k = st.slider("Contexts (top-k)", 1, 10, int(os.getenv("TOP_K", "4")))

    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.subheader("Ingest a document")
    st.caption("Enqueues to the Celery worker. A worker must be running.")
    doc_path = st.text_input("File path or URL", placeholder="/path/to/doc.pdf or https://…")
    if st.button("⬆️ Ingest", use_container_width=True, disabled=not doc_path):
        try:
            task = _celery().send_task("ingest_document", args=[doc_path])
            st.success(f"Queued (task {task.id[:8]}…)")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not enqueue: {exc}")

    st.divider()
    st.caption("Traces → Jaeger: http://localhost:16686")


# --- Chat -----------------------------------------------------------------

st.title("Ask Helix")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Replay history.
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            _render_sources(msg.get("sources", []))

if prompt := st.chat_input("Ask about your product…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving context…"):
            try:
                sources = retrieve(prompt, top_k=top_k)
            except Exception as exc:  # noqa: BLE001
                sources = []
                st.warning(f"Retrieval unavailable: {exc}")

        if not sources:
            st.info("No indexed context found. Ingest a document first (sidebar).")

        # Prior turns become chat history for the LLM (exclude the current prompt).
        history = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages[:-1]
        ]
        try:
            with st.spinner("Generating…"):
                answer = generate_answer(
                    prompt, [s["text"] for s in sources], history=history
                )
        except Exception as exc:  # noqa: BLE001
            answer = (
                "⚠️ Generation failed. Is `OPENROUTER_API_KEY` set in `.env`?\n\n"
                f"`{exc}`"
            )

        st.markdown(answer)
        _render_sources(sources)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
