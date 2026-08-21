"""Helix — conversational RAG UI (Streamlit).

Run from the project root:

    uv run streamlit run ui/app.py

Runs every question through the full chat graph (guardrail + router → rewrite
→ retrieve → rerank → generate → verify) — the same path `/chat` in the API
takes. Earlier this called `retrieve()` and `generate_answer()` directly,
which meant the demo UI didn't exercise the guardrail, the router, reranking,
or citations at all, and could silently drift from what production actually
does. Document ingestion is enqueued to the Celery worker (non-blocking).
"""

from __future__ import annotations

import os
import sys

# Make the `app` package importable regardless of the working directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st  # noqa: E402  (import app after sys.path is set up)

from app.graph import answer_question  # noqa: E402
from app.llm import provider as llm_provider  # noqa: E402

st.set_page_config(page_title="Helix", page_icon="🧬", layout="wide")

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")


@st.cache_resource
def _celery():
    """Lightweight Celery client to enqueue ingestion (no worker code imported)."""
    from celery import Celery

    return Celery("helix-ui", broker=REDIS_URL, backend=REDIS_URL)


def _render_sources(sources: list[dict], reranked: bool) -> None:
    if not sources:
        return
    label = f"📚 {len(sources)} source(s)"
    if not reranked:
        label += " (vector order — reranker unavailable)"
    with st.expander(label):
        for i, s in enumerate(sources, 1):
            trail = " › ".join(s.get("headings") or []) or "—"
            pages = s.get("pages") or []
            page_str = f" · p.{','.join(map(str, pages))}" if pages else ""
            score = s.get("rerank_score") if reranked else s.get("score")
            score_label = "rerank" if reranked else "vector"
            score_str = f" · {score_label} {score:.3f}" if isinstance(score, (int, float)) else ""
            st.markdown(f"**[{i}] {s.get('source', 'unknown')}**{page_str}{score_str}")
            st.caption(trail)
            st.markdown(f"> {(s.get('text') or '')[:400]}…")


# --- Sidebar: settings + ingestion ---------------------------------------

with st.sidebar:
    st.title("🧬 Helix")
    st.caption("Conversational product RAG")
    st.caption(f"LLM: {llm_provider().name}")

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
            _render_sources(msg.get("sources", []), msg.get("reranked", False))
            if msg.get("grounded") is False:
                st.caption("⚠️ This answer could not be fully verified against its sources.")

if prompt := st.chat_input("Ask about your product…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Prior turns as chat history for the graph (exclude the current prompt,
    # which answer_question takes separately).
    history = [
        {"role": m["role"], "content": m["content"]} for m in st.session_state.messages[:-1]
    ]

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                result = answer_question(prompt, history=history)
            except Exception as exc:  # noqa: BLE001
                result = {
                    "answer": (
                        f"⚠️ Something went wrong. Is `{llm_provider().key_env}` set in "
                        f"`.env`?\n\n`{exc}`"
                    ),
                    "sources": [],
                    "route": "error",
                    "reranked": False,
                    "grounded": None,
                }

        answer = result.get("answer", "")
        sources = result.get("sources", [])
        reranked = bool(result.get("reranked"))

        st.markdown(answer)
        if result.get("route") == "technical" and not sources:
            st.caption("No indexed context matched this question closely enough to answer.")
        _render_sources(sources, reranked)
        if result.get("grounded") is False:
            st.caption("⚠️ This answer could not be fully verified against its sources.")

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "reranked": reranked,
            "grounded": result.get("grounded"),
        }
    )
