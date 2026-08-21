"""Helix — agentic RAG over product documentation.

Loads ``.env`` on package import so every entry point (FastAPI, Celery worker,
Streamlit UI, eval scripts) sees the same configuration. Modules read their env
at import time, so this has to happen before any of them are imported — putting
it here makes that ordering automatic rather than something each caller has to
remember. Real environment variables take precedence over the file.
"""

from dotenv import load_dotenv

load_dotenv()
