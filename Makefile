# Makefile for Helix — production-grade agentic RAG

# Source of documents to ingest (override: `make ingest SRC=path/to/docs`)
SRC ?= data

.PHONY: help install services services-down worker api ui \
        ingest ingest-append reset-index migrate-hybrid clean \
        test test-eval eval eval-sweep \
        docker-build docker-up docker-down docker-logs

help:  ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

# --- Environment ----------------------------------------------------------

install:  ## Create the uv environment and install dependencies
	uv sync

# --- Infrastructure (host-run app, containerized dependencies) ------------

services:  ## Start Qdrant + Redis + Jaeger (docker compose)
	docker compose up -d qdrant redis jaeger

services-down:  ## Stop the infrastructure containers
	docker compose down

# --- Ingestion ------------------------------------------------------------

reset-index:  ## Delete ALL vectors (drop the Qdrant collection)
	uv run python scripts/ingest.py --reset

ingest:  ## Wipe ALL data, then ingest SRC (default: data/)
	uv run python scripts/ingest.py --reset $(SRC)

ingest-append:  ## Incrementally ingest SRC WITHOUT wiping (per-doc re-ingest)
	uv run python scripts/ingest.py $(SRC)

migrate-hybrid:  ## Migrate an existing dense-only collection to dense+sparse
	uv run python scripts/migrate_hybrid.py

# --- Runtime (host) ---------------------------------------------------------

worker:  ## Run the Celery ingestion worker (async / UI ingestion)
	PYTHONPATH=ingestion-workers:. uv run celery -A worker worker --loglevel=info

api:  ## Run the FastAPI app with autoreload
	uv run uvicorn app.main:app --reload

ui:  ## Launch the Streamlit UI
	uv run streamlit run ui/app.py

# --- Quality ----------------------------------------------------------------

test:  ## Run the unit test suite
	uv run pytest

test-eval:  ## Run tests plus the retrieval eval regression gate
	uv run pytest -m "not slow"
	uv run pytest tests/test_retrieval_eval.py -m slow

eval:  ## Run the retrieval eval against the golden set and print a report
	uv run python evals/run_eval.py

eval-sweep:  ## Sweep the rerank threshold and report false-answer/false-abstain rates
	uv run python evals/sweep_threshold.py

# --- Docker (full stack, including the API and worker) ---------------------

docker-build:  ## Build the API/worker image
	docker compose build

docker-up:  ## Start the full stack in containers
	docker compose up -d

docker-down:  ## Stop the full stack
	docker compose down

docker-logs:  ## Tail API + worker logs
	docker compose logs -f api worker

# --- Housekeeping ---------------------------------------------------------

clean:  ## Remove the venv and Python caches
	rm -rf .venv
	find . -type d -name __pycache__ -exec rm -rf {} +
