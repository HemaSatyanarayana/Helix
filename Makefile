# Makefile for Helix — production-grade agentic RAG

# Source of documents to ingest (override: `make ingest SRC=path/to/docs`)
SRC ?= data

.PHONY: help install services services-down worker ui \
        ingest ingest-append reset-index clean

help:  ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

# --- Environment ----------------------------------------------------------

install:  ## Create the uv environment and install dependencies
	uv sync

# --- Infrastructure -------------------------------------------------------

services:  ## Start Qdrant + Redis + Jaeger (docker compose)
	docker compose up -d

services-down:  ## Stop the infrastructure containers
	docker compose down

# --- Ingestion ------------------------------------------------------------

reset-index:  ## Delete ALL vectors (drop the Qdrant collection)
	uv run python scripts/ingest.py --reset

ingest:  ## Wipe ALL data, then ingest SRC (default: data/)
	uv run python scripts/ingest.py --reset $(SRC)

ingest-append:  ## Incrementally ingest SRC WITHOUT wiping (per-doc re-ingest)
	uv run python scripts/ingest.py $(SRC)

# --- Runtime --------------------------------------------------------------

worker:  ## Run the Celery ingestion worker (async / UI ingestion)
	PYTHONPATH=ingestion-workers:. uv run celery -A worker worker --loglevel=info

ui:  ## Launch the Streamlit UI
	uv run streamlit run ui/app.py

# --- Housekeeping ---------------------------------------------------------

clean:  ## Remove the venv and Python caches
	rm -rf .venv
	find . -type d -name __pycache__ -exec rm -rf {} +
