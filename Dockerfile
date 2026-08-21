# Helix API + ingestion worker — one image, two entrypoints (see docker-compose.yml).
#
# Multi-stage: `builder` resolves dependencies with uv into a venv that gets
# copied verbatim into `runtime`, so the final image carries no compiler
# toolchain, no uv, and no source download cache.
FROM python:3.12-slim AS builder

# Build tools for packages with native extensions (onnxruntime, torch deps).
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.9.9 /uv /uvx /bin/

WORKDIR /app

# Dependencies first, isolated from source, so editing app code doesn't bust
# the layer cache and force a full reinstall.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

COPY . .
RUN uv sync --frozen --no-dev


FROM python:3.12-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 helix

WORKDIR /app
COPY --from=builder --chown=helix:helix /app /app

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app:/app/ingestion-workers"

USER helix

# Model caches (sentence-transformers, fastembed, Docling) persist here across
# runs when the directory is a mounted volume — see docker-compose.yml.
ENV HF_HOME=/home/helix/.cache/huggingface

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# The worker service overrides this command (see docker-compose.yml); the
# image defaults to serving the API.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
