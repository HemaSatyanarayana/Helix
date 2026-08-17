# RAG Production Grade

A production-grade Retrieval-Augmented Generation (RAG) service.

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) and Docker Compose
- [uv](https://docs.astral.sh/uv/) for Python environment management

## Infrastructure (Qdrant + Redis)

The project uses **Qdrant** as the vector store and **Redis** as the broker/backend
for ingestion workers. Both run via Docker Compose.

### Start the services

```bash
docker compose up -d
```

This starts:

| Service | Image                  | Ports                | Purpose             |
|---------|------------------------|----------------------|---------------------|
| Qdrant  | `qdrant/qdrant:latest` | `6333` (REST), `6334` (gRPC) | Vector database     |
| Redis   | `redis:7-alpine`       | `6379`               | Task broker/backend |

Data persists in the `qdrant_data` and `redis_data` Docker volumes.

### Verify the services

```bash
# Check container status (should show "healthy")
docker compose ps

# Qdrant REST endpoint
curl http://localhost:6333/healthz

# Redis
docker compose exec redis redis-cli ping   # -> PONG
```

- Qdrant dashboard: http://localhost:6333/dashboard

### Stop the services

```bash
docker compose down          # stop and remove containers
docker compose down -v       # also delete the data volumes
```

## Python environment (uv)

```bash
uv sync        # create .venv and install dependencies from pyproject.toml
```

## Configuration

Copy the example env file and fill in your values:

```bash
cp .env.example .env
```

The application reads these variables (see `app/config.py`):

- `QDRANT_URL` (default `http://localhost:6333`)
- `REDIS_URL` (default `redis://localhost:6379/0`)
- `ANTHROPIC_API_KEY`
