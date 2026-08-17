"""Celery ingestion worker.

Pipeline for each document:

    parse (Docling)  ->  chunk (Docling HybridChunker)  ->  embed  ->  upsert (Qdrant)

Docling handles the first two stages: it converts PDFs / DOCX / PPTX / HTML /
images into a structured ``DoclingDocument`` (layout, tables, reading order,
provenance) and the ``HybridChunker`` splits it along that structure while
staying within the embedding model's token budget.

Heavy objects (Docling models, the embedder, the Qdrant client) are created
lazily and cached, so importing this module — which Celery does on every
worker — stays cheap and the ML models load only when the first task runs.
"""

from __future__ import annotations

import hashlib
import os
import uuid
from functools import lru_cache

from celery import Celery

# --- Configuration (env with sane local defaults) -------------------------

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "documents")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

celery_app = Celery("ingestion", broker=REDIS_URL, backend=REDIS_URL)


# --- Lazy singletons ------------------------------------------------------


@lru_cache(maxsize=1)
def get_converter():
    """Docling document converter (loads the layout + table models once)."""
    from docling.document_converter import DocumentConverter

    return DocumentConverter()


@lru_cache(maxsize=1)
def get_chunker():
    """Structure-aware, tokenizer-aware chunker tied to the embedding model."""
    from docling.chunking import HybridChunker

    # Tokenizer-aware chunking packs each chunk to fit EMBEDDING_MODEL's limit.
    return HybridChunker(tokenizer=EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def get_embedder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def get_qdrant():
    from qdrant_client import QdrantClient

    return QdrantClient(url=QDRANT_URL)


def _ensure_collection(dim: int) -> None:
    """Create the target collection on first use (idempotent)."""
    from qdrant_client.models import Distance, VectorParams

    client = get_qdrant()
    if not client.collection_exists(QDRANT_COLLECTION):
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )


# --- Task -----------------------------------------------------------------


@celery_app.task(name="ingest_document")
def ingest_document(path: str) -> dict:
    """Parse, chunk, embed, and upsert a single document into Qdrant."""
    # 1. Parse — Docling turns the file into a structured DoclingDocument.
    result = get_converter().convert(path)
    doc = result.document

    # 2. Chunk — split along document structure, tokenizer-aware.
    chunker = get_chunker()
    chunks = list(chunker.chunk(doc))
    if not chunks:
        return {"status": "empty", "path": path, "chunks": 0}

    # `contextualize` prepends the heading trail to each chunk, which improves
    # retrieval. That enriched text is what we embed.
    texts = [chunker.contextualize(chunk=c) for c in chunks]

    # 3. Embed.
    vectors = get_embedder().encode(texts, normalize_embeddings=True)
    _ensure_collection(dim=len(vectors[0]))

    # 4. Upsert into Qdrant, carrying provenance (headings + pages) so answers
    #    can be cited back to their source location.
    from qdrant_client.models import PointStruct

    doc_id = hashlib.sha256(path.encode()).hexdigest()[:16]
    points = []
    for i, (chunk, text, vector) in enumerate(zip(chunks, texts, vectors)):
        headings = list(getattr(chunk.meta, "headings", None) or [])
        pages = sorted(
            {
                prov.page_no
                for item in getattr(chunk.meta, "doc_items", []) or []
                for prov in getattr(item, "prov", []) or []
                if getattr(prov, "page_no", None) is not None
            }
        )
        points.append(
            PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc_id}:{i}")),
                vector=vector.tolist(),
                payload={
                    "text": text,
                    "source": path,
                    "doc_id": doc_id,
                    "chunk_index": i,
                    "headings": headings,
                    "pages": pages,
                },
            )
        )

    get_qdrant().upsert(collection_name=QDRANT_COLLECTION, points=points)

    return {"status": "ok", "path": path, "chunks": len(points)}
