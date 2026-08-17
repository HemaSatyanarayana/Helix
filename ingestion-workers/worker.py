"""Celery ingestion worker with incremental (per-document) re-ingestion.

Pipeline for each document:

    parse (Docling)  ->  chunk (Docling HybridChunker)  ->  embed  ->  upsert (Qdrant)

Docling handles the first two stages: it converts PDFs / DOCX / PPTX / HTML /
images into a structured ``DoclingDocument`` (layout, tables, reading order,
provenance) and the ``HybridChunker`` splits it along that structure while
staying within the embedding model's token budget.

Re-ingestion strategy (Qdrant is the source of truth — no external manifest):

* **Content-addressed chunk IDs** — ``id = uuid5(doc_id + sha256(normalized_text))``.
  Identical text ⇒ identical ID across re-ingests, so unchanged chunks are stable
  and writes are idempotent. (Replaces the old positional ``doc_id:i`` scheme,
  which re-embedded unchanged content and leaked stale chunks.)
* **Doc-hash gate (C)** — a cheap source hash short-circuits the whole pipeline
  when a local file is unchanged (no parse / chunk / embed).
* **Upsert-and-prune (B)** — diff this run's chunk IDs against what's stored for
  the ``doc_id``: embed only ``new − existing``, delete ``existing − new``, and
  leave the intersection untouched.
* **Corpus reconcile** — ``reconcile_corpus`` deletes documents that vanished
  from the source (orphan sweep); ``delete_document`` removes one on demand.

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

from app.telemetry import configure_logfire

# --- Configuration (env with sane local defaults) -------------------------

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "documents")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

celery_app = Celery("ingestion", broker=REDIS_URL, backend=REDIS_URL)

# --- Observability (Logfire: infra + pipeline tracing) --------------------
# Traces every Celery task and outbound HTTP call, plus the explicit pipeline
# spans below. No-ops safely when LOGFIRE_TOKEN is unset.
logfire = configure_logfire("helix-ingestion")
logfire.instrument_celery()
logfire.instrument_httpx()


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


# --- Identity / hashing helpers -------------------------------------------


def _doc_id(path: str) -> str:
    """Stable ID for a document, derived from its source path/URL."""
    return hashlib.sha256(path.encode()).hexdigest()[:16]


def _normalize(text: str) -> str:
    """Collapse whitespace so trivial reformatting isn't seen as a change."""
    return " ".join(text.split())


def _chunk_id(doc_id: str, text: str) -> str:
    """Content-addressed chunk ID: same text ⇒ same ID across re-ingests."""
    digest = hashlib.sha256(_normalize(text).encode()).hexdigest()
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc_id}:{digest}"))


def _source_hash(path: str) -> str | None:
    """Cheap pre-parse fingerprint for a local file (None for URLs)."""
    if os.path.isfile(path):
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(65536), b""):
                h.update(block)
        return h.hexdigest()
    return None


def _provenance(chunk) -> tuple[list[str], list[int]]:
    headings = list(getattr(chunk.meta, "headings", None) or [])
    pages = sorted(
        {
            prov.page_no
            for item in getattr(chunk.meta, "doc_items", []) or []
            for prov in getattr(item, "prov", []) or []
            if getattr(prov, "page_no", None) is not None
        }
    )
    return headings, pages


# --- Qdrant helpers -------------------------------------------------------


def _ensure_collection(dim: int) -> None:
    """Create the collection + doc_id payload index on first use (idempotent)."""
    from qdrant_client.models import Distance, PayloadSchemaType, VectorParams

    client = get_qdrant()
    if not client.collection_exists(QDRANT_COLLECTION):
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        # Keyword index makes per-doc filter/scroll/delete fast.
        client.create_payload_index(
            collection_name=QDRANT_COLLECTION,
            field_name="doc_id",
            field_schema=PayloadSchemaType.KEYWORD,
        )


def _doc_filter(doc_id: str):
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    return Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))])


def _scroll_existing(doc_id: str) -> dict[str, dict]:
    """Return {point_id: payload} for every chunk currently stored for a doc."""
    client = get_qdrant()
    if not client.collection_exists(QDRANT_COLLECTION):
        return {}

    result: dict[str, dict] = {}
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=QDRANT_COLLECTION,
            scroll_filter=_doc_filter(doc_id),
            with_payload=True,
            with_vectors=False,
            limit=256,
            offset=offset,
        )
        for p in points:
            result[str(p.id)] = p.payload or {}
        if offset is None:
            break
    return result


def _delete_doc(doc_id: str) -> None:
    client = get_qdrant()
    if client.collection_exists(QDRANT_COLLECTION):
        client.delete(collection_name=QDRANT_COLLECTION, points_selector=_doc_filter(doc_id))


def _all_doc_ids() -> set[str]:
    """Distinct doc_ids currently stored (for corpus reconciliation)."""
    client = get_qdrant()
    if not client.collection_exists(QDRANT_COLLECTION):
        return set()

    ids: set[str] = set()
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=QDRANT_COLLECTION,
            with_payload=["doc_id"],
            with_vectors=False,
            limit=512,
            offset=offset,
        )
        for p in points:
            did = (p.payload or {}).get("doc_id")
            if did:
                ids.add(did)
        if offset is None:
            break
    return ids


# --- Tasks ----------------------------------------------------------------


@celery_app.task(name="ingest_document")
def ingest_document(path: str) -> dict:
    """Incrementally (re)ingest a single document.

    Embeds only new/changed chunks, deletes removed ones, leaves unchanged
    chunks untouched, and skips the whole pipeline if the source is unchanged.
    """
    with logfire.span("ingest_document", path=path) as span:
        doc_id = _doc_id(path)
        existing = _scroll_existing(doc_id)
        stored_hash = next(iter(existing.values()), {}).get("doc_hash") if existing else None

        # --- Doc-hash gate (C): skip entirely if a local file is unchanged.
        src_hash = _source_hash(path)
        if src_hash and existing and stored_hash == src_hash:
            span.set_attribute("skipped", True)
            return {"status": "unchanged", "path": path, "chunks": len(existing)}

        # 1. Parse — Docling turns the file into a structured DoclingDocument.
        with logfire.span("parse", path=path):
            result = get_converter().convert(path)
            doc = result.document

        # For URLs (no cheap pre-parse hash) derive the doc hash from content.
        doc_hash = src_hash or hashlib.sha256(doc.export_to_markdown().encode()).hexdigest()

        # 2. Chunk — split along document structure, tokenizer-aware.
        with logfire.span("chunk") as chunk_span:
            chunker = get_chunker()
            chunks = list(chunker.chunk(doc))
            chunk_span.set_attribute("n_chunks", len(chunks))

        # Build content-addressed map: id -> (chunk, contextualized_text, index).
        # `contextualize` prepends the heading trail (better retrieval); that
        # enriched text is what we hash, embed, and store.
        new: dict[str, tuple] = {}
        for i, c in enumerate(chunks):
            text = chunker.contextualize(chunk=c)
            new[_chunk_id(doc_id, text)] = (c, text, i)

        existing_ids = set(existing)
        new_ids = set(new)
        to_add = new_ids - existing_ids
        to_delete = existing_ids - new_ids
        unchanged = new_ids & existing_ids
        span.set_attribute("added", len(to_add))
        span.set_attribute("deleted", len(to_delete))
        span.set_attribute("unchanged", len(unchanged))

        # 3. Embed ONLY the new/changed chunks.
        if to_add:
            from qdrant_client.models import PointStruct

            add_ids = list(to_add)
            add_texts = [new[cid][1] for cid in add_ids]
            with logfire.span("embed", n_texts=len(add_texts), model=EMBEDDING_MODEL):
                vectors = get_embedder().encode(add_texts, normalize_embeddings=True)
            _ensure_collection(dim=len(vectors[0]))

            points = []
            for cid, vector in zip(add_ids, vectors):
                chunk, text, index = new[cid]
                headings, pages = _provenance(chunk)
                points.append(
                    PointStruct(
                        id=cid,
                        vector=vector.tolist(),
                        payload={
                            "text": text,
                            "source": path,
                            "doc_id": doc_id,
                            "doc_hash": doc_hash,
                            "chunk_index": index,
                            "headings": headings,
                            "pages": pages,
                        },
                    )
                )
            with logfire.span("upsert", n_points=len(points)):
                get_qdrant().upsert(collection_name=QDRANT_COLLECTION, points=points)

        # 4. Prune chunks that no longer exist in the document.
        if to_delete:
            with logfire.span("prune", n_points=len(to_delete)):
                get_qdrant().delete(
                    collection_name=QDRANT_COLLECTION,
                    points_selector=list(to_delete),
                )

        # 5. Refresh doc_hash on untouched chunks so the gate stays consistent
        #    (updates metadata only — vectors are left alone).
        if unchanged and stored_hash != doc_hash:
            get_qdrant().set_payload(
                collection_name=QDRANT_COLLECTION,
                payload={"doc_hash": doc_hash},
                points=list(unchanged),
            )

        logfire.info(
            "ingested", path=path, added=len(to_add), deleted=len(to_delete), unchanged=len(unchanged)
        )
        return {
            "status": "ok",
            "path": path,
            "added": len(to_add),
            "deleted": len(to_delete),
            "unchanged": len(unchanged),
            "total": len(new_ids),
        }


@celery_app.task(name="delete_document")
def delete_document(path: str) -> dict:
    """Remove all vectors for a single document."""
    with logfire.span("delete_document", path=path):
        doc_id = _doc_id(path)
        _delete_doc(doc_id)
        return {"status": "deleted", "path": path, "doc_id": doc_id}


@celery_app.task(name="reconcile_corpus")
def reconcile_corpus(source_paths: list[str]) -> dict:
    """Orphan sweep: delete documents no longer present in the source set."""
    with logfire.span("reconcile_corpus", n_sources=len(source_paths)) as span:
        wanted = {_doc_id(p) for p in source_paths}
        stored = _all_doc_ids()
        orphaned = stored - wanted
        for doc_id in orphaned:
            _delete_doc(doc_id)
        span.set_attribute("orphaned", len(orphaned))
        return {"status": "ok", "stored": len(stored), "orphaned": len(orphaned)}
