"""Celery ingestion worker with incremental (per-document) re-ingestion.

Pipeline for each document:

    parse (Docling)  ->  chunk (Docling HybridChunker)  ->  embed  ->  upsert (Qdrant)

Docling handles the first two stages: it converts PDFs / DOCX / PPTX / HTML /
images into a structured ``DoclingDocument`` (layout, tables, reading order,
provenance) and the ``HybridChunker`` splits it along that structure while
staying within the embedding model's token budget.

Each chunk is stored with **two** vectors — a dense semantic one and a sparse
BM25 one — so retrieval can fuse semantic and lexical rankings (see
:mod:`app.embedding`).

Re-ingestion strategy (Qdrant is the source of truth — no external manifest):

* **Content-addressed chunk IDs** — ``id = uuid5(doc_id + sha256(normalized_text))``.
  Identical text ⇒ identical ID across re-ingests, so unchanged chunks are stable
  and writes are idempotent.
* **Doc-hash gate** — a cheap source hash short-circuits the whole pipeline
  when a local file is unchanged (no parse / chunk / embed).
* **Upsert-and-prune** — diff this run's chunk IDs against what's stored for
  the ``doc_id``: embed only ``new − existing``, delete ``existing − new``, and
  leave the intersection untouched.
* **Corpus reconcile** — ``reconcile_corpus`` deletes documents that vanished
  from the source (orphan sweep); ``delete_document`` removes one on demand.

Collections are addressed through an **alias**. ``QDRANT_COLLECTION`` names the
alias; the data lives in a timestamped physical collection behind it. A full
rebuild therefore builds into a *new* collection and repoints the alias in one
atomic step, so readers never observe a half-built or empty index — the old
``--reset`` flow dropped the collection first and left the API answering
"I couldn't find anything" for the length of the rebuild.

Heavy objects (Docling models, the encoders, the Qdrant client) are created
lazily and cached, so importing this module — which Celery does on every
worker — stays cheap and the models load only when the first task runs.
"""

from __future__ import annotations

import hashlib
import os
import time
import uuid
from functools import lru_cache

from celery import Celery

from app.embedding import (
    DENSE_VECTOR,
    SPARSE_VECTOR,
    dense_dimension,
    embed_dense,
    embed_sparse,
)
from app.telemetry import configure_logfire

# --- Configuration (env with sane local defaults) -------------------------

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or None
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "documents")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

celery_app = Celery("ingestion", broker=REDIS_URL, backend=REDIS_URL)

# --- Observability (Logfire: infra + pipeline tracing) --------------------
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
def get_qdrant():
    from qdrant_client import QdrantClient

    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


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


# --- Collection / alias management ----------------------------------------


def new_collection_name(alias: str = QDRANT_COLLECTION) -> str:
    """A fresh timestamped physical collection to build into."""
    return f"{alias}_{time.strftime('%Y%m%d%H%M%S')}"


def resolve_collection(alias: str = QDRANT_COLLECTION) -> str | None:
    """The physical collection currently serving ``alias``.

    Returns the name itself when it is a plain collection (the pre-alias
    layout), or None when nothing exists yet.
    """
    client = get_qdrant()
    try:
        for descriptor in client.get_aliases().aliases:
            if descriptor.alias_name == alias:
                return descriptor.collection_name
    except Exception as exc:  # noqa: BLE001 — fall through to the direct check.
        logfire.warning("alias_lookup_failed", alias=alias, error=str(exc))
    if client.collection_exists(alias):
        return alias
    return None


def active_collection(alias: str = QDRANT_COLLECTION) -> str:
    """Where writes go: the collection behind the alias, created if absent."""
    existing = resolve_collection(alias)
    if existing:
        return existing
    target = new_collection_name(alias)
    create_collection(target)
    point_alias_at(target, alias)
    return target


def create_collection(collection: str) -> None:
    """Create a collection with the hybrid schema and payload indexes."""
    from qdrant_client.models import (
        Distance,
        Modifier,
        PayloadSchemaType,
        SparseVectorParams,
        VectorParams,
    )

    client = get_qdrant()
    if client.collection_exists(collection):
        return

    client.create_collection(
        collection_name=collection,
        vectors_config={
            DENSE_VECTOR: VectorParams(size=dense_dimension(), distance=Distance.COSINE)
        },
        # IDF is computed by Qdrant across the collection: BM25 term weights
        # depend on corpus statistics the encoder cannot know on its own.
        sparse_vectors_config={SPARSE_VECTOR: SparseVectorParams(modifier=Modifier.IDF)},
    )
    # Keyword indexes make per-doc filter/scroll/delete and source filtering fast.
    for field in ("doc_id", "source"):
        client.create_payload_index(
            collection_name=collection,
            field_name=field,
            field_schema=PayloadSchemaType.KEYWORD,
        )
    logfire.info("collection_created", collection=collection)


def point_alias_at(collection: str, alias: str = QDRANT_COLLECTION) -> None:
    """Atomically repoint ``alias`` at ``collection``."""
    from qdrant_client.models import CreateAlias, CreateAliasOperation

    if collection == alias:  # legacy layout: the name *is* the collection
        return
    # Creating an alias that already exists repoints it, atomically.
    get_qdrant().update_collection_aliases(
        change_aliases_operations=[
            CreateAliasOperation(
                create_alias=CreateAlias(collection_name=collection, alias_name=alias)
            )
        ]
    )
    logfire.info("alias_swapped", alias=alias, collection=collection)


def _doc_filter(doc_id: str):
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    return Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))])


def _scroll_existing(doc_id: str, collection: str) -> dict[str, dict]:
    """Return {point_id: payload} for every chunk currently stored for a doc."""
    client = get_qdrant()
    if not client.collection_exists(collection):
        return {}

    result: dict[str, dict] = {}
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection,
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


def _delete_doc(doc_id: str, collection: str) -> None:
    client = get_qdrant()
    if client.collection_exists(collection):
        client.delete(collection_name=collection, points_selector=_doc_filter(doc_id))


def _all_doc_ids(collection: str) -> set[str]:
    """Distinct doc_ids currently stored (for corpus reconciliation)."""
    client = get_qdrant()
    if not client.collection_exists(collection):
        return set()

    ids: set[str] = set()
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection,
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
def ingest_document(path: str, collection: str | None = None) -> dict:
    """Incrementally (re)ingest a single document.

    Embeds only new/changed chunks, deletes removed ones, leaves unchanged
    chunks untouched, and skips the whole pipeline if the source is unchanged.
    ``collection`` targets a specific physical collection (used by rebuilds);
    by default it follows the alias.
    """
    with logfire.span("ingest_document", path=path) as span:
        target = collection or active_collection()
        span.set_attribute("collection", target)
        doc_id = _doc_id(path)
        existing = _scroll_existing(doc_id, target)
        stored_hash = next(iter(existing.values()), {}).get("doc_hash") if existing else None

        # --- Doc-hash gate: skip entirely if a local file is unchanged.
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

        # 3. Embed ONLY the new/changed chunks — dense and sparse together.
        if to_add:
            add_ids = list(to_add)
            add_texts = [new[cid][1] for cid in add_ids]
            with logfire.span("embed", n_texts=len(add_texts), model=EMBEDDING_MODEL):
                dense_vectors = embed_dense(add_texts)
                sparse_vectors = embed_sparse(add_texts)
            create_collection(target)

            points = _build_points(add_ids, new, dense_vectors, sparse_vectors, path, doc_id, doc_hash)
            with logfire.span("upsert", n_points=len(points)):
                get_qdrant().upsert(collection_name=target, points=points)

        # 4. Prune chunks that no longer exist in the document.
        if to_delete:
            with logfire.span("prune", n_points=len(to_delete)):
                get_qdrant().delete(
                    collection_name=target, points_selector=list(to_delete)
                )

        # 5. Refresh doc_hash on untouched chunks so the gate stays consistent
        #    (updates metadata only — vectors are left alone).
        if unchanged and stored_hash != doc_hash:
            get_qdrant().set_payload(
                collection_name=target,
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


def _build_points(add_ids, new, dense_vectors, sparse_vectors, path, doc_id, doc_hash):
    """Assemble Qdrant points carrying both vectors plus provenance payload."""
    from qdrant_client.models import PointStruct, SparseVector

    points = []
    for cid, dense, sparse in zip(add_ids, dense_vectors, sparse_vectors):
        chunk, text, index = new[cid]
        headings, pages = _provenance(chunk)
        points.append(
            PointStruct(
                id=cid,
                vector={
                    DENSE_VECTOR: dense,
                    SPARSE_VECTOR: SparseVector(
                        indices=sparse["indices"], values=sparse["values"]
                    ),
                },
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
    return points


@celery_app.task(name="delete_document")
def delete_document(path: str) -> dict:
    """Remove all vectors for a single document."""
    with logfire.span("delete_document", path=path):
        doc_id = _doc_id(path)
        _delete_doc(doc_id, active_collection())
        return {"status": "deleted", "path": path, "doc_id": doc_id}


@celery_app.task(name="reconcile_corpus")
def reconcile_corpus(source_paths: list[str]) -> dict:
    """Orphan sweep: delete documents no longer present in the source set."""
    with logfire.span("reconcile_corpus", n_sources=len(source_paths)) as span:
        target = active_collection()
        wanted = {_doc_id(p) for p in source_paths}
        stored = _all_doc_ids(target)
        orphaned = stored - wanted
        for doc_id in orphaned:
            _delete_doc(doc_id, target)
        span.set_attribute("orphaned", len(orphaned))
        return {"status": "ok", "stored": len(stored), "orphaned": len(orphaned)}


@celery_app.task(name="rebuild_corpus")
def rebuild_corpus(source_paths: list[str], keep_old: bool = False) -> dict:
    """Build a fresh index into a new collection, then swap the alias onto it.

    The live index keeps serving throughout; the swap is atomic and the old
    collection is dropped only afterwards. Nothing observes an empty index.
    """
    with logfire.span("rebuild_corpus", n_sources=len(source_paths)) as span:
        previous = resolve_collection()
        target = new_collection_name()
        span.set_attribute("target", target)
        create_collection(target)

        ingested = 0
        for path in source_paths:
            ingest_document(path, collection=target)
            ingested += 1

        point_alias_at(target)

        dropped = None
        if previous and previous != target and not keep_old:
            # Legacy plain collections share the alias name and cannot be
            # dropped without taking the alias with them.
            if previous != QDRANT_COLLECTION:
                get_qdrant().delete_collection(previous)
                dropped = previous
            else:
                logfire.warning(
                    "legacy_collection_kept",
                    collection=previous,
                    detail="drop it manually once the alias is verified",
                )

        logfire.info("rebuilt", collection=target, documents=ingested, dropped=dropped)
        return {
            "status": "ok",
            "collection": target,
            "documents": ingested,
            "previous": previous,
            "dropped": dropped,
        }
