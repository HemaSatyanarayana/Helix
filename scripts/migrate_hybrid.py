"""Migrate an existing dense-only collection to the hybrid schema.

    python scripts/migrate_hybrid.py [--alias documents] [--keep-old]

The hybrid schema stores two named vectors per chunk (``dense`` + ``bm25``),
which a collection created with a single unnamed vector cannot be altered into.
This copies the data across instead — and does it **without re-parsing**:

* dense vectors are read back out of the old collection, not recomputed, so
  Docling never runs and the sentence-transformer never sees a GPU;
* sparse vectors are computed from the stored chunk text, which is cheap
  because BM25 is a tokenizer, not a model;
* payloads carry over verbatim, so chunk IDs, provenance and the doc-hash gate
  all stay valid.

The alias is repointed only after the new collection is fully populated and
verified, so readers never see a partial index.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

BATCH = 256


def _load_worker():
    spec = importlib.util.spec_from_file_location(
        "worker", ROOT / "ingestion-workers" / "worker.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _dense_of(point, dense_name: str) -> list[float] | None:
    """Read the dense vector from either schema (unnamed legacy, or named)."""
    vector = point.vector
    if vector is None:
        return None
    if isinstance(vector, dict):
        return vector.get(dense_name)
    return vector  # legacy: a bare list


def main() -> int:
    ap = argparse.ArgumentParser(description="Migrate a collection to the hybrid schema.")
    ap.add_argument("--alias", default=None, help="alias to migrate (default: QDRANT_COLLECTION)")
    ap.add_argument("--keep-old", action="store_true", help="don't drop the old collection")
    args = ap.parse_args()

    worker = _load_worker()
    from app.embedding import DENSE_VECTOR, SPARSE_VECTOR, embed_sparse

    from qdrant_client.models import PointStruct, SparseVector

    alias = args.alias or worker.QDRANT_COLLECTION
    client = worker.get_qdrant()

    source = worker.resolve_collection(alias)
    if not source:
        print(f"nothing to migrate: no collection or alias named {alias!r}")
        return 1

    total = client.count(collection_name=source, exact=True).count
    if total == 0:
        print(f"{source!r} is empty — run `make ingest` instead")
        return 1

    target = worker.new_collection_name(alias)
    print(f"migrating {total} chunks: {source!r} -> {target!r}")
    worker.create_collection(target)

    migrated = 0
    skipped = 0
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=source,
            with_payload=True,
            with_vectors=True,
            limit=BATCH,
            offset=offset,
        )
        if points:
            texts, keep = [], []
            for p in points:
                dense = _dense_of(p, DENSE_VECTOR)
                payload = p.payload or {}
                text = payload.get("text")
                if dense is None or not text:
                    skipped += 1
                    continue
                keep.append((p, dense, payload))
                texts.append(text)

            if keep:
                sparse_vectors = embed_sparse(texts)
                client.upsert(
                    collection_name=target,
                    points=[
                        PointStruct(
                            id=p.id,
                            vector={
                                DENSE_VECTOR: dense,
                                SPARSE_VECTOR: SparseVector(
                                    indices=sparse["indices"], values=sparse["values"]
                                ),
                            },
                            payload=payload,
                        )
                        for (p, dense, payload), sparse in zip(keep, sparse_vectors)
                    ],
                )
                migrated += len(keep)
                print(f"  {migrated}/{total} chunks", end="\r", flush=True)

        if offset is None:
            break

    copied = client.count(collection_name=target, exact=True).count
    print(f"\ncopied {copied} chunks (skipped {skipped} without text or vector)")

    # Verify before swapping: a partial copy behind the alias is worse than no
    # migration at all, because it looks like a working index.
    if copied != total - skipped:
        print(f"ABORT: expected {total - skipped} chunks in {target!r}, found {copied}")
        print(f"the alias still points at {source!r}; inspect and delete {target!r} manually")
        return 1

    if source == alias:
        # Legacy layout: the old collection occupies the name the alias needs,
        # and Qdrant refuses an alias that collides with a collection name. The
        # old collection must therefore go first — a sub-second window during
        # which queries see no collection, which retrieval already handles by
        # returning no hits rather than erroring.
        if args.keep_old:
            print(
                f"ABORT: --keep-old cannot apply here — {source!r} is a plain collection\n"
                f"       occupying the alias name, so it must be dropped for the alias\n"
                f"       to take it. Re-run without --keep-old, or migrate to a different\n"
                f"       alias with --alias."
            )
            return 1
        print(f"dropping legacy collection {source!r} to free the alias name…")
        client.delete_collection(source)
        worker.point_alias_at(target, alias)
    else:
        worker.point_alias_at(target, alias)
        if not args.keep_old:
            client.delete_collection(source)
            print(f"dropped old collection {source!r}")

    print(f"alias {alias!r} now points at {target!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
