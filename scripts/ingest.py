"""Batch ingestion runner for Helix (synchronous — no Celery worker needed).

Usage:
    python scripts/ingest.py [--reset] [<path> ...]

* ``--reset`` deletes the ENTIRE Qdrant collection before ingesting, so you
  start from a clean index.
* Paths may be files or directories (directories are walked recursively). With
  ``--reset`` and no paths, it just clears the index and exits.

Loads the worker by file path to bypass the un-importable ``ingestion-workers``
directory name (hyphen).
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # make the `app` package importable

SUPPORTED = {
    ".pdf", ".docx", ".pptx", ".xlsx", ".md", ".html", ".htm",
    ".txt", ".csv", ".png", ".jpg", ".jpeg", ".adoc",
}


def _load_worker():
    spec = importlib.util.spec_from_file_location(
        "worker", ROOT / "ingestion-workers" / "worker.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _collect(paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for p in paths:
        pp = Path(p)
        if pp.is_dir():
            files += sorted(
                f for f in pp.rglob("*") if f.is_file() and f.suffix.lower() in SUPPORTED
            )
        elif pp.is_file():
            files.append(pp)
        else:
            print(f"  skip (not found): {p}")
    return files


def main() -> int:
    ap = argparse.ArgumentParser(description="Batch ingest documents into Qdrant.")
    ap.add_argument("paths", nargs="*", help="files or directories to ingest")
    ap.add_argument("--reset", action="store_true", help="delete all vectors first")
    args = ap.parse_args()

    if not args.paths and not args.reset:
        ap.error("provide at least one path, or --reset")

    worker = _load_worker()

    # --- Delete ALL existing data before ingesting new data. ---
    if args.reset:
        client = worker.get_qdrant()
        coll = worker.QDRANT_COLLECTION
        if client.collection_exists(coll):
            client.delete_collection(coll)
            print(f"reset: dropped collection '{coll}'")
        else:
            print(f"reset: collection '{coll}' did not exist")

    files = _collect(args.paths)
    if not files:
        print("nothing to ingest.")
        return 0

    print(f"ingesting {len(files)} file(s)...")
    total = 0
    for i, f in enumerate(files, 1):
        res = worker.ingest_document(str(f))
        added = res.get("added", res.get("chunks", 0)) or 0
        total += added
        print(f"  [{i}/{len(files)}] {res.get('status', '?'):9} +{added:<4} {f}")

    print(f"done. {total} chunk(s) indexed across {len(files)} file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
