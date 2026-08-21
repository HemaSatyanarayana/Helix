"""Encoders shared by ingestion and retrieval.

Both stages must agree exactly on how text becomes vectors — a query embedded
differently from the documents searches a different space and quietly returns
nothing useful. Keeping both encoders here makes that agreement structural
rather than a convention two modules remember to follow.

Two vectors per chunk:

* **dense** (``EMBEDDING_MODEL``, a sentence-transformer) — semantic similarity.
  Finds the passage that *means* the same thing as the question.
* **sparse** (BM25 via fastembed) — lexical term matching. Finds the passage
  that uses the same *words*.

Product documentation needs both. A question naming ``apxor.init()``, an error
string, or a config key is answered by exact-token overlap, which is precisely
what a 384-dimension bi-encoder blurs away; a question phrased in the user's own
words needs the semantic side. Qdrant fuses the two rankings with Reciprocal
Rank Fusion at query time.

BM25 here is a tokenizer plus IDF, not a model — no weights are downloaded and
no GPU is involved. Qdrant computes the IDF component server-side, which is why
the sparse vector config must declare ``Modifier.IDF``.
"""

from __future__ import annotations

import os
from functools import lru_cache

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
SPARSE_MODEL = os.getenv("SPARSE_MODEL", "Qdrant/bm25")

# Named vectors in the Qdrant collection. Named (rather than default/unnamed)
# vectors are required to hold more than one vector per point.
DENSE_VECTOR = "dense"
SPARSE_VECTOR = "bm25"

HYBRID_ENABLED = os.getenv("HYBRID_ENABLED", "true").lower() == "true"


@lru_cache(maxsize=1)
def dense_encoder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def sparse_encoder():
    from fastembed import SparseTextEmbedding

    return SparseTextEmbedding(model_name=SPARSE_MODEL)


def embed_dense(texts: list[str]) -> list[list[float]]:
    """Encode texts into normalized dense vectors."""
    vectors = dense_encoder().encode(texts, normalize_embeddings=True)
    return [v.tolist() for v in vectors]


def embed_dense_query(question: str) -> list[float]:
    return embed_dense([question])[0]


def _to_sparse(embedding) -> dict[str, list]:
    """fastembed SparseEmbedding -> the shape Qdrant wants."""
    return {
        "indices": embedding.indices.tolist(),
        "values": embedding.values.tolist(),
    }


def embed_sparse(texts: list[str]) -> list[dict[str, list]]:
    """Encode documents into BM25 sparse vectors."""
    return [_to_sparse(e) for e in sparse_encoder().embed(texts)]


def embed_sparse_query(question: str) -> dict[str, list]:
    """Encode a query into a BM25 sparse vector.

    Queries use ``query_embed``, not ``embed`` — BM25 weights document terms by
    frequency but query terms by presence, so encoding a query as a document
    skews the match toward whichever word the user happened to repeat.
    """
    return _to_sparse(next(iter(sparse_encoder().query_embed(question))))


def dense_dimension() -> int:
    return int(dense_encoder().get_sentence_embedding_dimension())
