"""
Semantic search, keyword (BM25) search, and hybrid search with RRF fusion.
"""
from __future__ import annotations

import math
from typing import Any

from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

from src.rag.ingest import EMBED_MODEL

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL)
    return _model


# ---------------------------------------------------------------------------
# Semantic search
# ---------------------------------------------------------------------------

def semantic_search(
    query: str,
    collection: Any,
    top_k: int = 10,
    language_filter: str | None = None,
    category_filter: str | None = None,
) -> list[dict]:
    model = _get_model()
    query_embedding = model.encode([query])[0].tolist()

    where: dict = {}
    if language_filter:
        where["language"] = language_filter
    if category_filter:
        where["category"] = category_filter

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where if where else None,
        include=["documents", "metadatas", "distances"],
    )

    hits = []
    for i in range(len(results["ids"][0])):
        score = 1 - results["distances"][0][i]  # cosine similarity
        hits.append({
            "id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "semantic_score": round(score, 4),
        })
    return hits


# ---------------------------------------------------------------------------
# Keyword search (BM25) with optional keyword upweighting
# ---------------------------------------------------------------------------

DOMAIN_KEYWORDS = {
    "portfolio", "risk", "return", "investment", "asset", "equity", "bond",
    "dividend", "yield", "diversification", "allocation", "rebalancing",
    "retirement", "tax", "estate", "inflation", "volatility", "hedge",
    "cartera", "riesgo", "inversión", "patrimonio",   # Spanish
    "portefeuille", "risque", "rendement", "patrimoine",  # French
}


def keyword_search(
    query: str,
    bm25: BM25Okapi,
    chunks: list[dict],
    top_k: int = 10,
    upweight_keywords: list[str] | None = None,
) -> list[dict]:
    tokens = query.lower().split()
    scores = bm25.get_scores(tokens)

    # Upweight domain keywords and caller-specified terms
    boost_terms = set(t.lower() for t in (upweight_keywords or [])) | DOMAIN_KEYWORDS
    for i, chunk in enumerate(chunks):
        text_tokens = set(chunk["text"].lower().split())
        overlap = boost_terms & text_tokens
        if overlap:
            scores[i] *= 1 + 0.1 * len(overlap)

    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
    return [
        {
            "id": chunks[idx]["id"],
            "text": chunks[idx]["text"],
            "metadata": chunks[idx]["metadata"],
            "bm25_score": round(float(score), 4),
        }
        for idx, score in ranked
        if score > 0
    ]


# ---------------------------------------------------------------------------
# Hybrid search via Reciprocal Rank Fusion (RRF)
# ---------------------------------------------------------------------------

def hybrid_search(
    query: str,
    collection: Any,
    bm25: BM25Okapi,
    chunks: list[dict],
    top_k: int = 10,
    rrf_k: int = 60,
    semantic_weight: float = 0.6,
    keyword_weight: float = 0.4,
    language_filter: str | None = None,
    category_filter: str | None = None,
) -> list[dict]:
    sem_hits = semantic_search(query, collection, top_k=top_k * 2,
                               language_filter=language_filter,
                               category_filter=category_filter)
    kw_hits = keyword_search(query, bm25, chunks, top_k=top_k * 2)

    rrf_scores: dict[str, float] = {}
    id_to_hit: dict[str, dict] = {}

    for rank, hit in enumerate(sem_hits):
        rrf_scores[hit["id"]] = rrf_scores.get(hit["id"], 0) + semantic_weight / (rrf_k + rank + 1)
        id_to_hit[hit["id"]] = hit

    for rank, hit in enumerate(kw_hits):
        rrf_scores[hit["id"]] = rrf_scores.get(hit["id"], 0) + keyword_weight / (rrf_k + rank + 1)
        if hit["id"] not in id_to_hit:
            id_to_hit[hit["id"]] = hit

    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [
        {**id_to_hit[chunk_id], "hybrid_score": round(score, 6)}
        for chunk_id, score in ranked
    ]
