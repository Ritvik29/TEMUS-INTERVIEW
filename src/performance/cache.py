"""
Semantic response cache — returns cached answers for semantically similar queries.

How it works:
  1. Every LLM response is stored with the embedding of its query.
  2. On a new query, we embed it and compute cosine similarity against all cached entries.
  3. If the nearest cached query exceeds SIMILARITY_THRESHOLD, return the cached response
     (cache HIT) — no LLM call needed.
  4. Otherwise call the LLM, store the response, and return it (cache MISS).

Why semantic rather than exact-match?
  "What is diversification?" and "Explain portfolio diversification" should share a cache
  entry. Exact-match caching would miss this and make redundant LLM calls.

Cache eviction: LRU by access time, max MAX_ENTRIES entries.
"""
from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np
from sentence_transformers import SentenceTransformer

from src.rag.ingest import EMBED_MODEL

SIMILARITY_THRESHOLD = 0.92   # cosine similarity above this → cache hit
MAX_ENTRIES = 512

_embed_model: SentenceTransformer | None = None


def _model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(EMBED_MODEL)
    return _embed_model


@dataclass
class CacheEntry:
    query: str
    response: str
    embedding: np.ndarray
    hits: int = 0
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)


class SemanticCache:
    def __init__(
        self,
        similarity_threshold: float = SIMILARITY_THRESHOLD,
        max_entries: int = MAX_ENTRIES,
    ):
        self.threshold = similarity_threshold
        self.max_entries = max_entries
        self._store: OrderedDict[str, CacheEntry] = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._total_latency_saved_ms = 0.0

    # ------------------------------------------------------------------
    def get(self, query: str) -> CacheEntry | None:
        """Return a cached entry if a semantically similar query exists."""
        if not self._store:
            return None
        q_emb = _model().encode([query])[0]
        best_key, best_sim = self._nearest(q_emb)
        if best_sim >= self.threshold:
            entry = self._store[best_key]
            entry.hits += 1
            entry.last_accessed = time.time()
            self._store.move_to_end(best_key)   # LRU update
            self._hits += 1
            return entry
        return None

    def put(self, query: str, response: str) -> CacheEntry:
        """Store a new query-response pair."""
        if len(self._store) >= self.max_entries:
            self._store.popitem(last=False)   # evict LRU
        emb = _model().encode([query])[0]
        entry = CacheEntry(query=query, response=response, embedding=emb)
        self._store[query] = entry
        self._misses += 1
        return entry

    def cached_call(self, query: str, llm_fn, avg_llm_ms: float = 800.0) -> tuple[str, bool]:
        """
        Return (response, was_cached).
        Tracks estimated latency saved when serving from cache.
        """
        hit = self.get(query)
        if hit:
            self._total_latency_saved_ms += avg_llm_ms
            return hit.response, True
        response = llm_fn(query)
        self.put(query, response)
        return response, False

    # ------------------------------------------------------------------
    def _nearest(self, q_emb: np.ndarray) -> tuple[str, float]:
        embeddings = np.stack([e.embedding for e in self._store.values()])
        keys = list(self._store.keys())
        q_norm = q_emb / (np.linalg.norm(q_emb) + 1e-9)
        e_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9)
        sims = e_norm @ q_norm
        best_idx = int(np.argmax(sims))
        return keys[best_idx], float(sims[best_idx])

    def stats(self) -> dict:
        total = self._hits + self._misses
        return {
            "entries": len(self._store),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / total, 3) if total else 0.0,
            "latency_saved_ms": round(self._total_latency_saved_ms, 1),
            "threshold": self.threshold,
        }

    def clear(self) -> None:
        self._store.clear()
        self._hits = self._misses = 0
        self._total_latency_saved_ms = 0.0
