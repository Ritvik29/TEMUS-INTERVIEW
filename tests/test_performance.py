"""
Performance tests — no API calls, no vector index required.
Tests cover: semantic cache logic, message queue, throughput math, benchmark queries.
"""
import asyncio
import pytest
import numpy as np

from src.performance.cache import SemanticCache
from src.performance.queue import MessageQueue, Priority, run_batch
from src.performance.benchmarks import BENCHMARK_QUERIES
from src.performance.vector_scaling import QPS_QUERIES


# ---------------------------------------------------------------------------
# Semantic cache
# ---------------------------------------------------------------------------

def test_cache_miss_on_empty_cache():
    cache = SemanticCache()
    result = cache.get("What is diversification?")
    assert result is None


def test_cache_hit_on_exact_repeat():
    cache = SemanticCache(similarity_threshold=0.90)
    cache.put("What is portfolio diversification?", "Spreading investments across assets.")
    hit = cache.get("What is portfolio diversification?")
    assert hit is not None
    assert hit.response == "Spreading investments across assets."


def test_cache_hit_on_semantically_similar_query():
    cache = SemanticCache(similarity_threshold=0.88)
    cache.put("What is portfolio diversification?", "Spreading investments across assets.")
    hit = cache.get("Explain portfolio diversification.")
    assert hit is not None  # semantically similar enough


def test_cache_miss_on_unrelated_query():
    cache = SemanticCache(similarity_threshold=0.92)
    cache.put("What is portfolio diversification?", "Spreading investments across assets.")
    hit = cache.get("What is the weather like today?")
    assert hit is None


def test_cache_stats_track_hits_and_misses():
    cache = SemanticCache(similarity_threshold=0.90)
    cache.put("What is a bond?", "A bond is a fixed income instrument.")
    cache.get("What is a bond?")   # hit
    cache.get("Unrelated question about cooking.")  # miss
    stats = cache.stats()
    assert stats["hits"] >= 1
    assert stats["misses"] >= 1


def test_cache_evicts_lru_when_full():
    cache = SemanticCache(max_entries=2)
    cache.put("query one about finance", "answer one")
    cache.put("query two about markets", "answer two")
    cache.put("query three about bonds", "answer three")  # should evict oldest
    assert len(cache._store) == 2


# ---------------------------------------------------------------------------
# Message queue
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_queue_processes_all_tasks():
    q = MessageQueue(n_workers=2)
    await q.start()

    results = []
    async def task_fn(val):
        await asyncio.sleep(0.01)
        return val * 2

    ids = [q.submit(lambda v=i: task_fn(v), name=f"task_{i}") for i in range(5)]
    outcomes = [await q.wait_for(tid) for tid in ids]
    await q.stop()

    assert all(o.success for o in outcomes)
    assert q.stats()["processed"] == 5


@pytest.mark.asyncio
async def test_queue_handles_task_failure_gracefully():
    q = MessageQueue(n_workers=2)
    await q.start()

    async def failing_task():
        raise ValueError("Deliberate failure")

    tid = q.submit(failing_task, name="bad_task")
    result = await q.wait_for(tid)
    await q.stop()

    assert not result.success
    assert "Deliberate failure" in result.error


@pytest.mark.asyncio
async def test_queue_priority_ordering():
    q = MessageQueue(n_workers=1)
    await q.start()

    order = []
    async def make_task(label):
        order.append(label)
        return label

    q.submit(lambda: make_task("low"),    priority=Priority.LOW)
    q.submit(lambda: make_task("high"),   priority=Priority.HIGH)
    q.submit(lambda: make_task("normal"), priority=Priority.NORMAL)

    await q._queue.join()
    await q.stop()
    assert order[0] == "high"


# ---------------------------------------------------------------------------
# Benchmark queries coverage
# ---------------------------------------------------------------------------

def test_benchmark_has_10_queries():
    assert len(BENCHMARK_QUERIES) == 10


def test_benchmark_queries_are_semantically_diverse():
    """All 10 queries should be distinct (low pairwise similarity)."""
    from sentence_transformers import SentenceTransformer
    from src.rag.ingest import EMBED_MODEL
    model = SentenceTransformer(EMBED_MODEL)
    embs = model.encode(BENCHMARK_QUERIES)
    # Compute pairwise similarities (excluding self)
    sims = []
    for i in range(len(embs)):
        for j in range(i + 1, len(embs)):
            sim = float(np.dot(embs[i], embs[j]) /
                        (np.linalg.norm(embs[i]) * np.linalg.norm(embs[j]) + 1e-9))
            sims.append(sim)
    avg_sim = np.mean(sims)
    assert avg_sim < 0.85, f"Queries too similar on average: {avg_sim:.3f}"


def test_qps_query_set_has_100_entries():
    assert len(QPS_QUERIES) == 100


def test_qps_queries_cover_multiple_languages():
    languages_detected = {"en": 0, "es": 0, "fr": 0}
    for q in QPS_QUERIES:
        if any(c in q for c in "¿áéíóúñ"):
            languages_detected["es"] += 1
        elif any(c in q for c in "àâçèêîôùûœæ") or "Qu'est" in q:
            languages_detected["fr"] += 1
        else:
            languages_detected["en"] += 1
    assert languages_detected["es"] >= 5
    assert languages_detected["fr"] >= 5
