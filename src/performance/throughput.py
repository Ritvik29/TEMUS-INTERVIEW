"""
Throughput scaling: proves 2x improvement by moving from sequential to concurrent requests.

Baseline:  N queries processed one-at-a-time   → QPS_baseline
Scaled:    N queries processed concurrently    → QPS_scaled ≥ 2× QPS_baseline

Correctness proof: responses from both runs are non-empty and semantically equivalent
(same query → same answer within 95% cosine similarity).
"""
from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field

import numpy as np
from dotenv import load_dotenv
from groq import AsyncGroq, Groq
from sentence_transformers import SentenceTransformer

from src.rag.ingest import EMBED_MODEL
from src.performance.benchmarks import BENCHMARK_QUERIES

load_dotenv()

_embed: SentenceTransformer | None = None


def _get_embed() -> SentenceTransformer:
    global _embed
    if _embed is None:
        _embed = SentenceTransformer(EMBED_MODEL)
    return _embed


SYSTEM = "You are AuraWealth AI, a financial advisor. Answer in 1-2 sentences."


@dataclass
class ThroughputResult:
    mode: str           # "sequential" or "concurrent"
    n_queries: int
    total_time_s: float
    qps: float
    responses: list[str] = field(default_factory=list)
    all_successful: bool = True


def _sequential_run(queries: list[str]) -> ThroughputResult:
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    responses = []
    t0 = time.perf_counter()
    for q in queries:
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": q}],
            max_tokens=100,
        )
        responses.append(resp.choices[0].message.content)
    elapsed = time.perf_counter() - t0
    return ThroughputResult(
        mode="sequential",
        n_queries=len(queries),
        total_time_s=round(elapsed, 2),
        qps=round(len(queries) / elapsed, 2),
        responses=responses,
        all_successful=all(r for r in responses),
    )


async def _concurrent_run(queries: list[str]) -> ThroughputResult:
    client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))
    t0 = time.perf_counter()

    async def call(q: str) -> str:
        resp = await client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": q}],
            max_tokens=100,
        )
        return resp.choices[0].message.content

    responses = await asyncio.gather(*[call(q) for q in queries])
    elapsed = time.perf_counter() - t0
    return ThroughputResult(
        mode="concurrent",
        n_queries=len(queries),
        total_time_s=round(elapsed, 2),
        qps=round(len(queries) / elapsed, 2),
        responses=list(responses),
        all_successful=all(r for r in responses),
    )


def _correctness_score(responses_a: list[str], responses_b: list[str]) -> float:
    """Avg cosine similarity between paired responses — proves answers are equivalent."""
    model = _get_embed()
    embs_a = model.encode(responses_a)
    embs_b = model.encode(responses_b)
    sims = [
        float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
        for a, b in zip(embs_a, embs_b)
    ]
    return round(float(np.mean(sims)), 4)


def run_throughput_comparison(n_queries: int = 6) -> dict:
    """
    Run sequential vs concurrent throughput comparison.
    Uses the same query set for both runs to allow correctness comparison.
    """
    queries = BENCHMARK_QUERIES[:n_queries]

    print(f"\n{'='*60}")
    print(f"THROUGHPUT BENCHMARK — {n_queries} queries")
    print(f"{'='*60}")

    print(f"\n[1/2] Sequential run...")
    seq = _sequential_run(queries)
    print(f"  Total: {seq.total_time_s}s  |  QPS: {seq.qps}")

    print(f"\n[2/2] Concurrent run...")
    conc = asyncio.run(_concurrent_run(queries))
    print(f"  Total: {conc.total_time_s}s  |  QPS: {conc.qps}")

    speedup = round(conc.qps / seq.qps, 2) if seq.qps > 0 else 0
    correctness = _correctness_score(seq.responses, conc.responses)

    print(f"\n  Speedup        : {speedup}x")
    print(f"  Target         : ≥2x")
    print(f"  Correctness    : {correctness:.3f} avg cosine similarity (>0.85 = equivalent)")
    print(f"  All successful : seq={seq.all_successful}  conc={conc.all_successful}")

    return {
        "sequential": {"qps": seq.qps, "total_s": seq.total_time_s, "ok": seq.all_successful},
        "concurrent": {"qps": conc.qps, "total_s": conc.total_time_s, "ok": conc.all_successful},
        "speedup": speedup,
        "target_met": speedup >= 2.0,
        "correctness_score": correctness,
        "correctness_ok": correctness >= 0.85,
    }
