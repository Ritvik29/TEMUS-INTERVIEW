"""
Performance benchmarks: time-to-first-token (TTFT) and end-to-end latency.

Targets:
  - <5s  time to first token (baseline requirement)
  - <1s  time to first token via streaming (stretch goal, +2 pts)

Groq's inference is fast enough to hit <1s TTFT on small prompts.
We use streaming to capture the exact moment the first token arrives.
"""
from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# 10 semantically different financial queries covering diverse topics
BENCHMARK_QUERIES = [
    "What is portfolio diversification?",                      # portfolio basics
    "Explain the yield curve inversion and recession signals.", # fixed income / macro
    "How does tax-loss harvesting reduce my tax bill?",        # tax
    "What retirement withdrawal rate is sustainable at 60?",   # retirement planning
    "Compare ESG investing vs traditional value investing.",    # ESG / analytical
    "¿Qué es la gestión de riesgos financieros?",              # multilingual (ES)
    "What are the risks of holding crypto in a portfolio?",    # crypto / risk
    "How does dollar-cost averaging protect against volatility?", # behavioral / strategy
    "Qu'est-ce que la diversification de portefeuille?",       # multilingual (FR)
    "What insurance products should retirees consider?",       # retirement / risk
]


@dataclass
class QueryBenchmark:
    query: str
    ttft_ms: float          # time to first token
    total_ms: float         # full response time
    tokens_approx: int      # approximate token count
    passed_1s: bool = False
    passed_5s: bool = False


@dataclass
class BenchmarkReport:
    results: list[QueryBenchmark] = field(default_factory=list)
    avg_ttft_ms: float = 0.0
    p50_ttft_ms: float = 0.0
    p95_ttft_ms: float = 0.0
    passed_1s_count: int = 0
    passed_5s_count: int = 0
    total_queries: int = 0


def measure_ttft(query: str, client: Groq) -> QueryBenchmark:
    """Measure time-to-first-token using the streaming API."""
    t_start = time.perf_counter()
    t_first_token = None
    full_response = []

    stream = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are AuraWealth AI, a financial advisor. Be concise."},
            {"role": "user", "content": query},
        ],
        max_tokens=200,
        stream=True,
    )

    for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        if delta and t_first_token is None:
            t_first_token = time.perf_counter()
        full_response.append(delta)

    t_end = time.perf_counter()
    ttft_ms = (t_first_token - t_start) * 1000 if t_first_token else (t_end - t_start) * 1000
    total_ms = (t_end - t_start) * 1000
    tokens_approx = len("".join(full_response).split())

    return QueryBenchmark(
        query=query,
        ttft_ms=round(ttft_ms, 1),
        total_ms=round(total_ms, 1),
        tokens_approx=tokens_approx,
        passed_1s=ttft_ms < 1000,
        passed_5s=ttft_ms < 5000,
    )


def run_ttft_benchmark(queries: list[str] | None = None) -> BenchmarkReport:
    """
    Run TTFT benchmark across all 10 semantically different queries.
    Reports per-query and aggregate latency statistics.
    """
    import numpy as np
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    queries = queries or BENCHMARK_QUERIES
    report = BenchmarkReport(total_queries=len(queries))
    results = []

    print(f"\n{'='*60}")
    print(f"TTFT BENCHMARK — {len(queries)} queries")
    print(f"{'='*60}")

    for q in queries:
        result = measure_ttft(q, client)
        results.append(result)
        status = "✅ <1s" if result.passed_1s else ("⚠️  <5s" if result.passed_5s else "❌ >5s")
        print(f"  {status}  TTFT={result.ttft_ms:.0f}ms  total={result.total_ms:.0f}ms  | {q[:55]}")

    ttfts = np.array([r.ttft_ms for r in results])
    report.results = results
    report.avg_ttft_ms = round(float(np.mean(ttfts)), 1)
    report.p50_ttft_ms = round(float(np.percentile(ttfts, 50)), 1)
    report.p95_ttft_ms = round(float(np.percentile(ttfts, 95)), 1)
    report.passed_1s_count = sum(1 for r in results if r.passed_1s)
    report.passed_5s_count = sum(1 for r in results if r.passed_5s)

    print(f"\n  avg TTFT : {report.avg_ttft_ms}ms")
    print(f"  p50 TTFT : {report.p50_ttft_ms}ms")
    print(f"  p95 TTFT : {report.p95_ttft_ms}ms")
    print(f"  <1s      : {report.passed_1s_count}/{report.total_queries}")
    print(f"  <5s      : {report.passed_5s_count}/{report.total_queries}")
    return report
