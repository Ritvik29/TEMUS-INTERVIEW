"""
Vector DB scaling: demonstrates 100 QPS against the full document corpus.

Method:
  - Run 100 concurrent semantic searches using asyncio + thread executor
    (ChromaDB is synchronous so we wrap it in run_in_executor)
  - Queries cover the full semantic range of the corpus (diverse topics + languages)
  - Report: QPS, latency percentiles, and result quality (non-empty results)
"""
from __future__ import annotations

import asyncio
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from src.rag.ingest import EMBED_MODEL, load_index
from src.rag.search import semantic_search

load_dotenv()

# 100 diverse queries covering the full corpus
QPS_QUERIES = [
    # English — portfolio (10)
    "What is portfolio diversification?",
    "Explain the efficient frontier.",
    "How does rebalancing work?",
    "What is beta in investing?",
    "Describe factor investing strategies.",
    "What is the Capital Market Line?",
    "How does Black-Litterman model work?",
    "What is risk parity?",
    "What is goals-based investing?",
    "How much equity should a 60 year old hold?",
    # English — risk (10)
    "What is sequence of returns risk?",
    "How to protect against inflation risk?",
    "What is longevity risk?",
    "What is concentration risk?",
    "How does dollar cost averaging work?",
    "What is liquidity risk?",
    "Compare systematic and unsystematic risk.",
    "What is the bucket strategy in retirement?",
    "How do I stress test a portfolio?",
    "What is behavioral risk in investing?",
    # English — retirement (10)
    "What is the 4 percent withdrawal rule?",
    "Difference between Traditional and Roth IRA?",
    "When should I claim Social Security?",
    "What are Required Minimum Distributions?",
    "How does compound interest affect savings?",
    "What is a Roth conversion ladder?",
    "How to plan for healthcare costs in retirement?",
    "What is a Qualified Charitable Distribution?",
    "Should I choose 401k or IRA?",
    "What is U-shaped retirement spending?",
    # English — tax (10)
    "What is tax loss harvesting?",
    "What is the wash sale rule?",
    "What is asset location in tax planning?",
    "What are long term capital gains rates?",
    "What is a Donor Advised Fund?",
    "How to donate appreciated stock to charity?",
    "What is a 1031 exchange?",
    "Compare Roth and traditional 401k contributions.",
    "What is the step up in basis at death?",
    "What is the Qualified Business Income deduction?",
    # English — fixed income (10)
    "Why do bond prices fall when rates rise?",
    "What is bond duration?",
    "What is an inverted yield curve?",
    "What are TIPS bonds?",
    "What is a bond ladder strategy?",
    "What are investment grade vs high yield bonds?",
    "What is negative convexity in MBS?",
    "What are municipal bonds?",
    "How does credit risk affect bond yields?",
    "What is a yield curve?",
    # English — ESG and crypto (10)
    "What does ESG stand for?",
    "What is negative screening in ESG?",
    "Does ESG investing sacrifice returns?",
    "What is greenwashing?",
    "What are the UN SDGs?",
    "What is Bitcoin halving?",
    "What is DeFi?",
    "How to custody cryptocurrency safely?",
    "What are stablecoins?",
    "What percentage of portfolio should be in crypto?",
    # English — behavioral and real estate (10)
    "What is loss aversion in investing?",
    "What is recency bias?",
    "What is overconfidence bias?",
    "What is the disposition effect?",
    "How does herding affect markets?",
    "What is a cap rate in real estate?",
    "What are REITs?",
    "How does leverage affect real estate returns?",
    "What is a revocable living trust?",
    "What is a power of attorney?",
    # Spanish (10)
    "¿Qué es la diversificación de cartera?",
    "¿Cómo protegerse del riesgo de inflación?",
    "¿Qué es el riesgo de longevidad?",
    "¿Cuándo empezar a ahorrar para la jubilación?",
    "¿Qué es la gestión de riesgos financieros?",
    "¿Qué es un bono de alto rendimiento?",
    "¿Cómo funciona el interés compuesto?",
    "¿Qué es la planificación patrimonial?",
    "¿Qué son los fondos indexados?",
    "¿Qué es el rebalanceo de cartera?",
    # French (10)
    "Qu'est-ce que la diversification de portefeuille?",
    "Comment fonctionne le marché obligataire?",
    "Qu'est-ce que la gestion de patrimoine?",
    "Comment réduire le risque de mon portefeuille?",
    "Qu'est-ce que la planification de la retraite?",
    "Qu'est-ce que l'investissement ESG?",
    "Qu'est-ce que le risque de taux d'intérêt?",
    "Comment fonctionne la diversification internationale?",
    "Qu'est-ce que l'analyse fondamentale?",
    "Qu'est-ce que la frontière efficiente?",
    # English — estate and advanced (10)
    "What is a GRAT in estate planning?",
    "How do beneficiary designations work?",
    "What is intestate succession?",
    "What is a 1031 exchange in real estate?",
    "How does the step up in basis work at death?",
    "What is the difference between a will and a trust?",
    "What is a special needs trust?",
    "How does portfolio rebalancing reduce risk over time?",
    "What is the equity risk premium?",
    "What is the Sharpe ratio?",
]

assert len(QPS_QUERIES) == 100, f"Expected 100 queries, got {len(QPS_QUERIES)}"


@dataclass
class QPSReport:
    total_queries: int = 100
    successful: int = 0
    failed: int = 0
    total_time_s: float = 0.0
    qps: float = 0.0
    avg_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    target_met: bool = False


async def run_100_qps_test(collection=None, bm25=None, chunks=None) -> QPSReport:
    """
    Fire 100 concurrent semantic searches and measure QPS + latency.
    Loads the index from disk if not provided.
    """
    if collection is None:
        print("Loading vector index...")
        collection, bm25, chunks = load_index()

    model = SentenceTransformer(EMBED_MODEL)
    loop = asyncio.get_event_loop()
    executor = ThreadPoolExecutor(max_workers=16)

    def search_one(query: str) -> tuple[float, int]:
        t0 = time.perf_counter()
        results = semantic_search(query, collection, top_k=3)
        elapsed = (time.perf_counter() - t0) * 1000
        return elapsed, len(results)

    print(f"\n{'='*60}")
    print(f"VECTOR DB QPS TEST — {len(QPS_QUERIES)} concurrent queries")
    print(f"{'='*60}")

    t_global = time.perf_counter()
    tasks = [
        loop.run_in_executor(executor, search_one, q)
        for q in QPS_QUERIES
    ]
    raw_results = await asyncio.gather(*tasks, return_exceptions=True)
    total_s = time.perf_counter() - t_global

    latencies = []
    successful = 0
    for r in raw_results:
        if isinstance(r, Exception):
            continue
        lat, n_results = r
        if n_results > 0:
            successful += 1
        latencies.append(lat)

    executor.shutdown(wait=False)

    arr = np.array(latencies) if latencies else np.array([0.0])
    qps = round(len(QPS_QUERIES) / total_s, 1)
    report = QPSReport(
        total_queries=len(QPS_QUERIES),
        successful=successful,
        failed=len(QPS_QUERIES) - successful,
        total_time_s=round(total_s, 2),
        qps=qps,
        avg_latency_ms=round(float(np.mean(arr)), 1),
        p50_latency_ms=round(float(np.percentile(arr, 50)), 1),
        p95_latency_ms=round(float(np.percentile(arr, 95)), 1),
        p99_latency_ms=round(float(np.percentile(arr, 99)), 1),
        target_met=qps >= 100,
    )

    status = "✅" if report.target_met else "⚠️ "
    print(f"\n  {status} QPS       : {report.qps} (target: 100)")
    print(f"  Successful  : {report.successful}/{report.total_queries}")
    print(f"  Total time  : {report.total_time_s}s")
    print(f"  avg latency : {report.avg_latency_ms}ms")
    print(f"  p50 latency : {report.p50_latency_ms}ms")
    print(f"  p95 latency : {report.p95_latency_ms}ms")
    print(f"  p99 latency : {report.p99_latency_ms}ms")
    return report
