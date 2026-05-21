"""
Custom reranker: combines semantic score, BM25 score, recency, and domain relevance.

Scoring rationale:
  - Semantic similarity captures meaning-level relevance.
  - BM25 boosts exact keyword matches the vector model may miss.
  - Recency rewards newer documents for fast-moving topics like crypto/markets.
  - Domain boost rewards chunks whose category matches inferred query domain.
"""
from __future__ import annotations

from datetime import date


CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "portfolio":   ["portfolio", "allocation", "diversification", "rebalancing", "cartera"],
    "risk":        ["risk", "volatility", "hedge", "drawdown", "riesgo"],
    "retirement":  ["retirement", "401k", "ira", "rmd", "social security", "jubilación"],
    "tax":         ["tax", "capital gains", "roth", "harvesting", "impuesto"],
    "fixed_income":["bond", "yield", "duration", "coupon", "treasury", "obligación"],
    "real_estate": ["real estate", "reit", "property", "mortgage", "rental", "inmueble"],
    "esg":         ["esg", "sustainable", "climate", "governance", "impact"],
    "crypto":      ["bitcoin", "ethereum", "crypto", "blockchain", "defi", "token"],
    "behavioral":  ["bias", "behavioral", "psychology", "overconfidence", "heuristic"],
    "estate":      ["estate", "trust", "inheritance", "will", "succession", "herencia"],
    "fundamentals":["market", "stock", "equity", "index", "marché", "bourse"],
}

TODAY = date.today()


def _recency_score(doc_date_str: str) -> float:
    """More recent docs score higher; linear decay over 3 years to 0.5."""
    try:
        doc_date = date.fromisoformat(doc_date_str)
        days_old = (TODAY - doc_date).days
        return max(0.5, 1.0 - days_old / (3 * 365))
    except Exception:
        return 0.75


def _domain_boost(chunk_text: str, query: str) -> float:
    """Boost if chunk category keywords appear in both query and chunk."""
    query_lower = query.lower()
    chunk_lower = chunk_text.lower()
    boost = 1.0
    for _, keywords in CATEGORY_KEYWORDS.items():
        matches_query = any(k in query_lower for k in keywords)
        matches_chunk = any(k in chunk_lower for k in keywords)
        if matches_query and matches_chunk:
            boost += 0.15
    return min(boost, 1.6)


def rerank(
    hits: list[dict],
    query: str,
    semantic_w: float = 0.50,
    bm25_w: float = 0.25,
    recency_w: float = 0.10,
    domain_w: float = 0.15,
) -> list[dict]:
    """
    Score each hit on four dimensions and return results sorted by composite score.
    All component weights sum to 1.0.
    """
    scored = []
    for hit in hits:
        sem = hit.get("semantic_score", hit.get("hybrid_score", 0.5))
        bm25 = min(hit.get("bm25_score", 0) / 10, 1.0)  # normalise BM25
        recency = _recency_score(hit["metadata"].get("date", "2023-01-01"))
        domain = _domain_boost(hit["text"], query)

        composite = (
            semantic_w * sem
            + bm25_w * bm25
            + recency_w * recency
            + domain_w * (domain - 1.0)  # domain boost is additive delta
        )
        scored.append({**hit, "rerank_score": round(composite, 4)})

    scored.sort(key=lambda x: x["rerank_score"], reverse=True)
    return scored
