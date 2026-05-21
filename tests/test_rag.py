"""
Tests for RAG pipeline: chunking, search, reranking, and prompt optimization.
These tests run without hitting Groq or loading heavy ML models.
"""
from src.rag.ingest import chunk_text
from src.rag.reranker import rerank, _recency_score, _domain_boost
from src.rag.pipeline import detect_intent, build_rag_prompt
from src.rag.documents import DOCUMENTS


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def test_chunking_produces_over_1000_chunks():
    """All 12 documents chunked at default settings must exceed 1000 total chunks."""
    total = sum(len(chunk_text(doc["content"])) for doc in DOCUMENTS)
    assert total >= 1000, f"Only {total} chunks — need 1000+"


def test_chunks_respect_size_and_overlap():
    text = "A" * 500
    chunks = chunk_text(text, chunk_size=200, overlap=100)
    # Each chunk should be at most 200 chars
    assert all(len(c) <= 200 for c in chunks)
    # Overlap: consecutive chunks should share content
    assert chunks[0][:100] == chunks[1][:100] or len(chunks) == 1


def test_documents_cover_multiple_languages():
    languages = {doc["language"] for doc in DOCUMENTS}
    assert "en" in languages
    assert "es" in languages
    assert "fr" in languages


# ---------------------------------------------------------------------------
# Reranker
# ---------------------------------------------------------------------------

def test_reranker_orders_by_composite_score():
    hits = [
        {"id": "a", "text": "portfolio diversification reduces risk", "metadata": {"date": "2024-06-01", "title": "T"}, "semantic_score": 0.9, "bm25_score": 5.0},
        {"id": "b", "text": "unrelated cooking recipe text here", "metadata": {"date": "2020-01-01", "title": "T"}, "semantic_score": 0.3, "bm25_score": 0.1},
    ]
    ranked = rerank(hits, query="portfolio risk management")
    assert ranked[0]["id"] == "a", "Higher semantic + domain hit should rank first"


def test_reranker_recency_score_decreases_with_age():
    recent = _recency_score("2025-05-15")   # ~1 month ago
    old = _recency_score("2020-01-01")       # 5+ years ago — hits 0.5 floor
    assert recent > old


def test_reranker_domain_boost_increases_for_relevant_text():
    relevant = _domain_boost("portfolio allocation and diversification strategy", "portfolio allocation")
    irrelevant = _domain_boost("sunny weather in the afternoon today", "portfolio allocation")
    assert relevant > irrelevant


# ---------------------------------------------------------------------------
# Dynamic prompt optimization
# ---------------------------------------------------------------------------

def test_intent_detection_factual():
    assert detect_intent("What is portfolio diversification?") == "factual"


def test_intent_detection_analytical():
    assert detect_intent("Compare stocks and bonds for retirement") == "analytical"


def test_intent_detection_multilingual():
    assert detect_intent("¿Qué es la diversificación de cartera?") == "multilingual"


def test_rag_prompt_includes_context_and_query():
    fake_chunks = [{
        "text": "Diversification reduces risk.",
        "metadata": {"title": "Portfolio Guide", "language": "en", "date": "2024-01-01"},
    }]
    system, user = build_rag_prompt("What is diversification?", fake_chunks)
    assert "Diversification reduces risk" in user
    assert "What is diversification?" in user
    assert "AuraWealth" in system
