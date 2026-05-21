"""
Full RAG pipeline:
  1. Dynamic prompt optimization based on query intent
  2. Hybrid retrieval (semantic + BM25)
  3. Custom reranking
  4. Grounded answer generation via Groq
  5. Semantic clustering utility
"""
from __future__ import annotations

import os
from typing import Any

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np

from src.rag.search import hybrid_search, semantic_search
from src.rag.reranker import rerank
from src.rag.ingest import EMBED_MODEL

# ---------------------------------------------------------------------------
# Dynamic prompt optimization
# ---------------------------------------------------------------------------

INTENT_PROFILES = {
    "factual": {
        "keywords": ["what is", "define", "explain", "how does", "what are"],
        "instruction": "Answer directly and concisely using the retrieved context. Cite specific facts.",
    },
    "analytical": {
        "keywords": ["compare", "analyze", "evaluate", "pros and cons", "tradeoffs", "why"],
        "instruction": "Provide a structured analysis. Compare perspectives from the context. Show reasoning.",
    },
    "planning": {
        "keywords": ["how to", "should i", "strategy", "plan", "steps", "recommend"],
        "instruction": "Give actionable, step-by-step guidance grounded in the context. Tailor to the user's situation.",
    },
    "multilingual": {
        "keywords": [],  # detected by non-ASCII characters
        "instruction": "Respond in the same language as the query. Use retrieved context in any language.",
    },
}


def detect_intent(query: str) -> str:
    q = query.lower()
    # Multilingual detection: non-ASCII suggests non-English
    if any(ord(c) > 127 for c in query):
        return "multilingual"
    for intent, profile in INTENT_PROFILES.items():
        if any(kw in q for kw in profile["keywords"]):
            return intent
    return "factual"


def build_rag_prompt(query: str, context_chunks: list[dict]) -> tuple[str, str]:
    intent = detect_intent(query)
    instruction = INTENT_PROFILES[intent]["instruction"]

    context_text = "\n\n---\n\n".join(
        f"[{c['metadata']['title']} | {c['metadata']['language'].upper()} | {c['metadata']['date']}]\n{c['text']}"
        for c in context_chunks[:5]
    )

    system = (
        "You are AuraWealth AI, an expert financial advisor assistant. "
        "Answer questions using ONLY the provided context. "
        "If the context is insufficient, say so clearly. "
        f"{instruction}"
    )
    user = f"Context:\n{context_text}\n\nQuestion: {query}"
    return system, user


# ---------------------------------------------------------------------------
# RAG pipeline
# ---------------------------------------------------------------------------

def run_rag(
    query: str,
    collection: Any,
    bm25: Any,
    chunks: list[dict],
    top_k: int = 10,
    language_filter: str | None = None,
) -> dict:
    """Retrieve, rerank, and generate a grounded answer."""
    hits = hybrid_search(
        query, collection, bm25, chunks,
        top_k=top_k,
        language_filter=language_filter,
    )
    ranked = rerank(hits, query)
    system_prompt, user_prompt = build_rag_prompt(query, ranked)

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY"),
    )
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])

    return {
        "query": query,
        "intent": detect_intent(query),
        "answer": response.content,
        "sources": [
            {
                "title": h["metadata"]["title"],
                "language": h["metadata"]["language"],
                "score": h["rerank_score"],
                "snippet": h["text"][:120] + "...",
            }
            for h in ranked[:3]
        ],
    }


# ---------------------------------------------------------------------------
# Semantic clustering
# ---------------------------------------------------------------------------

def cluster_documents(
    chunks: list[dict],
    n_clusters: int = 6,
    sample_size: int = 200,
) -> dict[int, list[str]]:
    """K-means clustering over a sample of chunks. Returns cluster → document titles."""
    model = SentenceTransformer(EMBED_MODEL)

    sample = chunks[:sample_size]
    texts = [c["text"] for c in sample]
    embeddings = model.encode(texts, show_progress_bar=False)

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    labels = km.fit_predict(embeddings)

    clusters: dict[int, list[str]] = {i: [] for i in range(n_clusters)}
    for chunk, label in zip(sample, labels):
        title = chunk["metadata"]["title"]
        if title not in clusters[label]:
            clusters[label].append(title)

    return clusters
