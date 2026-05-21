"""
Advanced evaluation technique: RAGAS-style metrics with LLM-as-judge.

Three metrics (as used in the RAGAS paper, adapted for our RAG pipeline):

1. Faithfulness
   — Is every claim in the generated answer supported by the retrieved context?
   — Score: fraction of claims that are grounded in context (0–1).

2. Answer Relevancy
   — Does the answer actually address the question asked?
   — Score: semantic similarity between generated answer and question (0–1).

3. Context Recall
   — Does the retrieved context contain the information needed to answer?
   — Score: fraction of reference answer sentences covered by context (0–1).

Composite RAGAS score = harmonic mean of the three metrics.

Why this technique?
  RAGAS provides orthogonal diagnostics:
  - Low faithfulness → model is hallucinating beyond the retrieved context.
  - Low relevancy   → model is retrieving correct info but not answering the question.
  - Low recall      → retriever is missing relevant documents.
  This separation helps us identify and fix the right component.
"""
from __future__ import annotations

import json
import os

from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer
import numpy as np

from src.rag.ingest import EMBED_MODEL

load_dotenv()

JUDGE_MODEL = "llama-3.3-70b-versatile"
_embed_model: SentenceTransformer | None = None


def _get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(EMBED_MODEL)
    return _embed_model


def _llm(prompt: str) -> str:
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    resp = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
        temperature=0.0,
    )
    return resp.choices[0].message.content.strip()


def _parse_json(text: str) -> dict:
    text = text.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(text)
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Metric 1: Faithfulness
# ---------------------------------------------------------------------------

FAITHFULNESS_PROMPT = """Given the context and the answer, identify each factual claim in the answer.
For each claim, determine if it is directly supported by the context.

Context:
{context}

Answer:
{answer}

Return ONLY valid JSON:
{{
  "claims": ["claim1", "claim2", ...],
  "supported": [true, false, ...],
  "faithfulness_score": <float 0-1>
}}"""


def faithfulness(answer: str, context: str) -> float:
    """Fraction of answer claims supported by context."""
    prompt = FAITHFULNESS_PROMPT.format(context=context[:2000], answer=answer)
    result = _parse_json(_llm(prompt))
    return float(result.get("faithfulness_score", 0.5))


# ---------------------------------------------------------------------------
# Metric 2: Answer Relevancy
# ---------------------------------------------------------------------------

def answer_relevancy(question: str, answer: str) -> float:
    """
    Semantic similarity between question and answer embeddings.
    High similarity means the answer directly addresses the question.
    """
    model = _get_embed_model()
    q_emb = model.encode([question])[0]
    a_emb = model.encode([answer])[0]
    sim = float(np.dot(q_emb, a_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(a_emb) + 1e-9))
    return round(max(0.0, sim), 4)


# ---------------------------------------------------------------------------
# Metric 3: Context Recall
# ---------------------------------------------------------------------------

RECALL_PROMPT = """Given the reference answer and the retrieved context, determine what fraction of
the reference answer's key information is present in the context.

Reference answer:
{reference}

Retrieved context:
{context}

Return ONLY valid JSON:
{{
  "covered_points": ["point1", ...],
  "missing_points": ["point1", ...],
  "recall_score": <float 0-1>
}}"""


def context_recall(reference_answer: str, context: str) -> float:
    """Fraction of reference answer information covered by context."""
    prompt = RECALL_PROMPT.format(reference=reference_answer, context=context[:2000])
    result = _parse_json(_llm(prompt))
    return float(result.get("recall_score", 0.5))


# ---------------------------------------------------------------------------
# Composite RAGAS score
# ---------------------------------------------------------------------------

def ragas_score(faithfulness_s: float, relevancy_s: float, recall_s: float) -> float:
    """Harmonic mean of the three RAGAS metrics (penalises weak components)."""
    scores = [faithfulness_s, relevancy_s, recall_s]
    if any(s <= 0 for s in scores):
        return 0.0
    return round(3 / sum(1 / s for s in scores), 4)


# ---------------------------------------------------------------------------
# Full RAGAS evaluation for a single RAG response
# ---------------------------------------------------------------------------

def evaluate_rag_response(
    question: str,
    generated_answer: str,
    retrieved_context: str,
    reference_answer: str,
) -> dict:
    """
    Run all three RAGAS metrics and return a structured report.

    Args:
        question: original user question
        generated_answer: answer produced by the RAG pipeline
        retrieved_context: concatenated retrieved chunks (text)
        reference_answer: ground truth answer from eval dataset
    """
    f = faithfulness(generated_answer, retrieved_context)
    r = answer_relevancy(question, generated_answer)
    c = context_recall(reference_answer, retrieved_context)
    composite = ragas_score(f, r, c)

    return {
        "question": question,
        "faithfulness": f,
        "answer_relevancy": r,
        "context_recall": c,
        "ragas_score": composite,
        "interpretation": {
            "faithfulness": "hallucination risk" if f < 0.6 else "grounded",
            "answer_relevancy": "off-topic risk" if r < 0.5 else "on-topic",
            "context_recall": "retriever gap" if c < 0.6 else "good coverage",
        },
        "diagnosis": (
            "Pipeline healthy" if composite >= 0.65
            else "Needs attention: "
            + (", ".join([
                "faithfulness" if f < 0.6 else "",
                "relevancy" if r < 0.5 else "",
                "recall" if c < 0.6 else "",
            ]).strip(", "))
        ),
    }


# ---------------------------------------------------------------------------
# Batch RAGAS evaluation
# ---------------------------------------------------------------------------

def batch_ragas_eval(
    samples: list[dict],
) -> dict:
    """
    Evaluate a list of RAG samples.

    Each sample: {question, generated_answer, retrieved_context, reference_answer}
    """
    results = []
    for s in samples:
        r = evaluate_rag_response(
            s["question"], s["generated_answer"],
            s["retrieved_context"], s["reference_answer"],
        )
        results.append(r)
        print(f"  RAGAS={r['ragas_score']:.3f} | {r['diagnosis'][:50]}")

    n = len(results)
    return {
        "n": n,
        "avg_faithfulness": round(sum(r["faithfulness"] for r in results) / n, 3),
        "avg_relevancy": round(sum(r["answer_relevancy"] for r in results) / n, 3),
        "avg_recall": round(sum(r["context_recall"] for r in results) / n, 3),
        "avg_ragas": round(sum(r["ragas_score"] for r in results) / n, 3),
        "results": results,
    }
