"""
Explainability for the RAG workflow.

Produces a human-readable trace of every decision made during a RAG call:
  1. Query intent detection
  2. Hybrid retrieval scores (semantic + BM25)
  3. Reranker scoring breakdown (semantic, BM25, recency, domain)
  4. Top-k context selection
  5. Prompt construction
  6. Guardrail check results
  7. Final answer

This lets users and auditors understand *why* the system said what it said.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from dotenv import load_dotenv

from src.rag.pipeline import detect_intent, build_rag_prompt
from src.rag.search import semantic_search, keyword_search, hybrid_search
from src.rag.reranker import rerank, _recency_score, _domain_boost
from src.governance.guardrails import GuardrailPipeline

load_dotenv()


@dataclass
class RetrievalExplanation:
    chunk_id: str
    title: str
    language: str
    snippet: str
    semantic_score: float
    bm25_score: float
    recency_score: float
    domain_boost: float
    rerank_score: float


@dataclass
class WorkflowTrace:
    query: str
    detected_intent: str
    input_guardrail_passed: bool
    input_guardrail_violations: list[str]
    retrieval_strategy: str
    top_chunks: list[RetrievalExplanation]
    system_prompt_preview: str
    output_guardrail_passed: bool
    output_guardrail_violations: list[str]
    final_answer: str
    answer_word_count: int
    explanation: str = ""


def explain_rag(
    query: str,
    collection: Any,
    bm25: Any,
    chunks: list[dict],
    top_k: int = 5,
) -> WorkflowTrace:
    """
    Run the full RAG pipeline with full tracing.
    Returns a WorkflowTrace explaining every step.
    """
    pipeline = GuardrailPipeline()

    # Step 1: Input guardrail
    input_result = pipeline.check_input(query)
    if not input_result.passed:
        return WorkflowTrace(
            query=query,
            detected_intent="blocked",
            input_guardrail_passed=False,
            input_guardrail_violations=[str(v) for v in input_result.violations],
            retrieval_strategy="none — blocked at input",
            top_chunks=[],
            system_prompt_preview="",
            output_guardrail_passed=True,
            output_guardrail_violations=[],
            final_answer=input_result.sanitised_text,
            answer_word_count=0,
            explanation="Query was blocked by the input guardrail before reaching the LLM.",
        )

    # Step 2: Intent detection
    intent = detect_intent(query)

    # Step 3: Hybrid retrieval with individual scores
    sem_hits = semantic_search(query, collection, top_k=top_k * 2)
    kw_hits = keyword_search(query, bm25, chunks, top_k=top_k * 2)
    hybrid_hits = hybrid_search(query, collection, bm25, chunks, top_k=top_k)

    # Step 4: Reranking with score breakdown
    ranked = rerank(hybrid_hits, query)[:top_k]

    sem_scores = {h["id"]: h.get("semantic_score", 0) for h in sem_hits}
    kw_scores  = {h["id"]: h.get("bm25_score", 0)    for h in kw_hits}

    chunk_explanations = []
    for hit in ranked:
        cid = hit["id"]
        chunk_explanations.append(RetrievalExplanation(
            chunk_id=cid,
            title=hit["metadata"]["title"],
            language=hit["metadata"]["language"],
            snippet=hit["text"][:100] + "...",
            semantic_score=round(sem_scores.get(cid, hit.get("semantic_score", 0)), 4),
            bm25_score=round(kw_scores.get(cid, hit.get("bm25_score", 0)), 4),
            recency_score=round(_recency_score(hit["metadata"].get("date", "2023-01-01")), 4),
            domain_boost=round(_domain_boost(hit["text"], query), 4),
            rerank_score=hit["rerank_score"],
        ))

    # Step 5: Prompt construction
    system_prompt, user_prompt = build_rag_prompt(query, ranked)

    # Step 6: LLM call
    from groq import Groq
    from langchain_core.messages import SystemMessage, HumanMessage
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    resp = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        max_tokens=512,
    )
    raw_answer = resp.choices[0].message.content

    # Step 7: Output guardrail
    output_result = pipeline.check_output(raw_answer)
    final_answer = output_result.sanitised_text

    explanation = (
        f"Intent '{intent}' detected → hybrid retrieval (semantic + BM25) → "
        f"{len(ranked)} chunks reranked by composite score → "
        f"prompt optimised for '{intent}' queries → "
        f"{'output blocked by guardrail' if not output_result.passed else 'answer delivered'}."
    )

    return WorkflowTrace(
        query=query,
        detected_intent=intent,
        input_guardrail_passed=True,
        input_guardrail_violations=[],
        retrieval_strategy=f"hybrid (semantic + BM25) → reranked top-{top_k}",
        top_chunks=chunk_explanations,
        system_prompt_preview=system_prompt[:200] + "...",
        output_guardrail_passed=output_result.passed,
        output_guardrail_violations=[str(v) for v in output_result.violations],
        final_answer=final_answer,
        answer_word_count=len(final_answer.split()),
        explanation=explanation,
    )


def print_trace(trace: WorkflowTrace) -> None:
    """Pretty-print a WorkflowTrace for demo/debug purposes."""
    print(f"\n{'='*60}")
    print(f"EXPLAINABILITY TRACE")
    print(f"{'='*60}")
    print(f"Query         : {trace.query}")
    print(f"Intent        : {trace.detected_intent}")
    print(f"Input Guard   : {'✅ passed' if trace.input_guardrail_passed else '🚫 blocked'}")
    print(f"Strategy      : {trace.retrieval_strategy}")
    print(f"\nTop Retrieved Chunks:")
    for i, c in enumerate(trace.top_chunks, 1):
        print(f"  {i}. [{c.rerank_score:.4f}] {c.title} ({c.language.upper()})")
        print(f"     semantic={c.semantic_score} bm25={c.bm25_score} "
              f"recency={c.recency_score} domain={c.domain_boost}")
        print(f"     \"{c.snippet}\"")
    print(f"\nOutput Guard  : {'✅ passed' if trace.output_guardrail_passed else '🚫 blocked'}")
    print(f"Answer ({trace.answer_word_count} words):\n{trace.final_answer}")
    print(f"\nExplanation   : {trace.explanation}")
    print(f"{'='*60}\n")
