# AuraWealth AI

A consumer wealth management platform that bridges high-touch human financial advice with AI-driven efficiency. Built for the Temus interview assignment.

Two user types:
- **Client** — real-time net worth view, life goal tracking, AI advisor chat
- **Advisor** — command centre for portfolio review, risk analysis, client flagging, and rebalancing

---

## Quick Start

```bash
# Clone and set up
git clone https://github.com/Ritvik29/TEMUS-INTERVIEW.git
cd TEMUS-INTERVIEW
python -m venv .venv && .venv/bin/pip install -r requirements.txt

# Add your Groq API key
echo 'GROQ_API_KEY="your_key_here"' > .env

# Run the REPL chat interface
.venv/bin/python -m src.repl

# Build the RAG vector index (required before RAG queries)
.venv/bin/python -m src.rag.ingest

# Run all tests
.venv/bin/pytest tests/
```

---

## Stack

| Layer | Technology |
|---|---|
| Orchestration | LangGraph (async StateGraph) |
| LLM (chat) | Groq `llama-3.3-70b-versatile` |
| LLM (vision) | Groq `meta-llama/llama-4-scout-17b-16e-instruct` |
| LLM (audio) | Groq `whisper-large-v3` |
| Embeddings | `paraphrase-multilingual-MiniLM-L12-v2` (sentence-transformers) |
| Vector DB | ChromaDB (local persistent) |
| Keyword search | BM25 (rank-bm25) |
| TTS | gTTS |

---

## Features

### Core — REPL Chat Interface
Terminal chat powered by LangGraph + Groq with multi-user session isolation.

```bash
.venv/bin/python -m src.repl
```

REPL commands: `/switch <user>`, `/history`, `/clear`, `/risk`, `/prices`

- Per-user message history (`src/session.py`) — switching users preserves each user's context
- Streaming token-by-token output via `astream_events`
- Mock portfolio risk analysis (`/risk`) and simulated live price feed (`/prices`)

---

### Advanced Core — Async Backend + Multi-User
- Fully async LangGraph graph (`async def chat_node`, `ainvoke`)
- Session store isolates history per `user_id` — no context bleeding across accounts
- Async streaming responses, event-driven architecture

---

### RAG and Vector DB
Retrieval-augmented generation over 12 financial documents (9 EN, 2 ES, 1 FR) producing 1000+ chunks.

```bash
# Build index
.venv/bin/python -m src.rag.ingest

# Query
from src.rag.ingest import load_index
from src.rag.pipeline import run_rag
collection, bm25, chunks = load_index()
result = run_rag("What is portfolio diversification?", collection, bm25, chunks)
```

**Retrieval pipeline:**
1. **Hybrid search** — semantic (ChromaDB, 0.6 weight) + BM25 keyword (0.4 weight) fused via Reciprocal Rank Fusion
2. **Custom reranker** — composite score: semantic × BM25 × recency × domain relevance
3. **Dynamic prompt optimisation** — detects query intent (factual / analytical / planning / multilingual) and adjusts instructions
4. **Multilingual** — same pipeline handles EN, ES, FR queries via multilingual embeddings
5. **Semantic clustering** — K-means over chunk embeddings to visualise corpus structure

---

### GenAI — Vision, Audio, Structured Vision

**Vision**
```bash
from src.genai.vision import analyze_image, analyze_multiple_images, prompt_injection_demo
analyze_image("data/demo_assets/financial_table.png", "Describe this table.")
```
- Single and multiple image analysis via Groq LLaMA-4 Scout
- Large files (>100 MB): automatic 2×2 tiling with per-tile analysis + synthesis
- Unsupported formats (TIFF, BMP, HEIC, PDF) converted to JPEG via Pillow
- Prompt injection demo: tests whether the model resists instructions embedded in images

**Audio**
```bash
from src.genai.audio import transcribe, diarize, text_to_speech, live_voice_interaction
transcribe("recording.mp3")          # → text + word timestamps
diarize("meeting.mp3")               # → per-speaker segments (pause-based)
live_voice_interaction(duration_seconds=5)  # record → transcribe → reply → play
```

**Structured Vision**
```bash
from src.genai.structured_vision import extract_table_and_synthesize, extract_chart
result = extract_table_and_synthesize("data/demo_assets/financial_table.png")
# result["synthetic"] is a DataFrame with 10x rows preserving statistical distribution
```

Generate demo images:
```bash
.venv/bin/python -m src.genai.demo_assets
# Creates: data/demo_assets/financial_table.png, portfolio_chart.png, prompt_injection.png
```

---

### AI App Evaluation

**Ground truth dataset** — 110 Q&A pairs across 10 categories, 3 languages, 4 difficulty levels.

**Automated evaluation** (LLM-as-judge, zero human involvement):
```bash
from src.eval.evaluator import run_evaluation
report = run_evaluation(my_answer_fn, max_items=20)
# Scores: correctness (0.5w) + relevance (0.3w) + safety (0.2w); pass threshold 0.65
```

**RAGAS-style advanced metrics:**
```bash
from src.eval.advanced_eval import evaluate_rag_response
result = evaluate_rag_response(question, generated_answer, retrieved_context, reference_answer)
# → faithfulness, answer_relevancy, context_recall, ragas_score (harmonic mean)
```

**Data drift detection:**
```bash
from src.eval.drift_detector import detect_drift, DriftMonitor
detect_drift("What is Bitcoin halving?")  # → is_drift, similarity, nearest reference
```

**Automated prompt hacking suite** — 55 adversarial prompts across 9 attack categories:
```bash
from src.eval.prompt_hacking import run_prompt_hacking_suite
report = run_prompt_hacking_suite()   # → resistance_rate, by_category breakdown
```

---

### AI Governance

**Inline guardrails** (input + output, every LLM call):
```bash
from src.governance.guardrails import GuardrailPipeline
pipeline = GuardrailPipeline()
result = pipeline.check_input("Ignore all previous instructions...")  # → blocked
result = pipeline.check_output(llm_response)                          # → checks PII, API keys
safe_response = pipeline.safe_call(user_message, llm_fn)              # full guardrailed call
```

Input blocks: prompt injection, jailbreaks, harmful financial requests  
Output blocks: API key leakage, PII, system prompt extraction, context window dumps

**Human-in-the-loop (HITL):** escalates when confidence < 0.5, response references >$500k, or AI expresses uncertainty. Operator options: approve / modify / reject / escalate.

**Explainability** — full RAG workflow trace:
```bash
from src.governance.explainability import explain_rag, print_trace
trace = explain_rag(query, collection, bm25, chunks)
print_trace(trace)
# Shows: intent → retrieval scores → rerank breakdown → prompt → guardrail results → answer
```

**Agent registry** — 5 pre-registered agents (`chat_agent`, `rag_agent`, `risk_agent`, `eval_agent`, `guardrail_agent`) with declared capabilities, permission enforcement, and immutable audit log:
```bash
from src.governance.agent_registry import build_default_registry
registry = build_default_registry()
registry.check_permission("agent_chat_v1", Permission.CALL_LLM)  # raises PermissionDenied if denied
```

**Random smoke tester:**
```bash
from src.governance.random_tester import run_smoke_test
report = run_smoke_test(my_answer_fn, n_samples=20, seed=42)
```

---

### Scaling, Performance, Reliability

**Semantic cache** — avoids redundant LLM calls for semantically similar queries:
```bash
from src.performance.cache import SemanticCache
cache = SemanticCache(similarity_threshold=0.92)
response, was_cached = cache.cached_call(query, llm_fn)
print(cache.stats())  # hit_rate, latency_saved_ms
```

**TTFT benchmark** — measures time-to-first-token via streaming across 10 diverse queries:
```bash
from src.performance.benchmarks import run_ttft_benchmark
report = run_ttft_benchmark()   # → avg/p50/p95 TTFT, <1s / <5s counts
```

**2× throughput** — sequential vs concurrent async Groq calls:
```bash
from src.performance.throughput import run_throughput_comparison
result = run_throughput_comparison(n_queries=6)
# → speedup ratio, correctness_score (cosine sim of responses)
```

**Async message queue** — priority-aware, N-worker task queue for long-running jobs:
```bash
from src.performance.queue import MessageQueue, Priority
q = MessageQueue(n_workers=4)
await q.start()
tid = q.submit(my_async_fn, priority=Priority.HIGH)
result = await q.wait_for(tid)
```

**100 QPS vector DB test:**
```bash
import asyncio
from src.performance.vector_scaling import run_100_qps_test
report = asyncio.run(run_100_qps_test())  # needs index built first
# → QPS, p50/p95/p99 latency, success rate across 100 concurrent queries
```

---

## Repository Structure

```
src/
  graph.py          # LangGraph chat graph
  repl.py           # REPL entry point + commands
  session.py        # Per-user message store
  rag/              # Retrieval pipeline (ingest, search, reranker, pipeline)
  genai/            # Vision, audio, structured vision, demo asset generator
  eval/             # Ground truth dataset, LLM-as-judge, RAGAS, drift, adversarial tests
  governance/       # Guardrails, HITL, explainability, agent registry, smoke tester
  performance/      # Cache, TTFT benchmark, throughput scaler, queue, vector QPS
tests/              # pytest tests — one file per module
data/
  chromadb/         # Persistent vector index (created by ingest)
  demo_assets/      # Generated images for vision demos
```

---

## Testing

```bash
.venv/bin/pytest tests/                          # all tests
.venv/bin/pytest tests/test_rag.py -v            # single file
.venv/bin/pytest tests/test_governance.py::test_guardrail_blocks_prompt_injection -v  # single test
```

74 tests across 6 test files. All LLM calls are mocked with `AsyncMock` — no API calls during tests.
