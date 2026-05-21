# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project
**AuraWealth** — Consumer wealth management app with two user types:
- **Client**: Real-time net worth view, life goal tracking, AI advisor chat interface
- **Advisor**: Command center for portfolio review, risk analysis, client flagging, and rebalancing

## Stack
- Orchestration: LangGraph (async graph, `ainvoke` / `astream_events`)
- LLM: Groq `llama-3.3-70b-versatile` (chat) and `whisper-large-v3` (audio) via `groq` SDK
- Vision: Groq `meta-llama/llama-4-scout-17b-16e-instruct`
- Embeddings: `paraphrase-multilingual-MiniLM-L12-v2` via `sentence-transformers`
- Vector DB: ChromaDB (persistent, local at `data/chromadb/`)
- Keyword search: BM25 via `rank-bm25`
- Python venv: `.venv/` — always use `.venv/bin/python` and `.venv/bin/pytest`
- Env: `GROQ_API_KEY` in `.env` (loaded via `load_dotenv()` at module level in each file that needs it)

## Commands

```bash
# Install dependencies
python -m venv .venv && .venv/bin/pip install -r requirements.txt

# Run all tests
.venv/bin/pytest tests/

# Run a single test file
.venv/bin/pytest tests/test_rag.py -v

# Run a single test by name
.venv/bin/pytest tests/test_governance.py::test_guardrail_blocks_prompt_injection -v

# Run the REPL chat interface
.venv/bin/python -m src.repl

# Build the RAG vector index (required before RAG queries)
.venv/bin/python -m src.rag.ingest

# Generate demo assets (table/chart/injection images)
.venv/bin/python -m src.genai.demo_assets
```

## Architecture

The codebase is split into six independent feature modules under `src/`:

**`src/graph.py` + `src/repl.py` + `src/session.py`** — Core chat layer. `graph.py` defines the LangGraph `StateGraph` with a single async `chat_node`. `session.py` is an in-memory store mapping `user_id → [messages]` for multi-turn, multi-user isolation. `repl.py` is the terminal entry point with commands: `/switch`, `/history`, `/clear`, `/risk`, `/prices`.

**`src/rag/`** — Retrieval-augmented generation pipeline.
- `documents.py` — 12 synthetic financial docs (EN/ES/FR)
- `ingest.py` — chunks (100-char size, 50-char overlap), embeds, stores in ChromaDB + BM25 pickle at `data/`
- `search.py` — semantic search, BM25 keyword search with domain upweighting, hybrid RRF fusion
- `reranker.py` — composite score: semantic × BM25 × recency × domain boost
- `pipeline.py` — intent detection (factual/analytical/planning/multilingual), prompt construction, K-means clustering

**`src/genai/`** — Multimodal features.
- `vision.py` — image analysis via Groq vision; large-file tiling (>100 MB), format conversion, prompt injection demo
- `audio.py` — Whisper transcription, speaker diarization, TTS via gTTS, live mic interaction
- `structured_vision.py` — table/chart extraction → pandas DataFrame, 10x synthetic data generation
- `demo_assets.py` — generates `data/demo_assets/` images for testing

**`src/eval/`** — Automated evaluation.
- `dataset.py` — 100 ground-truth Q&A pairs (10 categories, EN/ES/FR, easy/medium/hard/edge_case)
- `evaluator.py` — LLM-as-judge pipeline: correctness × relevance × safety → composite score
- `drift_detector.py` — cosine similarity vs reference distribution; `DriftMonitor` with rolling window alert
- `prompt_hacking.py` — 55 adversarial prompts across 9 attack categories; `_is_safe_response()` checker
- `advanced_eval.py` — RAGAS-style: Faithfulness, Answer Relevancy, Context Recall → harmonic mean

**`src/governance/`** — AI safety and governance.
- `guardrails.py` — `GuardrailPipeline` with input (injection/jailbreak/harmful) and output (API key/PII/exfiltration) pattern matching
- `hitl.py` — Human-in-the-loop: escalates on low confidence, large dollar amounts, or uncertainty phrases; operator approve/modify/reject/escalate
- `random_tester.py` — smoke tests random eval-dataset samples through the full guardrail pipeline
- `explainability.py` — full RAG workflow trace: intent → retrieval scores → rerank breakdown → prompt → guardrail results
- `agent_registry.py` — 5 pre-registered agents with IDs, capability declarations, permission enforcement, immutable audit log

**`src/performance/`** — Scaling and reliability.
- `cache.py` — `SemanticCache`: cosine similarity lookup, LRU eviction, hit-rate tracking
- `benchmarks.py` — TTFT (time-to-first-token) measurement via streaming across 10 diverse queries
- `throughput.py` — sequential vs concurrent async Groq calls; correctness verified via embedding similarity
- `queue.py` — `MessageQueue`: priority-aware asyncio queue, N workers, timeout handling, audit trail
- `vector_scaling.py` — 100 concurrent ChromaDB searches via `run_in_executor`; QPS + latency percentiles

## Commit discipline
- One feature per commit, no batching
- Commit message format: `feat: <Feature Name>`

## Testing
- Every feature needs 3 non-trivial tests
- Use pytest; async tests need `@pytest.mark.asyncio`
- Mock LLM calls with `AsyncMock` returning `AIMessage` — never hit the real API in tests
- The pytest-asyncio mode is `STRICT` — all async tests must have the decorator
