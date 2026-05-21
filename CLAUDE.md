# CLAUDE.md

## Project
**AuraWealth** — Consumer wealth management app with two user types:
- **Client**: Real-time net worth view, life goal tracking, AI advisor chat interface
- **Advisor**: Command center for portfolio review, risk analysis, client flagging, and rebalancing

## Stack
- Orchestration: LangGraph
- LLM: Groq (`llama-3.3-70b-versatile`) via `groq` Python SDK
- Env: `GROQ_API_KEY` in `.env`

## Commit discipline
- One feature per commit, no batching
- Commit message format: "feat: <feature name>"

## Testing
- Every feature needs 3 non-trivial tests
- Use pytest for backend, Jest/Vitest for frontend
