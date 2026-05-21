import asyncio
import random
from langchain_core.messages import HumanMessage
from src.graph import build_graph
from src.session import get_session, update_session, clear_session

MOCK_PORTFOLIO = {
    "alice": {"AAPL": 50, "MSFT": 30, "TSLA": 20, "AMZN": 10},
    "bob":   {"GOOG": 100, "META": 40, "NFLX": 15},
}

HELP_TEXT = """
Commands:
  /switch <username>  — switch to another user (preserves each user's history)
  /history            — show current conversation history
  /clear              — clear your conversation history
  /risk               — run AI risk analysis on your mock portfolio
  /prices             — stream live simulated price ticks
  quit / exit         — end the session
"""


async def stream_response(graph, state: dict) -> dict:
    """Stream tokens to stdout as they arrive, return updated state."""
    print("\nAuraWealth: ", end="", flush=True)
    final_state = None
    async for event in graph.astream_events(state, version="v2"):
        if event["event"] == "on_chat_model_stream":
            chunk = event["data"]["chunk"].content
            if chunk:
                print(chunk, end="", flush=True)
        elif event["event"] == "on_chain_end" and event["name"] == "LangGraph":
            final_state = event["data"]["output"]
    print("\n")
    return final_state if final_state else state


async def run_risk_analysis(graph, user_id: str, state: dict) -> dict:
    portfolio = MOCK_PORTFOLIO.get(user_id, {"SPY": 100})
    holdings = ", ".join(f"{qty} shares of {ticker}" for ticker, qty in portfolio.items())
    prompt = (
        f"Perform a brief risk analysis for a portfolio with: {holdings}. "
        "Assess concentration risk, sector exposure, and volatility. "
        "Give 2-3 actionable suggestions."
    )
    state["messages"].append(HumanMessage(content=prompt))
    return await stream_response(graph, state)


async def stream_prices(user_id: str):
    """Simulate a real-time price feed for the user's portfolio."""
    portfolio = MOCK_PORTFOLIO.get(user_id, {"SPY": 100})
    tickers = list(portfolio.keys())
    prices = {t: round(random.uniform(100, 500), 2) for t in tickers}

    print("\n[Price Stream — press Ctrl+C to stop]\n")
    try:
        for _ in range(10):
            for ticker in tickers:
                change = random.uniform(-3, 3)
                prices[ticker] = round(prices[ticker] + change, 2)
                direction = "▲" if change >= 0 else "▼"
                print(f"  {ticker:6s}  ${prices[ticker]:>8.2f}  {direction} {abs(change):.2f}")
            print()
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        pass
    print("[Stream ended]\n")


async def run_repl():
    graph = build_graph()

    print("=" * 50)
    print("  AuraWealth AI — Financial Advisor")
    print("=" * 50)
    user_id = input("Enter your username: ").strip() or "guest"
    print(f"\nWelcome, {user_id}!{HELP_TEXT}")

    while True:
        try:
            user_input = input(f"[{user_id}] You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        if user_input.startswith("/switch "):
            user_id = user_input.split(" ", 1)[1].strip() or user_id
            print(f"Switched to user: {user_id}\n")
            continue

        if user_input == "/history":
            messages = get_session(user_id)
            if not messages:
                print("  (no history yet)\n")
            for m in messages:
                role = "You" if isinstance(m, HumanMessage) else "AuraWealth"
                print(f"  {role}: {m.content}")
            print()
            continue

        if user_input == "/clear":
            clear_session(user_id)
            print("History cleared.\n")
            continue

        if user_input == "/risk":
            state = {"messages": get_session(user_id)}
            state = await run_risk_analysis(graph, user_id, state)
            update_session(user_id, state["messages"])
            continue

        if user_input == "/prices":
            price_task = asyncio.create_task(stream_prices(user_id))
            try:
                await price_task
            except asyncio.CancelledError:
                price_task.cancel()
            continue

        messages = get_session(user_id)
        messages.append(HumanMessage(content=user_input))
        state = {"messages": messages}
        state = await stream_response(graph, state)
        update_session(user_id, state["messages"])


def main():
    asyncio.run(run_repl())


if __name__ == "__main__":
    main()
