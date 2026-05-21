import pytest
from unittest.mock import MagicMock, AsyncMock
from langchain_core.messages import HumanMessage, AIMessage
from src.graph import build_graph
from src.session import get_session, update_session, clear_session
from src.repl import stream_prices, MOCK_PORTFOLIO


# --- Async graph tests ---

def make_async_mock_llm(response_text: str) -> MagicMock:
    llm = MagicMock()
    llm.ainvoke = AsyncMock(return_value=AIMessage(content=response_text))
    return llm


@pytest.mark.asyncio
async def test_async_graph_returns_ai_response():
    llm = make_async_mock_llm("Here is your financial summary.")
    graph = build_graph(llm=llm)

    result = await graph.ainvoke({"messages": [HumanMessage(content="Summarize my finances.")]})

    assert isinstance(result["messages"][-1], AIMessage)
    assert result["messages"][-1].content == "Here is your financial summary."


@pytest.mark.asyncio
async def test_async_graph_uses_ainvoke_not_invoke():
    """Graph node must call ainvoke (async), not the sync invoke."""
    llm = make_async_mock_llm("Async response.")
    graph = build_graph(llm=llm)

    result = await graph.ainvoke({"messages": [HumanMessage(content="Hello")]})

    llm.ainvoke.assert_called_once()
    assert result["messages"][-1].content == "Async response."


# --- Session / multi-turn tests ---

def test_session_accumulates_multi_turn_history():
    """Multi-turn history is preserved via the session store, not the graph state."""
    clear_session("test_mt")

    turn1 = [HumanMessage(content="My retirement goal is 55."), AIMessage(content="Got it.")]
    update_session("test_mt", turn1)

    session = get_session("test_mt")
    session.append(HumanMessage(content="What was my goal?"))
    session.append(AIMessage(content="Retire at 55."))
    update_session("test_mt", session)

    final = get_session("test_mt")
    assert len([m for m in final if isinstance(m, HumanMessage)]) == 2
    assert len([m for m in final if isinstance(m, AIMessage)]) == 2


def test_separate_users_have_isolated_sessions():
    clear_session("user_a")
    clear_session("user_b")

    update_session("user_a", [HumanMessage(content="I have $100k saved.")])
    update_session("user_b", [HumanMessage(content="I have $500k saved.")])

    assert get_session("user_a")[0].content == "I have $100k saved."
    assert get_session("user_b")[0].content == "I have $500k saved."
    assert get_session("user_a") is not get_session("user_b")


# --- Risk analysis tests ---

@pytest.mark.asyncio
async def test_risk_analysis_sends_portfolio_to_llm():
    """Risk analysis must include the user's holdings in the prompt sent to the LLM."""
    llm = make_async_mock_llm("High concentration in tech stocks.")
    graph = build_graph(llm=llm)

    from src.repl import run_risk_analysis
    state = {"messages": []}
    await run_risk_analysis(graph, "alice", state)

    prompt_sent = llm.ainvoke.call_args[0][0]
    prompt_text = " ".join(m.content for m in prompt_sent)
    for ticker in MOCK_PORTFOLIO["alice"]:
        assert ticker in prompt_text


@pytest.mark.asyncio
async def test_risk_analysis_falls_back_to_spy_for_unknown_user():
    """Unknown users get a default SPY portfolio, not an error."""
    llm = make_async_mock_llm("Diversified.")
    graph = build_graph(llm=llm)

    from src.repl import run_risk_analysis
    state = {"messages": []}
    await run_risk_analysis(graph, "unknown_user_xyz", state)

    prompt_sent = llm.ainvoke.call_args[0][0]
    prompt_text = " ".join(m.content for m in prompt_sent)
    assert "SPY" in prompt_text


# --- Price stream tests ---

@pytest.mark.asyncio
async def test_price_stream_covers_all_user_tickers(capsys):
    """Price stream must print every ticker in the user's portfolio."""
    await stream_prices("alice")

    output = capsys.readouterr().out
    for ticker in MOCK_PORTFOLIO["alice"]:
        assert ticker in output


@pytest.mark.asyncio
async def test_price_stream_uses_default_for_unknown_user(capsys):
    """Unknown user gets the SPY fallback portfolio in the price stream."""
    await stream_prices("unknown_xyz")

    output = capsys.readouterr().out
    assert "SPY" in output
