import pytest
from unittest.mock import MagicMock, AsyncMock
from langchain_core.messages import HumanMessage, AIMessage
from src.graph import build_graph, ChatState


def make_mock_llm(response_text: str) -> MagicMock:
    llm = MagicMock()
    llm.ainvoke = AsyncMock(return_value=AIMessage(content=response_text))
    return llm


def test_graph_compiles_and_has_chat_node():
    graph = build_graph(llm=make_mock_llm("hello"))
    assert "chat" in graph.get_graph().nodes


@pytest.mark.asyncio
async def test_graph_returns_ai_response_for_single_message():
    llm = make_mock_llm("Your net worth summary is ready.")
    graph = build_graph(llm=llm)

    result = await graph.ainvoke({"messages": [HumanMessage(content="What is my net worth?")]})

    assert len(result["messages"]) == 2  # human + AI
    assert isinstance(result["messages"][-1], AIMessage)
    assert result["messages"][-1].content == "Your net worth summary is ready."


@pytest.mark.asyncio
async def test_graph_includes_system_prompt_in_llm_call():
    """System prompt must be prepended to every LLM call."""
    from langchain_core.messages import SystemMessage
    llm = make_mock_llm("Acknowledged.")
    graph = build_graph(llm=llm)

    await graph.ainvoke({"messages": [HumanMessage(content="Hello")]})

    call_args = llm.ainvoke.call_args[0][0]
    assert isinstance(call_args[0], SystemMessage)
    assert "AuraWealth" in call_args[0].content
