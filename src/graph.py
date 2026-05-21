import os
from typing import Annotated
from typing_extensions import TypedDict
from dotenv import load_dotenv
from langchain_core.messages import AnyMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

load_dotenv()

SYSTEM_PROMPT = """You are AuraWealth AI, a personal financial advisor assistant.
You help clients understand their net worth, track progress toward life goals, and make
informed financial decisions. You also support wealth advisors with portfolio analysis,
risk insights, and client management. Be concise, clear, and always grounded in the
user's financial context. Never give specific investment recommendations without
qualifying them as general information."""


class ChatState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


def build_graph(llm=None):
    if llm is None:
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=os.getenv("GROQ_API_KEY"),
        )

    async def chat_node(state: ChatState) -> dict:
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
        response = await llm.ainvoke(messages)
        return {"messages": [response]}

    graph = StateGraph(ChatState)
    graph.add_node("chat", chat_node)
    graph.set_entry_point("chat")
    graph.add_edge("chat", END)
    return graph.compile()
