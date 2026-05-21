"""
In-memory session store: maps user_id -> conversation state.
Each user gets an isolated message history so context never bleeds across accounts.
"""
from langchain_core.messages import AnyMessage

_sessions: dict[str, list[AnyMessage]] = {}


def get_session(user_id: str) -> list[AnyMessage]:
    return _sessions.setdefault(user_id, [])


def update_session(user_id: str, messages: list[AnyMessage]) -> None:
    _sessions[user_id] = messages


def clear_session(user_id: str) -> None:
    _sessions.pop(user_id, None)


def list_users() -> list[str]:
    return list(_sessions.keys())
