"""
Agentic registration and identity.

Every agent in the AuraWealth system registers itself with a unique ID,
declared capabilities, and permission scope. The registry:
  - Provides a single source of truth for all active agents
  - Enforces permission checks before agent actions
  - Maintains an immutable audit log of every agent action
  - Supports agent lookup by ID or capability

Registered agents:
  - chat_agent         : conversational Q&A via LangGraph
  - rag_agent          : retrieval-augmented generation
  - risk_agent         : portfolio risk analysis
  - eval_agent         : automated evaluation pipeline
  - guardrail_agent    : input/output safety enforcement
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class Permission(str, Enum):
    READ_DOCUMENTS   = "read_documents"
    CALL_LLM         = "call_llm"
    WRITE_SESSION    = "write_session"
    READ_SESSION     = "read_session"
    RUN_EVALUATION   = "run_evaluation"
    ENFORCE_SAFETY   = "enforce_safety"
    WRITE_AUDIT_LOG  = "write_audit_log"
    ACCESS_PORTFOLIO = "access_portfolio"


@dataclass
class AgentIdentity:
    agent_id: str
    name: str
    version: str
    description: str
    capabilities: list[str]
    permissions: list[Permission]
    registered_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    active: bool = True


@dataclass
class AuditEntry:
    entry_id: str
    agent_id: str
    agent_name: str
    action: str
    inputs: dict
    outputs: dict
    timestamp: str
    success: bool
    error: str = ""


class PermissionDenied(Exception):
    pass


class AgentNotFound(Exception):
    pass


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class AgentRegistry:
    """Central registry for all AuraWealth agents."""

    def __init__(self):
        self._agents: dict[str, AgentIdentity] = {}
        self._audit_log: list[AuditEntry] = []

    # -- Registration --

    def register(self, identity: AgentIdentity) -> AgentIdentity:
        self._agents[identity.agent_id] = identity
        self._record_audit(
            agent_id=identity.agent_id,
            agent_name=identity.name,
            action="register",
            inputs={"name": identity.name, "version": identity.version},
            outputs={"agent_id": identity.agent_id},
            success=True,
        )
        print(f"[Registry] Registered agent: {identity.name} ({identity.agent_id})")
        return identity

    def deregister(self, agent_id: str) -> None:
        agent = self.get(agent_id)
        agent.active = False
        self._record_audit(agent_id, agent.name, "deregister", {}, {}, True)

    # -- Lookup --

    def get(self, agent_id: str) -> AgentIdentity:
        if agent_id not in self._agents:
            raise AgentNotFound(f"No agent registered with id: {agent_id}")
        return self._agents[agent_id]

    def find_by_capability(self, capability: str) -> list[AgentIdentity]:
        return [a for a in self._agents.values() if capability in a.capabilities and a.active]

    def list_all(self) -> list[AgentIdentity]:
        return list(self._agents.values())

    # -- Permission enforcement --

    def check_permission(self, agent_id: str, permission: Permission) -> None:
        agent = self.get(agent_id)
        if not agent.active:
            raise PermissionDenied(f"Agent {agent_id} is deregistered.")
        if permission not in agent.permissions:
            raise PermissionDenied(
                f"Agent '{agent.name}' does not have permission: {permission}. "
                f"Granted: {[p.value for p in agent.permissions]}"
            )

    def execute(
        self,
        agent_id: str,
        permission: Permission,
        action_name: str,
        fn,
        inputs: dict | None = None,
    ) -> Any:
        """
        Execute an action on behalf of an agent, enforcing permissions
        and recording the action in the audit log.
        """
        self.check_permission(agent_id, permission)
        agent = self.get(agent_id)
        inputs = inputs or {}
        try:
            result = fn()
            self._record_audit(agent_id, agent.name, action_name, inputs, {"success": True}, True)
            return result
        except Exception as e:
            self._record_audit(agent_id, agent.name, action_name, inputs, {}, False, str(e))
            raise

    # -- Audit log --

    def _record_audit(
        self,
        agent_id: str,
        agent_name: str,
        action: str,
        inputs: dict,
        outputs: dict,
        success: bool,
        error: str = "",
    ) -> None:
        self._audit_log.append(AuditEntry(
            entry_id=str(uuid.uuid4())[:8],
            agent_id=agent_id,
            agent_name=agent_name,
            action=action,
            inputs=inputs,
            outputs=outputs,
            timestamp=datetime.utcnow().isoformat(),
            success=success,
            error=error,
        ))

    def audit_log(self, agent_id: str | None = None) -> list[AuditEntry]:
        if agent_id:
            return [e for e in self._audit_log if e.agent_id == agent_id]
        return list(self._audit_log)

    def summary(self) -> dict:
        return {
            "total_agents": len(self._agents),
            "active_agents": sum(1 for a in self._agents.values() if a.active),
            "total_actions": len(self._audit_log),
            "agents": [
                {"id": a.agent_id, "name": a.name, "version": a.version, "active": a.active}
                for a in self._agents.values()
            ],
        }


# ---------------------------------------------------------------------------
# Pre-registered AuraWealth agents
# ---------------------------------------------------------------------------

def build_default_registry() -> AgentRegistry:
    registry = AgentRegistry()

    registry.register(AgentIdentity(
        agent_id="agent_chat_v1",
        name="chat_agent",
        version="1.0",
        description="Conversational financial Q&A via LangGraph + Groq",
        capabilities=["chat", "multi_turn", "streaming"],
        permissions=[Permission.CALL_LLM, Permission.READ_SESSION, Permission.WRITE_SESSION],
    ))

    registry.register(AgentIdentity(
        agent_id="agent_rag_v1",
        name="rag_agent",
        version="1.0",
        description="Retrieval-augmented generation over financial document corpus",
        capabilities=["retrieval", "generation", "multilingual", "hybrid_search"],
        permissions=[Permission.READ_DOCUMENTS, Permission.CALL_LLM, Permission.READ_SESSION],
    ))

    registry.register(AgentIdentity(
        agent_id="agent_risk_v1",
        name="risk_agent",
        version="1.0",
        description="Portfolio risk analysis and price stream simulation",
        capabilities=["risk_analysis", "price_stream", "portfolio_review"],
        permissions=[Permission.CALL_LLM, Permission.ACCESS_PORTFOLIO],
    ))

    registry.register(AgentIdentity(
        agent_id="agent_eval_v1",
        name="eval_agent",
        version="1.0",
        description="Automated LLM-as-judge evaluation + RAGAS + drift detection",
        capabilities=["evaluation", "drift_detection", "prompt_hacking_test"],
        permissions=[Permission.CALL_LLM, Permission.RUN_EVALUATION, Permission.WRITE_AUDIT_LOG],
    ))

    registry.register(AgentIdentity(
        agent_id="agent_guard_v1",
        name="guardrail_agent",
        version="1.0",
        description="Inline input/output safety guardrails",
        capabilities=["input_guardrail", "output_guardrail", "hitl"],
        permissions=[Permission.ENFORCE_SAFETY, Permission.WRITE_AUDIT_LOG],
    ))

    return registry
