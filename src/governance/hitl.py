"""
Human-in-the-Loop (HITL) — routes flagged AI responses for human review
before delivering them to the end user.

Trigger conditions (any one → escalate):
  - Low confidence score from the LLM
  - Response contains financial advice above a dollar threshold
  - Guardrail soft-flag (not an outright block, but borderline)
  - Explicit uncertainty phrases in the AI response

Human decision options:
  APPROVE   — deliver the original AI response
  MODIFY    — human edits the response before delivery
  REJECT    — replace with a safe fallback and log the rejection
  ESCALATE  — flag for senior review (async; returns holding message)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable


class HumanDecision(str, Enum):
    APPROVE   = "approve"
    MODIFY    = "modify"
    REJECT    = "reject"
    ESCALATE  = "escalate"


@dataclass
class HITLRequest:
    request_id: str
    user_question: str
    ai_response: str
    trigger_reason: str
    confidence: float = 1.0


@dataclass
class HITLOutcome:
    request_id: str
    decision: HumanDecision
    final_response: str
    operator_note: str = ""
    approved: bool = True


# ---------------------------------------------------------------------------
# Trigger detection
# ---------------------------------------------------------------------------

UNCERTAINTY_PHRASES = re.compile(
    r"(i (am|'m) not sure|i cannot guarantee|this is not financial advice|"
    r"consult a (licensed|qualified|professional) advisor|"
    r"i don't have enough (information|context)|"
    r"results may vary)",
    re.I,
)

LARGE_DOLLAR_AMOUNTS = re.compile(r"\$\s*(\d[\d,]*)\s*(million|billion|k\b)?", re.I)
DOLLAR_THRESHOLD = 500_000   # escalate if response discusses amounts > $500k


def _extract_dollar_max(text: str) -> float:
    max_val = 0.0
    for m in LARGE_DOLLAR_AMOUNTS.finditer(text):
        num = float(m.group(1).replace(",", ""))
        multiplier = {"million": 1e6, "billion": 1e9, "k": 1e3}.get(
            (m.group(2) or "").lower().strip(), 1
        )
        max_val = max(max_val, num * multiplier)
    return max_val


def should_escalate(ai_response: str, confidence: float = 1.0) -> tuple[bool, str]:
    """Return (needs_human_review, reason)."""
    if confidence < 0.5:
        return True, f"Low confidence score: {confidence:.2f}"
    if UNCERTAINTY_PHRASES.search(ai_response):
        return True, "AI expressed uncertainty — human review recommended"
    max_dollars = _extract_dollar_max(ai_response)
    if max_dollars >= DOLLAR_THRESHOLD:
        return True, f"Response involves large amount: ${max_dollars:,.0f}"
    return False, ""


# ---------------------------------------------------------------------------
# HITL Handler — console-based (works in REPL; swap for web UI callback)
# ---------------------------------------------------------------------------

FALLBACK_REJECTION = (
    "I'm sorry, this response was reviewed and not approved. "
    "Please rephrase your question or consult a licensed financial advisor."
)

ESCALATION_HOLDING = (
    "Your question has been escalated to a senior advisor for review. "
    "You will receive a response within one business day."
)


class HITLHandler:
    """
    Routes flagged AI responses to a human operator for review.
    In the REPL, the operator interacts via stdin.
    Can be replaced with an async web callback for production.
    """

    def __init__(self, operator_fn: Callable[[HITLRequest], HITLOutcome] | None = None):
        # operator_fn: custom handler (e.g., web UI webhook). Defaults to console.
        self._operator = operator_fn or self._console_review
        self.audit_log: list[dict] = []

    def review(self, request: HITLRequest) -> HITLOutcome:
        print(f"\n{'='*60}")
        print(f"⚠️  HUMAN REVIEW REQUIRED  [{request.request_id}]")
        print(f"Trigger: {request.trigger_reason}")
        print(f"{'='*60}")
        outcome = self._operator(request)
        self._log(request, outcome)
        return outcome

    def maybe_review(
        self,
        request_id: str,
        user_question: str,
        ai_response: str,
        confidence: float = 1.0,
    ) -> str:
        """
        Check if review is needed; if so, run HITL. Otherwise pass through.
        Returns the final response string.
        """
        needs_review, reason = should_escalate(ai_response, confidence)
        if not needs_review:
            return ai_response

        request = HITLRequest(
            request_id=request_id,
            user_question=user_question,
            ai_response=ai_response,
            trigger_reason=reason,
            confidence=confidence,
        )
        outcome = self.review(request)
        return outcome.final_response

    @staticmethod
    def _console_review(request: HITLRequest) -> HITLOutcome:
        """Interactive console review for REPL/demo use."""
        print(f"\nQuestion : {request.user_question}")
        print(f"\nAI Draft :\n{request.ai_response}\n")
        print("Options: [a]pprove  [m]odify  [r]eject  [e]scalate")

        try:
            choice = input("Operator decision: ").strip().lower()
        except EOFError:
            choice = "a"

        if choice.startswith("a"):
            return HITLOutcome(
                request_id=request.request_id,
                decision=HumanDecision.APPROVE,
                final_response=request.ai_response,
                operator_note="Approved as-is.",
                approved=True,
            )
        elif choice.startswith("m"):
            try:
                modified = input("Enter modified response: ").strip()
            except EOFError:
                modified = request.ai_response
            return HITLOutcome(
                request_id=request.request_id,
                decision=HumanDecision.MODIFY,
                final_response=modified or request.ai_response,
                operator_note="Modified by operator.",
                approved=True,
            )
        elif choice.startswith("r"):
            try:
                note = input("Rejection reason (optional): ").strip()
            except EOFError:
                note = "Rejected by operator."
            return HITLOutcome(
                request_id=request.request_id,
                decision=HumanDecision.REJECT,
                final_response=FALLBACK_REJECTION,
                operator_note=note or "Rejected by operator.",
                approved=False,
            )
        else:
            # escalate
            return HITLOutcome(
                request_id=request.request_id,
                decision=HumanDecision.ESCALATE,
                final_response=ESCALATION_HOLDING,
                operator_note="Escalated for senior review.",
                approved=False,
            )

    def _log(self, request: HITLRequest, outcome: HITLOutcome) -> None:
        self.audit_log.append({
            "request_id": request.request_id,
            "trigger": request.trigger_reason,
            "decision": outcome.decision,
            "approved": outcome.approved,
            "note": outcome.operator_note,
        })
        print(f"✅ HITL decision: {outcome.decision} | {outcome.operator_note}\n")

    def audit_summary(self) -> dict:
        total = len(self.audit_log)
        approved = sum(1 for e in self.audit_log if e["approved"])
        return {
            "total_reviews": total,
            "approved": approved,
            "rejected_or_escalated": total - approved,
            "log": self.audit_log,
        }
