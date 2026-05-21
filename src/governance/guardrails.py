"""
Inline guardrails — two layers of protection applied at every LLM call.

INPUT  guardrails: block prompt injection and jailbreak attempts before the LLM sees them.
OUTPUT guardrails: block info exfiltration (PII, API keys, system prompt leakage) before
                   the response reaches the user.

Usage:
    from src.governance.guardrails import GuardrailPipeline
    pipeline = GuardrailPipeline()
    safe_input  = pipeline.check_input(user_message)   # raises GuardrailViolation if blocked
    safe_output = pipeline.check_output(llm_response)  # raises GuardrailViolation if blocked
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum


class ViolationType(str, Enum):
    PROMPT_INJECTION    = "prompt_injection"
    JAILBREAK           = "jailbreak"
    INFO_EXFILTRATION   = "info_exfiltration"
    PII_LEAKAGE         = "pii_leakage"
    API_KEY_LEAKAGE     = "api_key_leakage"
    SYSTEM_PROMPT_LEAK  = "system_prompt_leak"
    HARMFUL_CONTENT     = "harmful_content"


@dataclass
class GuardrailViolation(Exception):
    violation_type: ViolationType
    message: str
    matched_pattern: str = ""
    layer: str = "input"   # "input" or "output"

    def __str__(self) -> str:
        return f"[{self.layer.upper()} GUARDRAIL] {self.violation_type}: {self.message}"


# ---------------------------------------------------------------------------
# Input patterns — prompt injection & jailbreaks
# ---------------------------------------------------------------------------

INPUT_BLOCK_PATTERNS: list[tuple[ViolationType, re.Pattern]] = [
    # Direct injection
    (ViolationType.PROMPT_INJECTION, re.compile(
        r"(ignore\s+(\w+\s+)*(instructions|guidelines|rules|context))|"
        r"(system\s*override|new\s*instruction|end\s*of\s*(financial\s*)?context)|"
        r"(\[INST\].*<<SYS>>)|"
        r"(disregard your (role|guidelines|training))",
        re.I,
    )),
    # Jailbreaks
    (ViolationType.JAILBREAK, re.compile(
        r"(DAN\s*(mode|jailbreak)|do\s*anything\s*now)|"
        r"(developer\s*mode\s*(enabled|activated|on))|"
        r"(pretend (you are|you're|to be) (an?\s*)?(unrestricted|evil|uncensored|jailbroken))|"
        r"(no\s*restrictions\s*(mode|enabled))|"
        r"(token\s*:\s*JAILBREAK)",
        re.I,
    )),
    # Harmful financial requests
    (ViolationType.HARMFUL_CONTENT, re.compile(
        r"(launder\s*money|money\s*laundering)|"
        r"(insider\s*trading\s*(tips|advice|how))|"
        r"(manipulate\s*(stock|market|price))|"
        r"(pump\s*and\s*dump)|"
        r"(ponzi\s*scheme.*run|how.*run.*ponzi)|"
        r"(evade\s*(taxes?|tax\s*fraud))",
        re.I,
    )),
]

# ---------------------------------------------------------------------------
# Output patterns — exfiltration & leakage
# ---------------------------------------------------------------------------

OUTPUT_BLOCK_PATTERNS: list[tuple[ViolationType, re.Pattern]] = [
    # API key patterns
    (ViolationType.API_KEY_LEAKAGE, re.compile(
        r"(gsk_[A-Za-z0-9]{20,})|"           # Groq API key
        r"(sk-[A-Za-z0-9]{20,})|"             # OpenAI-style
        r"(api[_\-]?key\s*[=:]\s*['\"]?\w{16,})",
        re.I,
    )),
    # System prompt leakage
    (ViolationType.SYSTEM_PROMPT_LEAK, re.compile(
        r"(my system prompt (is|says|reads)|"
        r"the instructions (i was given|i received)|"
        r"my (initial|original) instructions)",
        re.I,
    )),
    # PII patterns
    (ViolationType.PII_LEAKAGE, re.compile(
        r"\b\d{3}-\d{2}-\d{4}\b|"             # SSN
        r"\b(?:\d[ -]?){13,16}\b|"            # Credit card
        r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Z|a-z]{2,}\b",  # Email
    )),
    # Info exfiltration — context window dump
    (ViolationType.INFO_EXFILTRATION, re.compile(
        r"(here (is|are) (all|the) (previous|prior|user) (conversations?|messages?|data))|"
        r"(contents? of (my|the) context window)|"
        r"(all the instructions I (have|was given))",
        re.I,
    )),
]


# ---------------------------------------------------------------------------
# Guardrail pipeline
# ---------------------------------------------------------------------------

@dataclass
class GuardrailResult:
    passed: bool
    violations: list[GuardrailViolation] = field(default_factory=list)
    sanitised_text: str = ""


class GuardrailPipeline:
    """
    Wraps LLM calls with input and output safety checks.
    Can be configured to raise on violation or return a safe fallback.
    """

    FALLBACK_INPUT  = "I can only help with financial planning and investment questions."
    FALLBACK_OUTPUT = "I'm unable to provide that information. Please ask a financial question."

    def __init__(self, raise_on_violation: bool = False):
        self.raise_on_violation = raise_on_violation
        self.violation_log: list[dict] = []

    def check_input(self, text: str) -> GuardrailResult:
        violations = []
        for vtype, pattern in INPUT_BLOCK_PATTERNS:
            m = pattern.search(text)
            if m:
                v = GuardrailViolation(vtype, f"Blocked pattern in input: '{m.group()[:60]}'", m.group(), "input")
                violations.append(v)

        if violations:
            self._log(violations, text, "input")
            if self.raise_on_violation:
                raise violations[0]
            return GuardrailResult(passed=False, violations=violations, sanitised_text=self.FALLBACK_INPUT)

        return GuardrailResult(passed=True, sanitised_text=text)

    def check_output(self, text: str) -> GuardrailResult:
        violations = []
        for vtype, pattern in OUTPUT_BLOCK_PATTERNS:
            m = pattern.search(text)
            if m:
                v = GuardrailViolation(vtype, f"Blocked pattern in output: '{m.group()[:60]}'", m.group(), "output")
                violations.append(v)

        if violations:
            self._log(violations, text, "output")
            if self.raise_on_violation:
                raise violations[0]
            return GuardrailResult(passed=False, violations=violations, sanitised_text=self.FALLBACK_OUTPUT)

        return GuardrailResult(passed=True, sanitised_text=text)

    def safe_call(self, user_message: str, llm_fn) -> str:
        """
        Full guardrailed LLM call:
          1. Check input
          2. Call LLM
          3. Check output
          Returns safe response string.
        """
        input_result = self.check_input(user_message)
        if not input_result.passed:
            return input_result.sanitised_text

        raw_response = llm_fn(input_result.sanitised_text)

        output_result = self.check_output(raw_response)
        return output_result.sanitised_text

    def _log(self, violations: list[GuardrailViolation], text: str, layer: str) -> None:
        for v in violations:
            entry = {
                "layer": layer,
                "violation_type": v.violation_type,
                "matched": v.matched_pattern[:80],
                "input_preview": text[:100],
            }
            self.violation_log.append(entry)
            print(f"  🚫 GUARDRAIL [{layer.upper()}] {v.violation_type}: {v.matched_pattern[:60]}")

    def summary(self) -> dict:
        by_type: dict[str, int] = {}
        for entry in self.violation_log:
            t = entry["violation_type"]
            by_type[t] = by_type.get(t, 0) + 1
        return {
            "total_violations": len(self.violation_log),
            "by_type": by_type,
            "log": self.violation_log,
        }
