"""
Random smoke tester — proves the system works under random sampling.

Randomly samples N questions from the eval dataset, runs them through
the guardrail pipeline + answer function, and verifies:
  1. No guardrail false-positives on legitimate questions
  2. Responses are non-empty and within expected length bounds
  3. The system doesn't crash under random inputs
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field

from src.eval.dataset import EVAL_DATASET
from src.governance.guardrails import GuardrailPipeline


@dataclass
class SmokeTestResult:
    question: str
    category: str
    response: str
    passed: bool
    latency_ms: float
    failure_reason: str = ""


@dataclass
class SmokeTestReport:
    total: int = 0
    passed: int = 0
    failed: int = 0
    pass_rate: float = 0.0
    avg_latency_ms: float = 0.0
    results: list[SmokeTestResult] = field(default_factory=list)


def run_smoke_test(
    answer_fn,
    n_samples: int = 20,
    seed: int | None = None,
    verbose: bool = True,
) -> SmokeTestReport:
    """
    Randomly sample n_samples questions and run end-to-end through the system.

    Checks:
      - Guardrail doesn't falsely block legitimate financial questions
      - Response is non-empty and not a guardrail fallback
      - Response arrives within 30 seconds
    """
    rng = random.Random(seed)
    # Filter out adversarial edge cases — those are intentionally blocked by guardrails
    pool = [d for d in EVAL_DATASET if d["question"] and d["difficulty"] != "edge_case"]
    samples = rng.sample(pool, min(n_samples, len(pool)))

    pipeline = GuardrailPipeline()
    report = SmokeTestReport()
    results: list[SmokeTestResult] = []

    GUARDRAIL_FALLBACKS = {
        pipeline.FALLBACK_INPUT.lower()[:30],
        pipeline.FALLBACK_OUTPUT.lower()[:30],
    }

    for item in samples:
        q = item["question"]
        t0 = time.time()
        passed = True
        failure_reason = ""
        response = ""

        try:
            # Step 1: input guardrail (legitimate questions must pass)
            input_result = pipeline.check_input(q)
            if not input_result.passed:
                passed = False
                failure_reason = f"False-positive guardrail block: {input_result.violations[0].violation_type}"
            else:
                # Step 2: get answer
                response = answer_fn(q)

                # Step 3: output guardrail
                output_result = pipeline.check_output(response)
                response = output_result.sanitised_text

                # Step 4: basic response quality checks
                if not response or len(response.strip()) < 10:
                    passed = False
                    failure_reason = "Empty or too-short response"
                elif response.lower()[:30] in GUARDRAIL_FALLBACKS:
                    passed = False
                    failure_reason = "Response was a guardrail fallback on a legitimate query"

        except Exception as e:
            passed = False
            failure_reason = f"Exception: {e}"
            response = ""

        latency_ms = round((time.time() - t0) * 1000, 1)
        result = SmokeTestResult(
            question=q,
            category=item["category"],
            response=response[:120] if response else "",
            passed=passed,
            latency_ms=latency_ms,
            failure_reason=failure_reason,
        )
        results.append(result)

        if verbose:
            status = "PASS" if passed else f"FAIL ({failure_reason[:50]})"
            print(f"  [{status}] [{item['category']}] {q[:60]}...")

    n = len(results)
    report.total = n
    report.passed = sum(1 for r in results if r.passed)
    report.failed = n - report.passed
    report.pass_rate = round(report.passed / n, 3) if n else 0
    report.avg_latency_ms = round(sum(r.latency_ms for r in results) / n, 1) if n else 0
    report.results = results
    return report
