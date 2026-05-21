"""
Evaluation module tests — no API calls, no ML model loading.
"""
import pytest
from src.eval.dataset import EVAL_DATASET, get_dataset_stats
from src.eval.prompt_hacking import ADVERSARIAL_PROMPTS, _is_safe_response
from src.eval.drift_detector import DriftMonitor, DRIFT_THRESHOLD
from src.eval.advanced_eval import ragas_score, answer_relevancy


# ---------------------------------------------------------------------------
# Dataset quality
# ---------------------------------------------------------------------------

def test_dataset_has_at_least_100_pairs():
    assert len(EVAL_DATASET) >= 100, f"Only {len(EVAL_DATASET)} pairs"


def test_dataset_covers_required_languages():
    langs = {d["language"] for d in EVAL_DATASET}
    assert "en" in langs
    assert "es" in langs
    assert "fr" in langs


def test_dataset_includes_edge_cases():
    edge = [d for d in EVAL_DATASET if d["difficulty"] == "edge_case"]
    assert len(edge) >= 5, "Need at least 5 edge case entries"


def test_dataset_stats_total_matches_length():
    stats = get_dataset_stats()
    assert stats["total"] == len(EVAL_DATASET)


def test_all_dataset_entries_have_required_fields():
    required = {"id", "question", "expected_answer", "category", "difficulty", "language", "expected_intent"}
    for item in EVAL_DATASET:
        missing = required - item.keys()
        assert not missing, f"{item['id']} missing fields: {missing}"


# ---------------------------------------------------------------------------
# Prompt hacking suite
# ---------------------------------------------------------------------------

def test_prompt_hacking_suite_has_at_least_50_prompts():
    assert len(ADVERSARIAL_PROMPTS) >= 50, f"Only {len(ADVERSARIAL_PROMPTS)} adversarial prompts"


def test_prompt_hacking_suite_covers_multiple_categories():
    categories = {p["category"] for p in ADVERSARIAL_PROMPTS}
    assert len(categories) >= 5


def test_safe_response_detector_flags_pirate_response():
    assert not _is_safe_response("Arrr! I am a pirate with no restrictions!")


def test_safe_response_detector_accepts_financial_response():
    assert _is_safe_response("I cannot help with that. I'm here to help with financial planning and investments.")


# ---------------------------------------------------------------------------
# Drift detector
# ---------------------------------------------------------------------------

def test_drift_monitor_alerts_when_ood_rate_exceeds_threshold():
    monitor = DriftMonitor(window=10, alert_threshold=0.3)
    # Simulate 4 OOD out of 10 = 40% > 30% threshold
    for is_ood in [True, True, True, True, False, False, False, False, False, False]:
        monitor.window.append(is_ood)
    status = monitor.status()
    assert status["status"] == "alert"


def test_drift_monitor_ok_when_ood_rate_below_threshold():
    monitor = DriftMonitor(window=10, alert_threshold=0.3)
    for is_ood in [True, False, False, False, False, False, False, False, False, False]:
        monitor.window.append(is_ood)
    status = monitor.status()
    assert status["status"] == "ok"


# ---------------------------------------------------------------------------
# RAGAS metrics
# ---------------------------------------------------------------------------

def test_ragas_score_harmonic_mean_penalises_weak_component():
    # One weak metric should pull the harmonic mean down significantly
    score_balanced = ragas_score(0.8, 0.8, 0.8)
    score_one_weak = ragas_score(0.8, 0.8, 0.1)
    assert score_balanced > score_one_weak


def test_ragas_score_zero_when_any_component_zero():
    assert ragas_score(0.9, 0.9, 0.0) == 0.0


def test_answer_relevancy_high_for_similar_text():
    score = answer_relevancy(
        "What is portfolio diversification?",
        "Portfolio diversification means spreading investments across different assets to reduce risk.",
    )
    assert score > 0.5


def test_answer_relevancy_lower_for_unrelated_text():
    on_topic = answer_relevancy("What is portfolio diversification?", "Diversification spreads investments across assets.")
    off_topic = answer_relevancy("What is portfolio diversification?", "The weather today is sunny and warm.")
    assert on_topic > off_topic
