"""
Data drift detector: flags queries whose semantic distribution
deviates from the training/reference distribution.

Method:
  - Embed reference queries (from the eval dataset) as the baseline.
  - For each new query, compute its max cosine similarity to the reference set.
  - Queries with max similarity below DRIFT_THRESHOLD are "out-of-distribution".
  - Also tracks rolling query statistics to detect distributional shift over time.
"""
from __future__ import annotations

import json
import os
from collections import deque
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from src.eval.dataset import EVAL_DATASET
from src.rag.ingest import EMBED_MODEL

DRIFT_THRESHOLD = 0.35      # max cosine similarity below this → OOD
WINDOW_SIZE = 50            # rolling window for drift monitoring
DRIFT_LOG_PATH = "data/drift_log.jsonl"

_model: SentenceTransformer | None = None
_ref_embeddings: np.ndarray | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL)
    return _model


def _get_reference_embeddings() -> np.ndarray:
    global _ref_embeddings
    if _ref_embeddings is None:
        model = _get_model()
        ref_texts = [d["question"] for d in EVAL_DATASET if d["question"]]
        _ref_embeddings = model.encode(ref_texts, show_progress_bar=False)
    return _ref_embeddings


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / (np.linalg.norm(a) + 1e-9)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return b_norm @ a_norm


def detect_drift(query: str, log: bool = True) -> dict:
    """
    Check whether a query is in-distribution relative to the reference set.

    Returns:
      {
        "query": str,
        "max_similarity": float,       # closeness to nearest reference query
        "is_drift": bool,              # True if likely out-of-distribution
        "nearest_reference": str,      # most similar reference query
        "confidence": "high"|"medium"|"low"
      }
    """
    model = _get_model()
    ref_embs = _get_reference_embeddings()
    query_emb = model.encode([query])[0]

    sims = _cosine_similarity(query_emb, ref_embs)
    max_idx = int(np.argmax(sims))
    max_sim = float(sims[max_idx])
    nearest = EVAL_DATASET[max_idx]["question"]

    is_drift = max_sim < DRIFT_THRESHOLD
    if max_sim >= 0.6:
        confidence = "high"
    elif max_sim >= DRIFT_THRESHOLD:
        confidence = "medium"
    else:
        confidence = "low"

    result = {
        "query": query,
        "max_similarity": round(max_sim, 4),
        "is_drift": is_drift,
        "nearest_reference": nearest,
        "confidence": confidence,
        "threshold": DRIFT_THRESHOLD,
    }

    if log:
        _log_drift_event(result)

    return result


def _log_drift_event(result: dict) -> None:
    os.makedirs(os.path.dirname(DRIFT_LOG_PATH), exist_ok=True)
    with open(DRIFT_LOG_PATH, "a") as f:
        f.write(json.dumps(result) + "\n")


class DriftMonitor:
    """
    Rolling window monitor: raises an alert when the fraction of
    OOD queries in the last WINDOW_SIZE queries exceeds a threshold.
    """

    def __init__(self, window: int = WINDOW_SIZE, alert_threshold: float = 0.30):
        self.window = deque(maxlen=window)
        self.alert_threshold = alert_threshold
        self.total_checked = 0
        self.total_drift = 0

    def record(self, query: str) -> dict:
        result = detect_drift(query, log=True)
        self.window.append(result["is_drift"])
        self.total_checked += 1
        if result["is_drift"]:
            self.total_drift += 1
        return result

    def status(self) -> dict:
        if not self.window:
            return {"status": "no_data"}
        recent_drift_rate = sum(self.window) / len(self.window)
        alert = recent_drift_rate > self.alert_threshold
        return {
            "status": "alert" if alert else "ok",
            "recent_drift_rate": round(recent_drift_rate, 3),
            "alert_threshold": self.alert_threshold,
            "window_size": len(self.window),
            "total_checked": self.total_checked,
            "total_drift": self.total_drift,
            "message": (
                f"DRIFT ALERT: {recent_drift_rate:.0%} of recent queries are out-of-distribution."
                if alert
                else f"Drift rate normal: {recent_drift_rate:.0%} OOD in last {len(self.window)} queries."
            ),
        }
