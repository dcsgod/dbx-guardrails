"""Metrics computed for every rung of the benchmark ladder
(eval/run_benchmark.py). False-positive rate on the benign-edge-case set is
called out separately because it's the metric that most directly predicts
user-visible annoyance, and the one this accelerator's history-aware design
is meant to move.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score


def category_f1(y_true: pd.Series, y_pred: pd.Series, threshold: float = 0.5) -> float:
    return f1_score(y_true, (y_pred >= threshold).astype(int), zero_division=0)


def category_auprc(y_true: pd.Series, y_score: pd.Series) -> float:
    if y_true.nunique() < 2:
        return float("nan")
    return average_precision_score(y_true, y_score)


def severity_accuracy(y_true_severity: pd.Series, y_pred_severity: pd.Series) -> float:
    return float((y_true_severity == y_pred_severity).mean())


def false_positive_rate(y_true_benign: pd.Series, y_pred_blocked: pd.Series) -> float:
    """Fraction of ground-truth-benign/in-scope rows that were incorrectly
    blocked. y_true_benign is a boolean Series (True = should NOT be
    blocked); y_pred_blocked is a boolean Series (True = system blocked it).
    """
    if len(y_true_benign) == 0:
        return float("nan")
    benign_mask = y_true_benign.astype(bool)
    if benign_mask.sum() == 0:
        return float("nan")
    return float(y_pred_blocked[benign_mask].mean())


def latency_percentiles(latencies_ms: list[float]) -> dict[str, float]:
    if not latencies_ms:
        return {"p50": float("nan"), "p95": float("nan")}
    arr = np.array(latencies_ms)
    return {"p50": float(np.percentile(arr, 50)), "p95": float(np.percentile(arr, 95))}


def cost_per_1k_requests(total_cost: float, n_requests: int) -> float:
    if n_requests == 0:
        return float("nan")
    return (total_cost / n_requests) * 1000
