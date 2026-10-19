"""Evaluation utilities for discrimination, calibration and targeting."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def choose_f1_threshold(y_true: pd.Series | np.ndarray, scores: np.ndarray) -> dict[str, float]:
    """Select the probability threshold that maximizes F1 on validation data."""
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    if len(thresholds) == 0:
        return {"threshold": 0.5, "f1": 0.0, "precision": 0.0, "recall": 0.0}
    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    idx = int(np.nanargmax(f1))
    return {
        "threshold": float(thresholds[idx]),
        "f1": float(f1[idx]),
        "precision": float(precision[idx]),
        "recall": float(recall[idx]),
    }


def classification_metrics(
    y_true: pd.Series | np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    """Calculate threshold-free and threshold-dependent metrics."""
    y = np.asarray(y_true, dtype=int)
    pred = (np.asarray(scores) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, scores)),
        "pr_auc": float(average_precision_score(y, scores)),
        "brier_score": float(brier_score_loss(y, scores)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def top_k_metrics(
    y_true: pd.Series | np.ndarray,
    scores: np.ndarray,
    fractions: tuple[float, ...] = (0.05, 0.10, 0.20, 0.30),
) -> pd.DataFrame:
    """Compute precision, recall and lift among highest-risk customers."""
    y = np.asarray(y_true, dtype=int)
    order = np.argsort(-np.asarray(scores))
    base_rate = y.mean()
    total_positive = y.sum()
    rows = []
    for fraction in fractions:
        n = max(1, int(np.ceil(len(y) * fraction)))
        selected = y[order[:n]]
        captured = int(selected.sum())
        precision = captured / n
        recall = captured / total_positive if total_positive else 0.0
        lift = precision / base_rate if base_rate else 0.0
        rows.append(
            {
                "top_fraction": fraction,
                "customers_targeted": n,
                "churners_captured": captured,
                "precision": precision,
                "recall": recall,
                "lift": lift,
            }
        )
    return pd.DataFrame(rows)


def subgroup_metrics(
    frame: pd.DataFrame,
    y_true: pd.Series | np.ndarray,
    scores: np.ndarray,
    threshold: float,
    column: str,
) -> pd.DataFrame:
    """Diagnostic subgroup metrics; not a fairness certification."""
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores)
    values = frame[column].astype(str).to_numpy()
    rows = []
    for group in sorted(pd.unique(values)):
        mask = values == group
        if mask.sum() < 20 or len(np.unique(y[mask])) < 2:
            continue
        pred = (s[mask] >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y[mask], pred, labels=[0, 1]).ravel()
        rows.append(
            {
                "attribute": column,
                "group": group,
                "n": int(mask.sum()),
                "prevalence": float(y[mask].mean()),
                "precision": float(precision_score(y[mask], pred, zero_division=0)),
                "recall": float(recall_score(y[mask], pred, zero_division=0)),
                "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
                "predicted_positive_rate": float(pred.mean()),
                "roc_auc": float(roc_auc_score(y[mask], s[mask])),
            }
        )
    return pd.DataFrame(rows)
