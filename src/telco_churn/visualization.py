"""Small plotting helpers used by reports and notebooks."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, roc_curve


def save_discrimination_figure(
    y_true,
    model_scores: dict[str, np.ndarray],
    output: str | Path,
    *,
    split_label: str = "Validation",
) -> None:
    """Save ROC curves for the supplied evaluation split."""
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    for name, scores in model_scores.items():
        fpr, tpr, _ = roc_curve(y_true, scores)
        ax.plot(fpr, tpr, label=name)
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    ax.set(
        xlabel="False positive rate",
        ylabel="True positive rate",
        title=f"{split_label} ROC curves",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def save_precision_recall_figure(
    y_true,
    model_scores: dict[str, np.ndarray],
    output: str | Path,
    *,
    split_label: str = "Validation",
) -> None:
    """Save precision-recall curves for the supplied evaluation split."""
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    base = float(np.mean(y_true))
    for name, scores in model_scores.items():
        precision, recall, _ = precision_recall_curve(y_true, scores)
        ax.plot(recall, precision, label=name)
    ax.axhline(base, linestyle="--", linewidth=1, label="prevalence")
    ax.set(
        xlabel="Recall",
        ylabel="Precision",
        title=f"{split_label} precision-recall curves",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def save_calibration_figure(
    y_true,
    scores: np.ndarray,
    output: str | Path,
    *,
    model_label: str = "selected model",
    split_label: str = "Test",
) -> None:
    """Save a calibration curve for one already-selected model."""
    observed, predicted = calibration_curve(y_true, scores, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    ax.plot(predicted, observed, marker="o", label=model_label)
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="perfect calibration")
    ax.set(
        xlabel="Mean predicted probability",
        ylabel="Observed churn rate",
        title=f"{split_label} calibration",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def save_logistic_driver_figure(
    coefficients: pd.DataFrame,
    output: str | Path,
    n: int = 12,
) -> None:
    plot_df = coefficients.head(n).sort_values("coefficient")
    labels = (
        plot_df["feature"]
        .str.replace("numeric__", "", regex=False)
        .str.replace("categorical__", "", regex=False)
    )
    fig, ax = plt.subplots(figsize=(8.0, 5.5))
    ax.barh(labels, plot_df["coefficient"])
    ax.axvline(0, linewidth=1)
    ax.set(title="Largest regularized logistic coefficients", xlabel="Coefficient (model scale)")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
