"""Run the controlled model comparison and write portfolio evidence.

The protocol deliberately prevents candidate-model access to the test split. Candidate comparison
and threshold selection are completed on train/validation first. Only the locked selected model is
then scored once on the temporal test split.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from telco_churn.config import RANDOM_STATE  # noqa: E402
from telco_churn.data import load_splits  # noqa: E402
from telco_churn.evaluation import (  # noqa: E402
    choose_f1_threshold,
    classification_metrics,
    subgroup_metrics,
    top_k_metrics,
)
from telco_churn.features import add_active_product_count, split_xy  # noqa: E402
from telco_churn.modeling import (  # noqa: E402
    build_candidate_models,
    logistic_coefficients,
    predict_scores,
)
from telco_churn.visualization import (  # noqa: E402
    save_calibration_figure,
    save_discrimination_figure,
    save_logistic_driver_figure,
    save_precision_recall_figure,
)


def _jsonable(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    return value


def main() -> None:
    data_dir = ROOT / "data" / "synthetic"
    metrics_dir = ROOT / "reports" / "metrics"
    figures_dir = ROOT / "reports" / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    frames = load_splits(data_dir)
    train, validation, test = (frames[k] for k in ("train", "validation", "test"))
    X_train, y_train = split_xy(train)
    X_val, y_val = split_xy(validation)

    # Do not construct X_test/y_test until the model and threshold are locked.
    candidates = build_candidate_models(X_train)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
        "f1_at_050": "f1",
        "neg_brier": "neg_brier_score",
    }

    cv_rows = []
    validation_rows = []
    fitted = {}
    validation_scores = {}

    for name, model in candidates.items():
        if name != "dummy_prior":
            cv_result = cross_validate(
                model,
                X_train,
                y_train,
                cv=cv,
                scoring=scoring,
                n_jobs=1,
                error_score="raise",
            )
            cv_rows.append(
                {
                    "model": name,
                    "roc_auc_mean": cv_result["test_roc_auc"].mean(),
                    "roc_auc_sd": cv_result["test_roc_auc"].std(ddof=0),
                    "pr_auc_mean": cv_result["test_pr_auc"].mean(),
                    "pr_auc_sd": cv_result["test_pr_auc"].std(ddof=0),
                    "f1_at_050_mean": cv_result["test_f1_at_050"].mean(),
                    "f1_at_050_sd": cv_result["test_f1_at_050"].std(ddof=0),
                    "brier_mean": -cv_result["test_neg_brier"].mean(),
                    "brier_sd": cv_result["test_neg_brier"].std(ddof=0),
                }
            )

        model.fit(X_train, y_train)
        fitted[name] = model
        val_score = predict_scores(model, X_val)
        validation_scores[name] = val_score
        val_metrics = classification_metrics(y_val, val_score, threshold=0.5)
        validation_rows.append({"model": name, **val_metrics})

    cv_df = pd.DataFrame(cv_rows).sort_values("pr_auc_mean", ascending=False)
    validation_df = pd.DataFrame(validation_rows).sort_values("pr_auc", ascending=False)
    cv_df.to_csv(metrics_dir / "cross_validation_metrics.csv", index=False)
    validation_df.to_csv(metrics_dir / "validation_model_comparison.csv", index=False)

    trained_candidates = validation_df[validation_df["model"] != "dummy_prior"]
    selected_name = str(trained_candidates.iloc[0]["model"])
    selected_model = fitted[selected_name]
    selected_val_scores = validation_scores[selected_name]
    threshold_info = choose_f1_threshold(y_val, selected_val_scores)
    threshold = threshold_info["threshold"]

    # Candidate comparison figures are validation-only. This prevents test-set model comparison.
    trained_validation_score_map = {
        name: scores for name, scores in validation_scores.items() if name != "dummy_prior"
    }
    save_discrimination_figure(
        y_val,
        trained_validation_score_map,
        figures_dir / "validation_roc_curves.png",
        split_label="Validation",
    )
    save_precision_recall_figure(
        y_val,
        trained_validation_score_map,
        figures_dir / "validation_pr_curves.png",
        split_label="Validation",
    )

    # Test scoring begins here, after the candidate and threshold are locked.
    X_test, y_test = split_xy(test)
    selected_test_scores = predict_scores(selected_model, X_test)
    test_metrics = classification_metrics(y_test, selected_test_scores, threshold=threshold)
    test_metrics.update({"selected_model": selected_name})

    (metrics_dir / "model_selection.json").write_text(
        json.dumps(
            _jsonable(
                {
                    "primary_validation_metric": "pr_auc",
                    "selected_model": selected_name,
                    "threshold_selection": "maximum validation F1",
                    "validation_threshold": threshold_info,
                    "test_protocol": "selected model only; accessed after model and threshold lock",
                    "test_metrics": test_metrics,
                }
            ),
            indent=2,
        ),
        encoding="utf-8",
    )

    top_k = top_k_metrics(y_test, selected_test_scores)
    top_k.to_csv(metrics_dir / "test_top_k_metrics.csv", index=False)

    test_with_features = add_active_product_count(test)
    subgroup_tables = [
        subgroup_metrics(test_with_features, y_test, selected_test_scores, threshold, "Gender")
    ]
    age_band = pd.cut(
        test_with_features["Age"],
        bins=[17, 29, 44, 64, 120],
        labels=["18-29", "30-44", "45-64", "65+"],
        include_lowest=True,
    )
    tmp = test_with_features.copy()
    tmp["Age Band"] = age_band.astype(str)
    subgroup_tables.append(
        subgroup_metrics(tmp, y_test, selected_test_scores, threshold, "Age Band")
    )
    pd.concat(subgroup_tables, ignore_index=True).to_csv(
        metrics_dir / "test_subgroup_diagnostics.csv", index=False
    )

    if "logistic_regression" in fitted:
        coef = logistic_coefficients(fitted["logistic_regression"])
        coef.to_csv(metrics_dir / "logistic_coefficients.csv", index=False)
        save_logistic_driver_figure(coef, figures_dir / "logistic_key_drivers.png")

    save_calibration_figure(
        y_test,
        selected_test_scores,
        figures_dir / "test_calibration.png",
        model_label=selected_name,
        split_label="Test",
    )

    summary = {
        "dataset_rows": {name: len(frame) for name, frame in frames.items()},
        "selected_model": selected_name,
        "validation_pr_auc": float(
            validation_df.loc[validation_df["model"] == selected_name, "pr_auc"].iloc[0]
        ),
        "threshold": float(threshold),
        "test_protocol": "selected model only; test scored after model and threshold lock",
        "test_metrics": test_metrics,
        "top_10_percent": top_k.loc[top_k["top_fraction"] == 0.10].iloc[0].to_dict(),
        "interpretation": (
            "Model metrics are evidence for this synthetic benchmark only; they are not estimates "
            "of production telecom performance."
        ),
    }
    (metrics_dir / "benchmark_summary.json").write_text(
        json.dumps(_jsonable(summary), indent=2), encoding="utf-8"
    )

    print(json.dumps(_jsonable(summary), indent=2))


if __name__ == "__main__":
    main()
