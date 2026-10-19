"""Model construction and selection helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from .config import RANDOM_STATE


def infer_column_types(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric = [column for column in X.columns if pd.api.types.is_numeric_dtype(X[column])]
    categorical = [column for column in X.columns if column not in numeric]
    return numeric, categorical


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build a preprocessing graph fit only inside the model pipeline."""
    numeric, categorical = infer_column_types(X)
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=True),
                categorical,
            ),
        ],
        remainder="drop",
    )


def build_candidate_models(X: pd.DataFrame) -> dict[str, object]:
    """Return baseline, interpretable and nonlinear candidate models."""
    return {
        "dummy_prior": DummyClassifier(strategy="prior"),
        "logistic_regression": Pipeline(
            steps=[
                ("preprocess", build_preprocessor(X)),
                (
                    "model",
                    LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
                ),
            ]
        ),
        "xgboost": Pipeline(
            steps=[
                ("preprocess", build_preprocessor(X)),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=400,
                        max_depth=3,
                        learning_rate=0.035,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        min_child_weight=3,
                        reg_lambda=2.0,
                        reg_alpha=0.1,
                        objective="binary:logistic",
                        eval_metric="logloss",
                        random_state=RANDOM_STATE,
                        n_jobs=2,
                    ),
                ),
            ]
        ),
    }


def predict_scores(model: object, X: pd.DataFrame) -> np.ndarray:
    """Return positive-class scores/probabilities."""
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X))[:, 1]
    return np.asarray(model.predict(X), dtype=float)


def logistic_coefficients(model: Pipeline) -> pd.DataFrame:
    """Extract regularized logistic coefficients with transformed feature names."""
    preprocess = model.named_steps["preprocess"]
    estimator = model.named_steps["model"]
    names = preprocess.get_feature_names_out()
    coefficients = estimator.coef_[0]
    table = pd.DataFrame({"feature": names, "coefficient": coefficients})
    table["odds_ratio_per_model_unit"] = np.exp(table["coefficient"])
    table["abs_coefficient"] = table["coefficient"].abs()
    return table.sort_values("abs_coefficient", ascending=False).reset_index(drop=True)
