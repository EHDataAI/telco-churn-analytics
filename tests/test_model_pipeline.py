from pathlib import Path

from sklearn.metrics import average_precision_score, roc_auc_score

from telco_churn.data import load_splits
from telco_churn.features import split_xy
from telco_churn.modeling import build_candidate_models, predict_scores

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "synthetic"


def test_logistic_pipeline_fits_without_leakage_columns():
    frames = load_splits(DATA_DIR)
    X_train, y_train = split_xy(frames["train"].head(1000))
    X_val, y_val = split_xy(frames["validation"].head(500))
    model = build_candidate_models(X_train)["logistic_regression"]
    model.fit(X_train, y_train)
    scores = predict_scores(model, X_val)
    assert len(scores) == len(X_val)
    assert ((scores >= 0) & (scores <= 1)).all()
    assert roc_auc_score(y_val, scores) > 0.60
    assert average_precision_score(y_val, scores) > y_val.mean()
