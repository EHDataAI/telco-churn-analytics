from pathlib import Path

from telco_churn.config import TARGET
from telco_churn.data import load_splits

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "synthetic"


def test_splits_load_and_are_chronological():
    frames = load_splits(DATA_DIR)
    assert len(frames["train"]) == 4225
    assert len(frames["validation"]) == 1409
    assert len(frames["test"]) == 1409
    assert frames["train"]["Snapshot Date"].max() < frames["validation"]["Snapshot Date"].min()
    assert frames["validation"]["Snapshot Date"].max() < frames["test"]["Snapshot Date"].min()


def test_target_is_binary_and_non_degenerate():
    frames = load_splits(DATA_DIR)
    for frame in frames.values():
        assert set(frame[TARGET].unique()) == {0, 1}
        assert 0.20 < frame[TARGET].mean() < 0.35


def test_no_cross_split_customer_overlap():
    frames = load_splits(DATA_DIR)
    ids = {name: set(frame["Customer ID"]) for name, frame in frames.items()}
    assert ids["train"].isdisjoint(ids["validation"])
    assert ids["train"].isdisjoint(ids["test"])
    assert ids["validation"].isdisjoint(ids["test"])


def test_outcome_derived_source_columns_are_not_in_public_data():
    frames = load_splits(DATA_DIR)
    blocked = {"Customer Status", "Churn Score", "Churn Category", "Churn Reason"}
    for frame in frames.values():
        assert blocked.isdisjoint(frame.columns)
