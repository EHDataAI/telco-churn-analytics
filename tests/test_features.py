from pathlib import Path

from telco_churn.config import ACTIVE_PRODUCT_COLUMNS
from telco_churn.data import load_splits
from telco_churn.features import add_active_product_count, primary_feature_columns

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "synthetic"


def test_active_product_count_uses_one_definition_for_all_splits():
    frames = load_splits(DATA_DIR)
    for frame in frames.values():
        enriched = add_active_product_count(frame)
        expected = frame[ACTIVE_PRODUCT_COLUMNS].sum(axis=1).astype(int)
        assert enriched["Active Product Count"].equals(expected)


def test_paperless_billing_is_not_counted_as_product():
    assert "Paperless Billing" not in ACTIVE_PRODUCT_COLUMNS


def test_primary_model_excludes_outcome_and_sensitive_or_redundant_fields():
    train = add_active_product_count(load_splits(DATA_DIR)["train"])
    features = set(primary_feature_columns(train))
    blocked = {
        "Customer ID",
        "Snapshot Date",
        "Churn Next 30D",
        "Gender",
        "Married",
        "Offer",
        "Total Charges",
        "Total Extra Data Charges",
        "Total Long Distance Charges",
        "Total Refunds",
        "Total Revenue",
    }
    assert features.isdisjoint(blocked)
