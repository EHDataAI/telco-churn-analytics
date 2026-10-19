"""Feature engineering with explicit, testable semantics."""

from __future__ import annotations

import pandas as pd

from .config import ACTIVE_PRODUCT_COLUMNS, PRIMARY_EXCLUDED_COLUMNS, TARGET


def add_active_product_count(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with a deterministic count of active products/add-ons.

    `Paperless Billing` is deliberately not included because it is a billing
    preference, not a contracted product. The function is shared by all data
    splits to prevent train/validation/test semantic drift.
    """
    missing = [column for column in ACTIVE_PRODUCT_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing product columns: {missing}")

    out = df.copy()
    out["Active Product Count"] = out[ACTIVE_PRODUCT_COLUMNS].sum(axis=1).astype(int)
    return out


def primary_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the feature list for the primary decision model."""
    return [column for column in df.columns if column not in PRIMARY_EXCLUDED_COLUMNS]


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Apply deterministic feature engineering and return X/y."""
    enriched = add_active_product_count(df)
    if TARGET not in enriched.columns:
        raise ValueError(f"Target column '{TARGET}' not found")
    features = primary_feature_columns(enriched)
    return enriched[features], enriched[TARGET].astype(int)
