"""Data loading and contract validation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import DATE_COLUMN, ID_COLUMN, TARGET

EXPECTED_SPLITS = ("train", "validation", "test")


def load_splits(data_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Load train/validation/test synthetic CSV files."""
    base = Path(data_dir)
    frames: dict[str, pd.DataFrame] = {}
    for split in EXPECTED_SPLITS:
        path = base / f"telco_churn_synthetic_{split}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path, parse_dates=[DATE_COLUMN], keep_default_na=False)
        frames[split] = frame
    validate_splits(frames)
    return frames


def validate_splits(frames: dict[str, pd.DataFrame]) -> None:
    """Fail fast on schema, target, ID overlap, dates, or impossible service states."""
    if set(frames) != set(EXPECTED_SPLITS):
        raise ValueError(f"Expected splits {EXPECTED_SPLITS}, got {sorted(frames)}")

    columns = None
    id_sets: dict[str, set[str]] = {}
    date_ranges: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {}

    for name in EXPECTED_SPLITS:
        frame = frames[name]
        if columns is None:
            columns = list(frame.columns)
        elif list(frame.columns) != columns:
            raise ValueError(f"Column order/schema mismatch in split '{name}'")

        required = {ID_COLUMN, DATE_COLUMN, TARGET, "Internet Service", "Phone Service"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"Missing required columns in '{name}': {sorted(missing)}")

        if frame[ID_COLUMN].isna().any() or frame[ID_COLUMN].duplicated().any():
            raise ValueError(f"Missing or duplicate Customer ID in '{name}'")
        if not set(frame[TARGET].dropna().unique()).issubset({0, 1}):
            raise ValueError(f"Target must be binary in '{name}'")
        if frame.isna().any().any():
            raise ValueError(f"Unexpected missing values in '{name}'")

        id_sets[name] = set(frame[ID_COLUMN].astype(str))
        date_ranges[name] = (frame[DATE_COLUMN].min(), frame[DATE_COLUMN].max())

        internet_addons = [
            "Online Security",
            "Online Backup",
            "Device Protection Plan",
            "Premium Tech Support",
            "Streaming TV",
            "Streaming Movies",
            "Streaming Music",
            "Unlimited Data",
        ]
        if ((frame["Internet Service"] == 0) & (frame[internet_addons].sum(axis=1) > 0)).any():
            raise ValueError(f"Internet add-on without Internet Service in '{name}'")
        if ((frame["Phone Service"] == 0) & (frame["Multiple Lines"] == 1)).any():
            raise ValueError(f"Multiple Lines without Phone Service in '{name}'")

    if id_sets["train"] & id_sets["validation"]:
        raise ValueError("Customer ID overlap between train and validation")
    if id_sets["train"] & id_sets["test"]:
        raise ValueError("Customer ID overlap between train and test")
    if id_sets["validation"] & id_sets["test"]:
        raise ValueError("Customer ID overlap between validation and test")

    if not (
        date_ranges["train"][1] < date_ranges["validation"][0]
        and date_ranges["validation"][1] < date_ranges["test"][0]
    ):
        raise ValueError("Splits must respect train < validation < test chronology")
