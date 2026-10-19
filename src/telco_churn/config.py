"""Project configuration and feature contracts."""

from __future__ import annotations

RANDOM_STATE = 20260924
TARGET = "Churn Next 30D"
ID_COLUMN = "Customer ID"
DATE_COLUMN = "Snapshot Date"

ACTIVE_PRODUCT_COLUMNS = [
    "Phone Service",
    "Internet Service",
    "Multiple Lines",
    "Online Security",
    "Online Backup",
    "Device Protection Plan",
    "Premium Tech Support",
    "Streaming TV",
    "Streaming Movies",
    "Streaming Music",
    "Unlimited Data",
]

# Features excluded from the primary predictive model by design.
# Gender/Married remain available for subgroup/sensitivity analysis but are not
# required for the baseline decision model. Offer can reflect treatment-selection
# processes. Historical aggregates are intentionally excluded from the primary
# model because they are highly redundant with tenure and monthly charges.
PRIMARY_EXCLUDED_COLUMNS = [
    ID_COLUMN,
    DATE_COLUMN,
    TARGET,
    "Gender",
    "Married",
    "Offer",
    "Total Charges",
    "Total Extra Data Charges",
    "Total Long Distance Charges",
    "Total Refunds",
    "Total Revenue",
]

OUTCOME_DERIVED_SOURCE_COLUMNS = [
    "Customer Status",
    "Churn Score",
    "Churn Category",
    "Churn Reason",
]

SOURCE_GEO_COLUMNS = [
    "City",
    "Country",
    "State",
    "Zip Code",
    "Latitude",
    "Longitude",
    "Lat Long",
    "Population",
]
