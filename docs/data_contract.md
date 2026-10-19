# Data contract

## Prediction question

At each `Snapshot Date`, estimate the probability that a customer will have
`Churn Next 30D = 1` during the following 30 days.

All model features must be known at or before the snapshot. Anything observed only at,
during, or after cancellation is prohibited from the predictive feature set.

## Temporal partitions

| Split | Snapshot window | Rows | Intended use |
|---|---|---:|---|
| train | 2025-01-31 to 2025-06-30 | 4,225 | fitting and cross-validation |
| validation | 2025-07-31 to 2025-08-31 | 1,409 | model and threshold selection |
| test | 2025-09-30 to 2025-10-31 | 1,409 | one final locked evaluation |

Customer IDs do not overlap across partitions.

## Explicit exclusions

The public reconstruction does not contain the source-sample fields `Customer Status`,
`Churn Score`, `Churn Category`, or `Churn Reason`. These fields are outcome-derived or
model-derived and would create direct or proxy leakage for this prediction task.

Geographic identifiers from the source academic data are also excluded because they are not
needed to demonstrate the modeling method.

## Satisfaction timing

The public field is named `Preperiod Satisfaction Score` rather than `Satisfaction Score`.
Its contract explicitly states that it is measured before the 30-day outcome window.
This removes the timing ambiguity observed in the academic source data.

## Feature engineering contract

`Active Product Count` is never manually stored in the raw CSVs. It is derived in
`src/telco_churn/features.py` from one centrally defined list of service/product flags.
`Paperless Billing` is not counted as a product.

The same function is used for train, validation, and test and is protected by tests.

## Primary-model exclusions

The primary model does not use:

- Customer ID or Snapshot Date as predictors;
- Gender or Married status;
- Offer, because offer assignment may reflect a prior intervention/selection process;
- Total Charges, Total Revenue, refunds and other accumulated financial totals because they are
  highly redundant with tenure, monthly charge and usage history in this synthetic design.

Those columns remain available for EDA or explicitly labeled sensitivity analyses.
