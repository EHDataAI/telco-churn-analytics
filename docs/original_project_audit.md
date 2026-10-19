# Academic-source audit

This file records why the portfolio version was reconstructed instead of publishing the academic
artifacts unchanged.

## Source material reviewed

The academic delivery contained two Jupyter notebooks, an executive Word report, a PDF
presentation and a presentation video, plus three source CSV partitions later recovered for
reference.

The source partitions contain 7,043 rows in total: 4,225 train, 1,409 validation and 1,409 test.
All three contain 52 columns and have no repeated Customer IDs across partitions.

## Leakage and timing findings

The source data include `Customer Status`, `Churn Score`, `Churn Category` and `Churn Reason`.
These are unsuitable for the new prediction-time data contract. In the recovered source data,
`Customer Status` maps perfectly to the churn outcome, while `Churn Score` is strongly associated
with the target.

The source `Satisfaction Score` also shows a near-deterministic relationship with churn:

- score 1: 100% churn;
- score 2: 100% churn;
- score 3: about 16% churn;
- score 4: 0% churn;
- score 5: 0% churn.

Because the source materials do not establish a defensible pre-outcome measurement timestamp for
that variable, the public reconstruction replaces it with a synthetic, explicitly preperiod
measurement.

## Feature-engineering finding

The final academic notebook recalculated `N_Servicios` with a different column list for train than
for validation/test during the recorded execution sequence. In particular, `Paperless Billing`
was included in one definition and excluded in another. The portfolio version removes the stored
feature and derives a single `Active Product Count` through shared code and tests.

## Redundancy finding

The academic notebook used `Monthly Charge` together with two separate categorical derivatives of
the same variable. This adds redundant information and complicates coefficient interpretation.
The public primary model uses the continuous monthly charge without duplicate binnings.

## KPI finding

A business KPI in the academic notebook summarized `Total Revenue` after scaling, which means the
reported values were standardized scores rather than monetary revenue. The portfolio architecture
keeps EDA/business quantities separate from model preprocessing.

## Software/reproducibility findings

The academic notebooks preserve execution errors and non-monotonic execution histories and did not
ship with an environment definition, tests or CI. The portfolio version moves critical logic into
`src/`, adds deterministic synthetic data generation and validates the project with pytest and
notebook smoke tests.

## Portfolio decision

The academic artifacts remain reference material and are not intended for the public repository.
The public repository contains a new, synthetic, reproducible implementation that preserves the
analytical learning while making its limitations explicit.
