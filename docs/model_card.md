# Model card — portfolio benchmark

## Model selected

Regularized logistic regression with standardized numeric variables and one-hot encoded categorical
variables.

## Selection rule

Candidate models were compared on the temporal validation partition. PR-AUC was the primary model
selection metric. Logistic regression achieved validation PR-AUC 0.611 versus 0.597 for XGBoost,
so the simpler interpretable model was selected.

The classification threshold was then selected on validation by maximizing F1: approximately 0.355.
No candidate-model test comparison is generated; the test split is accessed only for the already
selected logistic model after the threshold is locked.

## Locked test results

| Metric | Test result |
|---|---:|
| ROC-AUC | 0.767 |
| PR-AUC | 0.541 |
| Brier score | 0.167 |
| Accuracy | 0.740 |
| Precision | 0.534 |
| Recall | 0.562 |
| F1 | 0.547 |

At the top 10% of predicted risk, the model captures 91 of 395 churners in the test partition,
with precision 0.645 and lift 2.30x over the test-set churn prevalence.

## Intended use

Educational/portfolio demonstration of leakage-aware churn modeling, reproducible evaluation and
business-oriented targeting metrics.

## Not intended for

- production customer decisions;
- real-world telecom performance claims;
- causal interpretation of coefficients;
- automated retention decisions without business validation and human oversight.

## Important limitation

The target-generating process is synthetic. Predictive performance therefore demonstrates that the
pipeline can recover designed signal under controlled drift; it does not validate transportability
to a real telecom population.
