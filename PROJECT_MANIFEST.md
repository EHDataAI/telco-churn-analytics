# Project manifest

## Public technical evidence

- `README.md` — recruiter-first technical overview.
- `data/` — synthetic train/validation/test splits, dictionary, hashes and dataset card.
- `scripts/generate_synthetic_data.py` — deterministic data generator.
- `src/telco_churn/` — reusable data validation, features, modeling, evaluation and plotting code.
- `scripts/run_benchmark.py` — controlled train/validation/locked-test workflow.
- `reports/metrics/` — cross-validation, validation selection, locked-test, lift, subgroup and coefficient evidence.
- `reports/figures/validation_*` — candidate-model visual comparison restricted to validation.
- `reports/figures/test_calibration.png` — selected-model calibration on the locked test split.
- `reports/figures/logistic_key_drivers.png` — predictive coefficient interpretation.
- `notebooks/` — three curated notebooks for EDA, modeling and interpretation.
- `tests/` — contract, feature, model-pipeline, evidence-protocol and generator-reproducibility tests.
- `docs/` — methodology, source audit, data contract, model card and public/private boundary.
- `.github/workflows/ci.yml` — quality, compatibility, dependency and secret-scanning controls.
- `.pre-commit-config.yaml` — local code-quality and secret-scan hooks.
- `SECURITY.md`, `.gitignore`, `LICENSE` — publication controls and project hygiene.
- `QA_REPORT.md` — ERL v2 local verification evidence.

## Explicitly excluded from the public project

- original academic train/validation/test CSVs;
- original academic notebooks;
- Word/PDF submission files;
- presentation video;
- personal filesystem paths;
- credentials/tokens;
- real customer data.

- `.gitattributes` — cross-platform text normalization with byte-preserved CSV artifacts
