# QA report — ERL v2 local release candidate

Status: **CONTENT QA PASS / GIT-REMOTE GATES NOT RUN**

This report covers the curated public project tree before Git publication. The ERL v2 adversarial
review identified and corrected two release-blocking issues from the previous package:

1. candidate models were being scored on the test split before the selection was formally locked;
2. GitHub Actions used action versions that were obsolete for the October 2026 runner runtime.

The benchmark now compares candidates on validation only. Test access begins only after the model
and threshold are locked, and only the selected model is scored on test.

## Automated local checks completed

| Check | Result |
|---|---|
| Python byte-compilation (`src`, `scripts`, `tests`) | PASS |
| pytest | PASS — 12 tests |
| `01_data_quality_eda.ipynb` smoke execution | PASS |
| `02_modeling.ipynb` smoke execution | PASS |
| `03_interpretation.ipynb` smoke execution | PASS |
| deterministic synthetic-data regeneration | PASS |
| cross-split Customer ID isolation | PASS |
| chronological train < validation < test contract | PASS |
| impossible service-state checks | PASS |
| outcome-derived source fields absent from public data | PASS |
| candidate comparison restricted to validation | PASS |
| selected-model-only test protocol | PASS |
| obvious credential/private-path static pattern scan | PASS |
| original academic artifact scan | PASS |

## Reproducible synthetic-data hashes

```text
train      9d581e64e313a41d5e9fcd26be58a09952772441d3e0140361610f7b6d107798
validation 73392960313d6470b850922ece9d15df9591a50892fe9ff99c54aed507087c37
test       0e17b05efb4ea666340b65eafdc7a448323d760b2d8161d026f9aa7f54ff224f
```

## Controlled modeling evidence

Candidate selection uses validation PR-AUC. The test split is not used for candidate comparison.

| Model | Validation PR-AUC | Validation ROC-AUC | Validation Brier |
|---|---:|---:|---:|
| Logistic regression | **0.611** | **0.797** | **0.152** |
| XGBoost | 0.597 | 0.792 | 0.155 |
| Prior baseline | 0.273 | 0.500 | 0.199 |

Selected model: **logistic regression**.

Validation-only F1 threshold: approximately **0.355**.

Locked temporal test results for the selected model:

| Metric | Result |
|---|---:|
| ROC-AUC | 0.767 |
| PR-AUC | 0.541 |
| Brier score | 0.167 |
| Precision | 0.534 |
| Recall | 0.562 |
| F1 | 0.547 |
| Top-10% lift | 2.30x |

These are synthetic benchmark metrics, not real telecom performance estimates.

## Toolchain corrections for October 2026

- GitHub Actions: `actions/checkout@v7.0.1`.
- GitHub Actions: `actions/setup-python@v7.0.0`.
- GitHub Actions: `gitleaks/gitleaks-action@v3.0.0`.
- pre-commit Ruff hook: `v0.16.10`.
- pre-commit Gitleaks hook: `v8.30.1`.
- CI dependency audit: `pip-audit`.
- CI compatibility: Python 3.12 reference plus Python 3.13 test job.

## Gates intentionally deferred to the Git / GitHub phase

The following controls remain `NOT_RUN` until the project exists inside the target Git repository:

- isolated environment install + `pip check`;
- `pre-commit run --all-files` against the exact staged tree;
- local Gitleaks against Git content/history;
- `pip-audit` with current vulnerability data;
- staged-diff inspection;
- signed commit verification;
- remote GitHub Actions;
- branch protection / required checks;
- Pull Request review and merge;
- signed release tag, if a release is warranted.

## Cross-platform Git reproducibility

`.gitattributes` enforces canonical LF for synthetic CSVs (`*.csv text eol=lf`). The generator also sets `lineterminator="\n"`, so committed bytes and SHA-256 evidence are reproducible across Windows, Linux, and macOS while remaining compatible with `git diff --check`.


## ERL runtime-environment scanner correction

A Windows H1 run exposed a false positive in the forbidden-artifact scan: the repository-local
`.venv` contains trusted package certificate bundles named `cacert.pem`. The scan now excludes only
generated/runtime directories (including `.venv`, `.git` and QA caches) while continuing to scan the
complete publication-boundary tree. A regression test verifies that `.venv/.../cacert.pem` is ignored
but a `*.pem` file under a public project directory is still rejected.


- ERL v2.6 hardening: root runtime directories are excluded only at repository root; nested lookalike directories remain inside the publication-boundary scan.

## ERL cross-platform path normalization

Windows H1R exposed a path-separator-only regression in the forbidden-artifact test. Forbidden-artifact findings are now emitted with `Path.as_posix()` so QA evidence is deterministic across Windows, Linux, and macOS without weakening the publication-boundary scan.


## ERL H2 line-ending hardening

An H2 staging gate exposed CRLF bytes in deterministic CSV artifacts. The generator now emits canonical LF explicitly, .gitattributes enforces LF for CSVs, regression tests reject carriage-return bytes, and the canonical SHA-256 evidence above was regenerated from unchanged tabular content.
