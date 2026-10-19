# Notebook publication contract (pilot: TELCO)

## Goal

The public notebook must include saved execution results. A successful run in CI
without committing its outputs is only a runtime smoke test; it is not evidence
that a GitHub visitor can view those results.

## Publication process

Use the project's validated Python 3.12 environment with `pip install -e ".[dev]"`.
Work on a new branch with a clean working tree. Never run publication on `main`.

```powershell
python scripts/run_qa.py
python scripts/execute_notebooks.py --publish
python scripts/execute_notebooks.py --check-public
python -m pytest -q
python -m ruff check src scripts tests
pre-commit run --all-files --show-diff-on-failure
```

Inspect `git diff -- notebooks/` and `git status --short`. Perform the existing
working-tree and Git-history secret checks, inspect embedded outputs for private
identifiers and sensitive details, and verify plots/tables from a guest browser.
Only then stage and submit the executed notebooks in a reviewed pull request.

The default `python scripts/execute_notebooks.py` remains a non-mutating smoke
test. The explicit `--publish` option executes and validates all notebooks before
replacing local copies. It does not change model-selection rules, rerun the
canonical benchmark, or automatically commit/push to GitHub. Saved outputs must
reflect the already frozen synthetic evidence and be compared to versioned metrics.

## Quality criteria

- Every code cell has an execution count; at least one result output is saved.
- No notebook contains an error output.
- Each notebook is valid nbformat v4 and under the repository's 2 MB file threshold.
- Only approved synthetic data/results are displayed.
- No credentials, private paths, identifiers or personal data appear in output.
- Notebook outputs match the locked reference metrics and scientific claims.
- HTML conversion or a separate gallery may be added after native GitHub output QA.

This is a pilot for TELCO. The same contract can be adapted to GeoAI and Mining
without overwriting frozen experiment artifacts. Healthcare MLOps uses a
separate curated-report exception because its original evidence dependencies
are deliberately not public.
