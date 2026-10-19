# Security and data-publication boundary

This repository is designed for public use and must not contain source academic files,
private datasets, credentials, access tokens, personal filesystem paths, or identifiable
customer data.

The committed TELCO datasets are synthetic. The original academic CSV files and original
notebooks/reports remain outside the repository and are used only as reference material.

Before every push:

1. run `pre-commit run --all-files`;
2. run `gitleaks git --redact` for Git-history scanning and `gitleaks dir . --redact` for the working tree when Gitleaks is installed separately;
3. verify that no files under `original/`, `private/`, or `source_data/` are staged;
4. inspect `git diff --cached` before committing.

If a secret is accidentally committed, revoke/rotate it first and then remove it from Git
history. Do not rely on deleting only the latest file revision.
