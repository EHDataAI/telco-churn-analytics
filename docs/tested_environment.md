# Tested environment

The portfolio package was validated locally with:

- Python 3.13.5
- pandas 2.2.3
- NumPy 2.3.5
- scikit-learn 1.8.0
- XGBoost 3.1.3
- matplotlib 3.10.8
- pytest 9.0.2
- nbformat 5.10.4
- nbclient 0.10.4

`pyproject.toml` uses compatible version ranges rather than locking the repository to one platform-specific environment. GitHub CI validates the reference Python 3.12 workflow and runs an additional Python 3.13 compatibility test. Dependency vulnerability scanning is performed in CI with `pip-audit`.
