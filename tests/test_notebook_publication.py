"""Regression tests for explicit notebook-publication checks."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import nbformat
import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "execute_notebooks.py"
SPEC = importlib.util.spec_from_file_location("notebook_publisher", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def make_notebook(*, executed: bool = True, error: bool = False) -> nbformat.NotebookNode:
    if error:
        outputs = [nbformat.v4.new_output("error", ename="ValueError", evalue="bad", traceback=[])]
    else:
        outputs = [nbformat.v4.new_output("stream", name="stdout", text="synthetic result\n")]
    cell = nbformat.v4.new_code_cell(
        source="print('synthetic result')",
        execution_count=1 if executed else None,
        outputs=outputs if executed else [],
    )
    return nbformat.v4.new_notebook(cells=[cell])


def test_published_notebook_has_executed_outputs() -> None:
    counts = MODULE.audit_executed_notebook(make_notebook())
    assert counts == {"code_cells": 1, "outputs": 1, "png_outputs": 0}


def test_rejects_notebook_without_execution() -> None:
    with pytest.raises(ValueError, match="unexecuted"):
        MODULE.audit_executed_notebook(make_notebook(executed=False))


def test_rejects_error_output() -> None:
    with pytest.raises(ValueError, match="error outputs"):
        MODULE.audit_executed_notebook(make_notebook(error=True))
