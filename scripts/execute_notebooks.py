"""Execute notebooks for QA; publish executed outputs only when explicitly requested."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"
MAX_PUBLIC_NOTEBOOK_BYTES = 2_000_000


def audit_executed_notebook(notebook: nbformat.NotebookNode) -> dict[str, int]:
    """Reject missing execution evidence and notebook error outputs."""
    nbformat.validate(notebook)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    if not code_cells:
        raise ValueError("Notebook contains no code cells")
    if any(cell.execution_count is None for cell in code_cells):
        raise ValueError("Notebook has unexecuted code cells")

    outputs = [output for cell in code_cells for output in cell.get("outputs", [])]
    if not outputs:
        raise ValueError("Notebook has no saved outputs")
    if any(output.get("output_type") == "error" for output in outputs):
        raise ValueError("Notebook contains execution error outputs")

    return {
        "code_cells": len(code_cells),
        "outputs": len(outputs),
        "png_outputs": sum("image/png" in output.get("data", {}) for output in outputs),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--publish",
        action="store_true",
        help="Replace source notebooks with validated executed versions on the current branch",
    )
    group.add_argument(
        "--check-public",
        action="store_true",
        help="Validate that each committed notebook includes executed outputs",
    )
    args = parser.parse_args()

    notebooks = sorted(NOTEBOOK_DIR.glob("*.ipynb"))
    if not notebooks:
        raise SystemExit("No notebooks found")

    if args.check_public:
        for path in notebooks:
            counts = audit_executed_notebook(nbformat.read(path, as_version=4))
            print(f"PUBLIC_PASS {path.name} {counts}")
        return

    prepared: list[tuple[Path, nbformat.NotebookNode]] = []
    for path in notebooks:
        notebook = nbformat.read(path, as_version=4)
        client = NotebookClient(
            notebook,
            timeout=180,
            kernel_name="python3",
            allow_errors=False,
        )
        executed = client.execute(cwd=str(ROOT))
        counts = audit_executed_notebook(executed)
        prepared.append((path, executed))
        print(f"PASS {path.name} {counts}")

    if not args.publish:
        return

    # Stage all files before replacing any source notebook, preventing a partial
    # publication if execution, validation, or output-size QA fails.
    with tempfile.TemporaryDirectory(prefix=".notebook-publication-", dir=ROOT) as directory:
        stage = Path(directory)
        for path, executed in prepared:
            candidate = stage / path.name
            with candidate.open("w", encoding="utf-8", newline="\n") as handle:
                nbformat.write(executed, handle)
            if candidate.stat().st_size >= MAX_PUBLIC_NOTEBOOK_BYTES:
                raise ValueError(
                    f"Public notebook exceeds {MAX_PUBLIC_NOTEBOOK_BYTES} bytes: {path.name}"
                )

        for path, _executed in prepared:
            os.replace(stage / path.name, path)
            print(f"PUBLISHED {path.relative_to(ROOT).as_posix()}")

    print("Publication files updated locally; review diffs and security QA before committing")


if __name__ == "__main__":
    main()
