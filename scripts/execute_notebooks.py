"""Execute curated notebooks as a reproducibility smoke test."""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"


def main() -> None:
    notebooks = sorted(NOTEBOOK_DIR.glob("*.ipynb"))
    if not notebooks:
        raise SystemExit("No notebooks found")

    for path in notebooks:
        notebook = nbformat.read(path, as_version=4)
        client = NotebookClient(
            notebook,
            timeout=180,
            kernel_name="python3",
            allow_errors=False,
        )
        client.execute(cwd=str(ROOT))
        print(f"PASS {path.name}")


if __name__ == "__main__":
    main()
