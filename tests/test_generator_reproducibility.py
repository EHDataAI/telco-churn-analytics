from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "synthetic"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_synthetic_generator_is_deterministic(tmp_path):
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "generate_synthetic_data.py"),
            "--output-dir",
            str(tmp_path),
        ],
        check=True,
        cwd=ROOT,
    )
    for split in ("train", "validation", "test"):
        name = f"telco_churn_synthetic_{split}.csv"
        assert sha256(tmp_path / name) == sha256(DATA_DIR / name)


def test_public_csvs_use_canonical_lf_line_endings():
    csv_files = sorted(path for path in ROOT.rglob("*.csv") if ".venv" not in path.parts)
    assert csv_files

    for path in csv_files:
        payload = path.read_bytes()
        relative = path.relative_to(ROOT).as_posix()
        assert b"\r" not in payload, relative
        assert payload.endswith(b"\n"), relative


def test_gitattributes_enforces_csv_lf_contract():
    attributes = (ROOT / ".gitattributes").read_text(encoding="utf-8").splitlines()
    assert "*.csv text eol=lf" in attributes
    assert "*.csv -text" not in attributes
