"""Cross-platform repository QA with ERL-oriented publication checks."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
DATA_DIR = ROOT / "data" / "synthetic"
ROOT_RUNTIME_EXCLUDED_DIRS = {
    ".git",
    ".venv",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
}
NESTED_RUNTIME_EXCLUDED_DIRS = {
    "__pycache__",
    ".ipynb_checkpoints",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def project_env() -> dict[str, str]:
    """Return a subprocess environment that can import the local src-layout package."""
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(SRC) if not existing else str(SRC) + os.pathsep + existing
    return env


def run(command: list[str]) -> None:
    subprocess.run(command, cwd=ROOT, env=project_env(), check=True)


def is_runtime_artifact_path(path: Path, root: Path = ROOT) -> bool:
    """Return True only for known runtime paths outside the publication boundary.

    Repository-level runtime directories such as .venv and .git are excluded only
    when they are the first path component. Nested directories with the same names
    remain inside the publication boundary and must still be scanned.
    """
    try:
        relative = path.relative_to(root)
    except ValueError:
        relative = path
    parts = relative.parts
    if not parts:
        return False
    if parts[0] in ROOT_RUNTIME_EXCLUDED_DIRS:
        return True
    return any(part in NESTED_RUNTIME_EXCLUDED_DIRS for part in parts)


def scan_text_files() -> list[dict[str, str]]:
    patterns = {
        "windows_user_path": re.compile(r"[A-Z]:\\\\Users\\\\[^\\\\\s]+", re.I),
        "posix_user_path": re.compile(r"/(?:home|Users)/[^/\s]+/"),
        "openai_key": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
        "github_token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
        "aws_key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
        "private_key": re.compile(r"BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY"),
        "private_ipv4_10": re.compile(r"\b10(?:\.\d{1,3}){3}\b"),
        "private_ipv4_172": re.compile(r"\b172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2}\b"),
        "private_ipv4_192": re.compile(r"\b192\.168(?:\.\d{1,3}){2}\b"),
    }
    findings: list[dict[str, str]] = []
    skip_suffixes = {".png", ".jpg", ".jpeg", ".pyc"}
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() in skip_suffixes:
            continue
        if is_runtime_artifact_path(path):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for name, pattern in patterns.items():
            if pattern.search(text):
                findings.append({"type": name, "file": str(path.relative_to(ROOT))})
    return findings


def find_forbidden_artifacts(root: Path = ROOT) -> list[str]:
    """Find prohibited source/private artifacts while excluding generated runtime environments."""
    patterns = (
        "*.docx",
        "*.mp4",
        "Datos Telecomunicaciones-*",
        "Entrega Final*",
        "Entrega 03*",
        ".env",
        ".env.*",
        "*.pem",
        "*.p12",
        "*.pfx",
    )
    artifacts: set[str] = set()
    for pattern in patterns:
        for path in root.rglob(pattern):
            if not path.is_file() or is_runtime_artifact_path(path, root=root):
                continue
            artifacts.add(path.relative_to(root).as_posix())
    return sorted(artifacts)


def verify_evidence_contract() -> dict[str, str]:
    """Verify that persisted evidence reflects the strict train/validation/test protocol."""
    figures = ROOT / "reports" / "figures"
    metrics = ROOT / "reports" / "metrics"

    required = [
        figures / "validation_roc_curves.png",
        figures / "validation_pr_curves.png",
        figures / "test_calibration.png",
        metrics / "validation_model_comparison.csv",
        metrics / "model_selection.json",
        metrics / "benchmark_summary.json",
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"Missing persisted evidence: {missing}")

    forbidden = [figures / "test_roc_curves.png", figures / "test_pr_curves.png"]
    leaked_comparison_artifacts = [
        str(path.relative_to(ROOT)) for path in forbidden if path.exists()
    ]
    if leaked_comparison_artifacts:
        raise RuntimeError(
            "Candidate-comparison artifacts must not be generated from test data: "
            f"{leaked_comparison_artifacts}"
        )

    selection = json.loads((metrics / "model_selection.json").read_text(encoding="utf-8"))
    protocol = str(selection.get("test_protocol", ""))
    if "selected model only" not in protocol.lower():
        raise RuntimeError(
            "model_selection.json does not document strict selected-model test access"
        )

    return {
        "candidate_comparison": "validation_only",
        "test_access": "selected_model_only_after_lock",
    }


def main() -> None:
    run([sys.executable, "-m", "compileall", "-q", "src", "scripts", "tests"])
    run([sys.executable, "-m", "pytest", "-q"])
    run([sys.executable, "scripts/execute_notebooks.py"])

    hashes = {}
    with tempfile.TemporaryDirectory() as directory:
        regenerated = Path(directory)
        run(
            [sys.executable, "scripts/generate_synthetic_data.py", "--output-dir", str(regenerated)]
        )
        for split in ("train", "validation", "test"):
            name = f"telco_churn_synthetic_{split}.csv"
            committed_hash = sha256(DATA_DIR / name)
            regenerated_hash = sha256(regenerated / name)
            if committed_hash != regenerated_hash:
                raise RuntimeError(f"Deterministic regeneration failed for {name}")
            hashes[name] = committed_hash

    forbidden_artifacts = find_forbidden_artifacts()

    findings = scan_text_files()
    if forbidden_artifacts:
        raise RuntimeError(f"Private/source artifacts found: {forbidden_artifacts}")
    if findings:
        raise RuntimeError(f"Suspicious text patterns found: {findings}")

    evidence_protocol = verify_evidence_contract()

    report = {
        "python": sys.version.split()[0],
        "compileall": "PASS",
        "pytest": "PASS",
        "notebooks": "PASS",
        "deterministic_regeneration": "PASS",
        "static_sensitive_pattern_scan": "PASS",
        "forbidden_source_artifact_scan": "PASS",
        "evidence_protocol": evidence_protocol,
        "data_sha256": hashes,
        "note": (
            "Pre-commit, dependency audit and Git-history Gitleaks remain release gates and must "
            "be rerun in the isolated project environment after Git initialization."
        ),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
