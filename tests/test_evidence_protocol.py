import json
import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_candidate_comparison_is_validation_only():
    figures = ROOT / "reports" / "figures"
    assert (figures / "validation_roc_curves.png").exists()
    assert (figures / "validation_pr_curves.png").exists()
    assert not (figures / "test_roc_curves.png").exists()
    assert not (figures / "test_pr_curves.png").exists()


def test_locked_test_protocol_is_documented():
    selection = json.loads(
        (ROOT / "reports" / "metrics" / "model_selection.json").read_text(encoding="utf-8")
    )
    assert selection["test_protocol"].startswith("selected model only")
    assert selection["selected_model"] == selection["test_metrics"]["selected_model"]


def test_forbidden_artifact_scan_excludes_runtime_environment(tmp_path):
    qa = runpy.run_path(str(ROOT / "scripts" / "run_qa.py"))
    find_forbidden_artifacts = qa["find_forbidden_artifacts"]

    runtime_cert = tmp_path / ".venv" / "Lib" / "site-packages" / "certifi" / "cacert.pem"
    runtime_cert.parent.mkdir(parents=True)
    runtime_cert.write_text("runtime certificate bundle", encoding="utf-8")

    source_key = tmp_path / "docs" / "leaked.pem"
    source_key.parent.mkdir(parents=True)
    source_key.write_text("publication-boundary artifact", encoding="utf-8")

    nested_fake_runtime = tmp_path / "docs" / ".venv" / "leaked.pem"
    nested_fake_runtime.parent.mkdir(parents=True)
    nested_fake_runtime.write_text("must remain in publication boundary", encoding="utf-8")

    findings = find_forbidden_artifacts(tmp_path)

    assert findings == ["docs/.venv/leaked.pem", "docs/leaked.pem"]
