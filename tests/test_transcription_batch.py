import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from scripts import report_transcription_batch as report
from scripts import run_transcription_batch as runner


def test_batch_stops_at_first_incomplete_pair_and_preserves_budget(tmp_path, monkeypatch):
    pdfs = [tmp_path / f"{name}.pdf" for name in ("first", "second", "third")]
    for pdf in pdfs:
        pdf.write_bytes(b"%PDF-" + pdf.stem.encode())
    plan = runner.make_plan(pdfs, runner.budget_cents("30.01"))
    calls = []

    def fake_pair(argv, **kwargs):
        def option(name):
            return argv[argv.index(name) + 1]

        assert option("--backend") == "api"
        assert option("--model") == "gpt-6.2-astra"
        assert option("--completion-policy") == "completion_gate_v1"
        calls.append(option("--budget-usd"))
        work = Path(option("--output"))
        work.mkdir()
        (work / "pair_status.json").write_text(json.dumps({"complete": len(calls) == 1}))
        # The existing command can exit zero and still produce an incomplete pair.
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(runner.subprocess, "run", fake_pair)
    output = tmp_path / "batch"
    assert runner.run_batch(plan, output) == 1
    assert [work["budget_usd"] for work in plan["works"]] == ["10.01", "10.00", "10.00"]
    assert calls == ["10.01", "10.00"]
    status = json.loads((output / "batch_status.json").read_text())
    assert status["state"] == "stopped_without_retry"
    assert [pair["state"] for pair in status["pairs"]] == ["complete", "incomplete"]
    assert not (output / "works/score_003").exists()


def test_report_distinguishes_saved_usage_from_missing_stages(tmp_path):
    pdf = tmp_path / "example.pdf"
    pdf.write_bytes(b"%PDF-example")
    plan = runner.make_plan([pdf], runner.budget_cents("20"))
    raw = json.dumps(plan).encode()
    (tmp_path / "plan.json").write_bytes(raw)
    (tmp_path / "batch_status.json").write_text(
        json.dumps(
            {
                "plan_sha256": hashlib.sha256(raw).hexdigest(),
                "pairs": [{"key": "score_001", "state": "incomplete"}],
            }
        )
    )
    stage = tmp_path / "works/score_001/initial"
    stage.mkdir(parents=True)
    provenance = {
        "auth_mode": "openai_api",
        "stage": "initial",
        "status": "failed",
        "budget_exhausted": True,
        "elapsed_seconds": 82,
        "cost_estimate_usd": "3.041",
        "usage": {"output_tokens": 212},
    }
    data = json.dumps(provenance).encode()
    provenance_path = stage / "stage_provenance.json"
    provenance_path.write_bytes(data)
    assert "No saved API cost estimate yet" in report.build_report(tmp_path)
    (stage / "seal.json").write_text(
        json.dumps(
            {
                "kind": "astra_reproduction_stage",
                "files": {provenance_path.name: hashlib.sha256(data).hexdigest()},
            }
        )
    )
    result = report.build_report(tmp_path)
    assert "0/1 scores finished both passes" in result
    assert "budget limit reached, 1m 22s | No saved result | $3.04 (partial)" in result
    assert "212 output tokens" in result
    provenance_path.write_bytes(data + b"\n")
    with pytest.raises(ValueError, match="Saved stage artifact changed"):
        report.build_report(tmp_path)
