import json
from pathlib import Path

from score2abc.evaluation import evaluate
from score2abc.manifest import write_manifest_jsonl
from score2abc.schemas import WorkItem, WorkMetadata


def _write_events(path: Path, time_signature: str = "4/4") -> None:
    payload = {
        "time_signature": time_signature,
        "notes": [
            {"measure": 1, "onset_beats": 0.0, "duration_beats": 1.0, "pitch_midi": 60},
            {"measure": 1, "onset_beats": 1.0, "duration_beats": 1.0, "pitch_midi": 62},
        ],
        "chords": [
            {"measure": 1, "onset_beats": 0.0, "symbol": "C"},
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def test_evaluate_writes_report(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    ground_truth_dir = tmp_path / "ground_truth"

    metadata = WorkMetadata(title="Demo", composer="Composer", rhythm="Pasillo")
    item = WorkItem(slug="demo", pdf_path=tmp_path / "demo.pdf", metadata=metadata)

    out_dir.mkdir(parents=True)
    write_manifest_jsonl([item], out_dir / "manifest.jsonl")

    pred_path = out_dir / "demo" / "intermediate" / "events.json"
    truth_path = ground_truth_dir / "demo.json"
    _write_events(pred_path)
    _write_events(truth_path)

    result = evaluate(out_dir, ground_truth_dir)
    assert result == 0

    report_path = out_dir / "eval" / "report.json"
    assert report_path.exists()

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["summary"]["works_evaluated"] == 1
    assert report["summary"]["evaluation_coverage"] == 1.0
    assert report["summary"]["coverage_gate_passed"] is True
    assert report["summary"]["note_count_match_rate"] == 1.0
    assert report["summary"]["note_f1_avg"] == 1.0
    assert report["summary"]["chord_f1_measure_only_avg"] == 1.0
    assert report["summary"]["chord_precision_measure_only_avg"] == 1.0
    assert report["summary"]["chord_recall_measure_only_avg"] == 1.0
    assert report["works"][0]["slug"] == "demo"


def test_evaluate_fails_when_coverage_is_too_low(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    ground_truth_dir = tmp_path / "ground_truth"

    item1 = WorkItem(
        slug="demo-1",
        pdf_path=tmp_path / "demo-1.pdf",
        metadata=WorkMetadata(title="Demo 1", composer="Composer", rhythm="Pasillo"),
    )
    item2 = WorkItem(
        slug="demo-2",
        pdf_path=tmp_path / "demo-2.pdf",
        metadata=WorkMetadata(title="Demo 2", composer="Composer", rhythm="Pasillo"),
    )

    out_dir.mkdir(parents=True)
    write_manifest_jsonl([item1, item2], out_dir / "manifest.jsonl")

    _write_events(out_dir / "demo-1" / "intermediate" / "events.json")
    _write_events(ground_truth_dir / "demo-1.json")
    _write_events(ground_truth_dir / "demo-2.json")

    result = evaluate(out_dir, ground_truth_dir)
    assert result == 1

    report = json.loads((out_dir / "eval" / "report.json").read_text(encoding="utf-8"))
    assert report["summary"]["evaluation_coverage"] == 0.5
    assert report["summary"]["coverage_gate_passed"] is False
