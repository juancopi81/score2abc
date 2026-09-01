from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from scripts.experiments import materialize_repaired_full_event_sidecar_v2 as sidecar


def test_materializes_and_replays_duration_aware_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    validated = _validated_fixture(tmp_path)
    _patch_runtime(monkeypatch, validated, _materialized_result)
    canonical_before = sidecar.v1._canonical_bytes(validated["inference_dir"])
    existing_v1 = validated["inference_dir"] / sidecar.v1.DEFAULT_OUTPUT_DIRNAME
    existing_v1.mkdir()
    v1_payload = existing_v1 / "manifest.json"
    v1_payload.write_text('{"version": "v1"}\n', encoding="utf-8")
    v1_bytes = v1_payload.read_bytes()

    result = sidecar.materialize_repaired_full_event_sidecar_v2(
        validated["inference_dir"], model_dir=validated["model_dir"]
    )

    output_dir = Path(result["output_dir"])
    assert output_dir.name == sidecar.DEFAULT_OUTPUT_DIRNAME
    assert result["duration_evidence_applied_count"] == 1
    assert sidecar.v1._canonical_bytes(validated["inference_dir"]) == canonical_before
    assert v1_payload.read_bytes() == v1_bytes
    manifest = _read_json(output_dir / "manifest.json")
    assert manifest["kind"] == sidecar.MANIFEST_KIND
    assert manifest["version"] == sidecar.SIDECAR_VERSION
    assert manifest["duration_evidence_applied_count"] == 1
    assert manifest["contract"]["existing_v1_artifacts_unchanged"] is True
    assert manifest["contract"]["duration_override_scope"] == ("sparse_shared_stem_dotted_half")
    assert set(manifest["upstream"]) == {
        "multihead_manifest",
        "multihead_lane",
        "sparse_manifest",
        "sparse_lane",
        "sparse_diagnostics",
    }
    prediction = _read_jsonl(output_dir / "predictions.jsonl")[0]
    assert prediction["notes"] == [
        {"duration_beats": 3.0, "onset_beats": 0.0, "pitch_midi": 60},
        {"duration_beats": 3.0, "onset_beats": 0.0, "pitch_midi": 64},
    ]
    verified = sidecar.verify_repaired_full_event_sidecar_v2(
        output_dir, model_dir=validated["model_dir"]
    )
    assert verified["verified"] is True
    assert verified["duration_evidence_applied_count"] == 1


def test_failed_row_publishes_no_sidecar_or_temporary_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    validated = _validated_fixture(tmp_path)

    def failed(*args: Any, **kwargs: Any) -> dict[str, Any]:
        result = _materialized_result(*args, **kwargs)
        result.update(
            status=sidecar.base_compositor.STATUS_REQUEST_METER_FALLBACK,
            prediction=None,
            meter_valid=False,
        )
        return result

    _patch_runtime(monkeypatch, validated, failed)
    output_dir = validated["inference_dir"] / sidecar.DEFAULT_OUTPUT_DIRNAME

    with pytest.raises(ValueError, match="Full-event v2 composition failed closed"):
        sidecar.materialize_repaired_full_event_sidecar_v2(
            validated["inference_dir"], model_dir=validated["model_dir"]
        )

    assert not output_dir.exists()
    assert not output_dir.with_name(f".{output_dir.name}.tmp").exists()


def test_validates_sparse_diagnostics_identity_and_truth_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    validated = _validated_fixture(tmp_path)
    base = dict(validated)
    base.pop("sparse_repair_decisions")
    base.pop("sparse_diagnostic_rows")
    base.pop("sparse_diagnostics_path")
    monkeypatch.setattr(sidecar.v1, "_validate_inputs", lambda *args, **kwargs: base)

    checked = sidecar._validate_inputs(validated["inference_dir"], model_dir=None)
    assert checked["sparse_repair_decisions"] == validated["sparse_repair_decisions"]

    diagnostics_path = validated["sparse_diagnostics_path"]
    row = _read_jsonl(diagnostics_path)[0]
    row["truth_used"] = True
    _write_jsonl(diagnostics_path, [row])
    with pytest.raises(ValueError, match="not explicitly truth-blind"):
        sidecar._validate_inputs(validated["inference_dir"], model_dir=None)


def _validated_fixture(tmp_path: Path) -> dict[str, Any]:
    inference_dir = tmp_path / "inference_v1"
    multihead_dir = inference_dir / sidecar.v1.runner.MULTIHEAD_RECOVERY_DIRNAME
    sparse_dir = inference_dir / sidecar.v1.runner.SPARSE_DYAD_REPAIR_DIRNAME
    model_dir = tmp_path / "model"
    out_dir = tmp_path / "out"
    for path in (inference_dir, multihead_dir, sparse_dir, model_dir, out_dir):
        path.mkdir(parents=True, exist_ok=True)

    identity = {
        "slug": "synthetic-score",
        "system_index": 1,
        "system_measure_index": 1,
        "automatic_measure_index": 1,
    }
    request = {
        "identity": identity,
        "allowed_context": {"expected_measure_beats": "3"},
        "truth_accessed": False,
    }
    inference_row = {"identity": identity, "truth_used": False}
    lane = [
        {"candidate_id": "d001", "center": {"x": 35.0, "y": 35.0}, "onset_group_index": 1},
        {"candidate_id": "d002", "center": {"x": 36.0, "y": 45.0}, "onset_group_index": 1},
    ]
    repair_row = {
        "identity": identity,
        "truth_used": False,
        "lanes": {"sparse_dyad_repair": {"candidate_lane": lane}},
    }
    decision = {
        "accepted": True,
        "reason": "accepted",
        "truth_used": False,
        "chosen_pair": {
            "candidate_ids": ["d001", "d002"],
            "augmentation_dot_pairs": [{"candidate_ids": ["d050", "d051"]}],
        },
    }
    diagnostic_row = {
        "identity": identity,
        "sparse_repair": decision,
        "truth_accessed": False,
        "truth_used": False,
    }

    _write_jsonl(inference_dir / "requests.jsonl", [request])
    _write_jsonl(inference_dir / "predictions.jsonl", [{"identity": identity}])
    _write_jsonl(inference_dir / "inference.jsonl", [inference_row])
    _write_json(inference_dir / "manifest.json", {"target": identity})
    _write_json(multihead_dir / "manifest.json", {})
    _write_jsonl(multihead_dir / "recovery_lane.jsonl", [{"identity": identity}])
    _write_json(sparse_dir / "manifest.json", {})
    _write_jsonl(sparse_dir / "repair_lane.jsonl", [repair_row])
    diagnostics_path = sparse_dir / "diagnostics.jsonl"
    _write_jsonl(diagnostics_path, [diagnostic_row])

    return {
        "inference_dir": inference_dir,
        "main": {"target": {"slug": identity["slug"], "system_index": 1}},
        "multihead_dir": multihead_dir,
        "sparse_dir": sparse_dir,
        "sparse_diagnostics_path": diagnostics_path,
        "model_dir": model_dir,
        "out_dir": out_dir,
        "pins": {},
        "selector_method_id": "synthetic-selector-v1",
        "pitch_predictor": object(),
        "requests": [request],
        "inference_rows": [inference_row],
        "repair_rows": [repair_row],
        "sparse_diagnostic_rows": [diagnostic_row],
        "sparse_repair_decisions": [decision],
    }


def _patch_runtime(
    monkeypatch: pytest.MonkeyPatch,
    validated: dict[str, Any],
    compose: Any,
) -> None:
    monkeypatch.setattr(sidecar, "_validate_inputs", lambda *args, **kwargs: validated)
    monkeypatch.setattr(sidecar.compositor, "compose_repaired_candidate_events_v2", compose)

    def write_overlay(*args: Any, output_path: Path, **kwargs: Any) -> None:
        Image.new("RGB", (80, 60), "white").save(output_path)

    def write_contact_sheet(paths: list[Path], output_path: Path) -> None:
        assert paths
        Image.new("RGB", (80, 80), "white").save(output_path)

    monkeypatch.setattr(sidecar.v1, "_write_repaired_overlay", write_overlay)
    monkeypatch.setattr(sidecar.v1.runner, "_write_contact_sheet", write_contact_sheet)


def _materialized_result(
    request: dict[str, Any],
    inference_row: dict[str, Any],
    lane: list[dict[str, Any]],
    decision: dict[str, Any],
    **kwargs: Any,
) -> dict[str, Any]:
    del inference_row, decision, kwargs
    identity = request["identity"]
    notes = [
        {"duration_beats": 3.0, "onset_beats": 0.0, "pitch_midi": 60},
        {"duration_beats": 3.0, "onset_beats": 0.0, "pitch_midi": 64},
    ]
    prediction = {
        "identity": identity,
        "notes": notes,
        "rests": [],
        "rhythm_tokens": [
            {"duration_beats": 3.0, "kind": "note", "note_count": 2, "onset_beats": 0.0}
        ],
        "measure_extent_beats": 3.0,
        "decoder_status": "duration-evidence",
    }
    anchors = [{"source": {"candidate_id": row["candidate_id"]}} for row in lane]
    return {
        "identity": identity,
        "status": sidecar.base_compositor.STATUS_MATERIALIZED,
        "prediction": prediction,
        "groups": [
            {
                "group_id": "g001",
                "onset_group_index": 1,
                "anchors": anchors,
            }
        ],
        "automatic_key_context": {"truth_used": False},
        "decoder_status": "duration-evidence",
        "expected_measure_beats": 3.0,
        "observed_extent_beats": 3.0,
        "decoded_extent_beats": 3.0,
        "meter_valid": True,
        "duration_evidence": {
            "schema_version": 1,
            "kind": "sparse_shared_stem_dotted_half",
            "applied": True,
            "reason": "accepted_visual_dotted_half_dyad",
            "truth_used": False,
        },
        "v1_base_composition": {
            "status": "not_materialized_request_only_meter_fallback",
            "decoder_status": "fallback",
            "observed_extent_beats": 1.0,
            "decoded_extent_beats": 3.0,
            "meter_valid": False,
        },
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
