import json
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from scripts.experiments import freeze_independent_dyad_recovery_gate as gate
from scripts.experiments import freeze_third_score_heldout as base
from scripts.experiments import run_third_score_heldout_inference as shared_runner


def test_prepares_fixed_eleven_crop_unknown_context_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    boundaries = [index / gate.EXPECTED_CROP_COUNT for index in range(12)]
    monkeypatch.setattr(base, "detect_barlines", lambda _path: boundaries)
    monkeypatch.setattr(base, "measure_boundaries_for_system", lambda _path, _raw: boundaries)
    _write_target(tmp_path)

    result = gate.prepare_independent_dyad_recovery_gate(tmp_path, namespace="test-v1")

    prepared_path = Path(result["prepared_manifest"])
    prepared = json.loads(prepared_path.read_text(encoding="utf-8"))
    context = json.loads(
        (
            prepared_path.parent / prepared["artifacts"]["context"]["allowed_context"]["path"]
        ).read_text(encoding="utf-8")
    )
    evaluator = json.loads(
        (prepared_path.parent / prepared["artifacts"]["evaluator"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    assert prepared["target"] == {
        "slug": gate.NO_LO_CREAS_SLUG,
        "system_index": gate.TARGET_SYSTEM_INDEX,
    }
    assert len(prepared["artifacts"]["crops"]) == 11
    assert prepared["independent_dyad_recovery_gate"]["expected_crop_count"] == 11
    assert context["allowed_context"] == {
        "allow_pickup": False,
        "clef": "treble",
        "expected_measure_beats": None,
        "key_hint": None,
        "time_signature": None,
    }
    assert evaluator["supported_metrics"] == [
        "candidate_localization",
        "note_count",
        "diatonic_pitch",
    ]
    assert "meter" in evaluator["unsupported_metrics"]
    assert any("out/local_restricted" in path for path in prepared["forbidden_truth_paths"])
    base._verify_prepared_manifest(
        prepared_path.parent,
        prepared_path,
        prepared,
        expected_kind=gate.INDEPENDENT_DYAD_RECOVERY_GATE.prepare_kind,
    )
    assert gate.INDEPENDENT_DYAD_RECOVERY_GATE.prepare_kind in shared_runner.GATE_CONFIGS
    assert (
        shared_runner.GATE_CONFIGS[gate.INDEPENDENT_DYAD_RECOVERY_GATE.prepare_kind][
            "inference_version"
        ]
        == "independent-dyad-baseline-inference-v1"
    )


def test_rejects_target_substitution_before_creating_namespace(tmp_path: Path) -> None:
    _write_target(tmp_path)
    substituted = (base.Candidate(gate.NO_LO_CREAS_SLUG, 7, "fixed_preregistered_target"),)

    with pytest.raises(ValueError, match="target is fixed"):
        gate.prepare_independent_dyad_recovery_gate(
            tmp_path,
            candidate_pool=substituted,
        )

    assert not (
        tmp_path / gate.NO_LO_CREAS_SLUG / gate.OUTPUT_SUBDIR / gate.DEFAULT_NAMESPACE
    ).exists()


def _write_target(root: Path) -> None:
    score_root = root / gate.NO_LO_CREAS_SLUG
    system_path = score_root / "systems" / f"system_{gate.TARGET_SYSTEM_INDEX:03d}.png"
    system_path.parent.mkdir(parents=True)
    image = Image.new("L", (1320, 120), "white")
    draw = ImageDraw.Draw(image)
    for y in (35, 45, 55, 65, 75):
        draw.line((0, y, 1319, y), fill="black", width=2)
    image.save(system_path)
    (score_root / "metadata.json").write_text(
        json.dumps(
            {
                "title": "No lo Creas",
                "composer": "A. Vasquez Pedrero",
                "rhythm": "Pasillo",
                "time_signature": "3/4",
                "key_hint": "withheld",
            }
        )
        + "\n",
        encoding="utf-8",
    )
