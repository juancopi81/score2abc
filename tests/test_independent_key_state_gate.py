import json
from pathlib import Path

from PIL import Image, ImageDraw

from scripts.experiments import freeze_independent_key_state_gates as freezer
from scripts.experiments import freeze_third_score_heldout as base
from scripts.experiments import run_independent_key_state_gate as runner
from scripts.experiments import run_third_score_heldout_inference as shared_runner


def test_prepare_pins_truth_blind_key_context(tmp_path: Path, monkeypatch) -> None:
    slug = "synthetic-score"
    case = freezer.KeyGateCase(
        case_id="synthetic_initial",
        slug=slug,
        target_system_index=1,
        key_source_system_index=1,
        detector_mode=freezer.detector.MODE_INITIAL,
        expected_crop_count=6,
        policy=base.LayoutPolicy(
            min_width_px=500,
            min_height_px=50,
            min_measure_count=6,
            max_measure_count=6,
            min_crop_width_px=50,
            max_spacing_cv=0.1,
        ),
        allow_pickup=True,
    )
    boundaries = tuple(index / 6 for index in range(7))
    monkeypatch.setattr(base, "detect_barlines", lambda _path: list(boundaries))
    monkeypatch.setattr(
        base,
        "measure_boundaries_for_system",
        lambda _path, _detected: list(boundaries),
    )
    system = _system_image(tmp_path, slug, 1)
    _metadata(tmp_path, slug)
    monkeypatch.setattr(
        freezer.detector,
        "detect_signature",
        lambda path, mode: {
            "input": {"path": str(path), "sha256": base._sha256(path)},
            "mode": mode,
            "fifths": -1,
            "gate_passed": True,
            "truth_used_for_prediction": False,
        },
    )
    monkeypatch.setattr(
        freezer.detector,
        "_draw_overlay",
        lambda _prediction, path: Image.open(system).save(path),
    )

    report = freezer.prepare_case(tmp_path, case=case)

    prepared_path = Path(report["prepared_manifest"])
    prepared = json.loads(prepared_path.read_text(encoding="utf-8"))
    context_record = prepared["artifacts"]["context"]["allowed_context"]
    context = json.loads(
        (prepared_path.parent / context_record["path"]).read_text(encoding="utf-8")
    )
    assert prepared["truth_accessed"] is False
    assert prepared["independent_key_gate"]["automatic_fifths"] == -1
    assert prepared["independent_key_gate"]["baseline_fifths"] is None
    assert context["allowed_context"]["key_hint"] == "1 flat(s): Bb"
    assert context["allowed_context"]["expected_measure_beats"] is None
    assert context["truth_used"] is False
    base._verify_prepared_manifest(
        prepared_path.parent,
        prepared_path,
        prepared,
        expected_kind=case.gate.prepare_kind,
    )


def test_build_paired_predictions_changes_only_key_pitch() -> None:
    rows = [
        {
            "identity": {
                "slug": "synthetic-score",
                "system_index": 1,
                "system_measure_index": 1,
            },
            "truth_used": False,
            "source": {"image": "synthetic.png", "sha256": "abc"},
            "staff_geometry": {"raw_staff_lines_y_px": [10, 20, 30, 40, 50]},
            "automatic_anchors": [
                {
                    "order": 1,
                    "pitch": "ignored",
                    "center": {"x": 25.0, "y": 30.0},
                    "source": {"candidate_id": "c001"},
                }
            ],
        }
    ]

    paired, invariance = runner.build_paired_predictions(rows, automatic_fifths=-1)

    baseline = paired[0]["lanes"]["global_no_key"]["notes"][0]
    automatic = paired[0]["lanes"]["global_automatic_key"]["notes"][0]
    assert baseline["candidate_id"] == automatic["candidate_id"] == "c001"
    assert baseline["center"] == automatic["center"] == {"x": 25.0, "y": 30.0}
    assert baseline["pitch"] == "B4"
    assert automatic["pitch"] == "Bb4"
    assert automatic["pitch_midi"] == baseline["pitch_midi"] - 1
    assert invariance["passed"] is True
    assert invariance["note_count"] == 1


def test_shared_runner_registers_both_independent_gates() -> None:
    prepare_kinds = {case.gate.prepare_kind for case in freezer.CASES.values()}
    assert prepare_kinds <= set(shared_runner.GATE_CONFIGS)


def _system_image(root: Path, slug: str, index: int) -> Path:
    path = root / slug / "systems" / f"system_{index:03d}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("L", (840, 120), "white")
    draw = ImageDraw.Draw(image)
    for y in (35, 45, 55, 65, 75):
        draw.line((0, y, 839, y), fill="black", width=2)
    image.save(path)
    return path


def _metadata(root: Path, slug: str) -> None:
    path = root / slug / "metadata.json"
    path.write_text(
        json.dumps(
            {
                "title": "Synthetic",
                "composer": "Test",
                "rhythm": "Danza",
                "time_signature": None,
                "key_hint": None,
            }
        )
        + "\n",
        encoding="utf-8",
    )
