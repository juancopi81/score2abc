import json
from pathlib import Path
from typing import Any, Sequence

import pytest

from scripts.experiments import evaluate_independent_dyad_recovery_gate as evaluator
from scripts.experiments import freeze_third_score_heldout as freezer
from scripts.experiments import run_independent_dyad_recovery_gate as dyad_runner


@pytest.fixture
def paired_rows_with_canonical_pitch_difference() -> list[dict[str, Any]]:
    rows = _paired_rows()
    baseline = rows[0]["lanes"][evaluator.LANE_BASELINE]
    baseline["canonical_prediction"]["notes"][0]["pitch"] = "G5"
    baseline["canonical_prediction_sha256"] = dyad_runner.inference._hash_json(
        baseline["canonical_prediction"]
    )
    return rows


def test_evaluates_paired_lanes_with_nontrivial_mapping_and_ignores_accidentals(
    tmp_path: Path,
) -> None:
    fixture = _frozen_fixture(tmp_path)
    musicxml = _write_musicxml(tmp_path / "truth.musicxml")
    mapping = _write_mapping(tmp_path / "mapping.json")

    result = evaluator.evaluate_independent_dyad_recovery_gate(
        fixture["sealed"],
        musicxml_path=musicxml,
        mapping_path=mapping,
    )

    report = _read_json(Path(result["report"]))
    materialized = _read_json(Path(result["mapping"]))
    manifest = _read_json(Path(result["manifest"]))
    baseline = report["lanes"][evaluator.LANE_BASELINE]["summary"]
    recovered = report["lanes"][evaluator.LANE_RECOVERED]["summary"]

    assert report["target"] == {
        "slug": evaluator.TARGET_SLUG,
        "system_index": evaluator.TARGET_SYSTEM_INDEX,
    }
    assert report["mapping_mode"] == "explicit_user_mapping"
    assert materialized["automatic_crops"][1]["physical_measure_spans"] == [
        {"measure_number": 1, "note_end": 3, "note_start": 2},
        {"measure_number": 2, "note_end": 2, "note_start": 0},
    ]
    assert baseline["predicted_note_count"] == 4
    assert recovered["predicted_note_count"] == 6
    assert recovered["truth_note_count"] == 6
    assert baseline["note_count_f1"] == 0.8
    assert recovered["note_count_f1"] == 1.0
    assert baseline["exact_diatonic_staff_position_matches"] == 4
    assert recovered["exact_diatonic_staff_position_matches"] == 6
    assert baseline["exact_chord_size_matches"] == 2
    assert recovered["exact_chord_size_matches"] == 4
    assert recovered["exact_structure_crops"] == 3
    assert report["comparison"] == {
        "exact_chord_size_match_delta": 2,
        "exact_diatonic_staff_position_match_delta": 2,
        "exact_structure_crop_delta": 2,
        "note_count_f1_delta": 0.2,
        "ordered_diatonic_alignment_accuracy_delta": 0.333333,
        "predicted_note_count_delta": 2,
    }
    assert report["metric_support"] == {
        "absolute_onset_and_rhythm": evaluator.NOT_SCORED_LOCALIZATION_GATE,
        "duration": evaluator.NOT_SCORED_LOCALIZATION_GATE,
        "key_signature_and_accidentals": evaluator.NOT_SCORED_LOCALIZATION_GATE,
        "meter": evaluator.NOT_SCORED_LOCALIZATION_GATE,
        "note_count_precision_recall_f1": "scored",
        "onset_group_chord_size_structure": "scored",
        "ordered_diatonic_pitch_staff_position": "scored_ignore_key_signature_accidentals",
        "rests": evaluator.NOT_SCORED_LOCALIZATION_GATE,
    }
    assert report["frozen_context"] == {
        "key_signature": "unknown",
        "meter": "unsupported_not_frozen",
        "rests": "unsupported_not_frozen",
        "rhythm": "unsupported_not_frozen",
    }
    assert report["source_musicxml_context_not_scored"]["key_fifths"] == -1
    assert manifest["truth_opened_after_all_frozen_hashes_verified"] is True
    assert manifest["pins"]["mapping"]["source_sha256"] == freezer._sha256(mapping)


def test_frozen_hash_drift_fails_before_opening_musicxml(tmp_path: Path) -> None:
    fixture = _frozen_fixture(tmp_path)
    fixture["model_snapshot"].write_text("drift\n", encoding="utf-8")
    opened = False

    def truth_loader(path: Path) -> evaluator.heldout.VisibleMusicXMLTruth:
        nonlocal opened
        opened = True
        return evaluator.heldout.load_visible_musicxml_truth(path)

    with pytest.raises(ValueError, match="snapshot hash drift"):
        evaluator.evaluate_independent_dyad_recovery_gate(
            fixture["sealed"],
            musicxml_path=_write_musicxml(tmp_path / "truth.musicxml"),
            mapping_path=_write_mapping(tmp_path / "mapping.json"),
            truth_loader=truth_loader,
        )

    assert opened is False


def test_contract_allows_canonical_pitch_to_differ_from_normalized_baseline(
    paired_rows_with_canonical_pitch_difference: list[dict[str, Any]],
) -> None:
    baseline = paired_rows_with_canonical_pitch_difference[0]["lanes"][evaluator.LANE_BASELINE]

    evaluator._validate_paired_prediction_contract(paired_rows_with_canonical_pitch_difference)

    assert baseline["canonical_prediction"]["notes"][0]["pitch"] == "G5"
    assert baseline["notes"][0]["pitch"] == "C4"


def test_existing_evaluation_refuses_overwrite_before_reopening_truth(tmp_path: Path) -> None:
    fixture = _frozen_fixture(tmp_path)
    existing = fixture["namespace_root"] / "evaluation_prior"
    existing.mkdir()
    opened = False

    def truth_loader(path: Path) -> evaluator.heldout.VisibleMusicXMLTruth:
        nonlocal opened
        opened = True
        return evaluator.heldout.load_visible_musicxml_truth(path)

    with pytest.raises(FileExistsError, match="already exists"):
        evaluator.evaluate_independent_dyad_recovery_gate(
            fixture["sealed"],
            musicxml_path=_write_musicxml(tmp_path / "truth.musicxml"),
            mapping_path=_write_mapping(tmp_path / "mapping.json"),
            truth_loader=truth_loader,
        )

    assert opened is False


def test_mapping_that_splits_a_chord_is_reported_as_partially_unsupported(
    tmp_path: Path,
) -> None:
    truth = evaluator.heldout.load_visible_musicxml_truth(
        _write_musicxml(tmp_path / "truth.musicxml")
    )
    mapping = evaluator.heldout.validate_and_materialize_mapping(
        {
            "schema_version": 1,
            "automatic_crops": [
                {
                    "automatic_crop_index": 1,
                    "physical_measure_spans": [
                        {"measure_number": 1, "note_start": 0, "note_end": 1}
                    ],
                },
                {
                    "automatic_crop_index": 2,
                    "physical_measure_spans": [
                        {"measure_number": 1, "note_start": 1, "note_end": 3},
                        {"measure_number": 2, "note_start": 0, "note_end": 2},
                    ],
                },
                {
                    "automatic_crop_index": 3,
                    "physical_measure_spans": [
                        {"measure_number": 2, "note_start": 2, "note_end": 3}
                    ],
                },
            ],
        },
        truth=truth,
        crop_indices=(1, 2, 3),
        mode="explicit_user_mapping",
    )

    support = evaluator._mapping_structure_support(mapping, truth)

    assert support[1]["well_defined"] is False
    assert support[2]["well_defined"] is False
    assert support[3]["well_defined"] is True
    assert support[1]["split_onset_group_boundaries"] == [
        {"boundary": "end", "measure_number": 1, "note_index": 1}
    ]


def test_help_names_expected_no_lo_creas_transcription_path(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        evaluator.main(["--help"])

    assert exc_info.value.code == 0
    assert evaluator.EXPECTED_TRANSCRIPTION_PATH.as_posix() in capsys.readouterr().out


def _frozen_fixture(tmp_path: Path) -> dict[str, Any]:
    out_dir = tmp_path / "out/local_restricted"
    namespace_root = (
        out_dir
        / evaluator.TARGET_SLUG
        / evaluator.OUTPUT_SUBDIR
        / evaluator.DEFAULT_NAMESPACE
        / f"system_{evaluator.TARGET_SYSTEM_INDEX:03d}"
    )
    namespace_root.mkdir(parents=True)
    source_system = (
        out_dir
        / evaluator.TARGET_SLUG
        / "systems"
        / f"system_{evaluator.TARGET_SYSTEM_INDEX:03d}.png"
    )
    source_system.parent.mkdir(parents=True)
    source_system.write_bytes(b"synthetic-system")

    selection = namespace_root / "selection.json"
    evaluator_spec = namespace_root / "evaluator_spec.json"
    requests_path = namespace_root / "requests.jsonl"
    selection.write_text('{"selected":"system_008"}\n', encoding="utf-8")
    evaluator_spec.write_text('{"version":"synthetic"}\n', encoding="utf-8")

    requests = []
    crops = []
    for index in range(1, 4):
        crop = namespace_root / "crops" / f"measure_{index:03d}.png"
        crop.parent.mkdir(exist_ok=True)
        crop.write_bytes(f"crop-{index}".encode())
        crops.append(
            {
                "measure_index": index,
                "path": crop.relative_to(namespace_root).as_posix(),
                "sha256": freezer._sha256(crop),
            }
        )
        requests.append(
            {
                "schema_version": 1,
                "split": freezer.SPLIT_NAME,
                "truth_accessed": False,
                "identity": {
                    "slug": evaluator.TARGET_SLUG,
                    "system_index": evaluator.TARGET_SYSTEM_INDEX,
                    "automatic_measure_index": index,
                },
            }
        )
    freezer._write_jsonl(requests_path, requests)
    prepared_path = namespace_root / "prepared_manifest.json"
    freezer._write_json(
        prepared_path,
        {
            "schema_version": 1,
            "kind": evaluator.GATE_SPEC.prepare_kind,
            "status": "prepared_awaiting_model_predictions",
            "split": freezer.SPLIT_NAME,
            "truth_accessed": False,
            "truth_must_be_added_only_after_freeze": True,
            "namespace": evaluator.DEFAULT_NAMESPACE,
            "target": {
                "slug": evaluator.TARGET_SLUG,
                "system_index": evaluator.TARGET_SYSTEM_INDEX,
            },
            "forbidden_truth_paths": [],
            "independent_dyad_recovery_gate": {
                "config_id": dyad_runner.recovery.EDGE_SAFE_STEM_DYAD_CONFIG_ID,
                "parameters": dict(dyad_runner.recovery.EDGE_SAFE_STEM_DYAD_PARAMETERS),
                "expected_crop_count": 3,
                "truth_accessed": False,
                "truth_used": False,
            },
            "artifacts": {
                "selection": {"path": selection.name, "sha256": freezer._sha256(selection)},
                "requests": {
                    "path": requests_path.name,
                    "sha256": freezer._sha256(requests_path),
                    "row_sha256": [freezer._hash_json(row) for row in requests],
                },
                "evaluator": {
                    "path": evaluator_spec.name,
                    "sha256": freezer._sha256(evaluator_spec),
                },
                "source_system": {
                    "path_relative_to_out": source_system.relative_to(out_dir).as_posix(),
                    "sha256": freezer._sha256(source_system),
                },
                "crops": crops,
            },
        },
    )

    artifacts = tmp_path / "blind-artifacts"
    artifacts.mkdir()
    paired_rows = _paired_rows()
    model = artifacts / "model.json"
    training = artifacts / "training.json"
    model.write_text('{"model":"synthetic-score-disjoint"}\n', encoding="utf-8")
    training.write_text('{"truth_used":false}\n', encoding="utf-8")
    baseline_inference = namespace_root / dyad_runner.DEFAULT_INFERENCE_DIRNAME
    baseline_inference.mkdir()
    baseline_predictions = baseline_inference / "predictions.jsonl"
    freezer._write_jsonl(
        baseline_predictions,
        [row["lanes"][evaluator.LANE_BASELINE]["canonical_prediction"] for row in paired_rows],
    )
    baseline_manifest = baseline_inference / "manifest.json"
    freezer._write_json(baseline_manifest, {"kind": "synthetic_generic_inference"})

    pair_dir = baseline_inference / dyad_runner.PAIR_DIRNAME
    pair_dir.mkdir()
    paired_predictions = pair_dir / "paired_predictions.jsonl"
    freezer._write_jsonl(paired_predictions, paired_rows)
    invariance = pair_dir / "additive_invariance.json"
    freezer._write_json(
        invariance,
        {
            "kind": "independent_dyad_recovery_additive_invariance",
            "passed": True,
            "truth_accessed": False,
            "truth_used": False,
        },
    )
    pair_manifest = pair_dir / "manifest.json"
    freezer._write_json(
        pair_manifest,
        {
            "schema_version": dyad_runner.SCHEMA_VERSION,
            "kind": "independent_dyad_recovery_paired_prediction_manifest",
            "version": dyad_runner.PAIR_VERSION,
            "target": {
                "slug": evaluator.TARGET_SLUG,
                "system_index": evaluator.TARGET_SYSTEM_INDEX,
            },
            "truth_accessed": False,
            "truth_used": False,
        },
    )

    paired_frozen = namespace_root / "dyad_recovery_frozen"
    paired_frozen.mkdir()
    groups = {
        "paired_predictions": [paired_predictions],
        "paired_artifacts": [pair_manifest, invariance],
        "prepared_and_source": [
            source_system,
            prepared_path,
            selection,
            requests_path,
            evaluator_spec,
            *[namespace_root / crop["path"] for crop in crops],
        ],
        "baseline_inference": [baseline_predictions, baseline_manifest],
        "model_and_training": [model, training],
        "implementations": [Path(evaluator.__file__), Path(dyad_runner.__file__)],
    }
    pins = {
        role: [
            freezer._snapshot_artifact(
                path,
                frozen_dir=paired_frozen,
                role=role,
                index=index,
            )
            for index, path in enumerate(paths, start=1)
        ]
        for role, paths in groups.items()
    }
    model_snapshot = (
        namespace_root / pins["model_and_training"][0]["snapshot_path_relative_to_namespace"]
    )
    paired_freeze_path = paired_frozen / "freeze.json"
    freezer._write_json(
        paired_freeze_path,
        {
            "schema_version": dyad_runner.SCHEMA_VERSION,
            "kind": "independent_dyad_recovery_paired_freeze",
            "status": "frozen_awaiting_truth",
            "truth_accessed": False,
            "truth_used": False,
            "target": {
                "slug": evaluator.TARGET_SLUG,
                "system_index": evaluator.TARGET_SYSTEM_INDEX,
            },
            "config_id": dyad_runner.recovery.EDGE_SAFE_STEM_DYAD_CONFIG_ID,
            "parameters": dict(dyad_runner.recovery.EDGE_SAFE_STEM_DYAD_PARAMETERS),
            "paired_manifest_sha256": freezer._sha256(pair_manifest),
            "generic_baseline_predictions_sha256": freezer._sha256(baseline_predictions),
            "prepared_manifest_sha256": freezer._sha256(prepared_path),
            "source_system_sha256": freezer._sha256(source_system),
            "additive_invariance_sha256": freezer._sha256(invariance),
            "pins": pins,
        },
    )
    sealed = paired_frozen / "sealed_manifest.json"
    freezer._write_json(
        sealed,
        {
            "schema_version": dyad_runner.SCHEMA_VERSION,
            "kind": "independent_dyad_recovery_paired_sealed_manifest",
            "status": "frozen_awaiting_truth",
            "truth_accessed": False,
            "truth_used": False,
            "target": {
                "slug": evaluator.TARGET_SLUG,
                "system_index": evaluator.TARGET_SYSTEM_INDEX,
            },
            "freeze": {"path": "freeze.json", "sha256": freezer._sha256(paired_freeze_path)},
        },
    )
    return {
        "namespace_root": namespace_root,
        "sealed": sealed,
        "model_snapshot": model_snapshot,
    }


def _paired_rows() -> list[dict[str, Any]]:
    groups = (
        (
            [_note("c1", -2, 10, 20, 1)],
            [_note("c1", -2, 10, 20, 1), _note("c2", 0, 10, 10, 1, recovered=True)],
        ),
        (
            [_note("c3", 2, 10, 10, 1), _note("c4", -1, 30, 20, 2)],
            [
                _note("c3", 2, 10, 10, 1),
                _note("c4", -1, 30, 20, 2),
                _note("c5", 1, 30, 10, 2, recovered=True),
            ],
        ),
        ([_note("c6", 3, 10, 10, 1)], [_note("c6", 3, 10, 10, 1)]),
    )
    return [
        {
            "schema_version": dyad_runner.SCHEMA_VERSION,
            "identity": {
                "slug": evaluator.TARGET_SLUG,
                "system_index": evaluator.TARGET_SYSTEM_INDEX,
                "automatic_measure_index": index,
            },
            "truth_accessed": False,
            "truth_used": False,
            "lanes": {
                evaluator.LANE_BASELINE: _baseline_lane(baseline),
                evaluator.LANE_RECOVERED: _recovered_lane(baseline, recovered),
            },
        }
        for index, (baseline, recovered) in enumerate(groups, start=1)
    ]


def _note(
    candidate_id: str,
    staff_position: int,
    x: float,
    y: float,
    onset_group_index: int,
    *,
    recovered: bool = False,
) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "center": {"x": x, "y": y},
        "staff_position": staff_position,
        "pitch": _pitch_for_staff_position(staff_position),
        "onset_group_index": onset_group_index,
        "recovered": recovered,
    }


def _baseline_lane(notes: list[dict[str, Any]]) -> dict[str, Any]:
    canonical_notes = [
        {
            key: value
            for key, value in note.items()
            if key not in {"staff_position", "onset_group_index", "recovered"}
        }
        for note in notes
    ]
    canonical = {"notes": canonical_notes, "rests": [], "rhythm_tokens": []}
    return {
        "canonical_prediction": canonical,
        "canonical_prediction_sha256": dyad_runner.inference._hash_json(canonical),
        "candidate_lane": [_candidate(note, recovered=False) for note in notes],
        "notes": notes,
        "recovered_head_count": 0,
    }


def _recovered_lane(
    baseline: list[dict[str, Any]], recovered: list[dict[str, Any]]
) -> dict[str, Any]:
    recovered_ids = [
        str(note["candidate_id"])
        for note in recovered
        if str(note["candidate_id"]) not in {str(item["candidate_id"]) for item in baseline}
    ]
    return {
        "config_id": dyad_runner.recovery.EDGE_SAFE_STEM_DYAD_CONFIG_ID,
        "candidate_lane": [
            _candidate(note, recovered=str(note["candidate_id"]) in recovered_ids)
            for note in recovered
        ],
        "notes": recovered,
        "recovered_head_count": len(recovered_ids),
        "recovered_candidate_ids": recovered_ids,
    }


def _candidate(note: dict[str, Any], *, recovered: bool) -> dict[str, Any]:
    return {
        "candidate_id": note["candidate_id"],
        "center": note["center"],
        "score": 0.9,
        "recovered": recovered,
        "onset_group_index": note["onset_group_index"],
    }


def _pitch_for_staff_position(position: int) -> str:
    return {-2: "C4", -1: "D4", 0: "E4", 1: "F4", 2: "G4", 3: "A4"}[position]


def _write_mapping(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "automatic_crops": [
                    {
                        "automatic_crop_index": 1,
                        "physical_measure_spans": [
                            {"measure_number": 1, "note_start": 0, "note_end": 2}
                        ],
                    },
                    {
                        "automatic_crop_index": 2,
                        "physical_measure_spans": [
                            {"measure_number": 1, "note_start": 2, "note_end": 3},
                            {"measure_number": 2, "note_start": 0, "note_end": 2},
                        ],
                    },
                    {
                        "automatic_crop_index": 3,
                        "physical_measure_spans": [
                            {"measure_number": 2, "note_start": 2, "note_end": 3}
                        ],
                    },
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _write_musicxml(path: Path) -> Path:
    measures: Sequence[Sequence[Sequence[tuple[str, int, int]]]] = (
        ((("C", 4, 0), ("E", 4, -1)), (("G", 4, 0),)),
        ((("D", 4, 0), ("F", 4, 0)), (("A", 4, 0),)),
    )
    measure_xml = []
    for measure_number, onset_groups in enumerate(measures, start=1):
        notes = []
        for group in onset_groups:
            for group_index, (step, octave, alter) in enumerate(group):
                chord = "<chord/>" if group_index else ""
                alter_xml = f"<alter>{alter}</alter>" if alter else ""
                notes.append(
                    "<note>"
                    f"{chord}<pitch><step>{step}</step>{alter_xml}<octave>{octave}</octave></pitch>"
                    "<duration>1</duration><type>quarter</type>"
                    "</note>"
                )
        attributes = ""
        if measure_number == 1:
            attributes = (
                "<attributes><divisions>1</divisions><key><fifths>-1</fifths></key>"
                "<time><beats>3</beats><beat-type>4</beat-type></time>"
                "<clef><sign>G</sign><line>2</line></clef></attributes>"
            )
        measure_xml.append(
            f'<measure number="{measure_number}">{attributes}{"".join(notes)}</measure>'
        )
    path.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<score-partwise version="4.0">'
        '<part-list><score-part id="P1"><part-name>Music</part-name></score-part></part-list>'
        f'<part id="P1">{"".join(measure_xml)}</part>'
        "</score-partwise>\n",
        encoding="utf-8",
    )
    return path


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
