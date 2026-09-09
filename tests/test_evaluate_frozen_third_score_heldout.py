import json
from pathlib import Path
from typing import Any, Sequence

import pytest

from scripts.experiments import evaluate_frozen_third_score_heldout as spike
from scripts.experiments import freeze_third_score_heldout as freezer
from scripts.experiments import run_third_score_heldout_inference as inference


def test_verifies_every_frozen_snapshot_before_opening_truth(tmp_path: Path) -> None:
    fixture = _frozen_fixture(tmp_path, predictions=[[60]] * 7)
    fixture["model_snapshot"].write_text("drift\n", encoding="utf-8")
    opened = False

    def truth_loader(path: Path) -> spike.VisibleMusicXMLTruth:
        nonlocal opened
        opened = True
        return spike.load_visible_musicxml_truth(path)

    with pytest.raises(ValueError, match="Frozen snapshot hash drift"):
        spike.evaluate_frozen_heldout(
            fixture["sealed"],
            musicxml_path=_write_musicxml(tmp_path / "truth.musicxml", [[60]] * 7),
            truth_loader=truth_loader,
        )

    assert opened is False


def test_v2_binding_hash_drift_fails_before_opening_truth(tmp_path: Path) -> None:
    fixture = _frozen_fixture(tmp_path, predictions=[[60]] * 7)
    sealed = json.loads(fixture["sealed"].read_text(encoding="utf-8"))
    sealed["inference_binding_sha256"] = "0" * 64
    freezer._write_json(fixture["sealed"], sealed)
    opened = False

    def truth_loader(path: Path) -> spike.VisibleMusicXMLTruth:
        nonlocal opened
        opened = True
        return spike.load_visible_musicxml_truth(path)

    with pytest.raises(ValueError, match="Sealed inference provenance binding hash drift"):
        spike.evaluate_frozen_heldout(
            fixture["sealed"],
            musicxml_path=_write_musicxml(tmp_path / "truth.musicxml", [[60]] * 7),
            truth_loader=truth_loader,
        )

    assert opened is False


def test_v2_cross_artifact_substitution_fails_before_opening_truth(tmp_path: Path) -> None:
    fixture = _frozen_fixture(tmp_path, predictions=[[60]] * 7)
    freeze = json.loads(fixture["freeze"].read_text(encoding="utf-8"))
    binding = freeze["inference_binding"]
    binding["inference"]["manifest"] = binding["selected_model"]["manifest"]
    freeze["inference_binding"] = binding
    freezer._write_json(fixture["freeze"], freeze)
    sealed = json.loads(fixture["sealed"].read_text(encoding="utf-8"))
    sealed["freeze"]["sha256"] = freezer._sha256(fixture["freeze"])
    sealed["inference_binding_sha256"] = inference._hash_json(binding)
    freezer._write_json(fixture["sealed"], sealed)
    opened = False

    def truth_loader(path: Path) -> spike.VisibleMusicXMLTruth:
        nonlocal opened
        opened = True
        return spike.load_visible_musicxml_truth(path)

    with pytest.raises(ValueError, match="Frozen inference/prepared-manifest binding mismatch"):
        spike.evaluate_frozen_heldout(
            fixture["sealed"],
            musicxml_path=_write_musicxml(tmp_path / "truth.musicxml", [[60]] * 7),
            truth_loader=truth_loader,
        )

    assert opened is False


def test_default_one_to_one_materializes_visible_tie_heads_and_sounding_accidentals(
    tmp_path: Path,
) -> None:
    pitches = [[60], [60], [70], [62], [64], [65], [67]]
    fixture = _frozen_fixture(tmp_path, predictions=pitches)
    musicxml = _write_musicxml(
        tmp_path / "seven.musicxml",
        pitches,
        ties={(1, 0): "start", (2, 0): "stop"},
        alters={(3, 0): -1},
    )
    frozen_before = {
        path: freezer._sha256(path) for path in fixture["frozen_dir"].rglob("*") if path.is_file()
    }

    result = spike.evaluate_frozen_heldout(fixture["sealed"], musicxml_path=musicxml)

    report = json.loads(Path(result["report"]).read_text(encoding="utf-8"))
    mapping = json.loads(Path(result["mapping"]).read_text(encoding="utf-8"))
    truth_rows = _read_jsonl(Path(result["truth"]))
    assert report["mapping_mode"] == "deterministic_default_one_to_one"
    assert report["metrics"]["summary"]["note_count_f1"] == 1.0
    assert report["metrics"]["summary"]["exact_pitch_matches"] == 7
    assert report["metrics"]["summary"]["exact_automatic_crops"] == 7
    assert report["metric_support"]["onset"] == spike.NOT_SCORED
    assert report["metric_support"]["duration"] == spike.NOT_SCORED
    assert report["metric_support"]["rests"] == spike.NOT_SCORED
    assert report["metric_support"]["meter"] == spike.NOT_SCORED
    assert mapping["automatic_crops"][0]["physical_measure_spans"] == [
        {"measure_number": 1, "note_end": 1, "note_start": 0}
    ]
    assert truth_rows[0]["notes"][0]["tie_start"] is True
    assert truth_rows[1]["notes"][0]["tie_stop"] is True
    assert truth_rows[2]["notes"][0]["pitch"] == "Bb4"
    assert truth_rows[2]["notes"][0]["pitch_midi"] == 70
    assert truth_rows[2]["notes"][0]["sounding_alter"] == -1
    assert frozen_before == {
        path: freezer._sha256(path) for path in fixture["frozen_dir"].rglob("*") if path.is_file()
    }


def test_visible_musicxml_truth_preserves_key_change_events(tmp_path: Path) -> None:
    musicxml = _write_musicxml(
        tmp_path / "key-change.musicxml",
        [[61], [63]],
        alters={(1, 0): 1, (2, 0): -1},
        key_changes={1: 2, 2: -2},
    )

    truth = spike.load_visible_musicxml_truth(musicxml)

    assert truth.key_fifths == -2
    assert truth.key_events == ((1, 2), (2, -2))


def test_explicit_mapping_supports_one_to_many_and_many_to_one(tmp_path: Path) -> None:
    fixture = _frozen_fixture(
        tmp_path,
        predictions=[[60], [62, 64], [65, 67]],
    )
    musicxml = _write_musicxml(tmp_path / "three.musicxml", [[60, 62], [64], [65, 67]])
    mapping = tmp_path / "mapping.json"
    mapping.write_text(
        json.dumps(
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
                            {"measure_number": 1, "note_start": 1, "note_end": 2},
                            {"measure_number": 2},
                        ],
                    },
                    {
                        "automatic_crop_index": 3,
                        "physical_measure_spans": [{"measure_number": 3}],
                    },
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    result = spike.evaluate_frozen_heldout(
        fixture["sealed"],
        musicxml_path=musicxml,
        mapping_path=mapping,
    )

    report = json.loads(Path(result["report"]).read_text(encoding="utf-8"))
    materialized = json.loads(Path(result["mapping"]).read_text(encoding="utf-8"))
    assert report["mapping_mode"] == "explicit_user_mapping"
    assert report["metrics"]["summary"]["exact_automatic_crops"] == 3
    assert materialized["automatic_crops"][1]["physical_measure_spans"] == [
        {"measure_number": 1, "note_end": 2, "note_start": 1},
        {"measure_number": 2, "note_end": 1, "note_start": 0},
    ]
    manifest = json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))
    assert manifest["pins"]["mapping"]["source_sha256"] == freezer._sha256(mapping)
    assert manifest["pins"]["mapping"]["snapshot_sha256"] == freezer._sha256(
        Path(result["mapping"])
    )


def test_non_seven_measure_musicxml_requires_explicit_mapping(tmp_path: Path) -> None:
    fixture = _frozen_fixture(tmp_path, predictions=[[60], [62], [64]])
    musicxml = _write_musicxml(tmp_path / "three.musicxml", [[60], [62], [64]])

    with pytest.raises(ValueError, match="provide --mapping"):
        spike.evaluate_frozen_heldout(fixture["sealed"], musicxml_path=musicxml)


def test_fourth_score_uses_six_crop_default_and_gate_specific_outputs(tmp_path: Path) -> None:
    pitches = [[60], [62, 64], [65], [67], [69, 71], [72]]
    fixture = _frozen_fixture(
        tmp_path,
        predictions=pitches,
        evaluation_spec=spike.FOURTH_SCORE_EVALUATION,
    )
    musicxml = _write_musicxml(tmp_path / "six.musicxml", pitches)

    result = spike.evaluate_frozen_heldout(fixture["sealed"], musicxml_path=musicxml)

    report = json.loads(Path(result["report"]).read_text(encoding="utf-8"))
    manifest = json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))
    assert report["kind"] == "fourth_score_pitch_only_one_shot_evaluation"
    assert report["metrics"]["summary"]["automatic_crop_count"] == 6
    assert report["metrics"]["summary"]["exact_automatic_crops"] == 6
    assert report["source_musicxml_context"] == {
        "clef": ["G", 2],
        "key_fifths": -1,
        "time_signature": "3/4",
    }
    assert manifest["kind"] == "fourth_score_post_freeze_evaluation_manifest"
    assert manifest["truth_opened_after_all_frozen_hashes_verified"] is True


def test_fifth_score_uses_six_crop_default_and_gate_specific_outputs(tmp_path: Path) -> None:
    pitches = [[60], [62, 64], [65], [67], [69, 71], [72]]
    fixture = _frozen_fixture(
        tmp_path,
        predictions=pitches,
        evaluation_spec=spike.FIFTH_SCORE_EVALUATION,
    )
    musicxml = _write_musicxml(tmp_path / "six.musicxml", pitches)

    result = spike.evaluate_frozen_heldout(fixture["sealed"], musicxml_path=musicxml)

    report = json.loads(Path(result["report"]).read_text(encoding="utf-8"))
    manifest = json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))
    assert report["kind"] == "fifth_score_pitch_only_one_shot_evaluation"
    assert report["metrics"]["summary"]["automatic_crop_count"] == 6
    assert report["metrics"]["summary"]["exact_automatic_crops"] == 6
    assert manifest["kind"] == "fifth_score_post_freeze_evaluation_manifest"
    assert manifest["truth_opened_after_all_frozen_hashes_verified"] is True


def test_one_shot_output_refuses_overwrite_before_reopening_truth(tmp_path: Path) -> None:
    fixture = _frozen_fixture(tmp_path, predictions=[[60]] * 7)
    musicxml = _write_musicxml(tmp_path / "seven.musicxml", [[60]] * 7)
    spike.evaluate_frozen_heldout(fixture["sealed"], musicxml_path=musicxml)
    opened = False

    def truth_loader(path: Path) -> spike.VisibleMusicXMLTruth:
        nonlocal opened
        opened = True
        return spike.load_visible_musicxml_truth(path)

    with pytest.raises(FileExistsError, match="already exists"):
        spike.evaluate_frozen_heldout(
            fixture["sealed"],
            musicxml_path=musicxml,
            evaluation_version="v2",
            truth_loader=truth_loader,
        )

    assert opened is False


def test_frozen_prediction_hash_drift_fails_closed(tmp_path: Path) -> None:
    fixture = _frozen_fixture(tmp_path, predictions=[[60]] * 7)
    fixture["prediction_snapshot"].write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Frozen snapshot hash drift"):
        spike.evaluate_frozen_heldout(
            fixture["sealed"],
            musicxml_path=_write_musicxml(tmp_path / "truth.musicxml", [[60]] * 7),
        )


def _frozen_fixture(
    tmp_path: Path,
    *,
    predictions: Sequence[Sequence[int]],
    evaluation_spec: spike.HeldoutEvaluationSpec = spike.THIRD_SCORE_EVALUATION,
) -> dict[str, Any]:
    slug = "synthetic-score"
    system_index = 7
    gate = evaluation_spec.gate
    gate_config = inference.GATE_CONFIGS[gate.prepare_kind]
    namespace_root = tmp_path / slug / gate.output_subdir / "v2" / f"system_{system_index:03d}"
    namespace_root.mkdir(parents=True)
    out_dir = namespace_root.parents[3]
    source_system = out_dir / slug / "systems/system_007.png"
    source_system.parent.mkdir(parents=True)
    source_system.write_bytes(b"synthetic-system")

    selection = namespace_root / "selection.json"
    evaluator = namespace_root / "evaluator_spec.json"
    requests_path = namespace_root / "requests.jsonl"
    selection.write_text('{"selected":"synthetic"}\n', encoding="utf-8")
    evaluator.write_text('{"version":"synthetic"}\n', encoding="utf-8")

    requests = []
    crops = []
    for index in range(1, len(predictions) + 1):
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
                "identity": {
                    "slug": slug,
                    "system_index": system_index,
                    "automatic_measure_index": index,
                }
            }
        )
    freezer._write_jsonl(requests_path, requests)
    prepared_path = namespace_root / "prepared_manifest.json"
    prepared = {
        "schema_version": 1,
        "kind": gate.prepare_kind,
        "status": "prepared_awaiting_model_predictions",
        "split": freezer.SPLIT_NAME,
        "truth_accessed": False,
        "truth_must_be_added_only_after_freeze": True,
        "namespace": "v2",
        "target": {"slug": slug, "system_index": system_index},
        "forbidden_truth_paths": [],
        "artifacts": {
            "selection": {"path": selection.name, "sha256": freezer._sha256(selection)},
            "requests": {
                "path": requests_path.name,
                "sha256": freezer._sha256(requests_path),
                "row_sha256": [freezer._hash_json(row) for row in requests],
            },
            "evaluator": {"path": evaluator.name, "sha256": freezer._sha256(evaluator)},
            "source_system": {
                "path_relative_to_out": source_system.relative_to(out_dir).as_posix(),
                "sha256": freezer._sha256(source_system),
            },
            "crops": crops,
        },
    }
    freezer._write_json(prepared_path, prepared)

    frozen_dir = namespace_root / "frozen"
    prediction_snapshot = frozen_dir / "artifacts/predictions/001_predictions.jsonl"
    model_snapshot = frozen_dir / "artifacts/model/001_model.json"
    model_implementation_snapshot = frozen_dir / "artifacts/model/002_model_implementation.py"
    training_selection_snapshot = frozen_dir / "artifacts/training/001_training_selection.json"
    model_manifest_snapshot = frozen_dir / "artifacts/training/002_model_manifest.json"
    inference_manifest_snapshot = frozen_dir / "artifacts/training/003_inference_manifest.json"
    inference_implementation_snapshot = frozen_dir / "artifacts/training/004_inference.py"
    metadata_snapshot = frozen_dir / "artifacts/training/005_metadata.json"
    assumptions_snapshot = frozen_dir / "artifacts/training/006_assumptions.json"
    inference_requests_snapshot = frozen_dir / "artifacts/training/007_requests.jsonl"
    replay_snapshot = frozen_dir / "artifacts/training/008_replay.json"
    detailed_inference_snapshot = frozen_dir / "artifacts/training/009_inference.jsonl"
    prediction_snapshot.parent.mkdir(parents=True)
    model_snapshot.parent.mkdir(parents=True)
    training_selection_snapshot.parent.mkdir(parents=True)
    prediction_rows = [
        {
            "identity": {
                "slug": slug,
                "system_index": system_index,
                "automatic_measure_index": index,
            },
            "measure_extent_beats": None,
            "notes": [
                {
                    "order": order,
                    "pitch_midi": pitch,
                    "onset_beats": None,
                    "duration_beats": None,
                }
                for order, pitch in enumerate(pitches, start=1)
            ],
            "rests": [],
        }
        for index, pitches in enumerate(predictions, start=1)
    ]
    freezer._write_jsonl(prediction_snapshot, prediction_rows)
    model_snapshot.write_text('{"model":"synthetic"}\n', encoding="utf-8")
    model_implementation_snapshot.write_text("# synthetic model\n", encoding="utf-8")
    training_selection_snapshot.write_text(
        '{"input_provenance":{},"selection":"synthetic"}\n', encoding="utf-8"
    )
    inference_implementation_snapshot.write_text("# synthetic inference\n", encoding="utf-8")
    metadata_snapshot.write_text('{"slug":"synthetic-score"}\n', encoding="utf-8")
    assumptions_snapshot.write_text('{"truth_used":false}\n', encoding="utf-8")
    inference_requests_snapshot.write_text('{"request":"synthetic"}\n', encoding="utf-8")
    replay_snapshot.write_text('{"replay":"synthetic"}\n', encoding="utf-8")
    detailed_inference_snapshot.write_text('{"inference":"synthetic"}\n', encoding="utf-8")

    def source_record(path: Path) -> dict[str, str]:
        return {"path": path.name, "sha256": freezer._sha256(path)}

    model_artifact_records = {
        "model.json": source_record(model_snapshot),
        "training_selection.json": source_record(training_selection_snapshot),
    }
    model_manifest_snapshot.write_text(
        json.dumps(
            {
                "artifacts": model_artifact_records,
                "implementation": source_record(model_implementation_snapshot),
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    context = {
        "metadata": source_record(metadata_snapshot),
        "assumptions": {
            "path": "assumptions.json",
            "sha256": freezer._sha256(assumptions_snapshot),
        },
        "requests": {
            "path": "requests.jsonl",
            "sha256": freezer._sha256(inference_requests_snapshot),
        },
        "replay": {
            "path": "replay.json",
            "sha256": freezer._sha256(replay_snapshot),
        },
        "detailed_inference": {
            "path": "inference.jsonl",
            "sha256": freezer._sha256(detailed_inference_snapshot),
        },
    }
    inference_artifacts = {
        "assumptions.json": context["assumptions"],
        "requests.jsonl": context["requests"],
        "replay.json": context["replay"],
        "inference.jsonl": context["detailed_inference"],
        "predictions.jsonl": {
            "path": "predictions.jsonl",
            "sha256": freezer._sha256(prediction_snapshot),
        },
    }

    def pin(path: Path) -> dict[str, str]:
        return {
            "source_path": path.name,
            "source_sha256": freezer._sha256(path),
            "snapshot_path_relative_to_namespace": path.relative_to(namespace_root).as_posix(),
            "snapshot_sha256": freezer._sha256(path),
        }

    inference_manifest_snapshot.write_text(
        json.dumps(
            {
                "prepared_manifest": {
                    "path": prepared_path.resolve().as_posix(),
                    "sha256": freezer._sha256(prepared_path),
                },
                "model_and_training": {
                    "model_manifest": source_record(model_manifest_snapshot),
                    "implementation": source_record(model_implementation_snapshot),
                    "artifacts": model_artifact_records,
                },
                "implementation": source_record(inference_implementation_snapshot),
                "context": context,
                "artifacts": inference_artifacts,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    prediction_pin = pin(prediction_snapshot)
    model_pin = pin(model_snapshot)
    model_implementation_pin = pin(model_implementation_snapshot)
    training_selection_pin = pin(training_selection_snapshot)
    model_manifest_pin = pin(model_manifest_snapshot)
    inference_manifest_pin = pin(inference_manifest_snapshot)
    inference_implementation_pin = pin(inference_implementation_snapshot)
    metadata_pin = pin(metadata_snapshot)
    assumptions_pin = pin(assumptions_snapshot)
    inference_requests_pin = pin(inference_requests_snapshot)
    replay_pin = pin(replay_snapshot)
    detailed_inference_pin = pin(detailed_inference_snapshot)
    binding = {
        "schema_version": 1,
        "kind": gate_config["binding_kind"],
        "version": gate_config["inference_version"],
        "prepared_manifest": {
            "path": prepared_path.resolve().as_posix(),
            "sha256": freezer._sha256(prepared_path),
        },
        "selected_model": {
            "manifest": model_manifest_pin,
            "artifacts": {
                "model.json": model_pin,
                "training_selection.json": training_selection_pin,
            },
            "implementation": model_implementation_pin,
            "training_inputs": [],
        },
        "inference": {
            "manifest": inference_manifest_pin,
            "implementation": inference_implementation_pin,
            "metadata": metadata_pin,
            "assumptions": assumptions_pin,
            "requests": inference_requests_pin,
            "replay": replay_pin,
            "detailed_inference": detailed_inference_pin,
            "predictions": prediction_pin,
        },
        "manifest_sha256": freezer._sha256(inference_manifest_snapshot),
        "context_sha256": inference._hash_json(context),
    }

    freeze_path = frozen_dir / "freeze.json"
    freeze = {
        "schema_version": 1,
        "kind": gate.freeze_kind,
        "status": "frozen_awaiting_truth",
        "split": freezer.SPLIT_NAME,
        "truth_accessed": False,
        "truth_must_be_added_only_after_freeze": True,
        "target": prepared["target"],
        "namespace": "v2",
        "prepared_manifest": {
            "path": "../prepared_manifest.json",
            "sha256": freezer._sha256(prepared_path),
        },
        "selection_sha256": prepared["artifacts"]["selection"]["sha256"],
        "requests": prepared["artifacts"]["requests"],
        "evaluator": prepared["artifacts"]["evaluator"],
        "forbidden_truth_paths": [],
        "predictions": prediction_pin,
        "model_artifacts": [model_pin, model_implementation_pin],
        "training_artifacts": [
            training_selection_pin,
            model_manifest_pin,
            inference_manifest_pin,
            inference_implementation_pin,
            metadata_pin,
            assumptions_pin,
            inference_requests_pin,
            replay_pin,
            detailed_inference_pin,
        ],
        "inference_binding": binding,
    }
    freezer._write_json(freeze_path, freeze)
    sealed_path = frozen_dir / "sealed_manifest.json"
    freezer._write_json(
        sealed_path,
        {
            "schema_version": 1,
            "kind": gate.sealed_kind,
            "status": "frozen_awaiting_truth",
            "split": freezer.SPLIT_NAME,
            "truth_accessed": False,
            "target": prepared["target"],
            "freeze": {"path": "freeze.json", "sha256": freezer._sha256(freeze_path)},
            "prepared_manifest_sha256": freezer._sha256(prepared_path),
            "inference_binding_sha256": inference._hash_json(binding),
        },
    )
    return {
        "sealed": sealed_path,
        "freeze": freeze_path,
        "frozen_dir": frozen_dir,
        "prediction_snapshot": prediction_snapshot,
        "model_snapshot": model_snapshot,
    }


def _write_musicxml(
    path: Path,
    measures: Sequence[Sequence[int]],
    *,
    ties: dict[tuple[int, int], str] | None = None,
    alters: dict[tuple[int, int], int] | None = None,
    key_changes: dict[int, int] | None = None,
) -> Path:
    ties = ties or {}
    alters = alters or {}
    key_changes = key_changes or {}
    measure_xml = []
    for measure_number, pitches in enumerate(measures, start=1):
        notes = []
        for note_index, midi in enumerate(pitches):
            alter = alters.get((measure_number, note_index), _default_alter(midi))
            step, octave = _step_and_octave(midi, alter)
            alter_xml = f"<alter>{alter}</alter>" if alter else ""
            tie_type = ties.get((measure_number, note_index))
            tie_xml = f'<tie type="{tie_type}"/>' if tie_type else ""
            tied_xml = f'<notations><tied type="{tie_type}"/></notations>' if tie_type else ""
            notes.append(
                "<note>"
                f"<pitch><step>{step}</step>{alter_xml}<octave>{octave}</octave></pitch>"
                "<duration>1</duration>"
                f"{tie_xml}{tied_xml}"
                "</note>"
            )
        attributes = ""
        if measure_number == 1:
            fifths = key_changes.get(1, -1)
            attributes = (
                f"<attributes><divisions>1</divisions><key><fifths>{fifths}</fifths></key>"
                "<time><beats>3</beats><beat-type>4</beat-type></time>"
                "<clef><sign>G</sign><line>2</line></clef></attributes>"
            )
        elif measure_number in key_changes:
            attributes = (
                f"<attributes><key><fifths>{key_changes[measure_number]}</fifths></key>"
                "</attributes>"
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


def _default_alter(midi: int) -> int:
    return 0


def _step_and_octave(midi: int, alter: int) -> tuple[str, int]:
    natural_midi = midi - alter
    pitch_class = natural_midi % 12
    steps = {0: "C", 2: "D", 4: "E", 5: "F", 7: "G", 9: "A", 11: "B"}
    if pitch_class not in steps:
        raise ValueError(f"Synthetic pitch {midi} cannot be spelled with alter {alter}")
    return steps[pitch_class], natural_midi // 12 - 1


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
