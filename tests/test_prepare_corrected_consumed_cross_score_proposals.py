import hashlib
import json
import shutil
from pathlib import Path

import pytest
from PIL import Image

from scripts.experiments import prepare_consumed_cross_score_proposals as prepare


def test_materializes_eight_unreviewed_physical_measure_proposals(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)

    output = prepare.prepare_corrected_consumed_cross_score_proposals(
        fixture["out_dir"],
        mapping_path=fixture["mapping"],
        consumption_mapping_path=fixture["consumption_mapping"],
        repo_root=tmp_path,
    )

    payload = _load(output)
    assert payload["kind"] == prepare.CORRECTED_OUTPUT_KIND
    assert payload["review_status"] == "unreviewed"
    assert payload["eligible_for_training"] is False
    assert payload["eligible_for_promotion"] is False
    assert len(payload["tasks"]) == 8
    assert [task["identity"]["physical_measure_number"] for task in payload["tasks"]] == list(
        range(1, 9)
    )
    second = payload["tasks"][1]
    assert second["expected"]["ordered_sounding_pitches"] == ["Bb4", "D4"]
    assert second["expected"]["notes"][0]["tie_types"] == ["stop"]
    assert [item["candidate_id"] for item in second["proposal"]["assignments"]] == ["c001"]
    assert second["proposal"]["assignments"][0]["expected_sounding_pitch"] == "Bb4"
    assert second["proposal"]["unresolved"]["expected_notes"][0]["sounding_pitch"] == "D4"
    assert second["proposal"]["unresolved"]["candidates"][0]["candidate_id"] == "c002"

    manifest = _load(output.parent / "manifest.json")
    assert manifest["review_status"] == "unreviewed"
    assert manifest["outputs"]["proposals"]["sha256"] == _sha256(output)
    assert manifest["outputs"]["coverage"]["sha256"] == _sha256(output.parent / "coverage.md")
    assert manifest["inputs"]["namespace_manifest"]["sha256"] == _sha256(
        fixture["namespace_manifest"]
    )
    assert manifest["inputs"]["mapping"]["sha256"] == _sha256(fixture["mapping"])
    assert manifest["inputs"]["musicxml"]["sha256"] == _sha256(fixture["musicxml"])
    assert manifest["inputs"]["candidates_manifest"]["sha256"] == _sha256(
        fixture["candidates_manifest"]
    )


def test_corrected_proposals_are_deterministic(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    first = prepare.prepare_corrected_consumed_cross_score_proposals(
        fixture["out_dir"],
        mapping_path=fixture["mapping"],
        consumption_mapping_path=fixture["consumption_mapping"],
        repo_root=tmp_path,
    )
    first_payload = first.read_bytes()
    first_coverage = (first.parent / "coverage.md").read_bytes()
    shutil.rmtree(first.parent)

    second = prepare.prepare_corrected_consumed_cross_score_proposals(
        fixture["out_dir"],
        mapping_path=fixture["mapping"],
        consumption_mapping_path=fixture["consumption_mapping"],
        repo_root=tmp_path,
    )

    assert second.read_bytes() == first_payload
    assert (second.parent / "coverage.md").read_bytes() == first_coverage


def test_corrected_proposals_refuse_overwrite(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    prepare.prepare_corrected_consumed_cross_score_proposals(
        fixture["out_dir"],
        mapping_path=fixture["mapping"],
        consumption_mapping_path=fixture["consumption_mapping"],
        repo_root=tmp_path,
    )

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        prepare.prepare_corrected_consumed_cross_score_proposals(
            fixture["out_dir"],
            mapping_path=fixture["mapping"],
            consumption_mapping_path=fixture["consumption_mapping"],
            repo_root=tmp_path,
        )


def test_corrected_proposals_do_not_read_legacy_heldout_requests(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    missing_requests = tmp_path / "fresh_heldout" / "must_not_be_read.jsonl"
    consumption = _load(fixture["consumption_mapping"])
    consumption["source"]["requests"] = {
        "path": str(missing_requests),
        "sha256": "0" * 64,
    }
    _write(fixture["consumption_mapping"], consumption)

    output = prepare.prepare_corrected_consumed_cross_score_proposals(
        fixture["out_dir"],
        mapping_path=fixture["mapping"],
        consumption_mapping_path=fixture["consumption_mapping"],
        repo_root=tmp_path,
    )

    assert output.is_file()
    assert _load(output)["provenance"]["heldout_discovery_or_globbing"] is False


def test_corrected_proposals_reject_non_blind_candidate_lineage(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    rows = _read_jsonl(fixture["candidates_manifest"])
    rows[0]["candidate_generation"]["ground_truth_files_read"] = ["forbidden.json"]
    _write_jsonl(fixture["candidates_manifest"], rows)
    mapping = _load(fixture["mapping"])
    mapping["source"]["candidates_manifest"]["sha256"] = _sha256(fixture["candidates_manifest"])
    _write(fixture["mapping"], mapping)
    manifest = _load(fixture["namespace_manifest"])
    manifest["artifacts"]["mapping"]["sha256"] = _sha256(fixture["mapping"])
    _write(fixture["namespace_manifest"], manifest)

    with pytest.raises(ValueError, match="without ground-truth access"):
        prepare.prepare_corrected_consumed_cross_score_proposals(
            fixture["out_dir"],
            mapping_path=fixture["mapping"],
            consumption_mapping_path=fixture["consumption_mapping"],
            repo_root=tmp_path,
        )


def test_corrected_mapping_supports_seven_contiguous_measures() -> None:
    crops = [
        {
            "system_measure_index": index,
            "physical_measure_numbers": [index],
        }
        for index in range(1, 8)
    ]

    result = prepare._validate_corrected_crop_mapping(
        crops,
        physical_measure_numbers=tuple(range(1, 8)),
        candidate_crop_indices=set(range(1, 8)),
    )

    assert result == crops


def _fixture(tmp_path: Path) -> dict[str, Path]:
    out_dir = tmp_path / "out"
    slug = "carrizal"
    namespace = out_dir / slug / "vlm_melody_training_inputs" / "carrizal_system_004_seg_v2"
    namespace.mkdir(parents=True)
    approved = tmp_path / "approved_consumed"
    approved.mkdir()
    musicxml = approved / "carrizal_system_004.musicxml"
    musicxml.write_text(_musicxml(), encoding="utf-8")
    evaluation = approved / "evaluation.json"
    _write(evaluation, {"status": "evaluated_once_after_frozen_predictions"})

    candidate_rows = []
    crop_rows = []
    pitches = ["C4", "B4", "E4", "F4", "G4", "A4", "B4", "C5"]
    for measure, pitch in enumerate(pitches, start=1):
        raw = namespace / "measure_inputs" / f"measure_{measure:03d}_raw.png"
        raw.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (100, 80), "white").save(raw)
        candidate_path = namespace / "candidates" / f"measure_{measure:03d}" / "candidates.json"
        candidate_path.parent.mkdir(parents=True, exist_ok=True)
        specs = [("c001", 20, pitch, 0.9)]
        if measure == 2:
            specs.append(("c002", 40, "F4", 0.8))
        _write(candidate_path, _candidate_artifact(slug, measure, raw, tmp_path, specs))
        candidate_record = {
            "schema_version": 1,
            "kind": prepare.CORRECTED_CANDIDATE_RECORD_KIND,
            "split_status": "consumed_training",
            "review_status": "unreviewed",
            "eligible_for_training": False,
            "eligible_for_promotion": False,
            "identity": {
                "slug": slug,
                "system_index": 4,
                "system_measure_index": measure,
            },
            "source": {"measure_raw": _file_record(raw, tmp_path)},
            "candidate_generation": {
                "candidate_count": len(specs),
                "ground_truth_files_read": [],
            },
            "artifacts": {"candidates": _file_record(candidate_path, tmp_path)},
        }
        candidate_rows.append(candidate_record)
        crop_rows.append(
            {
                "system_measure_index": measure,
                "physical_measure_numbers": [measure],
                "candidate_artifact": _file_record(candidate_path, tmp_path),
            }
        )

    candidates_manifest = namespace / "candidates_manifest.jsonl"
    _write_jsonl(candidates_manifest, candidate_rows)
    mapping = namespace / "mapping.json"
    _write(
        mapping,
        {
            "schema_version": 1,
            "kind": prepare.CORRECTED_MAPPING_KIND,
            "split_status": "consumed_training",
            "review_status": "unreviewed",
            "eligible_for_training": False,
            "eligible_for_promotion": False,
            "identity": {"slug": slug, "system_index": 4},
            "segmentation_namespace": "carrizal_system_004_seg_v2",
            "source": {"candidates_manifest": _file_record(candidates_manifest, tmp_path)},
            "crops": crop_rows,
        },
    )
    namespace_manifest = namespace / "manifest.json"
    _write(
        namespace_manifest,
        {
            "schema_version": 1,
            "kind": prepare.CORRECTED_NAMESPACE_KIND,
            "split_status": "consumed_training",
            "review_status": "unreviewed",
            "eligible_for_training": False,
            "eligible_for_promotion": False,
            "identity": {"slug": slug, "system_index": 4},
            "segmentation_namespace": "carrizal_system_004_seg_v2",
            "artifacts": {"mapping": _file_record(mapping, tmp_path)},
        },
    )
    consumption_mapping = approved / "mapping.json"
    _write(
        consumption_mapping,
        {
            "schema_version": 1,
            "kind": prepare.MAPPING_KIND,
            "split_status": "consumed_training",
            "identity": {"slug": slug, "system_index": 4},
            "consumption": {
                "source_split_status": "fresh_heldout",
                "reason": "The sealed result is explicitly consumed for training proposals.",
                "evaluation_evidence": _file_record(evaluation, tmp_path),
            },
            "source": {
                "musicxml": _file_record(musicxml, tmp_path),
                "requests": {"path": "not-read.jsonl", "sha256": "0" * 64},
            },
            "crops": [],
        },
    )
    return {
        "out_dir": out_dir,
        "mapping": mapping,
        "namespace_manifest": namespace_manifest,
        "candidates_manifest": candidates_manifest,
        "consumption_mapping": consumption_mapping,
        "musicxml": musicxml,
    }


def _candidate_artifact(
    slug: str,
    measure: int,
    raw: Path,
    tmp_path: Path,
    specs: list[tuple[str, int, str, float]],
) -> dict:
    pitch_to_y = {
        "C4": 70,
        "D4": 65,
        "E4": 60,
        "F4": 55,
        "G4": 50,
        "A4": 45,
        "B4": 40,
        "C5": 35,
    }
    return {
        "schema_version": 2,
        "kind": "vlm_notehead_candidates",
        "strategy": "staff-grid-density",
        "strategy_version": 2,
        "slug": slug,
        "system_index": 4,
        "system_measure_index": measure,
        "source_image_path": raw.relative_to(tmp_path).as_posix(),
        "staff_lines_y_px": [20, 30, 40, 50, 60],
        "candidates": [
            {
                "id": candidate_id,
                "center": {"x": x, "y": pitch_to_y[pitch]},
                "score": score,
            }
            for candidate_id, x, pitch, score in specs
        ],
    }


def _musicxml() -> str:
    notes = [
        "<pitch><step>C</step><octave>4</octave></pitch>",
        "<pitch><step>B</step><alter>-1</alter><octave>4</octave></pitch>",
        "<pitch><step>E</step><octave>4</octave></pitch>",
        "<pitch><step>F</step><octave>4</octave></pitch>",
        "<pitch><step>G</step><octave>4</octave></pitch>",
        "<pitch><step>A</step><octave>4</octave></pitch>",
        "<pitch><step>B</step><octave>4</octave></pitch>",
        "<pitch><step>C</step><octave>5</octave></pitch>",
    ]
    measures = []
    for number, pitch in enumerate(notes, start=1):
        tie = '<tie type="stop"/>' if number == 2 else ""
        extra = (
            "<note><pitch><step>D</step><octave>4</octave></pitch><duration>1</duration></note>"
            if number == 2
            else ""
        )
        measures.append(
            f'<measure number="{number}"><note>{pitch}<duration>1</duration>{tie}</note>'
            f"{extra}</measure>"
        )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<score-partwise version="4.0"><part-list><score-part id="P1">'
        '<part-name>Music</part-name></score-part></part-list><part id="P1">'
        + "".join(measures)
        + "</part></score-partwise>\n"
    )


def _file_record(path: Path, root: Path) -> dict[str, str]:
    return {"path": path.relative_to(root).as_posix(), "sha256": _sha256(path)}


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
