import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image

from scripts.experiments import prepare_consumed_cross_score_proposals as prepare


def test_prepares_unreviewed_consumed_queue_with_merged_physical_measures(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)

    output = prepare.prepare_consumed_cross_score_proposals(
        fixture["out_dir"],
        mapping_path=fixture["mapping"],
        repo_root=tmp_path,
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["kind"] == prepare.OUTPUT_KIND
    assert payload["split_status"] == "consumed_training"
    assert payload["proposal_status"] == "unreviewed_consumed_cross_score_proposals_only"
    assert payload["segmentation_namespace"] == "fresh_score_system_004_frozen_seg_v1"
    assert payload["eligible_for_training"] is False
    assert payload["eligible_for_promotion"] is False
    assert payload["source"]["source_split_status"] == "fresh_heldout"
    assert payload["source"]["musicxml_sha256"] == _sha256(fixture["musicxml"])
    assert payload["source"]["requests_sha256"] == _sha256(fixture["requests"])

    first, second = payload["tasks"]
    assert first["physical_measure_numbers"] == [1, 2]
    assert first["expected"]["note_count"] == 3
    assert first["expected"]["ordered_sounding_pitches"] == ["C4", "Bb4", "D4"]
    assert first["expected"]["ordered_staff_pitches"] == ["C4", "B4", "D4"]
    assert first["proposal"]["status"] == "available"
    assert [item["candidate_id"] for item in first["proposal"]["assignments"]] == [
        "c002",
        "c003",
        "c004",
    ]
    assert first["proposal"]["human_reviewed"] is False
    assert first["review_state"]["status"] == "human_review_required"

    assert second["expected"]["ordered_sounding_pitches"] == ["E4"]
    assert second["proposal"]["status"] == "review_queue_only"
    assert second["proposal"]["reason"] == "no_full_exact_staff_pitch_alignment"
    assert second["review_state"]["eligible_for_promotion"] is False
    assert len(first["source"]["request_row_sha256"]) == 64
    assert len(first["source"]["image_sha256"]) == 64
    assert len(first["source"]["candidate_artifact_sha256"]) == 64


@pytest.mark.parametrize("status", ["heldout", "fresh_heldout", "training"])
def test_rejects_any_output_status_other_than_consumed_training(
    tmp_path: Path, status: str
) -> None:
    fixture = _fixture(tmp_path)
    mapping = _load(fixture["mapping"])
    mapping["split_status"] = status
    _write(fixture["mapping"], mapping)

    with pytest.raises(ValueError, match="split_status"):
        prepare.prepare_consumed_cross_score_proposals(
            fixture["out_dir"],
            mapping_path=fixture["mapping"],
            repo_root=tmp_path,
        )


def test_rejects_changed_musicxml_source_hash(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    fixture["musicxml"].write_text(
        fixture["musicxml"].read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="MusicXML source hash changed"):
        prepare.prepare_consumed_cross_score_proposals(
            fixture["out_dir"],
            mapping_path=fixture["mapping"],
            repo_root=tmp_path,
        )


def test_rejects_changed_request_image(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    Image.new("RGB", (40, 40), "black").save(fixture["images"][0])

    with pytest.raises(ValueError, match="Raw request image source hash changed"):
        prepare.prepare_consumed_cross_score_proposals(
            fixture["out_dir"],
            mapping_path=fixture["mapping"],
            repo_root=tmp_path,
        )


def test_fresh_heldout_lineage_requires_sealed_evaluation_evidence(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    mapping = _load(fixture["mapping"])
    del mapping["consumption"]["evaluation_evidence"]
    _write(fixture["mapping"], mapping)

    with pytest.raises(ValueError, match="requires hash-pinned evaluation_evidence"):
        prepare.prepare_consumed_cross_score_proposals(
            fixture["out_dir"],
            mapping_path=fixture["mapping"],
            repo_root=tmp_path,
        )


def test_rejects_incomplete_physical_measure_mapping(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    mapping = _load(fixture["mapping"])
    mapping["crops"][0]["physical_measure_numbers"] = [1]
    _write(fixture["mapping"], mapping)

    with pytest.raises(ValueError, match="cover MusicXML measures exactly once"):
        prepare.prepare_consumed_cross_score_proposals(
            fixture["out_dir"],
            mapping_path=fixture["mapping"],
            repo_root=tmp_path,
        )


@pytest.mark.parametrize("protected", ["vlm_melody_inputs", "notehead_reviews"])
def test_refuses_to_write_into_source_or_human_review_namespaces(
    tmp_path: Path, protected: str
) -> None:
    fixture = _fixture(tmp_path)

    with pytest.raises(ValueError, match="protected namespace"):
        prepare.prepare_consumed_cross_score_proposals(
            fixture["out_dir"],
            mapping_path=fixture["mapping"],
            output_path=fixture["out_dir"] / "fresh-score" / protected / "proposals.json",
            repo_root=tmp_path,
        )


def _fixture(tmp_path: Path) -> dict:
    out_dir = tmp_path / "out"
    slug = "fresh-score"
    source_dir = tmp_path / "sources"
    source_dir.mkdir()
    musicxml = source_dir / "system.musicxml"
    musicxml.write_text(_musicxml(), encoding="utf-8")
    evaluation = source_dir / "evaluation.json"
    _write(evaluation, {"status": "evaluated_once_after_frozen_predictions"})

    images = []
    candidates = []
    request_rows = []
    candidate_specs = [
        [
            ("c001", 10, "G4", 0.95),
            ("c002", 20, "C4", 0.8),
            ("c003", 30, "B4", 0.7),
            ("c004", 40, "D4", 0.9),
        ],
        [("c001", 20, "F4", 0.8)],
    ]
    for crop_index, candidate_spec in enumerate(candidate_specs, start=1):
        image = out_dir / slug / "inputs" / f"measure_{crop_index:03d}.png"
        image.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (80, 80), "white").save(image)
        images.append(image)
        request_rows.append(
            {
                "schema_version": 1,
                "split": "fresh_heldout",
                "identity": {
                    "slug": slug,
                    "system_index": 4,
                    "system_measure_index": crop_index,
                    "global_measure_index": 20 + crop_index,
                },
                "images": {
                    "raw": {
                        "path_relative_to_out": image.relative_to(out_dir).as_posix(),
                        "sha256": _sha256(image),
                        "width_px": 80,
                        "height_px": 80,
                    }
                },
            }
        )
        candidate = source_dir / f"candidates_{crop_index}.json"
        _write(
            candidate,
            _candidate_artifact(
                slug,
                crop_index,
                image,
                tmp_path=tmp_path,
                candidate_spec=candidate_spec,
            ),
        )
        candidates.append(candidate)

    requests = source_dir / "requests.jsonl"
    requests.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in request_rows),
        encoding="utf-8",
    )
    mapping = source_dir / "mapping.json"
    _write(
        mapping,
        {
            "schema_version": 1,
            "kind": prepare.MAPPING_KIND,
            "split_status": "consumed_training",
            "segmentation_namespace": "fresh_score_system_004_frozen_seg_v1",
            "identity": {"slug": slug, "system_index": 4},
            "consumption": {
                "source_split_status": "fresh_heldout",
                "reason": "One-shot evaluation completed; approved truth is now training data.",
                "evaluation_evidence": {
                    "path": evaluation.relative_to(tmp_path).as_posix(),
                    "sha256": _sha256(evaluation),
                },
            },
            "source": {
                "musicxml": {
                    "path": musicxml.relative_to(tmp_path).as_posix(),
                    "sha256": _sha256(musicxml),
                },
                "requests": {
                    "path": requests.relative_to(tmp_path).as_posix(),
                    "sha256": _sha256(requests),
                },
            },
            "crops": [
                {
                    "system_measure_index": 1,
                    "physical_measure_numbers": [1, 2],
                    "candidate_artifact": {
                        "path": candidates[0].relative_to(tmp_path).as_posix(),
                        "sha256": _sha256(candidates[0]),
                    },
                },
                {
                    "system_measure_index": 2,
                    "physical_measure_numbers": [3],
                    "candidate_artifact": {
                        "path": candidates[1].relative_to(tmp_path).as_posix(),
                        "sha256": _sha256(candidates[1]),
                    },
                },
            ],
        },
    )
    return {
        "out_dir": out_dir,
        "mapping": mapping,
        "musicxml": musicxml,
        "evaluation": evaluation,
        "requests": requests,
        "images": images,
    }


def _candidate_artifact(
    slug: str,
    crop_index: int,
    image: Path,
    *,
    tmp_path: Path,
    candidate_spec: list[tuple[str, int, str, float]],
) -> dict:
    staff_lines = [20, 30, 40, 50, 60]
    pitch_to_y = {
        "G4": 50,
        "F4": 55,
        "E4": 60,
        "D4": 65,
        "C4": 70,
        "B4": 40,
    }
    return {
        "schema_version": 2,
        "kind": "vlm_notehead_candidates",
        "strategy": "staff-grid-density",
        "strategy_version": 2,
        "slug": slug,
        "system_index": 4,
        "system_measure_index": crop_index,
        "global_measure_index": 20 + crop_index,
        "source_image_path": image.relative_to(tmp_path).as_posix(),
        "source_image_size_px": {"width": 80, "height": 80},
        "staff_lines_y_px": staff_lines,
        "candidates": [
            {
                "id": candidate_id,
                "rank": rank,
                "center": {"x": x, "y": pitch_to_y[pitch]},
                "score": score,
            }
            for rank, (candidate_id, x, pitch, score) in enumerate(candidate_spec, start=1)
        ],
    }


def _musicxml() -> str:
    return """<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="4.0">
  <part-list><score-part id="P1"><part-name>Music</part-name></score-part></part-list>
  <part id="P1">
    <measure number="1">
      <note><pitch><step>C</step><octave>4</octave></pitch><duration>2</duration></note>
      <note><rest/><duration>2</duration></note>
    </measure>
    <measure number="2">
      <note><pitch><step>B</step><alter>-1</alter><octave>4</octave></pitch><duration>2</duration></note>
      <note><pitch><step>D</step><octave>4</octave></pitch><duration>2</duration></note>
    </measure>
    <measure number="3">
      <note><pitch><step>E</step><octave>4</octave></pitch><duration>2</duration></note>
    </measure>
  </part>
</score-partwise>
"""


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
