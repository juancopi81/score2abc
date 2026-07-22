import copy
import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image

from scripts.experiments import materialize_agent_visual_adjudication as materialize


def test_materializes_exact_reviews_confidences_and_overlays(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)

    output = materialize.materialize_agent_visual_adjudication(
        fixture["out_dir"],
        decision_fixture=fixture["decision_fixture"],
        repo_root=tmp_path,
    )

    manifest = _load(output / "manifest.json")
    assert manifest["counts"] == {
        "candidate_selections": 14,
        "high_confidence": 18,
        "manual_selections": 6,
        "measures": 8,
        "medium_confidence": 2,
        "noteheads": 20,
    }
    assert manifest["training_selection"]["medium_confidence_default"] is False
    assert manifest["eligible_for_human_promotion"] is False
    assert manifest["human_reviewed"] is False
    reviews = [_load(output / f"measure_{measure:03d}.json") for measure in range(1, 9)]
    assert [head["sounding_pitch"] for head in reviews[1]["heads"]] == [
        "C5",
        "A4",
        "C5",
        "Bb4",
        "G4",
        "G#4",
    ]
    assert [head["confidence"] for head in reviews[1]["heads"]].count("medium") == 2
    assert all(review["reviewer_type"] == "agent_visual_adjudication" for review in reviews)
    overlay = Image.open(output / "overlays" / "measure_002.png").convert("RGB")
    colors = {color for _, color in overlay.getcolors(maxcolors=overlay.width * overlay.height)}
    assert materialize.HIGH_COLOR in colors
    assert materialize.MEDIUM_COLOR in colors
    assert materialize.MANUAL_COLOR in colors
    assert materialize.REJECTED_COLOR in colors
    assert Image.open(output / "contact_sheet.png").size[0] > 0


def test_rejects_manual_coordinate_out_of_bounds(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    decisions = _load(fixture["decision_fixture"])
    decisions["measures"][0]["heads"][0]["selection"]["center"]["x"] = 999
    _write(fixture["decision_fixture"], decisions)

    with pytest.raises(ValueError, match="outside image bounds"):
        _materialize(fixture, tmp_path)


def test_rejects_candidate_mismatch(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    decisions = _load(fixture["decision_fixture"])
    decisions["measures"][1]["heads"][0]["selection"]["candidate_id"] = "c999"
    _write(fixture["decision_fixture"], decisions)

    with pytest.raises(ValueError, match="Candidate mismatch"):
        _materialize(fixture, tmp_path)


@pytest.mark.parametrize("mutation", ["pitch", "count"])
def test_rejects_exact_pitch_or_count_mismatch(tmp_path: Path, mutation: str) -> None:
    fixture = _fixture(tmp_path)
    decisions = _load(fixture["decision_fixture"])
    if mutation == "pitch":
        decisions["measures"][3]["heads"][2]["sounding_pitch"] = "B4"
    else:
        decisions["measures"][3]["heads"].pop()
    _write(fixture["decision_fixture"], decisions)

    with pytest.raises(ValueError, match="Pitch/order mismatch|Pitch/count mismatch"):
        _materialize(fixture, tmp_path)


def test_refuses_overwrite(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    _materialize(fixture, tmp_path)

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        _materialize(fixture, tmp_path)


def test_rejects_hash_drift(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    fixture["proposals"].write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Proposals hash drift"):
        _materialize(fixture, tmp_path)


def _materialize(fixture: dict[str, Path], repo_root: Path) -> Path:
    return materialize.materialize_agent_visual_adjudication(
        fixture["out_dir"],
        decision_fixture=fixture["decision_fixture"],
        repo_root=repo_root,
    )


def _fixture(tmp_path: Path) -> dict[str, Path]:
    source_fixture = (
        Path(__file__).parent
        / "fixtures"
        / "vlm_melody"
        / "agent_visual_adjudication"
        / "carrizal_system_004_seg_v2.json"
    )
    decisions = copy.deepcopy(_load(source_fixture))
    slug = decisions["identity"]["slug"]
    out_dir = tmp_path / "out"
    namespace = out_dir / slug / "vlm_melody_training_inputs" / materialize.NAMESPACE
    namespace.mkdir(parents=True)

    raw_records = []
    candidate_records = []
    candidate_rows = []
    for measure in range(1, 9):
        raw_path = namespace / "measure_inputs" / f"measure_{measure:03d}_raw.png"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (320, 221), "white").save(raw_path)
        candidate_path = namespace / "candidates" / f"measure_{measure:03d}" / "candidates.json"
        candidate_path.parent.mkdir(parents=True, exist_ok=True)
        candidates = []
        for index in range(1, 25):
            x = 12 + index * 5
            y = 35 + (index % 8) * 18
            candidates.append(
                {
                    "id": f"c{index:03d}",
                    "center": {"x": x, "y": y},
                    "bbox": {
                        "left": x - 5,
                        "top": y - 4,
                        "right": x + 5,
                        "bottom": y + 4,
                    },
                    "score": 0.8,
                }
            )
        _write(
            candidate_path,
            {
                "candidates": candidates,
                "staff_lines_y_px": [45, 70, 95, 120, 145],
            },
        )
        raw_record = {"system_measure_index": measure, **_record(raw_path, tmp_path)}
        candidate_record = {
            "system_measure_index": measure,
            **_record(candidate_path, tmp_path),
        }
        raw_records.append(raw_record)
        candidate_records.append(candidate_record)
        candidate_rows.append(
            {
                "identity": {
                    "slug": slug,
                    "system_index": 4,
                    "system_measure_index": measure,
                },
                "source": {"measure_raw": _record(raw_path, tmp_path)},
                "artifacts": {"candidates": _record(candidate_path, tmp_path)},
            }
        )

    candidates_manifest = namespace / "candidates_manifest.jsonl"
    candidates_manifest.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in candidate_rows),
        encoding="utf-8",
    )
    namespace_manifest = namespace / "manifest.json"
    _write(
        namespace_manifest,
        {
            "kind": "vlm_melody_consumed_cross_score_training_inputs",
            "identity": decisions["identity"],
            "segmentation_namespace": materialize.NAMESPACE,
        },
    )
    musicxml = tmp_path / "truth" / "carrizal.musicxml"
    musicxml.parent.mkdir()
    musicxml.write_text(_musicxml(), encoding="utf-8")
    fixture_source = {
        "namespace_manifest": _record(namespace_manifest, tmp_path),
        "candidates_manifest": _record(candidates_manifest, tmp_path),
        "musicxml": _record(musicxml, tmp_path),
    }
    proposals = namespace / "proposals" / "proposals.json"
    proposals.parent.mkdir()
    pitches = _expected_pitches()
    _write(
        proposals,
        {
            "kind": materialize.PROPOSALS_KIND,
            "identity": decisions["identity"],
            "segmentation_namespace": materialize.NAMESPACE,
            "source": {
                **fixture_source,
                "candidate_artifacts": candidate_records,
                "raw_images": raw_records,
            },
            "tasks": [
                {
                    "identity": {"physical_measure_number": measure},
                    "expected": {"ordered_sounding_pitches": pitch_list},
                }
                for measure, pitch_list in pitches.items()
            ],
        },
    )
    decisions["source"] = {**fixture_source, "proposals": _record(proposals, tmp_path)}
    decision_fixture = tmp_path / "decisions.json"
    _write(decision_fixture, decisions)
    return {
        "out_dir": out_dir,
        "decision_fixture": decision_fixture,
        "proposals": proposals,
    }


def _expected_pitches() -> dict[int, list[str]]:
    return {
        1: ["C5"],
        2: ["C5", "A4", "C5", "Bb4", "G4", "G#4"],
        3: ["A4"],
        4: ["A4", "Bb4", "A4", "G#4", "A4"],
        5: ["C#5", "A4"],
        6: ["E5", "A4"],
        7: ["F#5"],
        8: ["E5", "D5"],
    }


def _musicxml() -> str:
    measures = []
    for measure, pitches in _expected_pitches().items():
        notes = []
        for value in pitches:
            step = value[0]
            accidental = value[1:-1]
            octave = value[-1]
            alter = {"": "", "b": "<alter>-1</alter>", "#": "<alter>1</alter>"}[accidental]
            notes.append(
                f"<note><pitch><step>{step}</step>{alter}<octave>{octave}</octave>"
                "</pitch><duration>1</duration><type>quarter</type></note>"
            )
        measures.append(f'<measure number="{measure}">{"".join(notes)}</measure>')
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<score-partwise version="4.0"><part-list><score-part id="P1">'
        '<part-name>Music</part-name></score-part></part-list><part id="P1">'
        + "".join(measures)
        + "</part></score-partwise>\n"
    )


def _record(path: Path, repo_root: Path) -> dict[str, str]:
    return {
        "path": path.resolve().relative_to(repo_root).as_posix(),
        "sha256": _sha256(path),
    }


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
