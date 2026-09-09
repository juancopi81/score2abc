import json
from pathlib import Path

import pytest

from scripts.experiments import evaluate_second_score_heldout as spike
from scripts.experiments import spike_composed_melody_chain as composed


def test_committed_carrizal_evidence_matches_sealed_hashes() -> None:
    root = Path(__file__).parent / "fixtures/vlm_melody/fresh_heldout"
    musicxml = root / "carrizal_system_004.musicxml"
    truth = root / "carrizal_system_004_truth.jsonl"
    evaluation = json.loads(
        (root / "carrizal_system_004_evaluation.json").read_text(encoding="utf-8")
    )

    assert composed._sha256(musicxml) == evaluation["musicxml_sha256"]
    assert composed._sha256(truth) == evaluation["truth_sha256"]
    assert evaluation["freeze_sha256"] == spike.EXPECTED_FREEZE_SHA256
    assert evaluation["metrics"]["summary"]["note_f1"] == 0.325581
    assert evaluation["metrics"]["summary"]["exact_measures"] == 0


def test_one_shot_evaluation_maps_two_physical_measures_to_one_crop(
    tmp_path: Path,
) -> None:
    fixture = _frozen_fixture(tmp_path)

    report = spike.evaluate_fresh_heldout(
        tmp_path,
        musicxml_path=fixture["musicxml"],
        slug="fresh-score",
        system_index=4,
        crop_to_physical_measures={1: (1, 2), 2: (3,)},
        expected_freeze_sha256=fixture["freeze_sha256"],
        expected_requests_sha256=fixture["requests_sha256"],
        expected_predictions_sha256=fixture["predictions_sha256"],
    )

    assert report["status"] == "evaluated_once_after_frozen_predictions"
    assert report["segmentation"]["automatic_crop_count"] == 2
    assert report["segmentation"]["physical_measure_count"] == 3
    assert report["segmentation"]["missed_barline_count"] == 1
    assert report["segmentation"]["automatic_crop_to_physical_measures"]["1"] == [1, 2]
    assert report["metrics"]["summary"]["note_f1"] == 1.0
    assert report["metrics"]["summary"]["rest_f1"] == 1.0
    assert report["metrics"]["summary"]["exact_measures"] == 2
    assert report["pipeline_integration_decision"]["status"] == "evidence_present"

    truth_rows = composed._read_jsonl(
        tmp_path
        / "fresh-score"
        / spike.OUTPUT_SUBDIR
        / "system_004"
        / spike.SPLIT_NAME
        / "truth.jsonl"
    )
    assert truth_rows[0]["physical_measure_numbers"] == [1, 2]
    assert truth_rows[0]["measure_extent_beats"] == 6
    assert truth_rows[0]["notes"] == [
        {"duration_beats": 2.0, "onset_beats": 0, "pitch_midi": 60},
        {"duration_beats": 3.0, "onset_beats": 3, "pitch_midi": 62},
    ]


def test_truth_loader_runs_only_after_every_frozen_artifact_is_verified(
    tmp_path: Path,
) -> None:
    fixture = _frozen_fixture(tmp_path)
    fixture["inference"].write_text("changed\n", encoding="utf-8")
    called = False

    def truth_loader(path: Path) -> spike.MusicXMLTruth:
        nonlocal called
        called = True
        return spike.load_musicxml_truth(path)

    with pytest.raises(ValueError, match="Frozen prediction artifact changed"):
        spike.evaluate_fresh_heldout(
            tmp_path,
            musicxml_path=fixture["musicxml"],
            slug="fresh-score",
            system_index=4,
            crop_to_physical_measures={1: (1, 2), 2: (3,)},
            expected_freeze_sha256=fixture["freeze_sha256"],
            expected_requests_sha256=fixture["requests_sha256"],
            expected_predictions_sha256=fixture["predictions_sha256"],
            truth_loader=truth_loader,
        )

    assert called is False


def test_existing_one_shot_rejects_changed_musicxml(tmp_path: Path) -> None:
    fixture = _frozen_fixture(tmp_path)
    kwargs = {
        "musicxml_path": fixture["musicxml"],
        "slug": "fresh-score",
        "system_index": 4,
        "crop_to_physical_measures": {1: (1, 2), 2: (3,)},
        "expected_freeze_sha256": fixture["freeze_sha256"],
        "expected_requests_sha256": fixture["requests_sha256"],
        "expected_predictions_sha256": fixture["predictions_sha256"],
    }
    first = spike.evaluate_fresh_heldout(tmp_path, **kwargs)
    second = spike.evaluate_fresh_heldout(tmp_path, **kwargs)
    assert second == first

    fixture["musicxml"].write_text(
        fixture["musicxml"].read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="MusicXML changed after the one-shot evaluation"):
        spike.evaluate_fresh_heldout(tmp_path, **kwargs)


def test_mapping_must_exactly_cover_musicxml_measures(tmp_path: Path) -> None:
    fixture = _frozen_fixture(tmp_path)

    with pytest.raises(ValueError, match="does not exactly cover MusicXML measures"):
        spike.evaluate_fresh_heldout(
            tmp_path,
            musicxml_path=fixture["musicxml"],
            slug="fresh-score",
            system_index=4,
            crop_to_physical_measures={1: (1,), 2: (3,)},
            expected_freeze_sha256=fixture["freeze_sha256"],
            expected_requests_sha256=fixture["requests_sha256"],
            expected_predictions_sha256=fixture["predictions_sha256"],
        )


def _frozen_fixture(tmp_path: Path) -> dict:
    root = tmp_path / "fresh-score" / spike.OUTPUT_SUBDIR / "system_004"
    split_dir = root / spike.SPLIT_NAME
    split_dir.mkdir(parents=True)
    requests_path = root / "requests.jsonl"
    predictions_path = split_dir / "predictions.jsonl"
    inference_path = split_dir / "inference.jsonl"
    musicxml_path = root / spike.DEFAULT_MUSICXML_NAME

    identities = [
        {
            "slug": "fresh-score",
            "system_index": 4,
            "system_measure_index": 1,
            "global_measure_index": 21,
        },
        {
            "slug": "fresh-score",
            "system_index": 4,
            "system_measure_index": 2,
            "global_measure_index": 22,
        },
    ]
    requests = [
        {
            "identity": identity,
            "allowed_context": {
                "time_signature": "3/4",
                "expected_measure_beats": "3",
                "allow_pickup": False,
            },
        }
        for identity in identities
    ]
    predictions = [
        {
            "identity": identities[0],
            "measure_extent_beats": 6,
            "notes": [
                {"onset_beats": 0, "duration_beats": 2, "pitch_midi": 60},
                {"onset_beats": 3, "duration_beats": 3, "pitch_midi": 62},
            ],
            "rests": [{"onset_beats": 2, "duration_beats": 1}],
        },
        {
            "identity": identities[1],
            "measure_extent_beats": 3,
            "notes": [{"onset_beats": 1, "duration_beats": 2, "pitch_midi": 64}],
            "rests": [{"onset_beats": 0, "duration_beats": 1}],
        },
    ]
    composed._write_jsonl(requests_path, requests)
    composed._write_jsonl(predictions_path, predictions)
    composed._write_jsonl(inference_path, [{"identity": identity} for identity in identities])
    musicxml_path.write_text(_musicxml(), encoding="utf-8")

    freeze = {
        "schema_version": 1,
        "status": "frozen_before_truth",
        "split": spike.SPLIT_NAME,
        "target_count": 2,
        "requests": {
            "path": str(requests_path),
            "sha256": composed._sha256(requests_path),
        },
        "artifacts": [
            {"path": str(predictions_path), "sha256": composed._sha256(predictions_path)},
            {"path": str(inference_path), "sha256": composed._sha256(inference_path)},
        ],
    }
    freeze_path = split_dir / "freeze.json"
    freeze_path.write_text(json.dumps(freeze, indent=2) + "\n", encoding="utf-8")
    return {
        "musicxml": musicxml_path,
        "inference": inference_path,
        "freeze_sha256": composed._sha256(freeze_path),
        "requests_sha256": composed._sha256(requests_path),
        "predictions_sha256": composed._sha256(predictions_path),
    }


def _musicxml() -> str:
    return """<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="4.0">
  <part-list><score-part id="P1"><part-name>Music</part-name></score-part></part-list>
  <part id="P1">
    <measure number="1">
      <attributes>
        <divisions>2</divisions>
        <key><fifths>-1</fifths></key>
        <time><beats>3</beats><beat-type>4</beat-type></time>
        <clef><sign>G</sign><line>2</line></clef>
      </attributes>
      <note><pitch><step>C</step><octave>4</octave></pitch><duration>4</duration></note>
      <note><rest/><duration>2</duration></note>
    </measure>
    <measure number="2">
      <note><pitch><step>D</step><octave>4</octave></pitch><duration>6</duration></note>
    </measure>
    <measure number="3">
      <note><rest/><duration>2</duration></note>
      <note><pitch><step>E</step><octave>4</octave></pitch><duration>4</duration></note>
    </measure>
  </part>
</score-partwise>
"""
