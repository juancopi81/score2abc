from __future__ import annotations

import json
import threading
from pathlib import Path
from urllib.request import urlopen

import pytest
from PIL import Image

from scripts import review_vlm_notehead_candidates as reviewer


def _candidate(candidate_id: str, rank: int, x: float, y: float) -> dict[str, object]:
    return {
        "id": candidate_id,
        "rank": rank,
        "bbox": {
            "left": x - 5,
            "top": y - 4,
            "right": x + 5,
            "bottom": y + 4,
        },
        "center": {"x": x, "y": y},
        "score": 1.0 / rank,
        "features": {},
    }


def _measure(tmp_path: Path) -> reviewer.ReviewMeasure:
    image_path = tmp_path / "measure.png"
    Image.new("RGB", (100, 60), "white").save(image_path)
    artifact_path = tmp_path / "candidates.json"
    artifact_path.write_text("{}\n", encoding="utf-8")
    return reviewer.ReviewMeasure(
        slug="example",
        system_index=1,
        measure_index=3,
        global_measure_index=3,
        title="Example",
        source_image_path=image_path,
        candidate_artifact_path=artifact_path,
        review_dir=tmp_path / "reviews/measure_003",
        image_sha256=reviewer._sha256(image_path),
        candidate_artifact_sha256=reviewer._sha256(artifact_path),
        image_width=100,
        image_height=60,
        staff_lines=(10.0, 20.0, 30.0, 40.0, 50.0),
        candidates=(
            _candidate("c001", 1, 20.0, 30.0),
            _candidate("c002", 2, 60.0, 20.0),
        ),
    )


def _payload(measure: reviewer.ReviewMeasure) -> dict[str, object]:
    return {
        "source_hashes": {
            "image_sha256": measure.image_sha256,
            "candidate_artifact_sha256": measure.candidate_artifact_sha256,
        },
        "selected_candidate_ids": ["c001"],
        "manual_noteheads": [],
        "pitch_overrides": {},
        "active_review_ms": 1500,
    }


def test_public_state_does_not_read_ground_truth(tmp_path: Path, monkeypatch) -> None:
    measure = _measure(tmp_path)

    def fail(*args, **kwargs):
        raise AssertionError("ground truth must not be read while building browser state")

    monkeypatch.setattr(reviewer.evaluator, "_ground_truth_path", fail)
    monkeypatch.setattr(reviewer.evaluator, "_load_ground_truth_fixture", fail)

    state = reviewer.build_public_measure_state(measure)

    assert state["measure_index"] == 3
    assert [candidate["id"] for candidate in state["candidates"]] == ["c001", "c002"]
    assert "metrics" not in json.dumps(state)


def test_save_writes_review_and_hidden_post_save_metrics(tmp_path: Path, monkeypatch) -> None:
    measure = _measure(tmp_path)
    gt_dir = tmp_path / "gt"
    monkeypatch.setattr(reviewer, "GT_DIR", gt_dir)
    gt_path = reviewer.evaluator._ground_truth_path(
        gt_dir,
        slug=measure.slug,
        system_index=measure.system_index,
        measure=measure.measure_index,
    )
    gt_path.parent.mkdir(parents=True, exist_ok=True)
    gt_path.write_text(
        json.dumps(
            {
                "noteheads": [
                    {
                        "id": "n001",
                        "pitch": "B4",
                        "center": {"x": 20.0, "y": 30.0},
                        "annotation_geometry": {
                            "radius_x_px": 5.0,
                            "radius_y_px": 4.0,
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = reviewer.save_review(measure, _payload(measure))
    saved = json.loads(measure.review_path.read_text(encoding="utf-8"))

    assert "metrics" not in result
    assert result["final_notehead_count"] == 1
    assert saved["metrics"]["selection"]["f1"] == 1.0
    assert saved["metrics"]["automatic_natural_pitch"]["accuracy"] == 1.0
    assert saved["candidates"][0]["label"] == "accepted"
    assert saved["candidates"][1]["label"] == "rejected"
    assert measure.overlay_path.exists()


def test_review_payload_rejects_stale_unknown_and_invalid_values(tmp_path: Path) -> None:
    measure = _measure(tmp_path)
    stale = _payload(measure)
    stale["source_hashes"] = {
        "image_sha256": "stale",
        "candidate_artifact_sha256": measure.candidate_artifact_sha256,
    }
    try:
        reviewer.validate_review_payload(measure, stale)
    except ValueError as exc:
        assert "stale" in str(exc)
    else:
        raise AssertionError("stale image hash should fail")

    unknown = _payload(measure)
    unknown["selected_candidate_ids"] = ["c999"]
    try:
        reviewer.validate_review_payload(measure, unknown)
    except ValueError as exc:
        assert "Unknown candidate" in str(exc)
    else:
        raise AssertionError("unknown candidate should fail")

    invalid_manual = _payload(measure)
    invalid_manual["manual_noteheads"] = [
        {"center": {"x": 200.0, "y": 10.0}, "pitch": "not-a-pitch"}
    ]
    try:
        reviewer.validate_review_payload(measure, invalid_manual)
    except ValueError as exc:
        assert "outside" in str(exc)
    else:
        raise AssertionError("out-of-range manual point should fail")


def test_http_server_serves_blind_state_and_source_image(tmp_path: Path) -> None:
    measure = _measure(tmp_path)
    app = reviewer.ReviewApp(
        out_dir=tmp_path,
        slug=measure.slug,
        system_index=measure.system_index,
        measures={measure.measure_index: measure},
    )
    try:
        server = reviewer.create_server(app, host="127.0.0.1", port=0)
    except PermissionError:
        pytest.skip("local sandbox denies socket binding")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    try:
        with urlopen(f"http://{host}:{port}/", timeout=2) as response:
            assert response.status == 200
            assert b"Notehead review" in response.read()
        with urlopen(f"http://{host}:{port}/api/state", timeout=2) as response:
            state = json.load(response)
        with urlopen(f"http://{host}:{port}/assets/measure_003.png", timeout=2) as response:
            assert response.headers["Content-Type"] == "image/png"
            assert response.read().startswith(b"\x89PNG")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert state["measures"][0]["measure_index"] == 3
    assert "metrics" not in json.dumps(state)
