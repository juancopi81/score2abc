from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from urllib.request import urlopen

import pytest
from PIL import Image

from scripts.experiments import review_hollow_notehead_unseen_gate as reviewer


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _record(path: Path, manifest_dir: Path) -> dict[str, str]:
    return {
        "path": path.relative_to(manifest_dir).as_posix(),
        "sha256": _sha256(path),
    }


def _identity(measure: int = 1) -> dict[str, object]:
    return {
        "slug": "unseen-example",
        "system_index": 5,
        "system_measure_index": measure,
        "global_measure_index": 40 + measure,
    }


def _build_gate(tmp_path: Path, *, measure_count: int = 1) -> Path:
    gate_dir = tmp_path / "gate"
    gate_dir.mkdir(parents=True)
    rows = []
    for measure_index in range(1, measure_count + 1):
        identity = _identity(measure_index)
        source_dir = gate_dir / "sources" / f"measure_{measure_index:03d}"
        source_dir.mkdir(parents=True)
        raw_path = source_dir / "raw.png"
        Image.new("RGB", (120, 80), "white").save(raw_path)
        candidate_path = source_dir / "candidates.json"
        _write_json(
            candidate_path,
            {
                **identity,
                "candidate_count": 91,
                "candidates": [
                    {
                        "id": "hidden-candidate-z91",
                        "bbox": {
                            "left": 11.111,
                            "top": 22.222,
                            "right": 33.333,
                            "bottom": 44.444,
                        },
                    }
                ],
            },
        )
        proposal_path = source_dir / "frozen_proposals.json"
        _write_json(
            proposal_path,
            {
                "kind": "frozen-hollow-proposals",
                "identity": identity,
                "proposal_count": 77,
                "proposals": [{"center": {"x": 71.234, "y": 63.987}}],
            },
        )
        rows.append(
            {
                "identity": identity,
                "raw_image": _record(raw_path, gate_dir),
                "candidate_artifact": _record(candidate_path, gate_dir),
                "proposal_artifact": _record(proposal_path, gate_dir),
            }
        )
    manifest_path = gate_dir / "sealed_manifest.json"
    _write_json(
        manifest_path,
        {
            "schema_version": 1,
            "kind": reviewer.MANIFEST_KIND,
            "status": "frozen_awaiting_human_review",
            "split": "fresh_heldout_morphology",
            "create_once": True,
            "truth_accessed": False,
            "measure_count": measure_count,
            "measures": rows,
            "provenance": {
                "ground_truth_files_read": [],
                "review_files_read": [],
                "musicxml_files_read": [],
            },
        },
    )
    return manifest_path


def _payload(*centers: tuple[float, float]) -> dict[str, object]:
    return {
        "centers": [{"x": x, "y": y} for x, y in centers],
        "completion_confirmed": True,
    }


def _rewrite_manifest_record_hash(
    manifest_path: Path,
    *,
    source_key: str,
    source_path: Path,
) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["measures"][0][source_key]["sha256"] = _sha256(source_path)
    _write_json(manifest_path, manifest)


def test_public_state_is_raw_only_and_preserves_manifest_order(tmp_path: Path) -> None:
    manifest_path = _build_gate(tmp_path, measure_count=2)

    state = reviewer.load_review_app(manifest_path).public_state()
    serialized = json.dumps(state)

    assert [row["identity"]["system_measure_index"] for row in state["measures"]] == [1, 2]
    assert state["measures"][0]["status"] == "pending"
    assert state["measures"][0]["existing_review"] is None
    assert "candidate" not in serialized.lower()
    assert "proposal" not in serialized.lower()
    assert "center" not in serialized.lower()
    assert "count" not in serialized.lower()
    assert "hidden-candidate-z91" not in serialized
    assert "71.234" not in serialized
    assert "63.987" not in serialized
    assert "11.111" not in serialized


def test_load_validates_candidate_and_proposal_identity(tmp_path: Path) -> None:
    manifest_path = _build_gate(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    proposal_path = manifest_path.parent / manifest["measures"][0]["proposal_artifact"]["path"]
    proposal = json.loads(proposal_path.read_text(encoding="utf-8"))
    proposal["identity"]["system_measure_index"] = 99
    _write_json(proposal_path, proposal)
    _rewrite_manifest_record_hash(
        manifest_path,
        source_key="proposal_artifact",
        source_path=proposal_path,
    )

    with pytest.raises(ValueError, match="Frozen proposal artifact identity mismatch"):
        reviewer.load_review_app(manifest_path)


def test_load_rejects_hash_drift(tmp_path: Path) -> None:
    manifest_path = _build_gate(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    candidate_path = manifest_path.parent / manifest["measures"][0]["candidate_artifact"]["path"]
    candidate_path.write_text(candidate_path.read_text(encoding="utf-8") + "\n")

    with pytest.raises(ValueError, match="candidate_artifact hash mismatch"):
        reviewer.load_review_app(manifest_path)


def test_load_rejects_nonblind_or_unfrozen_manifest(tmp_path: Path) -> None:
    manifest_path = _build_gate(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["truth_accessed"] = True
    manifest["provenance"]["ground_truth_files_read"] = ["truth.json"]
    _write_json(manifest_path, manifest)

    with pytest.raises(ValueError, match="create-once truth-blind"):
        reviewer.load_review_app(manifest_path)


def test_save_is_create_once_and_reopens_read_only(tmp_path: Path) -> None:
    manifest_path = _build_gate(tmp_path)
    app = reviewer.load_review_app(manifest_path)

    result = app.save(0, _payload((20.0, 30.0), (70.0, 45.0)))
    review_path = Path(result["review_path"])
    overlay_path = Path(result["overlay_path"])
    saved = json.loads(review_path.read_text(encoding="utf-8"))

    assert saved["kind"] == reviewer.REVIEW_KIND
    assert saved["identity"] == _identity()
    assert saved["source"]["coordinate_space"] == reviewer.COORDINATE_SPACE
    assert set(saved["source"]) == {
        "sealed_manifest",
        "raw_image",
        "candidate_artifact",
        "proposal_artifact",
        "coordinate_space",
    }
    assert saved["centers"] == [{"x": 20.0, "y": 30.0}, {"x": 70.0, "y": 45.0}]
    assert saved["completion_confirmed"] is True
    assert saved["completed_at"]
    assert overlay_path.is_file()

    with Image.open(overlay_path) as overlay:
        assert overlay.getpixel((20, 30)) != (255, 255, 255)
        assert overlay.getpixel((71, 64)) == (255, 255, 255)

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        app.save(0, _payload((25.0, 35.0)))

    reopened = reviewer.load_review_app(manifest_path).public_state()["measures"][0]
    assert reopened["status"] == "completed"
    assert reopened["existing_review"]["centers"] == saved["centers"]
    assert reopened["image_url"].endswith("/truth-overlay")


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (None, "JSON object"),
        ({"centers": []}, "only centers and completion_confirmed"),
        (
            {"centers": [], "completion_confirmed": False},
            "completion_confirmed must be true",
        ),
        ({"centers": {}, "completion_confirmed": True}, "centers must be an array"),
        (
            {"centers": [{"x": 10.0}], "completion_confirmed": True},
            "must contain only x and y",
        ),
        (
            {"centers": [{"x": True, "y": 10.0}], "completion_confirmed": True},
            "must be a finite number",
        ),
        (
            {"centers": [{"x": 120.0, "y": 10.0}], "completion_confirmed": True},
            "outside the raw image",
        ),
        (
            {
                "centers": [{"x": 10.0, "y": 10.0}, {"x": 12.0, "y": 12.0}],
                "completion_confirmed": True,
            },
            "duplicates or nearly duplicates",
        ),
    ],
)
def test_payload_validation_rejects_malformed_values(
    tmp_path: Path,
    payload: object,
    message: str,
) -> None:
    measure = reviewer.load_review_app(_build_gate(tmp_path)).measures[0]

    with pytest.raises(ValueError, match=message):
        reviewer.validate_review_payload(measure, payload)


def test_empty_truth_is_a_valid_completed_review(tmp_path: Path) -> None:
    app = reviewer.load_review_app(_build_gate(tmp_path))

    result = app.save(0, _payload())

    assert result["center_count"] == 0


def test_save_rechecks_source_and_manifest_hashes(tmp_path: Path) -> None:
    manifest_path = _build_gate(tmp_path)
    app = reviewer.load_review_app(manifest_path)
    app.measures[0].candidate_path.write_text(
        app.measures[0].candidate_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Candidate artifact hash is stale"):
        app.save(0, _payload())

    manifest_path = _build_gate(tmp_path / "second")
    app = reviewer.load_review_app(manifest_path)
    manifest_path.write_text(
        manifest_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Sealed manifest hash is stale"):
        app.public_state()


def test_http_state_and_page_do_not_expose_automatic_artifacts(tmp_path: Path) -> None:
    app = reviewer.load_review_app(_build_gate(tmp_path))
    try:
        server = reviewer.create_server(app, host="127.0.0.1", port=0)
    except PermissionError:
        pytest.skip("local sandbox denies socket binding")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    try:
        with urlopen(f"http://{host}:{port}/", timeout=2) as response:
            page = response.read().decode("utf-8")
        with urlopen(f"http://{host}:{port}/api/state", timeout=2) as response:
            state = response.read().decode("utf-8")
        with urlopen(f"http://{host}:{port}/assets/0/raw", timeout=2) as response:
            raw = response.read()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    combined = (page + state).lower()
    assert "candidate" not in combined
    assert "proposal" not in combined
    assert "hidden-candidate-z91" not in combined
    assert "71.234" not in combined
    assert raw.startswith(b"\x89PNG")
