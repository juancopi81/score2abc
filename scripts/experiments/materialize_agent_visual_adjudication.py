"""Materialize accepted spike-only notehead reviews from a pinned decision fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiments.prepare_consumed_cross_score_proposals import (  # noqa: E402
    load_notated_pitches,
)

FIXTURE_KIND = "vlm_melody_agent_visual_adjudication"
PROPOSALS_KIND = "vlm_melody_corrected_consumed_cross_score_proposals"
REVIEW_KIND = "vlm_melody_agent_visual_adjudication_review"
MANIFEST_KIND = "vlm_melody_agent_visual_adjudication_manifest"
REVIEWER_TYPE = "agent_visual_adjudication"
PARENT_STATUS = "accepted_for_spike_training"
NAMESPACE = "carrizal_system_004_seg_v2"
EXPECTED_MEASURES = tuple(range(1, 9))
CONFIDENCES = {"high", "medium"}
HIGH_COLOR = (0, 166, 80)
MEDIUM_COLOR = (240, 160, 0)
MANUAL_COLOR = (142, 68, 173)
REJECTED_COLOR = (220, 45, 45)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--decision-fixture", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        destination = materialize_agent_visual_adjudication(
            args.out_dir,
            decision_fixture=args.decision_fixture,
        )
    except (FileExistsError, FileNotFoundError, OSError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(destination)
    return 0


def materialize_agent_visual_adjudication(
    out_dir: Path,
    *,
    decision_fixture: Path,
    repo_root: Path = REPO_ROOT,
) -> Path:
    root = repo_root.resolve()
    out_root = out_dir.resolve()
    fixture_path = decision_fixture.resolve()
    fixture = _load_object(fixture_path, "Decision fixture")
    identity = _validate_fixture_header(fixture)
    slug = identity["slug"]

    source = _required_object(fixture, "source", "Decision fixture")
    namespace_manifest_path = _validate_source_record(
        source, "namespace_manifest", root, "Namespace manifest"
    )
    proposals_path = _validate_source_record(source, "proposals", root, "Proposals")
    candidates_manifest_path = _validate_source_record(
        source, "candidates_manifest", root, "Candidates manifest"
    )
    musicxml_path = _validate_source_record(source, "musicxml", root, "MusicXML")

    namespace_dir = namespace_manifest_path.parent
    expected_namespace_dir = (out_root / slug / "vlm_melody_training_inputs" / NAMESPACE).resolve()
    if namespace_dir != expected_namespace_dir:
        raise ValueError("Decision fixture does not target the corrected Carrizal namespace")
    destination = namespace_dir / "agent_reviews"
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing agent reviews: {destination}")

    namespace_manifest = _load_object(namespace_manifest_path, "Namespace manifest")
    _validate_namespace_manifest(namespace_manifest, identity)
    proposals = _load_object(proposals_path, "Proposals")
    _validate_proposals(proposals, identity, source)
    candidate_rows = _read_jsonl(candidates_manifest_path)
    candidates_by_measure = _validate_candidate_rows(
        candidate_rows,
        proposals=proposals,
        identity=identity,
        repo_root=root,
    )
    expected_pitches = load_notated_pitches(musicxml_path)
    decisions = _validate_decisions(
        fixture,
        proposals=proposals,
        candidates_by_measure=candidates_by_measure,
        expected_pitches=expected_pitches,
    )

    fixture_record = _file_record(fixture_path, root)
    shared_source = {
        "decision_fixture": fixture_record,
        "namespace_manifest": _file_record(namespace_manifest_path, root),
        "proposals": _file_record(proposals_path, root),
        "candidates_manifest": _file_record(candidates_manifest_path, root),
        "musicxml": _file_record(musicxml_path, root),
    }
    with tempfile.TemporaryDirectory(prefix=".agent_reviews-", dir=namespace_dir) as temp_name:
        temp_dir = Path(temp_name)
        overlays_dir = temp_dir / "overlays"
        overlays_dir.mkdir()
        reviews = []
        overlay_paths = []
        for measure in EXPECTED_MEASURES:
            item = decisions[measure]
            raw_path = item["raw_image_path"]
            candidate_path = item["candidate_path"]
            overlay_path = overlays_dir / f"measure_{measure:03d}.png"
            _render_overlay(
                raw_path,
                candidate_artifact=item["candidate_artifact"],
                heads=item["heads"],
                destination=overlay_path,
                measure=measure,
            )
            review_path = temp_dir / f"measure_{measure:03d}.json"
            review = _review_payload(
                identity=identity,
                measure=measure,
                heads=item["heads"],
                raw_path=raw_path,
                candidate_path=candidate_path,
                overlay_path=overlay_path,
                final_overlay_path=destination / "overlays" / overlay_path.name,
                shared_source=shared_source,
                fixture_record=fixture_record,
                repo_root=root,
            )
            _write_json(review_path, review)
            reviews.append(review_path)
            overlay_paths.append(overlay_path)

        contact_sheet = temp_dir / "contact_sheet.png"
        _render_contact_sheet(overlay_paths, contact_sheet)
        confidence_counts = Counter(
            head["confidence"] for item in decisions.values() for head in item["heads"]
        )
        selection_counts = Counter(
            head["selection"]["kind"] for item in decisions.values() for head in item["heads"]
        )
        manifest = {
            "schema_version": 1,
            "kind": MANIFEST_KIND,
            "split_status": "consumed_training",
            "reviewer_type": REVIEWER_TYPE,
            "parent_review_status": PARENT_STATUS,
            "eligible_for_spike_training": True,
            "eligible_for_human_promotion": False,
            "human_reviewed": False,
            "identity": identity,
            "segmentation_namespace": NAMESPACE,
            "counts": {
                "measures": len(EXPECTED_MEASURES),
                "noteheads": sum(confidence_counts.values()),
                "high_confidence": confidence_counts["high"],
                "medium_confidence": confidence_counts["medium"],
                "candidate_selections": selection_counts["candidate"],
                "manual_selections": selection_counts["manual"],
            },
            "training_selection": {
                "high_confidence_default": True,
                "medium_confidence_default": False,
                "medium_confidence_separately_selectable": True,
            },
            "source": shared_source,
            "outputs": {
                "reviews": [
                    _file_record_at(path, destination / path.name, root) for path in reviews
                ],
                "overlays": [
                    _file_record_at(path, destination / "overlays" / path.name, root)
                    for path in overlay_paths
                ],
                "contact_sheet": _file_record_at(
                    contact_sheet, destination / contact_sheet.name, root
                ),
            },
            "provenance": {
                "scope": "spike_only",
                "decision_fixture_sha256": fixture_record["sha256"],
                "materializer": _file_record(Path(__file__).resolve(), root),
                "visual_judgments_added_during_materialization": False,
            },
        }
        _write_json(temp_dir / "manifest.json", manifest)
        temp_dir.rename(destination)
    return destination


def _validate_fixture_header(fixture: Mapping[str, Any]) -> dict[str, Any]:
    if fixture.get("kind") != FIXTURE_KIND:
        raise ValueError(f"Decision fixture kind must be {FIXTURE_KIND!r}")
    required = {
        "split_status": "consumed_training",
        "reviewer_type": REVIEWER_TYPE,
        "parent_review_status": PARENT_STATUS,
        "eligible_for_spike_training": True,
        "eligible_for_human_promotion": False,
        "human_reviewed": False,
        "segmentation_namespace": NAMESPACE,
    }
    for key, expected in required.items():
        if fixture.get(key) != expected:
            raise ValueError(f"Decision fixture {key} must be {expected!r}")
    identity = _required_object(fixture, "identity", "Decision fixture")
    slug = _required_string(identity, "slug", "Decision fixture identity")
    system_index = _required_int(identity, "system_index", "Decision fixture identity")
    if system_index != 4:
        raise ValueError("Decision fixture must target Carrizal system 4")
    return {"slug": slug, "system_index": system_index}


def _validate_namespace_manifest(manifest: Mapping[str, Any], identity: Mapping[str, Any]) -> None:
    if manifest.get("kind") != "vlm_melody_consumed_cross_score_training_inputs":
        raise ValueError("Unexpected namespace manifest kind")
    if manifest.get("identity") != identity:
        raise ValueError("Namespace manifest identity mismatch")
    if manifest.get("segmentation_namespace") != NAMESPACE:
        raise ValueError("Namespace manifest segmentation mismatch")


def _validate_proposals(
    proposals: Mapping[str, Any],
    identity: Mapping[str, Any],
    fixture_source: Mapping[str, Any],
) -> None:
    if proposals.get("kind") != PROPOSALS_KIND:
        raise ValueError("Unexpected proposals kind")
    if (
        proposals.get("identity") != identity
        or proposals.get("segmentation_namespace") != NAMESPACE
    ):
        raise ValueError("Proposals identity or namespace mismatch")
    source = _required_object(proposals, "source", "Proposals")
    for key in ("namespace_manifest", "candidates_manifest", "musicxml"):
        if source.get(key) != fixture_source.get(key):
            raise ValueError(f"Proposals {key} source does not match the decision fixture")


def _validate_candidate_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    proposals: Mapping[str, Any],
    identity: Mapping[str, Any],
    repo_root: Path,
) -> dict[int, dict[str, Any]]:
    proposal_source = _required_object(proposals, "source", "Proposals")
    candidate_records = _indexed_source_records(proposal_source, "candidate_artifacts")
    raw_records = _indexed_source_records(proposal_source, "raw_images")
    result = {}
    for row in rows:
        row_identity = _required_object(row, "identity", "Candidate row")
        measure = _required_int(row_identity, "system_measure_index", "Candidate identity")
        if {
            "slug": row_identity.get("slug"),
            "system_index": row_identity.get("system_index"),
        } != identity:
            raise ValueError(f"Candidate identity mismatch for measure {measure}")
        if measure in result:
            raise ValueError(f"Duplicate candidate row for measure {measure}")
        artifact_record = _required_object(
            _required_object(row, "artifacts", "Candidate row"),
            "candidates",
            "Candidate row artifacts",
        )
        raw_record = _required_object(
            _required_object(row, "source", "Candidate row"),
            "measure_raw",
            "Candidate row source",
        )
        if dict(artifact_record) != candidate_records.get(measure):
            raise ValueError(f"Candidate artifact mismatch for measure {measure}")
        if dict(raw_record) != raw_records.get(measure):
            raise ValueError(f"Raw image mismatch for measure {measure}")
        candidate_path = _validate_record(artifact_record, repo_root, "Candidate artifact")
        raw_path = _validate_record(raw_record, repo_root, "Raw image")
        artifact = _load_object(candidate_path, f"Measure {measure} candidates")
        result[measure] = {
            "candidate_path": candidate_path,
            "candidate_artifact": artifact,
            "raw_image_path": raw_path,
        }
    if set(result) != set(EXPECTED_MEASURES):
        raise ValueError("Candidates manifest must contain measures 1 through 8")
    return result


def _validate_decisions(
    fixture: Mapping[str, Any],
    *,
    proposals: Mapping[str, Any],
    candidates_by_measure: Mapping[int, Mapping[str, Any]],
    expected_pitches: Mapping[int, Sequence[Any]],
) -> dict[int, dict[str, Any]]:
    measures = fixture.get("measures")
    if not isinstance(measures, list):
        raise ValueError("Decision fixture measures must be a list")
    proposal_tasks = {
        int(task["identity"]["physical_measure_number"]): task
        for task in _required_list(proposals, "tasks", "Proposals")
    }
    result = {}
    for expected_measure, measure_payload in zip(EXPECTED_MEASURES, measures, strict=True):
        if not isinstance(measure_payload, dict):
            raise ValueError(f"Measure {expected_measure} decision must be an object")
        measure = _required_int(measure_payload, "physical_measure_number", "Decision measure")
        if measure != expected_measure:
            raise ValueError("Decision measures must be exactly 1 through 8 in order")
        expected = list(expected_pitches.get(measure, ()))
        proposal_task = proposal_tasks.get(measure)
        if proposal_task is None:
            raise ValueError(f"Missing proposal task for measure {measure}")
        proposal_expected = _required_object(proposal_task, "expected", "Proposal task")
        expected_sound = [pitch.sounding_pitch for pitch in expected]
        if proposal_expected.get("ordered_sounding_pitches") != expected_sound:
            raise ValueError(f"Proposal pitch order drift for measure {measure}")
        raw_path = Path(candidates_by_measure[measure]["raw_image_path"])
        with Image.open(raw_path) as image:
            width, height = image.size
        artifact = candidates_by_measure[measure]["candidate_artifact"]
        candidates = artifact.get("candidates")
        if not isinstance(candidates, list):
            raise ValueError(f"Candidate artifact {measure} has no candidates list")
        by_id = {}
        for candidate in candidates:
            candidate_id = _required_string(candidate, "id", f"Measure {measure} candidate")
            if candidate_id in by_id:
                raise ValueError(f"Duplicate candidate id {candidate_id} in measure {measure}")
            by_id[candidate_id] = candidate
        raw_heads = measure_payload.get("heads")
        if not isinstance(raw_heads, list):
            raise ValueError(f"Measure {measure} heads must be a list")
        if len(raw_heads) != len(expected):
            raise ValueError(f"Pitch/count mismatch for measure {measure}")
        heads = []
        used_candidates = set()
        for order, (raw_head, pitch) in enumerate(zip(raw_heads, expected, strict=True), start=1):
            if not isinstance(raw_head, dict) or raw_head.get("order") != order:
                raise ValueError(f"Head order mismatch in measure {measure}")
            if raw_head.get("sounding_pitch") != pitch.sounding_pitch:
                raise ValueError(f"Pitch/order mismatch in measure {measure} at head {order}")
            confidence = raw_head.get("confidence")
            if confidence not in CONFIDENCES:
                raise ValueError(f"Invalid confidence in measure {measure} at head {order}")
            selection = _required_object(raw_head, "selection", f"Measure {measure} head {order}")
            kind = selection.get("kind")
            if kind == "candidate":
                candidate_id = _required_string(selection, "candidate_id", "Candidate selection")
                if candidate_id not in by_id:
                    raise ValueError(
                        f"Candidate mismatch for measure {measure}: {candidate_id!r} not found"
                    )
                if candidate_id in used_candidates:
                    raise ValueError(
                        f"Candidate {candidate_id} selected twice in measure {measure}"
                    )
                used_candidates.add(candidate_id)
                candidate = by_id[candidate_id]
                center = _point(candidate.get("center"), f"Candidate {candidate_id} center")
                bbox = _bbox(candidate.get("bbox"), width, height, f"Candidate {candidate_id}")
                selection_payload = {"kind": "candidate", "candidate_id": candidate_id}
            elif kind == "manual":
                center = _point(selection.get("center"), "Manual center")
                _validate_point_in_bounds(center, width, height, f"Measure {measure} manual center")
                bbox = _manual_bbox(center, width, height)
                selection_payload = {"kind": "manual", "center": center}
            else:
                raise ValueError(f"Unknown selection kind for measure {measure} at head {order}")
            heads.append(
                {
                    "order": order,
                    "staff_pitch": pitch.staff_pitch,
                    "sounding_pitch": pitch.sounding_pitch,
                    "pitch_midi": pitch.pitch_midi,
                    "alter": pitch.alter,
                    "tie_types": list(pitch.tie_types),
                    "confidence": confidence,
                    "include_in_high_confidence_training": confidence == "high",
                    "include_in_medium_confidence_training": confidence == "medium",
                    "selection": selection_payload,
                    "center": center,
                    "bbox": bbox,
                }
            )
        result[measure] = {
            **candidates_by_measure[measure],
            "heads": heads,
        }
    confidence_counts = Counter(
        head["confidence"] for item in result.values() for head in item["heads"]
    )
    if confidence_counts != Counter({"high": 18, "medium": 2}):
        raise ValueError("Decision fixture confidence counts must be exactly 18 high and 2 medium")
    return result


def _review_payload(
    *,
    identity: Mapping[str, Any],
    measure: int,
    heads: Sequence[Mapping[str, Any]],
    raw_path: Path,
    candidate_path: Path,
    overlay_path: Path,
    final_overlay_path: Path,
    shared_source: Mapping[str, Any],
    fixture_record: Mapping[str, str],
    repo_root: Path,
) -> dict[str, Any]:
    counts = Counter(head["confidence"] for head in heads)
    return {
        "schema_version": 1,
        "kind": REVIEW_KIND,
        "split_status": "consumed_training",
        "reviewer_type": REVIEWER_TYPE,
        "parent_review_status": PARENT_STATUS,
        "eligible_for_spike_training": True,
        "eligible_for_human_promotion": False,
        "human_reviewed": False,
        "identity": {
            **identity,
            "system_measure_index": measure,
            "physical_measure_number": measure,
        },
        "heads": list(heads),
        "confidence_counts": {"high": counts["high"], "medium": counts["medium"]},
        "training_selection": {
            "high_confidence_default": True,
            "medium_confidence_default": False,
            "medium_confidence_separately_selectable": True,
        },
        "source": {
            **shared_source,
            "raw_image": _file_record(raw_path, repo_root),
            "candidate_artifact": _file_record(candidate_path, repo_root),
        },
        "artifacts": {"overlay": _file_record_at(overlay_path, final_overlay_path, repo_root)},
        "provenance": {
            "scope": "spike_only",
            "decision_fixture_sha256": fixture_record["sha256"],
            "human_judgment_claimed": False,
        },
    }


def _render_overlay(
    source: Path,
    *,
    candidate_artifact: Mapping[str, Any],
    heads: Sequence[Mapping[str, Any]],
    destination: Path,
    measure: int,
) -> None:
    with Image.open(source) as opened:
        image = opened.convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    selected_ids = {
        head["selection"].get("candidate_id")
        for head in heads
        if head["selection"]["kind"] == "candidate"
    }
    for candidate in candidate_artifact["candidates"]:
        if candidate["id"] not in selected_ids:
            draw.rectangle(_bbox_tuple(candidate["bbox"]), outline=REJECTED_COLOR, width=1)
    for head in heads:
        bbox = _bbox_tuple(head["bbox"])
        color = HIGH_COLOR if head["confidence"] == "high" else MEDIUM_COLOR
        draw.rectangle(bbox, outline=color, width=3)
        if head["selection"]["kind"] == "manual":
            draw.rectangle(_expand_bbox(bbox, 4, image.size), outline=MANUAL_COLOR, width=2)
        label = f'{head["order"]}:{head["sounding_pitch"]}:{head["confidence"][0].upper()}'
        label_y = max(0, bbox[1] - 12)
        draw.text(
            (bbox[0], label_y), label, fill=color, font=font, stroke_width=1, stroke_fill="white"
        )
    draw.text((4, 4), f"M{measure}", fill="black", font=font, stroke_width=1, stroke_fill="white")
    image.save(destination)


def _render_contact_sheet(overlays: Sequence[Path], destination: Path) -> None:
    columns = 2
    cell_width = 620
    label_height = 24
    rendered = []
    cell_heights = []
    for path in overlays:
        with Image.open(path) as opened:
            image = opened.convert("RGB")
        image.thumbnail((cell_width - 16, 300), Image.Resampling.LANCZOS)
        rendered.append(image)
        cell_heights.append(image.height + label_height + 16)
    row_heights = [
        max(cell_heights[index : index + columns]) for index in range(0, len(cell_heights), columns)
    ]
    sheet = Image.new("RGB", (cell_width * columns, sum(row_heights)), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    y = 0
    for row, row_height in enumerate(row_heights):
        for column in range(columns):
            index = row * columns + column
            if index >= len(rendered):
                break
            x = column * cell_width
            draw.text((x + 8, y + 5), f"Measure {index + 1}", fill="black", font=font)
            sheet.paste(rendered[index], (x + 8, y + label_height))
        y += row_height
    sheet.save(destination)


def _indexed_source_records(source: Mapping[str, Any], key: str) -> dict[int, dict[str, Any]]:
    rows = _required_list(source, key, "Proposals source")
    result = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"Proposals source {key} entries must be objects")
        measure = _required_int(row, "system_measure_index", f"Proposals source {key}")
        result[measure] = {"path": row.get("path"), "sha256": row.get("sha256")}
    return result


def _validate_source_record(
    source: Mapping[str, Any], key: str, repo_root: Path, label: str
) -> Path:
    return _validate_record(_required_object(source, key, "Decision source"), repo_root, label)


def _validate_record(record: Mapping[str, Any], repo_root: Path, label: str) -> Path:
    path_text = _required_string(record, "path", label)
    expected_hash = _required_string(record, "sha256", label)
    path = (
        (repo_root / path_text).resolve() if not Path(path_text).is_absolute() else Path(path_text)
    )
    try:
        path.relative_to(repo_root)
    except ValueError as exc:
        raise ValueError(f"{label} must stay under the repository root") from exc
    if not path.is_file():
        raise FileNotFoundError(f"{label} missing: {path}")
    actual = _sha256(path)
    if actual != expected_hash:
        raise ValueError(f"{label} hash drift: expected {expected_hash}, got {actual}")
    return path


def _point(value: Any, label: str) -> dict[str, float | int]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    x, y = value.get("x"), value.get("y")
    if not isinstance(x, (int, float)) or isinstance(x, bool):
        raise ValueError(f"{label}.x must be numeric")
    if not isinstance(y, (int, float)) or isinstance(y, bool):
        raise ValueError(f"{label}.y must be numeric")
    return {"x": x, "y": y}


def _bbox(value: Any, width: int, height: int, label: str) -> dict[str, int]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} bbox must be an object")
    result = {key: int(value[key]) for key in ("left", "top", "right", "bottom")}
    if not (0 <= result["left"] < result["right"] < width):
        raise ValueError(f"{label} bbox is outside image width")
    if not (0 <= result["top"] < result["bottom"] < height):
        raise ValueError(f"{label} bbox is outside image height")
    return result


def _manual_bbox(center: Mapping[str, float | int], width: int, height: int) -> dict[str, int]:
    x, y = float(center["x"]), float(center["y"])
    return {
        "left": max(0, round(x - 10)),
        "top": max(0, round(y - 9)),
        "right": min(width - 1, round(x + 10)),
        "bottom": min(height - 1, round(y + 9)),
    }


def _validate_point_in_bounds(
    point: Mapping[str, float | int], width: int, height: int, label: str
) -> None:
    if not (0 <= float(point["x"]) < width and 0 <= float(point["y"]) < height):
        raise ValueError(f"{label} is outside image bounds {width}x{height}")


def _bbox_tuple(bbox: Mapping[str, Any]) -> tuple[int, int, int, int]:
    return tuple(int(bbox[key]) for key in ("left", "top", "right", "bottom"))


def _expand_bbox(
    bbox: tuple[int, int, int, int], amount: int, size: tuple[int, int]
) -> tuple[int, int, int, int]:
    width, height = size
    return (
        max(0, bbox[0] - amount),
        max(0, bbox[1] - amount),
        min(width - 1, bbox[2] + amount),
        min(height - 1, bbox[3] + amount),
    )


def _file_record(path: Path, repo_root: Path) -> dict[str, str]:
    resolved = path.resolve()
    try:
        display = resolved.relative_to(repo_root).as_posix()
    except ValueError:
        display = str(resolved)
    return {"path": display, "sha256": _sha256(resolved)}


def _file_record_at(path: Path, final_path: Path, repo_root: Path) -> dict[str, str]:
    try:
        display = final_path.resolve().relative_to(repo_root).as_posix()
    except ValueError:
        display = str(final_path.resolve())
    return {"path": display, "sha256": _sha256(path)}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{number} must contain a JSON object")
        rows.append(value)
    return rows


def _load_object(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _required_object(payload: Mapping[str, Any], key: str, label: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{label}.{key} must be an object")
    return value


def _required_list(payload: Mapping[str, Any], key: str, label: str) -> list[Any]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise ValueError(f"{label}.{key} must be a list")
    return value


def _required_string(payload: Mapping[str, Any], key: str, label: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label}.{key} must be a non-empty string")
    return value


def _required_int(payload: Mapping[str, Any], key: str, label: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{label}.{key} must be an integer")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
