"""Materialize a pinned human notehead-candidate review for consumed evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_vlm_notehead_localization_spike import treble_pitch_for_y  # noqa: E402

DECISION_KIND = "vlm_melody_consumed_human_candidate_review_decision"
REVIEW_KIND = "vlm_melody_consumed_human_candidate_review"
MANIFEST_KIND = "vlm_melody_consumed_human_candidate_review_manifest"
PROPOSALS_KIND = "vlm_melody_corrected_consumed_cross_score_proposals"
CANDIDATES_KIND = "vlm_notehead_candidates"
PITCH_RE = re.compile(r"^[A-G](?:#|b)?-?\d+$")
COMMON_COLOR = (0, 145, 70)
HUMAN_ONLY_COLOR = (20, 100, 210)
AUTO_ONLY_COLOR = (215, 45, 45)
REJECTED_COLOR = (155, 155, 155)
SUPPORT_COLOR = (145, 75, 185)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--decision-fixture", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = materialize_human_candidate_review(
            args.out_dir,
            decision_fixture=args.decision_fixture,
        )
    except (FileExistsError, FileNotFoundError, OSError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(result["review"])
    print(result["comparison_overlay"])
    return 0


def materialize_human_candidate_review(
    out_dir: Path,
    *,
    decision_fixture: Path,
    repo_root: Path = REPO_ROOT,
) -> dict[str, str]:
    """Validate one complete human decision and publish a create-once review."""
    root = repo_root.resolve()
    output_root = out_dir.resolve()
    fixture_path = decision_fixture.resolve()
    decision = _load_object(fixture_path, "Decision fixture")
    identity, namespace = _validate_header(decision)
    slug = identity["slug"]
    measure = identity["system_measure_index"]
    namespace_root = (output_root / slug / "vlm_melody_training_inputs" / namespace).resolve()
    destination = namespace_root / "human_reviews" / f"measure_{measure:03d}"
    temp_dir = destination.with_name(f".{destination.name}.tmp")
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite human review: {destination}")
    if temp_dir.exists():
        raise FileExistsError(f"Refusing stale human review temp: {temp_dir}")

    source = _required_object(decision, "source", "Decision fixture")
    raw_path = _validate_source_record(source, "raw_image", root, "Raw image")
    candidate_path = _validate_source_record(
        source,
        "candidate_artifact",
        root,
        "Candidate artifact",
    )
    proposals_path = _validate_source_record(source, "proposals", root, "Proposals")

    candidate_artifact = _load_object(candidate_path, "Candidate artifact")
    candidates = _validate_candidates(
        candidate_artifact,
        identity=identity,
        raw_path=raw_path,
        repo_root=root,
    )
    proposals = _load_object(proposals_path, "Proposals")
    automatic_ids = _validate_and_load_automatic_ids(
        proposals,
        identity=identity,
        namespace=namespace,
        candidates=candidates,
    )
    accepted, rejected = _validate_decisions(
        decision,
        candidates=candidates,
        staff_lines=tuple(float(value) for value in candidate_artifact["staff_lines_y_px"]),
    )
    comparison = _comparison(accepted, automatic_ids)

    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir()
    try:
        overlay_path = temp_dir / "comparison_overlay.png"
        _render_overlay(
            raw_path,
            candidates=candidates,
            accepted=accepted,
            rejected=rejected,
            automatic_ids=automatic_ids,
            comparison=comparison,
            destination=overlay_path,
        )
        fixture_record = _file_record(fixture_path, root)
        review = {
            "schema_version": 2,
            "kind": REVIEW_KIND,
            "split_status": "consumed_training",
            "reviewer_type": "human_candidate_adjudication",
            "human_reviewed": True,
            "eligible_for_spike_training": True,
            "eligible_for_heldout_claim": False,
            "identity": identity,
            "segmentation_namespace": namespace,
            "accepted_heads": accepted,
            "rejected_candidates": rejected,
            "comparison_to_automatic_proposal": comparison,
            "source": {
                "decision_fixture": fixture_record,
                "raw_image": _file_record(raw_path, root),
                "candidate_artifact": _file_record(candidate_path, root),
                "proposals": _file_record(proposals_path, root),
            },
            "provenance": {
                "scope": "spike_only_consumed_training",
                "visual_judgments_added_during_materialization": False,
                "candidate_partition_complete": True,
                "derived_head_count": sum(
                    head["localization_kind"] == "derived" for head in accepted
                ),
            },
        }
        _write_json(temp_dir / "review.json", review)
        manifest = {
            "schema_version": 2,
            "kind": MANIFEST_KIND,
            "create_once": True,
            "human_reviewed": True,
            "identity": identity,
            "segmentation_namespace": namespace,
            "outputs": {
                "review": _local_record(temp_dir / "review.json", temp_dir),
                "comparison_overlay": _local_record(overlay_path, temp_dir),
            },
            "source": review["source"],
        }
        _write_json(temp_dir / "manifest.json", manifest)
        temp_dir.rename(destination)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise

    return {
        "output_dir": str(destination),
        "review": str(destination / "review.json"),
        "comparison_overlay": str(destination / "comparison_overlay.png"),
        "manifest": str(destination / "manifest.json"),
    }


def _validate_header(decision: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    if decision.get("kind") != DECISION_KIND:
        raise ValueError(f"Decision fixture kind must be {DECISION_KIND!r}")
    required = {
        "split_status": "consumed_training",
        "reviewer_type": "human_candidate_adjudication",
        "human_reviewed": True,
        "eligible_for_spike_training": True,
        "eligible_for_heldout_claim": False,
    }
    for key, expected in required.items():
        if decision.get(key) != expected:
            raise ValueError(f"Decision fixture {key} must be {expected!r}")
    identity_raw = _required_object(decision, "identity", "Decision fixture")
    identity = {
        "slug": _required_string(identity_raw, "slug", "Decision identity"),
        "system_index": _required_positive_int(identity_raw, "system_index", "Decision identity"),
        "system_measure_index": _required_positive_int(
            identity_raw,
            "system_measure_index",
            "Decision identity",
        ),
        "physical_measure_number": _required_positive_int(
            identity_raw,
            "physical_measure_number",
            "Decision identity",
        ),
    }
    namespace = _required_string(
        decision,
        "segmentation_namespace",
        "Decision fixture",
    )
    return identity, namespace


def _validate_candidates(
    artifact: Mapping[str, Any],
    *,
    identity: Mapping[str, Any],
    raw_path: Path,
    repo_root: Path,
) -> dict[str, dict[str, Any]]:
    if artifact.get("kind") != CANDIDATES_KIND:
        raise ValueError("Unexpected candidate artifact kind")
    for field in ("slug", "system_index", "system_measure_index"):
        if artifact.get(field) != identity[field]:
            raise ValueError(f"Candidate artifact {field} mismatch")
    recorded_raw = _resolve_repo_path(
        _required_string(artifact, "source_image_path", "Candidate artifact"),
        repo_root,
    )
    if recorded_raw != raw_path:
        raise ValueError("Candidate artifact points at a different raw image")
    staff_lines = artifact.get("staff_lines_y_px")
    if not isinstance(staff_lines, list) or len(staff_lines) != 5:
        raise ValueError("Candidate artifact must provide five staff lines")
    rows = artifact.get("candidates")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Candidate artifact has no candidates")
    result = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"Candidate {index} must be an object")
        candidate_id = _required_string(row, "id", f"Candidate {index}")
        if candidate_id in result:
            raise ValueError(f"Duplicate candidate id: {candidate_id}")
        _validate_bbox(row.get("bbox"), f"Candidate {candidate_id}")
        _validate_point(row.get("center"), f"Candidate {candidate_id}")
        result[candidate_id] = dict(row)
    if int(artifact.get("candidate_count", -1)) != len(result):
        raise ValueError("Candidate artifact count mismatch")
    return result


def _validate_and_load_automatic_ids(
    proposals: Mapping[str, Any],
    *,
    identity: Mapping[str, Any],
    namespace: str,
    candidates: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    if proposals.get("kind") != PROPOSALS_KIND:
        raise ValueError("Unexpected proposals kind")
    if proposals.get("identity") != {
        "slug": identity["slug"],
        "system_index": identity["system_index"],
    }:
        raise ValueError("Proposals identity mismatch")
    if proposals.get("segmentation_namespace") != namespace:
        raise ValueError("Proposals namespace mismatch")
    tasks = proposals.get("tasks")
    if not isinstance(tasks, list):
        raise ValueError("Proposals tasks must be a list")
    matches = []
    for index, task in enumerate(tasks):
        if not isinstance(task, dict):
            raise ValueError(f"Automatic proposal task {index} must be an object")
        task_identity = task.get("identity")
        if not isinstance(task_identity, dict):
            raise ValueError(f"Automatic proposal task {index} identity must be an object")
        if task_identity.get("system_measure_index") == identity["system_measure_index"]:
            matches.append(task)
    if len(matches) != 1:
        raise ValueError("Expected exactly one automatic proposal for the reviewed measure")
    proposal = _required_object(matches[0], "proposal", "Automatic proposal task")
    assignments = proposal.get("assignments")
    if not isinstance(assignments, list):
        raise ValueError("Automatic proposal assignments must be a list")
    result = []
    for index, row in enumerate(assignments):
        if not isinstance(row, dict):
            raise ValueError(f"Automatic assignment {index} must be an object")
        candidate_id = _required_string(
            row,
            "candidate_id",
            f"Automatic assignment {index}",
        )
        if candidate_id not in candidates:
            raise ValueError(f"Unknown automatic candidate id: {candidate_id}")
        if candidate_id in result:
            raise ValueError(f"Duplicate automatic candidate id: {candidate_id}")
        result.append(candidate_id)
    return result


def _validate_decisions(
    decision: Mapping[str, Any],
    *,
    candidates: Mapping[str, Mapping[str, Any]],
    staff_lines: Sequence[float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    raw_heads = decision.get("accepted_heads")
    if not isinstance(raw_heads, list) or not raw_heads:
        raise ValueError("Decision fixture accepted_heads must be a non-empty list")
    accepted = []
    classified_positive_ids = set()
    for index, raw in enumerate(raw_heads, start=1):
        if not isinstance(raw, dict):
            raise ValueError(f"Accepted head {index} must be an object")
        if raw.get("order") != index:
            raise ValueError("Accepted head order must be contiguous from 1")
        sounding_pitch = _pitch(raw.get("sounding_pitch"), f"Accepted head {index}")
        staff_pitch = _pitch(raw.get("staff_pitch"), f"Accepted head {index}")
        candidate_id = raw.get("candidate_id")
        derived_from = raw.get("derived_from")
        if (candidate_id is None) == (derived_from is None):
            raise ValueError(
                f"Accepted head {index} must provide exactly one of candidate_id or derived_from"
            )
        if candidate_id is not None:
            if (
                not isinstance(candidate_id, str)
                or candidate_id not in candidates
                or candidate_id in classified_positive_ids
            ):
                raise ValueError(f"Invalid accepted candidate id: {candidate_id!r}")
            candidate = candidates[candidate_id]
            center = dict(candidate["center"])
            bbox = dict(candidate["bbox"])
            localization = {
                "localization_kind": "candidate",
                "candidate_id": candidate_id,
                "candidate_score": candidate.get("score"),
            }
            classified_positive_ids.add(candidate_id)
        else:
            localization, center, bbox, support_ids = _derived_head(
                derived_from,
                candidates=candidates,
                label=f"Accepted head {index}",
            )
            overlap = classified_positive_ids.intersection(support_ids)
            if overlap:
                raise ValueError(
                    f"Accepted head {index} reuses positive candidates: {sorted(overlap)}"
                )
            classified_positive_ids.update(support_ids)
        automatic_staff_pitch = treble_pitch_for_y(
            float(center["y"]),
            staff_lines,
        )
        accepted.append(
            {
                "order": index,
                "sounding_pitch": sounding_pitch,
                "staff_pitch": staff_pitch,
                "automatic_staff_pitch": automatic_staff_pitch,
                "automatic_pitch_corrected": automatic_staff_pitch != staff_pitch,
                "center": center,
                "bbox": bbox,
                **localization,
            }
        )
    x_values = [float(head["center"]["x"]) for head in accepted]
    if x_values != sorted(x_values) or len(x_values) != len(set(x_values)):
        raise ValueError("Accepted heads must be strictly ordered left to right")

    raw_classes = decision.get("rejected_candidate_classes")
    if not isinstance(raw_classes, list) or not raw_classes:
        raise ValueError("Decision fixture rejected_candidate_classes must be a non-empty list")
    rejected = []
    rejected_ids = set()
    seen_classes = set()
    for class_index, raw_class in enumerate(raw_classes):
        if not isinstance(raw_class, dict):
            raise ValueError(f"Rejected class {class_index} must be an object")
        class_name = _required_string(raw_class, "class", f"Rejected class {class_index}")
        if class_name in seen_classes:
            raise ValueError(f"Duplicate rejection class: {class_name}")
        seen_classes.add(class_name)
        description = _required_string(
            raw_class,
            "description",
            f"Rejected class {class_name}",
        )
        candidate_ids = raw_class.get("candidate_ids")
        if not isinstance(candidate_ids, list) or not candidate_ids:
            raise ValueError(f"Rejected class {class_name} must list candidate ids")
        for candidate_id in candidate_ids:
            if (
                not isinstance(candidate_id, str)
                or candidate_id not in candidates
                or candidate_id in classified_positive_ids
                or candidate_id in rejected_ids
            ):
                raise ValueError(f"Invalid rejected candidate id: {candidate_id!r}")
            candidate = candidates[candidate_id]
            rejected.append(
                {
                    "candidate_id": candidate_id,
                    "rejection_class": class_name,
                    "description": description,
                    "center": candidate["center"],
                    "bbox": candidate["bbox"],
                    "candidate_score": candidate.get("score"),
                }
            )
            rejected_ids.add(candidate_id)
    known_ids = set(candidates)
    if classified_positive_ids | rejected_ids != known_ids:
        missing = sorted(known_ids - classified_positive_ids - rejected_ids)
        raise ValueError(f"Human review does not classify every candidate: {missing}")
    return accepted, rejected


def _derived_head(
    value: Any,
    *,
    candidates: Mapping[str, Mapping[str, Any]],
    label: str,
) -> tuple[dict[str, Any], dict[str, float], dict[str, int], list[str]]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} derived_from must be an object")
    method = value.get("method")
    if method != "mean_candidate_centers":
        raise ValueError(f"{label} derived_from method is unsupported: {method!r}")
    support_ids = value.get("support_candidate_ids")
    if (
        not isinstance(support_ids, list)
        or len(support_ids) < 2
        or len(support_ids) != len(set(support_ids))
        or any(not isinstance(item, str) or item not in candidates for item in support_ids)
    ):
        raise ValueError(f"{label} must reference at least two unique support candidates")
    support = [candidates[candidate_id] for candidate_id in support_ids]
    center = {
        "x": round(sum(float(item["center"]["x"]) for item in support) / len(support), 6),
        "y": round(sum(float(item["center"]["y"]) for item in support) / len(support), 6),
    }
    bbox = {
        "left": min(int(item["bbox"]["left"]) for item in support),
        "top": min(int(item["bbox"]["top"]) for item in support),
        "right": max(int(item["bbox"]["right"]) for item in support),
        "bottom": max(int(item["bbox"]["bottom"]) for item in support),
    }
    localization = {
        "localization_kind": "derived",
        "derivation_method": method,
        "support_candidate_ids": list(support_ids),
        "support_candidates": [
            {
                "candidate_id": candidate_id,
                "center": dict(candidates[candidate_id]["center"]),
                "bbox": dict(candidates[candidate_id]["bbox"]),
                "candidate_score": candidates[candidate_id].get("score"),
            }
            for candidate_id in support_ids
        ],
    }
    return localization, center, bbox, support_ids


def _comparison(
    accepted: Sequence[Mapping[str, Any]],
    automatic_ids: Sequence[str],
) -> dict[str, Any]:
    unused_automatic = list(automatic_ids)
    matches = []
    false_negative_refs = []
    human_heads = []
    direct_human_ids = []
    for head in accepted:
        reference = (
            str(head["candidate_id"])
            if head["localization_kind"] == "candidate"
            else f"derived:{head['order']}"
        )
        eligible_ids = (
            [str(head["candidate_id"])]
            if head["localization_kind"] == "candidate"
            else list(head["support_candidate_ids"])
        )
        if head["localization_kind"] == "candidate":
            direct_human_ids.append(str(head["candidate_id"]))
        human_heads.append(
            {
                "order": head["order"],
                "reference": reference,
                "localization_kind": head["localization_kind"],
                "eligible_candidate_ids": eligible_ids,
            }
        )
        matched_id = next(
            (candidate_id for candidate_id in unused_automatic if candidate_id in eligible_ids),
            None,
        )
        if matched_id is None:
            false_negative_refs.append(reference)
            continue
        unused_automatic.remove(matched_id)
        matches.append(
            {
                "human_order": head["order"],
                "human_reference": reference,
                "automatic_candidate_id": matched_id,
                "match_kind": head["localization_kind"],
            }
        )
    tp = len(matches)
    fp = len(unused_automatic)
    fn = len(false_negative_refs)
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    return {
        "comparison_unit": "human_notehead",
        "metric_semantics": (
            "one automatic assignment can match one human notehead; additional rim "
            "assignments for the same derived head remain false positives"
        ),
        "human_heads": human_heads,
        "human_candidate_ids": direct_human_ids,
        "automatic_candidate_ids": list(automatic_ids),
        "matches": matches,
        "true_positive_candidate_ids": [match["automatic_candidate_id"] for match in matches],
        "false_positive_candidate_ids": list(unused_automatic),
        "false_negative_head_references": false_negative_refs,
        "false_negative_candidate_ids": false_negative_refs,
        "metrics": {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": _ratio(2 * precision * recall, precision + recall),
        },
    }


def _render_overlay(
    source: Path,
    *,
    candidates: Mapping[str, Mapping[str, Any]],
    accepted: Sequence[Mapping[str, Any]],
    rejected: Sequence[Mapping[str, Any]],
    automatic_ids: Sequence[str],
    comparison: Mapping[str, Any],
    destination: Path,
) -> None:
    with Image.open(source) as opened:
        raw = opened.convert("RGB")
    scale = max(1, (420 + raw.width - 1) // raw.width)
    if scale > 1:
        raw = raw.resize(
            (raw.width * scale, raw.height * scale),
            Image.Resampling.NEAREST,
        )
    legend_height = 104
    image = Image.new("RGB", (raw.width, raw.height + legend_height), "white")
    image.paste(raw, (0, legend_height))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text((6, 5), "Human vs automatic candidate proposal", fill="black", font=font)
    legend = (
        (6, 24, COMMON_COLOR, "agreed"),
        (102, 24, HUMAN_ONLY_COLOR, "human-only"),
        (206, 24, AUTO_ONLY_COLOR, "automatic FP"),
        (6, 42, REJECTED_COLOR, "rejected"),
        (102, 42, SUPPORT_COLOR, "derived-head support"),
    )
    for x, y, color, label in legend:
        draw.rectangle((x, y + 1, x + 10, y + 11), fill=color)
        draw.text((x + 14, y), label, fill="black", font=font)
    metrics = comparison["metrics"]
    draw.text(
        (6, 83),
        (
            f"proposal TP={metrics['tp']} FP={metrics['fp']} FN={metrics['fn']} "
            f"F1={metrics['f1']:.3f}"
        ),
        fill="black",
        font=font,
    )

    human_ids = {
        candidate_id
        for head in accepted
        for candidate_id in (
            [head["candidate_id"]]
            if head["localization_kind"] == "candidate"
            else head["support_candidate_ids"]
        )
    }
    automatic = set(automatic_ids)
    for rejected_item in rejected:
        candidate_id = rejected_item["candidate_id"]
        candidate = candidates[candidate_id]
        color = AUTO_ONLY_COLOR if candidate_id in automatic else REJECTED_COLOR
        width = 3 if candidate_id in automatic else 1
        label = (
            f"{candidate_id} AUTO FP {rejected_item['rejection_class']}"
            if candidate_id in automatic
            else None
        )
        _draw_candidate_box(
            draw,
            candidate,
            color=color,
            width=width,
            y_offset=legend_height,
            scale=scale,
            label=label,
            font=font,
        )
    for head in accepted:
        if head["localization_kind"] == "candidate":
            candidate_id = head["candidate_id"]
            color = COMMON_COLOR if candidate_id in automatic else HUMAN_ONLY_COLOR
            _draw_candidate_box(
                draw,
                candidates[candidate_id],
                color=color,
                width=3,
                y_offset=legend_height,
                scale=scale,
                label=f"{head['order']}:{candidate_id} {head['sounding_pitch']}",
                font=font,
            )
            continue
        for candidate_id in head["support_candidate_ids"]:
            _draw_candidate_box(
                draw,
                candidates[candidate_id],
                color=SUPPORT_COLOR,
                width=2,
                y_offset=legend_height,
                scale=scale,
                label=f"{candidate_id} rim",
                font=font,
            )
        matched_orders = {match["human_order"] for match in comparison["matches"]}
        color = COMMON_COLOR if head["order"] in matched_orders else HUMAN_ONLY_COLOR
        _draw_derived_center(
            draw,
            head,
            color=color,
            y_offset=legend_height,
            scale=scale,
            font=font,
        )
    if set(candidates) != human_ids | {item["candidate_id"] for item in rejected}:
        raise ValueError("Overlay candidate partition changed after validation")
    image.save(destination)


def _draw_derived_center(
    draw: ImageDraw.ImageDraw,
    head: Mapping[str, Any],
    *,
    color: tuple[int, int, int],
    y_offset: int,
    scale: int,
    font: ImageFont.ImageFont,
) -> None:
    x = int(round(float(head["center"]["x"]) * scale))
    y = int(round(float(head["center"]["y"]) * scale)) + y_offset
    radius = max(6, 4 * scale)
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=3)
    draw.line((x - radius - 3, y, x + radius + 3, y), fill=color, width=2)
    draw.line((x, y - radius - 3, x, y + radius + 3), fill=color, width=2)
    draw.text(
        (x + radius + 2, max(y_offset, y - 12)),
        f"{head['order']}:derived {head['sounding_pitch']}",
        fill=color,
        font=font,
        stroke_width=1,
        stroke_fill="white",
    )


def _draw_candidate_box(
    draw: ImageDraw.ImageDraw,
    candidate: Mapping[str, Any],
    *,
    color: tuple[int, int, int],
    width: int,
    y_offset: int,
    scale: int,
    label: str | None,
    font: ImageFont.ImageFont,
) -> None:
    bbox = candidate["bbox"]
    box = (
        int(bbox["left"]) * scale,
        int(bbox["top"]) * scale + y_offset,
        int(bbox["right"]) * scale,
        int(bbox["bottom"]) * scale + y_offset,
    )
    draw.rectangle(box, outline=color, width=width)
    if label is not None:
        draw.text(
            (box[0], max(y_offset, box[1] - 10)),
            label,
            fill=color,
            font=font,
            stroke_width=1,
            stroke_fill="white",
        )


def _validate_source_record(
    source: Mapping[str, Any],
    key: str,
    repo_root: Path,
    label: str,
) -> Path:
    record = _required_object(source, key, "Decision source")
    path = _resolve_repo_path(_required_string(record, "path", label), repo_root)
    try:
        path.relative_to(repo_root)
    except ValueError as exc:
        raise ValueError(f"{label} must stay under the repository root") from exc
    if not path.is_file():
        raise FileNotFoundError(f"{label} missing: {path}")
    expected_hash = _required_string(record, "sha256", label)
    actual_hash = _sha256(path)
    if actual_hash != expected_hash:
        raise ValueError(f"{label} hash drift: expected {expected_hash}, got {actual_hash}")
    return path


def _resolve_repo_path(value: str, repo_root: Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _validate_bbox(value: Any, label: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{label} bbox must be an object")
    coordinates = [value.get(key) for key in ("left", "top", "right", "bottom")]
    if any(isinstance(number, bool) or not isinstance(number, int) for number in coordinates):
        raise ValueError(f"{label} bbox values must be integers")
    left, top, right, bottom = coordinates
    if not left < right or not top < bottom:
        raise ValueError(f"{label} bbox must have positive area")


def _validate_point(value: Any, label: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{label} center must be an object")
    for axis in ("x", "y"):
        coordinate = value.get(axis)
        if isinstance(coordinate, bool) or not isinstance(coordinate, (int, float)):
            raise ValueError(f"{label} center {axis} must be numeric")


def _required_object(value: Mapping[str, Any], key: str, label: str) -> dict[str, Any]:
    result = value.get(key)
    if not isinstance(result, dict):
        raise ValueError(f"{label} {key} must be an object")
    return result


def _required_string(value: Mapping[str, Any], key: str, label: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result:
        raise ValueError(f"{label} {key} must be a non-empty string")
    return result


def _required_positive_int(value: Mapping[str, Any], key: str, label: str) -> int:
    result = value.get(key)
    if isinstance(result, bool) or not isinstance(result, int) or result < 1:
        raise ValueError(f"{label} {key} must be a positive integer")
    return result


def _pitch(value: Any, label: str) -> str:
    if not isinstance(value, str) or not PITCH_RE.fullmatch(value):
        raise ValueError(f"{label} pitch is invalid: {value!r}")
    return value


def _ratio(numerator: float, denominator: float) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def _load_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{label} must be a JSON object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _file_record(path: Path, repo_root: Path) -> dict[str, Any]:
    resolved = path.resolve()
    try:
        display_path = resolved.relative_to(repo_root).as_posix()
    except ValueError:
        display_path = str(resolved)
    return {
        "path": display_path,
        "sha256": _sha256(resolved),
        "bytes": resolved.stat().st_size,
    }


def _local_record(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
