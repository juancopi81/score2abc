"""Score a sealed hollow-notehead gate after create-once human review.

The evaluator preserves the heldout order of operations:

1. verify the truth-blind freeze and all hash-pinned artifacts;
2. require every raw-only human review to be complete;
3. score baseline candidates and frozen proposals against those reviews;
4. publish a create-once report and visual overlays.

The gate promotes a rule only when the review contains enough hollow heads,
the baseline leaves at least one recovery opportunity, and the frozen proposals
recover those opportunities with high precision. An inapplicable or undersized
gate is reported as ``not_promoted`` rather than silently treated as a pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiments import freeze_hollow_notehead_unseen_gate as freezer  # noqa: E402
from scripts.experiments import review_hollow_notehead_unseen_gate as reviewer  # noqa: E402
from scripts.experiments import spike_consumed_hollow_notehead_proposals as hollow  # noqa: E402

SCHEMA_VERSION = 1
REPORT_KIND = "vlm_melody_hollow_notehead_unseen_gate_evaluation"
MANIFEST_KIND = f"{REPORT_KIND}_manifest"
DEFAULT_VERSION = "evaluation_v1"
MIN_TRUTH_NOTEHEADS = 5
MIN_RECOVERY_OPPORTUNITIES = 1
MIN_PROPOSAL_PRECISION = 0.90
MIN_RECOVERY_RATE = 0.70


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sealed_manifest", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to <sealed-manifest-dir>/evaluation_v1.",
    )
    args = parser.parse_args(argv)
    try:
        output = evaluate_hollow_notehead_unseen_gate(
            args.sealed_manifest,
            output_dir=args.output_dir,
        )
    except (FileExistsError, FileNotFoundError, OSError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(output)
    return 0


def evaluate_hollow_notehead_unseen_gate(
    sealed_manifest: Path,
    *,
    output_dir: Path | None = None,
) -> Path:
    """Evaluate one immutable freeze without mutating its predictions or reviews."""
    manifest_path = sealed_manifest.resolve()
    freezer.verify_sealed_manifest(manifest_path)
    app = reviewer.load_review_app(manifest_path)
    public_state = app.public_state()
    pending = [
        row["identity"]["system_measure_index"]
        for row in public_state["measures"]
        if row["status"] != "completed"
    ]
    if pending:
        raise ValueError(f"Human review is incomplete for measures: {pending}")

    destination = (
        output_dir.resolve()
        if output_dir is not None
        else (manifest_path.parent / DEFAULT_VERSION).resolve()
    )
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Refusing to overwrite heldout evaluation: {destination}")
    temp_dir = destination.with_name(f".{destination.name}.tmp")
    if temp_dir.exists() or temp_dir.is_symlink():
        raise FileExistsError(f"Refusing stale evaluation temp directory: {temp_dir}")

    evaluated_rows = []
    for measure, state_row in zip(app.measures, public_state["measures"], strict=True):
        review_state = state_row["existing_review"]
        truth_centers = [_point(center) for center in review_state["centers"]]
        candidates = _load_object(measure.candidate_path, "Candidate artifact")
        proposals = _load_object(measure.proposal_path, "Proposal artifact")
        evaluated_rows.append(
            _evaluate_measure(
                identity=measure.identity,
                truth_centers=truth_centers,
                candidate_payload=candidates,
                proposal_payload=proposals,
                source_records={
                    "raw_image": measure.raw_record,
                    "candidate_artifact": measure.candidate_record,
                    "proposal_artifact": measure.proposal_record,
                    "human_review": _external_record(measure.review_path),
                },
            )
        )

    summary = _summarize(evaluated_rows)
    decision = _gate_decision(summary)
    report = {
        "schema_version": SCHEMA_VERSION,
        "kind": REPORT_KIND,
        "split": "fresh_heldout_morphology",
        "status": "evaluated",
        "source": {
            "sealed_manifest": _external_record(manifest_path),
            "sealed_manifest_kind": freezer.SEALED_KIND,
            "review_kind": reviewer.REVIEW_KIND,
        },
        "evaluation": {
            "match_tolerance_staff_spacing": hollow.MATCH_TOLERANCE_SPACING,
            "baseline_definition": "all centers in the frozen staff-grid candidate artifact",
            "recovery_definition": (
                "a frozen hollow proposal matched one human center not already matched "
                "by the baseline candidate artifact"
            ),
        },
        "promotion_gate": {
            "thresholds": {
                "minimum_truth_noteheads": MIN_TRUTH_NOTEHEADS,
                "minimum_recovery_opportunities": MIN_RECOVERY_OPPORTUNITIES,
                "minimum_proposal_precision": MIN_PROPOSAL_PRECISION,
                "minimum_recovery_rate": MIN_RECOVERY_RATE,
                "maximum_false_proposals": 0,
            },
            **decision,
        },
        "summary": summary,
        "measures": evaluated_rows,
        "provenance": {
            "create_once": True,
            "predictions_frozen_before_truth": True,
            "truth_accessed_only_after_freeze_verification": True,
            "eligible_for_end_to_end_transcription_claim": False,
            "evaluator": _external_record(Path(__file__).resolve()),
        },
    }

    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir()
    try:
        overlay_root = temp_dir / "overlays"
        for measure, row in zip(app.measures, evaluated_rows, strict=True):
            overlay_path = (
                overlay_root
                / f"measure_{measure.identity['system_measure_index']:03d}_evaluation.png"
            )
            _render_overlay(
                raw_path=measure.raw_path,
                candidate_payload=_load_object(measure.candidate_path, "Candidate artifact"),
                row=row,
                destination=overlay_path,
            )
            row["artifacts"] = {"evaluation_overlay": _local_record(overlay_path, temp_dir)}
        report_path = temp_dir / "report.json"
        _write_json(report_path, report)
        _write_json(
            temp_dir / "manifest.json",
            {
                "schema_version": SCHEMA_VERSION,
                "kind": MANIFEST_KIND,
                "create_once": True,
                "source_sealed_manifest": _external_record(manifest_path),
                "report": _local_record(report_path, temp_dir),
                "decision": decision["decision"],
                "eligible_for_candidate_pipeline_integration": decision[
                    "eligible_for_candidate_pipeline_integration"
                ],
            },
        )
        temp_dir.rename(destination)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    return destination


def _evaluate_measure(
    *,
    identity: Mapping[str, Any],
    truth_centers: Sequence[tuple[float, float]],
    candidate_payload: Mapping[str, Any],
    proposal_payload: Mapping[str, Any],
    source_records: Mapping[str, Any],
) -> dict[str, Any]:
    spacing = _staff_spacing(candidate_payload)
    tolerance = spacing * hollow.MATCH_TOLERANCE_SPACING
    candidate_rows = _object_rows(candidate_payload.get("candidates"), "Candidate rows")
    proposal_rows = _object_rows(proposal_payload.get("proposals"), "Proposal rows")
    considered_pair_count = _nonnegative_int(
        proposal_payload.get("considered_pair_count"),
        f"Proposal considered_pair_count for {dict(identity)}",
    )
    if candidate_payload.get("candidate_count") != len(candidate_rows):
        raise ValueError(f"Candidate count mismatch for {dict(identity)}")
    if proposal_payload.get("proposal_count") != len(proposal_rows):
        raise ValueError(f"Proposal count mismatch for {dict(identity)}")

    candidate_centers = [_point(row["center"]) for row in candidate_rows]
    proposal_centers = [_point(row["center"]) for row in proposal_rows]
    baseline_matches = _match_points(candidate_centers, truth_centers, tolerance=tolerance)
    baseline_truth_indices = {truth_index for _, truth_index in baseline_matches}
    opportunity_indices = [
        index for index in range(len(truth_centers)) if index not in baseline_truth_indices
    ]
    opportunity_centers = [truth_centers[index] for index in opportunity_indices]
    proposal_matches = _match_points(
        proposal_centers,
        opportunity_centers,
        tolerance=tolerance,
    )
    recovered_by_proposal = {
        proposal_index: opportunity_indices[opportunity_index]
        for proposal_index, opportunity_index in proposal_matches
    }

    assessed_proposals = []
    for proposal_index, (proposal, center) in enumerate(
        zip(proposal_rows, proposal_centers, strict=True)
    ):
        nearest_index, nearest_distance = _nearest(center, truth_centers)
        if proposal_index in recovered_by_proposal:
            outcome = "recovered_truth"
            matched_truth_index = recovered_by_proposal[proposal_index]
        elif nearest_index is not None and nearest_distance <= tolerance:
            outcome = "duplicate_truth"
            matched_truth_index = nearest_index
        else:
            outcome = "false_proposal"
            matched_truth_index = None
        assessed_proposals.append(
            {
                **proposal,
                "outcome": outcome,
                "matched_truth_index": matched_truth_index,
                "nearest_truth_index": nearest_index,
                "nearest_truth_distance_px": (
                    round(nearest_distance, 6) if nearest_index is not None else None
                ),
            }
        )

    baseline_by_truth = {
        truth_index: candidate_index for candidate_index, truth_index in baseline_matches
    }
    truth_assessment = []
    for truth_index, center in enumerate(truth_centers):
        proposal_index = next(
            (
                index
                for index, matched_truth_index in recovered_by_proposal.items()
                if matched_truth_index == truth_index
            ),
            None,
        )
        truth_assessment.append(
            {
                "truth_index": truth_index,
                "center": _point_json(center),
                "baseline_candidate_id": (
                    str(candidate_rows[baseline_by_truth[truth_index]].get("id"))
                    if truth_index in baseline_by_truth
                    else None
                ),
                "recovered_by_proposal_index": proposal_index,
                "outcome": (
                    "baseline_covered"
                    if truth_index in baseline_truth_indices
                    else "proposal_recovered" if proposal_index is not None else "unrecovered"
                ),
            }
        )
    outcome_counts = Counter(item["outcome"] for item in assessed_proposals)
    return {
        "identity": dict(identity),
        "sources": dict(source_records),
        "truth": {
            "count": len(truth_centers),
            "centers": truth_assessment,
        },
        "baseline": {
            "candidate_count": len(candidate_rows),
            "matched_truth_count": len(baseline_truth_indices),
        },
        "frozen_hollow_rule": {
            "considered_pair_count": considered_pair_count,
            "proposal_count": len(proposal_rows),
            "proposals": assessed_proposals,
            "recovered_truth_count": len(recovered_by_proposal),
            "outcomes": {
                "recovered_truth": outcome_counts["recovered_truth"],
                "duplicate_truth": outcome_counts["duplicate_truth"],
                "false_proposal": outcome_counts["false_proposal"],
            },
        },
        "metrics": {
            "staff_spacing_px": round(spacing, 6),
            "match_tolerance_px": round(tolerance, 6),
            "recovery_opportunity_count": len(opportunity_indices),
            "unrecovered_truth_count": len(opportunity_indices) - len(recovered_by_proposal),
            "augmented_matched_truth_count": len(baseline_truth_indices)
            + len(recovered_by_proposal),
        },
    }


def _summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    truth_count = sum(int(row["truth"]["count"]) for row in rows)
    baseline_matched = sum(int(row["baseline"]["matched_truth_count"]) for row in rows)
    proposal_count = sum(int(row["frozen_hollow_rule"]["proposal_count"]) for row in rows)
    considered_pair_count = sum(
        int(row["frozen_hollow_rule"]["considered_pair_count"]) for row in rows
    )
    recovered = sum(int(row["frozen_hollow_rule"]["recovered_truth_count"]) for row in rows)
    duplicate = sum(int(row["frozen_hollow_rule"]["outcomes"]["duplicate_truth"]) for row in rows)
    false = sum(int(row["frozen_hollow_rule"]["outcomes"]["false_proposal"]) for row in rows)
    opportunities = truth_count - baseline_matched
    augmented = baseline_matched + recovered
    return {
        "measure_count": len(rows),
        "truth_notehead_count": truth_count,
        "baseline_matched_truth_count": baseline_matched,
        "baseline_recall": _ratio(baseline_matched, truth_count),
        "recovery_opportunity_count": opportunities,
        "considered_pair_count": considered_pair_count,
        "frozen_proposal_count": proposal_count,
        "recovered_truth_count": recovered,
        "duplicate_proposal_count": duplicate,
        "false_proposal_count": false,
        "proposal_precision": _ratio(recovered, proposal_count),
        "recovery_rate": _ratio(recovered, opportunities),
        "augmented_matched_truth_count": augmented,
        "augmented_recall": _ratio(augmented, truth_count),
    }


def _gate_decision(summary: Mapping[str, Any]) -> dict[str, Any]:
    checks = {
        "sample_sufficient": summary["truth_notehead_count"] >= MIN_TRUTH_NOTEHEADS,
        "has_recovery_opportunity": (
            summary["recovery_opportunity_count"] >= MIN_RECOVERY_OPPORTUNITIES
        ),
        "proposal_precision_pass": (
            summary["proposal_precision"] is not None
            and summary["proposal_precision"] >= MIN_PROPOSAL_PRECISION
        ),
        "recovery_rate_pass": (
            summary["recovery_rate"] is not None and summary["recovery_rate"] >= MIN_RECOVERY_RATE
        ),
        "false_proposal_gate_pass": summary["false_proposal_count"] == 0,
    }
    reasons = []
    if not checks["sample_sufficient"]:
        reasons.append("review contains fewer than the minimum independent hollow noteheads")
    if not checks["has_recovery_opportunity"]:
        reasons.append("baseline candidates already cover all reviewed hollow noteheads")
    if summary["frozen_proposal_count"] == 0:
        reasons.append("frozen rule emitted no heldout proposals")
    elif not checks["proposal_precision_pass"]:
        reasons.append("frozen proposal precision is below threshold")
    if checks["has_recovery_opportunity"] and not checks["recovery_rate_pass"]:
        reasons.append("frozen proposal recovery rate is below threshold")
    if not checks["false_proposal_gate_pass"]:
        reasons.append("frozen rule emitted false proposals")
    passed = all(checks.values())
    return {
        "decision": "promote" if passed else "not_promoted",
        "eligible_for_candidate_pipeline_integration": passed,
        "checks": checks,
        "reasons": reasons,
    }


def _render_overlay(
    *,
    raw_path: Path,
    candidate_payload: Mapping[str, Any],
    row: Mapping[str, Any],
    destination: Path,
) -> None:
    with Image.open(raw_path) as opened:
        raw = opened.convert("RGB")
    scale = max(1, (520 + raw.width - 1) // raw.width)
    if scale > 1:
        raw = raw.resize((raw.width * scale, raw.height * scale), Image.Resampling.NEAREST)
    header = 72
    canvas = Image.new("RGB", (raw.width, raw.height + header), "white")
    canvas.paste(raw, (0, header))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    identity = row["identity"]
    draw.text(
        (6, 6),
        (
            f"{identity['slug']} s{identity['system_index']:03d} "
            f"m{identity['system_measure_index']:03d}"
        ),
        fill="black",
        font=font,
    )
    draw.text(
        (6, 26),
        "green=human truth  blue=baseline match  magenta=recovery  red=miss/false",
        fill="black",
        font=font,
    )
    draw.text(
        (6, 46),
        (
            f"truth={row['truth']['count']} "
            f"baseline={row['baseline']['matched_truth_count']} "
            f"proposals={row['frozen_hollow_rule']['proposal_count']}"
        ),
        fill="black",
        font=font,
    )
    candidate_by_id = {
        str(candidate["id"]): candidate
        for candidate in _object_rows(candidate_payload.get("candidates"), "Candidate rows")
    }
    for truth in row["truth"]["centers"]:
        center = _point(truth["center"])
        x = round(center[0] * scale)
        y = round(center[1] * scale) + header
        radius = max(7, 4 * scale)
        outcome = truth["outcome"]
        color = (0, 145, 70) if outcome != "unrecovered" else (215, 45, 45)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=3)
        candidate_id = truth["baseline_candidate_id"]
        if candidate_id is not None:
            candidate = candidate_by_id[candidate_id]
            bbox = candidate["bbox"]
            draw.rectangle(
                (
                    round(float(bbox["left"]) * scale),
                    round(float(bbox["top"]) * scale) + header,
                    round(float(bbox["right"]) * scale),
                    round(float(bbox["bottom"]) * scale) + header,
                ),
                outline=(60, 105, 220),
                width=2,
            )
            draw.text(
                (x + radius + 2, y - 8),
                candidate_id,
                fill=(60, 105, 220),
                font=font,
                stroke_width=1,
                stroke_fill="white",
            )
    for proposal in row["frozen_hollow_rule"]["proposals"]:
        center = _point(proposal["center"])
        x = round(center[0] * scale)
        y = round(center[1] * scale) + header
        radius = max(6, 3 * scale)
        color = (
            (185, 45, 185)
            if proposal["outcome"] == "recovered_truth"
            else (225, 125, 0) if proposal["outcome"] == "duplicate_truth" else (215, 45, 45)
        )
        draw.line((x - radius, y, x + radius, y), fill=color, width=3)
        draw.line((x, y - radius, x, y + radius), fill=color, width=3)
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination)


def _match_points(
    candidates: Sequence[tuple[float, float]],
    truth: Sequence[tuple[float, float]],
    *,
    tolerance: float,
) -> list[tuple[int, int]]:
    pairs = sorted(
        (
            (math.dist(candidate, target), candidate_index, truth_index)
            for candidate_index, candidate in enumerate(candidates)
            for truth_index, target in enumerate(truth)
            if math.dist(candidate, target) <= tolerance
        ),
        key=lambda row: (row[0], row[1], row[2]),
    )
    used_candidates: set[int] = set()
    used_truth: set[int] = set()
    matches = []
    for _, candidate_index, truth_index in pairs:
        if candidate_index in used_candidates or truth_index in used_truth:
            continue
        used_candidates.add(candidate_index)
        used_truth.add(truth_index)
        matches.append((candidate_index, truth_index))
    return matches


def _nearest(
    point: tuple[float, float],
    targets: Sequence[tuple[float, float]],
) -> tuple[int | None, float]:
    if not targets:
        return None, math.inf
    distances = [math.dist(point, target) for target in targets]
    index = min(range(len(distances)), key=distances.__getitem__)
    return index, distances[index]


def _staff_spacing(payload: Mapping[str, Any]) -> float:
    value = payload.get("staff_spacing_px")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Candidate artifact staff_spacing_px must be numeric")
    spacing = float(value)
    if not math.isfinite(spacing) or spacing <= 0:
        raise ValueError("Candidate artifact staff_spacing_px must be positive")
    return spacing


def _point(value: Any) -> tuple[float, float]:
    if not isinstance(value, Mapping):
        raise ValueError("Point must be an object")
    x = value.get("x")
    y = value.get("y")
    if (
        isinstance(x, bool)
        or isinstance(y, bool)
        or not isinstance(x, (int, float))
        or not isinstance(y, (int, float))
        or not math.isfinite(float(x))
        or not math.isfinite(float(y))
    ):
        raise ValueError("Point x and y must be finite numbers")
    return float(x), float(y)


def _point_json(point: tuple[float, float]) -> dict[str, float]:
    return {"x": round(point[0], 6), "y": round(point[1], 6)}


def _object_rows(value: Any, label: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or any(not isinstance(row, Mapping) for row in value):
        raise ValueError(f"{label} must be an array of objects")
    return [dict(row) for row in value]


def _ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def _nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _load_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload


def _external_record(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Source artifact is missing: {resolved}")
    return {"path": str(resolved), "sha256": _sha256(resolved)}


def _local_record(path: Path, root: Path) -> dict[str, str]:
    return {
        "path": path.resolve().relative_to(root.resolve()).as_posix(),
        "sha256": _sha256(path),
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as output:
        json.dump(payload, output, indent=2, ensure_ascii=False)
        output.write("\n")


if __name__ == "__main__":
    raise SystemExit(main())
