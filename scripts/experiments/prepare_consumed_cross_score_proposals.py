"""Prepare consumed cross-score notehead proposals for later human review.

This spike utility is deliberately downstream of a completed heldout evaluation.
It consumes a hash-pinned mapping manifest, approved partial-system MusicXML,
frozen request rows, measure images, and blind candidate artifacts. Its output is
an unreviewed proposal queue, never human ground truth, a human review, or a
promotable training fixture.

The mapping manifest must declare ``split_status: consumed_training`` and map
every physical MusicXML measure to one automatic crop. One crop may contain
multiple consecutive physical measures.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_vlm_notehead_localization_spike import (  # noqa: E402
    treble_pitch_for_y,
)

MAPPING_KIND = "vlm_melody_consumed_training_mapping"
CORRECTED_MAPPING_KIND = "vlm_melody_consumed_training_segmentation_mapping"
CORRECTED_NAMESPACE_KIND = "vlm_melody_consumed_cross_score_training_inputs"
CORRECTED_CANDIDATE_RECORD_KIND = "vlm_melody_cross_score_candidate_record"
OUTPUT_KIND = "vlm_melody_consumed_cross_score_proposals"
CORRECTED_OUTPUT_KIND = "vlm_melody_corrected_consumed_cross_score_proposals"
SPLIT_STATUS = "consumed_training"
HELDOUT_STATUSES = {"heldout", "fresh_heldout"}
PROTECTED_OUTPUT_NAMESPACES = {"vlm_melody_inputs", "notehead_reviews"}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
NAMESPACE_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
PITCH_CLASS = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
ALTER_SUFFIX = {-2: "bb", -1: "b", 0: "", 1: "#", 2: "##"}


@dataclass(frozen=True)
class NotatedPitch:
    physical_measure_number: int
    order_in_measure: int
    staff_pitch: str
    sounding_pitch: str
    pitch_midi: int
    alter: int
    tie_types: tuple[str, ...]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", type=Path, help="Pipeline output directory.")
    parser.add_argument("--mapping", type=Path, required=True, help="Hash-pinned mapping JSON.")
    parser.add_argument(
        "--consumption-mapping",
        type=Path,
        help=(
            "Approved consumed-truth lineage. Required when --mapping points to a corrected "
            "versioned training-input namespace."
        ),
    )
    parser.add_argument("--output", type=Path, help="Optional review-queue output path.")
    args = parser.parse_args(argv)
    try:
        mapping = _load_json_object(args.mapping.resolve(), "Mapping manifest")
        if mapping.get("kind") == CORRECTED_MAPPING_KIND:
            if args.output is not None:
                raise ValueError(
                    "Corrected namespace proposals use their fixed create-once proposals/ path"
                )
            if args.consumption_mapping is None:
                raise ValueError(
                    "--consumption-mapping is required for corrected namespace proposals"
                )
            output_path = prepare_corrected_consumed_cross_score_proposals(
                args.out_dir,
                mapping_path=args.mapping,
                consumption_mapping_path=args.consumption_mapping,
            )
        else:
            output_path = prepare_consumed_cross_score_proposals(
                args.out_dir,
                mapping_path=args.mapping,
                output_path=args.output,
            )
    except (
        FileExistsError,
        FileNotFoundError,
        OSError,
        TypeError,
        ValueError,
        ET.ParseError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(output_path)
    return 0


def prepare_consumed_cross_score_proposals(
    out_dir: Path,
    *,
    mapping_path: Path,
    output_path: Path | None = None,
    repo_root: Path = REPO_ROOT,
) -> Path:
    """Validate consumed evidence and write explicitly unreviewed proposals."""
    root = repo_root.resolve()
    output_root = out_dir.resolve()
    mapping_file = mapping_path.resolve()
    mapping = _load_json_object(mapping_file, "Mapping manifest")
    _validate_mapping_header(mapping)

    identity = _required_object(mapping, "identity", "Mapping manifest")
    slug = _required_string(identity, "slug", "Mapping identity")
    system_index = _required_positive_int(identity, "system_index", "Mapping identity")
    segmentation_namespace = _required_string(mapping, "segmentation_namespace", "Mapping manifest")
    if not NAMESPACE_RE.fullmatch(segmentation_namespace):
        raise ValueError(
            "Mapping manifest.segmentation_namespace must contain lowercase letters, "
            "digits, underscores, or hyphens"
        )
    destination = (
        output_path.resolve()
        if output_path is not None
        else output_root
        / slug
        / "vlm_melody_consumed_cross_score_proposals"
        / segmentation_namespace
        / "proposals.json"
    )
    _validate_output_path(destination)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing review queue: {destination}")

    source = _required_object(mapping, "source", "Mapping manifest")
    musicxml_path, musicxml_sha256 = _validated_source_file(
        _required_object(source, "musicxml", "Mapping source"),
        label="MusicXML",
        mapping_dir=mapping_file.parent,
        repo_root=root,
    )
    requests_path, requests_sha256 = _validated_source_file(
        _required_object(source, "requests", "Mapping source"),
        label="Requests",
        mapping_dir=mapping_file.parent,
        repo_root=root,
    )
    consumption = _validate_consumption_evidence(
        mapping,
        mapping_dir=mapping_file.parent,
        repo_root=root,
    )

    physical_pitches = load_notated_pitches(musicxml_path)
    physical_measure_numbers = tuple(physical_pitches)
    requests = _read_jsonl(requests_path)
    requests_by_crop = _requests_for_identity(
        requests,
        slug=slug,
        system_index=system_index,
        source_split_status=consumption["source_split_status"],
    )
    crop_specs = _validate_crop_mapping(
        _required_list(mapping, "crops", "Mapping manifest"),
        physical_measure_numbers=physical_measure_numbers,
        request_crop_indices=set(requests_by_crop),
    )

    tasks = []
    for crop_spec in crop_specs:
        crop_index = crop_spec["system_measure_index"]
        request = requests_by_crop[crop_index]
        raw_image = _required_object(
            _required_object(request, "images", f"Request crop {crop_index}"),
            "raw",
            f"Request crop {crop_index} images",
        )
        image_path = _resolve_out_relative_path(
            _required_string(raw_image, "path_relative_to_out", "Raw request image"),
            out_dir=output_root,
        )
        image_sha256 = _required_sha256(raw_image, "sha256", "Raw request image")
        _validate_file_hash(image_path, image_sha256, "Raw request image")

        candidate_path, candidate_sha256 = _validated_source_file(
            _required_object(crop_spec, "candidate_artifact", f"Crop {crop_index}"),
            label=f"Crop {crop_index} candidate artifact",
            mapping_dir=mapping_file.parent,
            repo_root=root,
        )
        candidate_artifact = _load_json_object(
            candidate_path, f"Crop {crop_index} candidate artifact"
        )
        _validate_candidate_artifact(
            candidate_artifact,
            slug=slug,
            system_index=system_index,
            crop_index=crop_index,
            image_path=image_path,
            out_dir=output_root,
            repo_root=root,
        )

        expected = [
            pitch
            for number in crop_spec["physical_measure_numbers"]
            for pitch in physical_pitches[number]
        ]
        proposal = build_exact_pitch_alignment_proposal(candidate_artifact, expected)
        identity_payload = _required_object(request, "identity", f"Request crop {crop_index}")
        tasks.append(
            {
                "identity": dict(identity_payload),
                "physical_measure_numbers": list(crop_spec["physical_measure_numbers"]),
                "expected": {
                    "note_count": len(expected),
                    "ordered_sounding_pitches": [pitch.sounding_pitch for pitch in expected],
                    "ordered_staff_pitches": [pitch.staff_pitch for pitch in expected],
                    "notes": [_pitch_payload(pitch) for pitch in expected],
                },
                "source": {
                    "request_row_sha256": _json_sha256(request),
                    "image_path": _display_path(image_path, root),
                    "image_sha256": image_sha256,
                    "candidate_artifact_path": _display_path(candidate_path, root),
                    "candidate_artifact_sha256": candidate_sha256,
                    "candidate_strategy": candidate_artifact.get("strategy"),
                    "candidate_strategy_version": candidate_artifact.get("strategy_version"),
                },
                "proposal": proposal,
                "review_state": {
                    "status": "human_review_required",
                    "human_reviewed": False,
                    "eligible_for_promotion": False,
                    "instruction": (
                        "Confirm/reject candidate assignments and add missing noteheads in the "
                        "human reviewer before creating any training fixture."
                    ),
                },
            }
        )

    script_path = Path(__file__).resolve()
    payload = {
        "schema_version": 1,
        "kind": OUTPUT_KIND,
        "split_status": SPLIT_STATUS,
        "identity": {"slug": slug, "system_index": system_index},
        "segmentation_namespace": segmentation_namespace,
        "proposal_status": "unreviewed_consumed_cross_score_proposals_only",
        "eligible_for_training": False,
        "eligible_for_promotion": False,
        "source": {
            "mapping_path": _display_path(mapping_file, root),
            "mapping_sha256": _sha256(mapping_file),
            "musicxml_path": _display_path(musicxml_path, root),
            "musicxml_sha256": musicxml_sha256,
            "requests_path": _display_path(requests_path, root),
            "requests_sha256": requests_sha256,
            "source_split_status": consumption["source_split_status"],
            "consumption_reason": consumption["reason"],
            "evaluation_evidence": consumption.get("evaluation_evidence"),
        },
        "tasks": tasks,
        "provenance": {
            "scope": "spike_only",
            "builder_path": _display_path(script_path, root),
            "builder_sha256": _sha256(script_path),
            "supervision_usage": (
                "Approved consumed MusicXML supplies ordered pitch/count constraints after the "
                "source split was explicitly consumed; proposals are neither human ground truth "
                "nor heldout evidence."
            ),
            "proposal_method": "monotonic_exact_staff_pitch_alignment_v1",
            "proposal_confidence_is_calibrated": False,
            "human_review_required_before_training": True,
        },
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    _write_json_exclusive(destination, payload)
    return destination


def prepare_corrected_consumed_cross_score_proposals(
    out_dir: Path,
    *,
    mapping_path: Path,
    consumption_mapping_path: Path,
    repo_root: Path = REPO_ROOT,
) -> Path:
    """Materialize create-once proposals for a corrected consumed namespace."""
    root = repo_root.resolve()
    output_root = out_dir.resolve()
    mapping_file = mapping_path.resolve()
    namespace_dir = mapping_file.parent
    destination = namespace_dir / "proposals"
    _validate_output_path(destination)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing proposal namespace: {destination}")

    mapping = _load_json_object(mapping_file, "Corrected mapping manifest")
    identity = _validate_corrected_mapping(mapping)
    slug = identity["slug"]
    system_index = identity["system_index"]
    namespace = _required_string(mapping, "segmentation_namespace", "Corrected mapping")
    namespace_manifest_path = namespace_dir / "manifest.json"
    namespace_manifest = _load_json_object(namespace_manifest_path, "Namespace manifest")
    _validate_corrected_namespace_manifest(
        namespace_manifest,
        slug=slug,
        system_index=system_index,
        namespace=namespace,
        mapping_path=mapping_file,
        repo_root=root,
    )

    source = _required_object(mapping, "source", "Corrected mapping")
    candidates_manifest_path, candidates_manifest_sha256 = _validated_source_file(
        _required_object(source, "candidates_manifest", "Corrected mapping source"),
        label="Candidates manifest",
        mapping_dir=namespace_dir,
        repo_root=root,
    )
    candidate_rows = _read_jsonl(candidates_manifest_path)
    candidates_by_crop = _validate_corrected_candidate_records(
        candidate_rows,
        slug=slug,
        system_index=system_index,
        out_dir=output_root,
        repo_root=root,
    )

    consumption_file = consumption_mapping_path.resolve()
    consumption_mapping = _load_json_object(consumption_file, "Consumption mapping")
    _validate_mapping_header(consumption_mapping)
    consumption_identity = _required_object(consumption_mapping, "identity", "Consumption mapping")
    if (
        consumption_identity.get("slug") != slug
        or consumption_identity.get("system_index") != system_index
    ):
        raise ValueError("Consumption mapping identity does not match corrected namespace")
    approved_source = _required_object(consumption_mapping, "source", "Consumption mapping")
    musicxml_path, musicxml_sha256 = _validated_source_file(
        _required_object(approved_source, "musicxml", "Consumption mapping source"),
        label="Approved consumed MusicXML",
        mapping_dir=consumption_file.parent,
        repo_root=root,
    )
    consumption = _validate_consumption_evidence(
        consumption_mapping,
        mapping_dir=consumption_file.parent,
        repo_root=root,
    )
    physical_pitches = load_notated_pitches(musicxml_path)
    crop_specs = _validate_corrected_crop_mapping(
        _required_list(mapping, "crops", "Corrected mapping"),
        physical_measure_numbers=tuple(physical_pitches),
        candidate_crop_indices=set(candidates_by_crop),
    )

    tasks = []
    input_candidate_hashes = []
    input_image_hashes = []
    for crop_spec in crop_specs:
        crop_index = crop_spec["system_measure_index"]
        candidate_record = candidates_by_crop[crop_index]
        candidate_artifact_record = _required_object(
            _required_object(candidate_record, "artifacts", f"Candidate record {crop_index}"),
            "candidates",
            f"Candidate record {crop_index} artifacts",
        )
        candidate_path, candidate_sha256 = _validated_source_file(
            candidate_artifact_record,
            label=f"Crop {crop_index} candidate artifact",
            mapping_dir=candidates_manifest_path.parent,
            repo_root=root,
        )
        mapped_candidate = _required_object(
            crop_spec, "candidate_artifact", f"Corrected crop {crop_index}"
        )
        mapped_candidate_path, mapped_candidate_sha256 = _validated_source_file(
            mapped_candidate,
            label=f"Corrected crop {crop_index} candidate artifact",
            mapping_dir=namespace_dir,
            repo_root=root,
        )
        if mapped_candidate_path != candidate_path or mapped_candidate_sha256 != candidate_sha256:
            raise ValueError(f"Crop {crop_index} candidate artifact differs between manifests")
        candidate_artifact = _load_json_object(
            candidate_path, f"Crop {crop_index} candidate artifact"
        )

        raw_record = _required_object(
            _required_object(candidate_record, "source", f"Candidate record {crop_index}"),
            "measure_raw",
            f"Candidate record {crop_index} source",
        )
        image_path, image_sha256 = _validated_source_file(
            raw_record,
            label=f"Crop {crop_index} raw image",
            mapping_dir=candidates_manifest_path.parent,
            repo_root=root,
        )
        _validate_candidate_artifact(
            candidate_artifact,
            slug=slug,
            system_index=system_index,
            crop_index=crop_index,
            image_path=image_path,
            out_dir=output_root,
            repo_root=root,
        )

        physical_measure_number = crop_spec["physical_measure_numbers"][0]
        expected = list(physical_pitches[physical_measure_number])
        proposal = build_partial_exact_pitch_alignment_proposal(candidate_artifact, expected)
        tasks.append(
            {
                "identity": {
                    "slug": slug,
                    "system_index": system_index,
                    "system_measure_index": crop_index,
                    "physical_measure_number": physical_measure_number,
                },
                "review_status": "unreviewed",
                "eligible_for_training": False,
                "eligible_for_promotion": False,
                "expected": {
                    "note_count": len(expected),
                    "ordered_sounding_pitches": [pitch.sounding_pitch for pitch in expected],
                    "ordered_staff_pitches": [pitch.staff_pitch for pitch in expected],
                    "notes": [_pitch_payload(pitch) for pitch in expected],
                },
                "source": {
                    "raw_image_path": _display_path(image_path, root),
                    "raw_image_sha256": image_sha256,
                    "candidate_artifact_path": _display_path(candidate_path, root),
                    "candidate_artifact_sha256": candidate_sha256,
                },
                "proposal": proposal,
            }
        )
        input_candidate_hashes.append(
            {
                "system_measure_index": crop_index,
                "path": _display_path(candidate_path, root),
                "sha256": candidate_sha256,
            }
        )
        input_image_hashes.append(
            {
                "system_measure_index": crop_index,
                "path": _display_path(image_path, root),
                "sha256": image_sha256,
            }
        )

    payload = {
        "schema_version": 1,
        "kind": CORRECTED_OUTPUT_KIND,
        "split_status": SPLIT_STATUS,
        "review_status": "unreviewed",
        "eligible_for_training": False,
        "eligible_for_promotion": False,
        "identity": {"slug": slug, "system_index": system_index},
        "segmentation_namespace": namespace,
        "proposal_method": "partial_monotonic_exact_staff_pitch_alignment_v1",
        "tasks": tasks,
        "source": {
            "namespace_manifest": _file_hash_payload(namespace_manifest_path, root),
            "mapping": _file_hash_payload(mapping_file, root),
            "consumption_mapping": _file_hash_payload(consumption_file, root),
            "musicxml": {
                "path": _display_path(musicxml_path, root),
                "sha256": musicxml_sha256,
            },
            "candidates_manifest": {
                "path": _display_path(candidates_manifest_path, root),
                "sha256": candidates_manifest_sha256,
            },
            "candidate_artifacts": input_candidate_hashes,
            "raw_images": input_image_hashes,
            "consumption": consumption,
        },
        "provenance": {
            "scope": "spike_only",
            "builder": _file_hash_payload(Path(__file__).resolve(), root),
            "approved_truth_usage": "consumed Carrizal MusicXML only",
            "heldout_discovery_or_globbing": False,
            "visual_adjudication_performed": False,
            "human_review_required_before_training": True,
        },
    }
    coverage = _coverage_rows(tasks)
    coverage_markdown = _coverage_markdown(coverage)

    destination.mkdir(parents=False)
    try:
        proposals_path = destination / "proposals.json"
        coverage_path = destination / "coverage.md"
        _write_json_exclusive(proposals_path, payload)
        coverage_path.write_text(coverage_markdown, encoding="utf-8")
        output_manifest = {
            "schema_version": 1,
            "kind": "vlm_melody_corrected_consumed_cross_score_proposal_manifest",
            "split_status": SPLIT_STATUS,
            "review_status": "unreviewed",
            "eligible_for_training": False,
            "eligible_for_promotion": False,
            "identity": {"slug": slug, "system_index": system_index},
            "segmentation_namespace": namespace,
            "inputs": payload["source"],
            "outputs": {
                "proposals": _file_hash_payload(proposals_path, root),
                "coverage": _file_hash_payload(coverage_path, root),
            },
        }
        _write_json_exclusive(destination / "manifest.json", output_manifest)
    except Exception:
        shutil.rmtree(destination)
        raise
    return proposals_path


def build_partial_exact_pitch_alignment_proposal(
    candidate_artifact: Mapping[str, Any],
    expected: Sequence[NotatedPitch],
) -> dict[str, Any]:
    """Return the best partial exact-pitch alignment plus an explicit review queue."""
    candidates = candidate_artifact.get("candidates")
    staff_lines = candidate_artifact.get("staff_lines_y_px")
    if (
        not isinstance(candidates, list)
        or not isinstance(staff_lines, list)
        or len(staff_lines) != 5
    ):
        raise ValueError("Candidate artifact requires candidates and five staff lines")
    enriched = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise ValueError("Candidate entries must be objects")
        candidate_id = _required_string(candidate, "id", "Candidate")
        center = _required_object(candidate, "center", f"Candidate {candidate_id}")
        enriched.append(
            {
                "candidate": candidate,
                "id": candidate_id,
                "x": float(center["x"]),
                "staff_pitch": treble_pitch_for_y(
                    float(center["y"]), tuple(float(y) for y in staff_lines)
                ),
                "score": float(candidate.get("score", 0.0)),
            }
        )
    enriched.sort(key=lambda item: (item["x"], item["id"]))

    @lru_cache(maxsize=None)
    def solve(
        expected_index: int, candidate_index: int
    ) -> tuple[int, float, tuple[tuple[int, int], ...]]:
        if expected_index == len(expected) or candidate_index == len(enriched):
            return 0, 0.0, ()
        options = [solve(expected_index + 1, candidate_index)]
        options.append(solve(expected_index, candidate_index + 1))
        if expected[expected_index].staff_pitch == enriched[candidate_index]["staff_pitch"]:
            count, score, pairs = solve(expected_index + 1, candidate_index + 1)
            options.append(
                (
                    count + 1,
                    score + enriched[candidate_index]["score"],
                    ((expected_index, candidate_index),) + pairs,
                )
            )
        return _best_partial_alignment(options)

    matched_count, score_sum, selected_pairs = solve(0, 0)
    selected_expected = {expected_index for expected_index, _ in selected_pairs}
    selected_candidates = {candidate_index for _, candidate_index in selected_pairs}
    assignments = []
    for expected_index, candidate_index in selected_pairs:
        pitch = expected[expected_index]
        item = enriched[candidate_index]
        assignments.append(
            {
                "expected_order": expected_index + 1,
                "candidate_id": item["id"],
                "candidate_center": item["candidate"].get("center"),
                "candidate_score": round(item["score"], 6),
                "candidate_staff_pitch": item["staff_pitch"],
                "expected_staff_pitch": pitch.staff_pitch,
                "expected_sounding_pitch": pitch.sounding_pitch,
                "physical_measure_number": pitch.physical_measure_number,
                "tie_types": list(pitch.tie_types),
            }
        )
    unresolved_notes = [
        {"expected_order": index + 1, **_pitch_payload(pitch)}
        for index, pitch in enumerate(expected)
        if index not in selected_expected
    ]
    unresolved_candidates = [
        {
            "candidate_id": item["id"],
            "candidate_center": item["candidate"].get("center"),
            "candidate_score": round(item["score"], 6),
            "candidate_staff_pitch": item["staff_pitch"],
        }
        for index, item in enumerate(enriched)
        if index not in selected_candidates
    ]
    return {
        "status": "unreviewed_partial_proposal",
        "method": "partial_monotonic_exact_staff_pitch_alignment_v1",
        "matched_count": matched_count,
        "assignments": assignments,
        "mean_matched_candidate_score": (
            round(score_sum / matched_count, 6) if matched_count else None
        ),
        "unresolved": {
            "expected_note_count": len(unresolved_notes),
            "expected_notes": unresolved_notes,
            "candidate_count": len(unresolved_candidates),
            "candidates": unresolved_candidates,
        },
        "human_reviewed": False,
    }


def _best_partial_alignment(
    options: Sequence[tuple[int, float, tuple[tuple[int, int], ...]]],
) -> tuple[int, float, tuple[tuple[int, int], ...]]:
    best = options[0]
    for option in options[1:]:
        if option[0] > best[0] or (option[0] == best[0] and option[1] > best[1]):
            best = option
        elif option[:2] == best[:2] and option[2] < best[2]:
            best = option
    return best


def _validate_corrected_mapping(mapping: Mapping[str, Any]) -> dict[str, Any]:
    if mapping.get("kind") != CORRECTED_MAPPING_KIND:
        raise ValueError(f"Corrected mapping kind must be {CORRECTED_MAPPING_KIND!r}")
    if mapping.get("split_status") != SPLIT_STATUS:
        raise ValueError(f"Corrected mapping split_status must be {SPLIT_STATUS!r}")
    if mapping.get("review_status") != "unreviewed":
        raise ValueError("Corrected mapping must remain unreviewed")
    if mapping.get("eligible_for_training") is not False:
        raise ValueError("Corrected mapping must not be training eligible")
    identity = _required_object(mapping, "identity", "Corrected mapping")
    return {
        "slug": _required_string(identity, "slug", "Corrected mapping identity"),
        "system_index": _required_positive_int(
            identity, "system_index", "Corrected mapping identity"
        ),
    }


def _validate_corrected_namespace_manifest(
    manifest: Mapping[str, Any],
    *,
    slug: str,
    system_index: int,
    namespace: str,
    mapping_path: Path,
    repo_root: Path,
) -> None:
    if manifest.get("kind") != CORRECTED_NAMESPACE_KIND:
        raise ValueError(f"Namespace manifest kind must be {CORRECTED_NAMESPACE_KIND!r}")
    if manifest.get("split_status") != SPLIT_STATUS:
        raise ValueError(f"Namespace manifest split_status must be {SPLIT_STATUS!r}")
    if manifest.get("review_status") != "unreviewed":
        raise ValueError("Namespace manifest must remain unreviewed")
    if manifest.get("eligible_for_training") is not False:
        raise ValueError("Namespace manifest must not be training eligible")
    identity = _required_object(manifest, "identity", "Namespace manifest")
    if identity.get("slug") != slug or identity.get("system_index") != system_index:
        raise ValueError("Namespace manifest identity does not match corrected mapping")
    if manifest.get("segmentation_namespace") != namespace:
        raise ValueError("Namespace manifest segmentation namespace changed")
    recorded_mapping = _required_object(
        _required_object(manifest, "artifacts", "Namespace manifest"),
        "mapping",
        "Namespace manifest artifacts",
    )
    recorded_path, recorded_sha256 = _validated_source_file(
        recorded_mapping,
        label="Namespace mapping",
        mapping_dir=mapping_path.parent,
        repo_root=repo_root,
    )
    if recorded_path != mapping_path or recorded_sha256 != _sha256(mapping_path):
        raise ValueError("Namespace manifest does not pin the supplied corrected mapping")


def _validate_corrected_candidate_records(
    rows: Sequence[Mapping[str, Any]],
    *,
    slug: str,
    system_index: int,
    out_dir: Path,
    repo_root: Path,
) -> dict[int, Mapping[str, Any]]:
    result = {}
    for row in rows:
        if row.get("kind") != CORRECTED_CANDIDATE_RECORD_KIND:
            raise ValueError("Candidates manifest contains an unexpected record kind")
        if row.get("split_status") != SPLIT_STATUS or row.get("review_status") != "unreviewed":
            raise ValueError("Corrected candidate records must remain unreviewed consumed training")
        if row.get("eligible_for_training") is not False:
            raise ValueError("Corrected candidate records must not be training eligible")
        identity = _required_object(row, "identity", "Candidate record")
        if identity.get("slug") != slug or identity.get("system_index") != system_index:
            raise ValueError("Candidate record identity does not match corrected namespace")
        crop_index = _required_positive_int(identity, "system_measure_index", "Candidate identity")
        if crop_index in result:
            raise ValueError(f"Duplicate corrected candidate record: {crop_index}")
        generation = _required_object(row, "candidate_generation", "Candidate record")
        if generation.get("ground_truth_files_read") != []:
            raise ValueError("Corrected candidates must be generated without ground-truth access")
        for label, record in (
            (
                "Candidate artifact",
                _required_object(
                    _required_object(row, "artifacts", "Candidate record"),
                    "candidates",
                    "Candidate record artifacts",
                ),
            ),
            (
                "Raw measure image",
                _required_object(
                    _required_object(row, "source", "Candidate record"),
                    "measure_raw",
                    "Candidate record source",
                ),
            ),
        ):
            path, sha256 = _validated_source_file(
                record,
                label=f"Crop {crop_index} {label}",
                mapping_dir=repo_root,
                repo_root=repo_root,
            )
            _validate_file_hash(path, sha256, f"Crop {crop_index} {label}")
            try:
                path.relative_to(out_dir)
            except ValueError as exc:
                raise ValueError(
                    f"Crop {crop_index} {label} must stay under the supplied output directory"
                ) from exc
        result[crop_index] = row
    if not result:
        raise ValueError("Candidates manifest contains no corrected candidate records")
    return result


def _validate_corrected_crop_mapping(
    raw_crops: Sequence[Any],
    *,
    physical_measure_numbers: Sequence[int],
    candidate_crop_indices: set[int],
) -> list[dict[str, Any]]:
    expected_measures = list(physical_measure_numbers)
    if expected_measures != list(range(1, len(expected_measures) + 1)):
        raise ValueError(
            "Corrected one-to-one mapping requires contiguous physical measures from 1"
        )
    if len(raw_crops) != len(expected_measures):
        raise ValueError(
            "Corrected mapping crop count must match the physical MusicXML measure count"
        )
    result = []
    for index, raw in enumerate(raw_crops, start=1):
        if not isinstance(raw, dict):
            raise ValueError(f"Corrected crops[{index - 1}] must be an object")
        crop_index = _required_positive_int(raw, "system_measure_index", f"Crop {index}")
        numbers = raw.get("physical_measure_numbers")
        if numbers != [expected_measures[index - 1]] or crop_index != index:
            raise ValueError("Corrected mapping must be one-to-one for every physical measure")
        result.append(dict(raw))
    if candidate_crop_indices != set(range(1, len(expected_measures) + 1)):
        raise ValueError("Corrected candidates manifest must contain contiguous crops from 1")
    return result


def _coverage_rows(tasks: Sequence[Mapping[str, Any]]) -> list[dict[str, int]]:
    rows = []
    for task in tasks:
        identity = _required_object(task, "identity", "Proposal task")
        expected = _required_object(task, "expected", "Proposal task")
        proposal = _required_object(task, "proposal", "Proposal task")
        unresolved = _required_object(proposal, "unresolved", "Proposal")
        rows.append(
            {
                "physical_measure": int(identity["physical_measure_number"]),
                "expected": int(expected["note_count"]),
                "matched": int(proposal["matched_count"]),
                "unresolved_notes": int(unresolved["expected_note_count"]),
                "unresolved_candidates": int(unresolved["candidate_count"]),
            }
        )
    return rows


def _coverage_markdown(rows: Sequence[Mapping[str, int]]) -> str:
    lines = [
        "# Corrected Consumed-System Proposal Coverage",
        "",
        "Unreviewed deterministic proposals only. These rows are not training eligible.",
        "",
        "| Physical measure | Expected | Matched | Unresolved notes | Unresolved candidates |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    lines.extend(
        "| {physical_measure} | {expected} | {matched} | {unresolved_notes} | "
        "{unresolved_candidates} |".format(**row)
        for row in rows
    )
    return "\n".join(lines) + "\n"


def _file_hash_payload(path: Path, repo_root: Path) -> dict[str, str]:
    return {"path": _display_path(path, repo_root), "sha256": _sha256(path)}


def load_notated_pitches(path: Path) -> dict[int, tuple[NotatedPitch, ...]]:
    """Return visible pitched-note order per MusicXML measure, excluding rests."""
    root = ET.parse(path).getroot()
    _strip_namespaces(root)
    parts = root.findall("part")
    if len(parts) != 1:
        raise ValueError(f"Expected exactly one MusicXML part, found {len(parts)}")

    result: dict[int, tuple[NotatedPitch, ...]] = {}
    for fallback, measure in enumerate(parts[0].findall("measure"), start=1):
        number = _measure_number(measure, fallback)
        if number in result:
            raise ValueError(f"Duplicate MusicXML measure number: {number}")
        if measure.find("backup") is not None:
            raise ValueError(f"Polyphonic MusicXML is unsupported in measure {number}")
        notes = []
        for note in measure.findall("note"):
            if note.find("rest") is not None:
                continue
            if note.find("chord") is not None:
                raise ValueError(f"Chordal MusicXML is unsupported in measure {number}")
            pitch = note.find("pitch")
            if pitch is None:
                continue
            step = (pitch.findtext("step") or "").strip()
            octave_text = (pitch.findtext("octave") or "").strip()
            if step not in PITCH_CLASS or not octave_text:
                raise ValueError(f"Invalid pitch in MusicXML measure {number}")
            octave = int(octave_text)
            alter = _integer_alter(pitch.findtext("alter"), measure_number=number)
            suffix = ALTER_SUFFIX.get(alter)
            if suffix is None:
                raise ValueError(f"Unsupported pitch alteration {alter} in measure {number}")
            tie_types = tuple(
                sorted(
                    {
                        value
                        for tie in note.findall("tie")
                        if (value := tie.get("type")) in {"start", "stop"}
                    }
                )
            )
            notes.append(
                NotatedPitch(
                    physical_measure_number=number,
                    order_in_measure=len(notes) + 1,
                    staff_pitch=f"{step}{octave}",
                    sounding_pitch=f"{step}{suffix}{octave}",
                    pitch_midi=12 * (octave + 1) + PITCH_CLASS[step] + alter,
                    alter=alter,
                    tie_types=tie_types,
                )
            )
        result[number] = tuple(notes)
    if not result:
        raise ValueError("MusicXML contains no measures")
    return result


def build_exact_pitch_alignment_proposal(
    candidate_artifact: Mapping[str, Any],
    expected: Sequence[NotatedPitch],
) -> dict[str, Any]:
    """Build a conservative x-monotonic proposal using exact staff positions only."""
    candidates = candidate_artifact.get("candidates")
    staff_lines = candidate_artifact.get("staff_lines_y_px")
    if (
        not isinstance(candidates, list)
        or not isinstance(staff_lines, list)
        or len(staff_lines) != 5
    ):
        raise ValueError("Candidate artifact requires candidates and five staff lines")
    enriched = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise ValueError("Candidate entries must be objects")
        center = _required_object(candidate, "center", "Candidate")
        center_x = float(center["x"])
        center_y = float(center["y"])
        enriched.append(
            {
                "candidate": candidate,
                "x": center_x,
                "staff_pitch": treble_pitch_for_y(center_y, tuple(float(y) for y in staff_lines)),
                "score": float(candidate.get("score", 0.0)),
            }
        )
    enriched.sort(key=lambda item: (item["x"], str(item["candidate"].get("id", ""))))

    if not expected:
        return {
            "status": "available",
            "method": "monotonic_exact_staff_pitch_alignment_v1",
            "assignments": [],
            "confidence": 1.0,
            "confidence_basis": "No sounding noteheads are expected in this crop.",
            "human_reviewed": False,
        }
    if len(enriched) < len(expected):
        return _unavailable_proposal("candidate_count_below_expected_note_count")

    paths: list[dict[int, tuple[float, tuple[int, ...]]]] = []
    for expected_index, pitch in enumerate(expected):
        current: dict[int, tuple[float, tuple[int, ...]]] = {}
        for candidate_index, item in enumerate(enriched):
            if item["staff_pitch"] != pitch.staff_pitch:
                continue
            if expected_index == 0:
                current[candidate_index] = (item["score"], (candidate_index,))
                continue
            previous = paths[expected_index - 1]
            options = [
                (score + item["score"], indices + (candidate_index,))
                for prior_index, (score, indices) in previous.items()
                if prior_index < candidate_index
            ]
            if options:
                current[candidate_index] = max(options, key=lambda value: (value[0], value[1]))
        paths.append(current)
        if not current:
            return _unavailable_proposal("no_full_exact_staff_pitch_alignment")

    best_score, selected_indices = max(paths[-1].values(), key=lambda value: (value[0], value[1]))
    assignments = []
    for order, (pitch, candidate_index) in enumerate(
        zip(expected, selected_indices, strict=True), start=1
    ):
        item = enriched[candidate_index]
        candidate = item["candidate"]
        assignments.append(
            {
                "order": order,
                "candidate_id": candidate.get("id"),
                "candidate_center": candidate.get("center"),
                "candidate_score": round(item["score"], 6),
                "candidate_staff_pitch": item["staff_pitch"],
                "expected_staff_pitch": pitch.staff_pitch,
                "expected_sounding_pitch": pitch.sounding_pitch,
                "physical_measure_number": pitch.physical_measure_number,
            }
        )
    mean_score = best_score / len(assignments)
    return {
        "status": "available",
        "method": "monotonic_exact_staff_pitch_alignment_v1",
        "assignments": assignments,
        "confidence": round(max(0.0, min(1.0, mean_score)), 6),
        "confidence_basis": (
            "Mean blind candidate-detector score after exact diatonic staff-position and "
            "left-to-right constraints; heuristic and not calibrated."
        ),
        "human_reviewed": False,
    }


def _unavailable_proposal(reason: str) -> dict[str, Any]:
    return {
        "status": "review_queue_only",
        "method": "monotonic_exact_staff_pitch_alignment_v1",
        "assignments": [],
        "confidence": None,
        "confidence_basis": "No automatic label proposal was emitted.",
        "reason": reason,
        "human_reviewed": False,
    }


def _validate_mapping_header(mapping: Mapping[str, Any]) -> None:
    if mapping.get("kind") != MAPPING_KIND:
        raise ValueError(f"Mapping kind must be {MAPPING_KIND!r}")
    status = mapping.get("split_status")
    if status in HELDOUT_STATUSES:
        raise ValueError(f"Refusing output split_status {status!r}; use {SPLIT_STATUS!r}")
    if status != SPLIT_STATUS:
        raise ValueError(f"split_status must be {SPLIT_STATUS!r}, got {status!r}")


def _validate_consumption_evidence(
    mapping: Mapping[str, Any], *, mapping_dir: Path, repo_root: Path
) -> dict[str, Any]:
    payload = _required_object(mapping, "consumption", "Mapping manifest")
    source_status = _required_string(payload, "source_split_status", "Consumption evidence")
    reason = _required_string(payload, "reason", "Consumption evidence")
    result: dict[str, Any] = {"source_split_status": source_status, "reason": reason}
    evidence = payload.get("evaluation_evidence")
    if source_status in HELDOUT_STATUSES and not isinstance(evidence, dict):
        raise ValueError(
            f"Source split {source_status!r} requires hash-pinned evaluation_evidence before reuse"
        )
    if isinstance(evidence, dict):
        path, sha256 = _validated_source_file(
            evidence,
            label="Evaluation evidence",
            mapping_dir=mapping_dir,
            repo_root=repo_root,
        )
        result["evaluation_evidence"] = {
            "path": _display_path(path, repo_root),
            "sha256": sha256,
        }
    return result


def _validate_crop_mapping(
    raw_crops: list[Any],
    *,
    physical_measure_numbers: Sequence[int],
    request_crop_indices: set[int],
) -> list[dict[str, Any]]:
    if not raw_crops:
        raise ValueError("Mapping manifest must contain at least one crop")
    result = []
    seen_crops = set()
    flattened = []
    for index, raw in enumerate(raw_crops):
        if not isinstance(raw, dict):
            raise ValueError(f"crops[{index}] must be an object")
        crop = _required_positive_int(raw, "system_measure_index", f"crops[{index}]")
        if crop in seen_crops:
            raise ValueError(f"Duplicate automatic crop mapping: {crop}")
        seen_crops.add(crop)
        numbers = raw.get("physical_measure_numbers")
        if (
            not isinstance(numbers, list)
            or not numbers
            or not all(isinstance(number, int) and number > 0 for number in numbers)
        ):
            raise ValueError(f"crops[{index}].physical_measure_numbers must be positive integers")
        flattened.extend(numbers)
        result.append(
            {
                "system_measure_index": crop,
                "physical_measure_numbers": tuple(numbers),
                "candidate_artifact": raw.get("candidate_artifact"),
            }
        )
    if tuple(flattened) != tuple(physical_measure_numbers):
        raise ValueError(
            "Crop mapping must cover MusicXML measures exactly once in score order: "
            f"mapping={flattened}, musicxml={list(physical_measure_numbers)}"
        )
    if seen_crops != request_crop_indices:
        raise ValueError(
            "Crop mapping must exactly match source request crops: "
            f"mapping={sorted(seen_crops)}, requests={sorted(request_crop_indices)}"
        )
    return sorted(result, key=lambda item: item["system_measure_index"])


def _requests_for_identity(
    rows: Sequence[Mapping[str, Any]],
    *,
    slug: str,
    system_index: int,
    source_split_status: str,
) -> dict[int, Mapping[str, Any]]:
    result = {}
    for row in rows:
        identity = row.get("identity")
        if not isinstance(identity, dict):
            continue
        if identity.get("slug") != slug or identity.get("system_index") != system_index:
            continue
        row_split = row.get("split")
        if row_split != source_split_status:
            raise ValueError(
                f"Request source split changed for {slug} system {system_index}: "
                f"expected {source_split_status!r}, got {row_split!r}"
            )
        crop = int(identity["system_measure_index"])
        if crop in result:
            raise ValueError(f"Duplicate request crop identity: {crop}")
        result[crop] = row
    if not result:
        raise ValueError(f"No request rows found for {slug} system {system_index}")
    return result


def _validate_candidate_artifact(
    artifact: Mapping[str, Any],
    *,
    slug: str,
    system_index: int,
    crop_index: int,
    image_path: Path,
    out_dir: Path,
    repo_root: Path,
) -> None:
    expected = (slug, system_index, crop_index)
    actual = (
        artifact.get("slug"),
        artifact.get("system_index"),
        artifact.get("system_measure_index"),
    )
    if actual != expected:
        raise ValueError(f"Candidate identity mismatch: expected {expected}, got {actual}")
    recorded_source = artifact.get("source_image_path")
    if not isinstance(recorded_source, str):
        raise ValueError("Candidate artifact source_image_path is required")
    candidate_image_path = _resolve_flexible_path(
        recorded_source,
        mapping_dir=repo_root,
        repo_root=repo_root,
        out_dir=out_dir,
    )
    if candidate_image_path != image_path.resolve():
        raise ValueError(
            f"Candidate source image mismatch: {candidate_image_path} != {image_path.resolve()}"
        )


def _validated_source_file(
    payload: Mapping[str, Any], *, label: str, mapping_dir: Path, repo_root: Path
) -> tuple[Path, str]:
    path_value = _required_string(payload, "path", label)
    sha256 = _required_sha256(payload, "sha256", label)
    path = _resolve_flexible_path(
        path_value,
        mapping_dir=mapping_dir,
        repo_root=repo_root,
    )
    _validate_file_hash(path, sha256, label)
    return path, sha256


def _validate_file_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(f"{label} source hash changed: expected {expected}, got {actual}")


def _resolve_flexible_path(
    value: str,
    *,
    mapping_dir: Path,
    repo_root: Path,
    out_dir: Path | None = None,
) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    candidates = [mapping_dir / path, repo_root / path]
    if out_dir is not None:
        candidates.append(out_dir / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def _resolve_out_relative_path(value: str, *, out_dir: Path) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Request image path must be safely relative to out_dir: {value!r}")
    return (out_dir / path).resolve()


def _validate_output_path(path: Path) -> None:
    if any(part.lower() in HELDOUT_STATUSES for part in path.parts):
        raise ValueError(f"Refusing to write consumed training output under a heldout path: {path}")
    protected = PROTECTED_OUTPUT_NAMESPACES.intersection(part.lower() for part in path.parts)
    if protected:
        raise ValueError(
            "Refusing to write cross-score proposals into protected namespace(s): "
            + ", ".join(sorted(protected))
        )


def _pitch_payload(pitch: NotatedPitch) -> dict[str, Any]:
    return {
        "physical_measure_number": pitch.physical_measure_number,
        "order_in_measure": pitch.order_in_measure,
        "staff_pitch": pitch.staff_pitch,
        "sounding_pitch": pitch.sounding_pitch,
        "pitch_midi": pitch.pitch_midi,
        "alter": pitch.alter,
        "tie_types": list(pitch.tie_types),
    }


def _integer_alter(value: str | None, *, measure_number: int) -> int:
    if value is None:
        return 0
    number = float(value)
    if not number.is_integer():
        raise ValueError(f"Microtonal alteration is unsupported in measure {measure_number}")
    return int(number)


def _measure_number(measure: ET.Element, fallback: int) -> int:
    value = measure.get("number")
    if value is None:
        return fallback
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"MusicXML measure number must be an integer: {value!r}") from exc


def _strip_namespaces(root: ET.Element) -> None:
    for element in root.iter():
        if "}" in element.tag:
            element.tag = element.tag.split("}", 1)[1]


def _required_object(payload: Mapping[str, Any], key: str, label: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{label}.{key} must be an object")
    return value


def _required_list(payload: Mapping[str, Any], key: str, label: str) -> list[Any]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise ValueError(f"{label}.{key} must be an array")
    return value


def _required_string(payload: Mapping[str, Any], key: str, label: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label}.{key} must be a non-empty string")
    return value.strip()


def _required_positive_int(payload: Mapping[str, Any], key: str, label: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{label}.{key} must be a positive integer")
    return value


def _required_sha256(payload: Mapping[str, Any], key: str, label: str) -> str:
    value = _required_string(payload, key, label)
    if not SHA256_RE.fullmatch(value):
        raise ValueError(f"{label}.{key} must be a lowercase SHA256")
    return value


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"Expected JSON object at {path}:{line_number}")
        rows.append(row)
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(payload: Mapping[str, Any]) -> str:
    content = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _display_path(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    content = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(content)


if __name__ == "__main__":
    raise SystemExit(main())
