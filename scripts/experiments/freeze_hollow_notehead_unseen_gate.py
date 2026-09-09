"""Create a truth-blind unseen-score freeze for the hollow-notehead proposer.

The command is intentionally create-once. It reads only pipeline layout/image
artifacts, generates measure crops and staff-grid candidates, runs the fixed
hollow-notehead rule, validates every hash-pinned artifact, and atomically
publishes one sealed morphology gate. It never reads MusicXML, note ground
truth, dataset ground truth, or human review files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import sys
from collections.abc import Mapping, Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from score2abc.chord_ocr.alignment import (  # noqa: E402
    detect_barlines,
    measure_boundaries_for_system,
)
from score2abc.manifest import load_manifest_jsonl  # noqa: E402
from scripts.build_vlm_melody_inputs import _estimate_staff  # noqa: E402
from scripts.build_vlm_notehead_candidates import (  # noqa: E402
    GridCandidate,
    detect_staff_grid_density_candidates,
)
from scripts.experiments import spike_consumed_hollow_notehead_proposals as hollow  # noqa: E402

SCHEMA_VERSION = 1
DEFAULT_SLUG = "jaime-llanos_25_chispazo_pasillo_pedro-morales-pino"
DEFAULT_SYSTEM_INDEX = 4
DEFAULT_EXPECTED_MEASURES = 8
DEFAULT_VERSION = "v1"
OUTPUT_SUBDIR = "vlm_melody_hollow_notehead_gate"
MAX_CANDIDATES = 24
SEALED_KIND = "vlm_melody_hollow_notehead_unseen_gate_sealed_manifest"
SELECTION_KIND = "vlm_melody_hollow_notehead_unseen_gate_selection"
SEGMENTATION_KIND = "vlm_melody_hollow_notehead_unseen_gate_segmentation"
CANDIDATE_KIND = "vlm_notehead_candidates"
PROPOSAL_KIND = "vlm_melody_hollow_notehead_unseen_gate_proposals"
VERSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

DEFAULT_SELECTION_MANIFEST = (
    REPO_ROOT / "tests/fixtures/vlm_melody/hollow_notehead_unseen_gate/"
    "chispazo_system_004_selection.json"
)
RULE_SOURCE = REPO_ROOT / "scripts/experiments/spike_consumed_hollow_notehead_proposals.py"
CANDIDATE_DETECTOR_SOURCE = REPO_ROOT / "scripts/build_vlm_notehead_candidates.py"
STAFF_GEOMETRY_SOURCE = REPO_ROOT / "scripts/build_vlm_melody_inputs.py"
ALIGNMENT_SOURCE = REPO_ROOT / "score2abc/chord_ocr/alignment.py"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", nargs="?", type=Path, default=Path("out"))
    parser.add_argument("--slug", default=DEFAULT_SLUG)
    parser.add_argument("--system", type=int, default=DEFAULT_SYSTEM_INDEX)
    parser.add_argument("--expected-measures", type=int, default=DEFAULT_EXPECTED_MEASURES)
    parser.add_argument("--version", default=DEFAULT_VERSION)
    parser.add_argument(
        "--selection-manifest",
        type=Path,
        default=DEFAULT_SELECTION_MANIFEST,
        help="Committed target preregistration; required and hash-pinned into the freeze.",
    )
    args = parser.parse_args(argv)
    try:
        manifest = freeze_hollow_notehead_unseen_gate(
            args.out_dir,
            slug=args.slug,
            system_index=args.system,
            expected_measure_count=args.expected_measures,
            version=args.version,
            selection_manifest=args.selection_manifest,
        )
    except (FileExistsError, FileNotFoundError, OSError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(manifest)
    return 0


def freeze_hollow_notehead_unseen_gate(
    out_dir: Path,
    *,
    slug: str = DEFAULT_SLUG,
    system_index: int = DEFAULT_SYSTEM_INDEX,
    expected_measure_count: int = DEFAULT_EXPECTED_MEASURES,
    version: str = DEFAULT_VERSION,
    selection_manifest: Path = DEFAULT_SELECTION_MANIFEST,
) -> Path:
    """Build and atomically publish one truth-blind morphology freeze."""
    if not slug or Path(slug).name != slug:
        raise ValueError(f"Unsafe slug: {slug!r}")
    if system_index <= 0:
        raise ValueError("system_index must be positive")
    if expected_measure_count <= 0:
        raise ValueError("expected_measure_count must be positive")
    if not VERSION_RE.fullmatch(version):
        raise ValueError(f"Unsafe gate version: {version!r}")

    out_root = out_dir.resolve()
    work_dir = out_root / slug
    system_path = work_dir / "systems" / f"system_{system_index:03d}.png"
    destination = work_dir / OUTPUT_SUBDIR / version / f"system_{system_index:03d}" / "frozen"
    temp_dir = destination.with_name(".frozen.tmp")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Refusing to overwrite unseen-score freeze: {destination}")
    if temp_dir.exists() or temp_dir.is_symlink():
        raise FileExistsError(f"Refusing stale unseen-score temporary freeze: {temp_dir}")

    _require_pipeline_work(out_root, slug=slug)
    if not system_path.is_file():
        raise FileNotFoundError(f"Source system not found: {system_path}")

    selection_path = selection_manifest.resolve()
    selected_boundaries = _validate_selection_manifest(
        selection_path,
        slug=slug,
        system_index=system_index,
        expected_measure_count=expected_measure_count,
        system_path=system_path,
    )
    barlines = sorted(float(value) for value in detect_barlines(system_path))
    detected_boundaries = [
        float(value) for value in measure_boundaries_for_system(system_path, barlines)
    ]
    measure_count = len(detected_boundaries) - 1
    if measure_count != expected_measure_count:
        raise ValueError(
            f"Expected {expected_measure_count} measures for {slug} system "
            f"{system_index}, found {measure_count}: {detected_boundaries}"
        )
    if len(detected_boundaries) != len(selected_boundaries) or any(
        not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12)
        for actual, expected in zip(detected_boundaries, selected_boundaries, strict=True)
    ):
        raise ValueError(
            "Detected measure boundaries changed from preregistration: "
            f"expected {selected_boundaries}, got {detected_boundaries}"
        )
    boundaries = selected_boundaries

    with Image.open(system_path) as opened:
        system_image = opened.convert("RGB")
    staff = _estimate_staff(system_image)
    staff_lines = [int(value) for value in staff.line_ys]
    _validate_staff_lines(staff_lines)
    staff_spacing = _staff_spacing(staff_lines)
    boundary_pixels = _boundary_pixels(system_image.width, boundaries)

    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=False, exist_ok=False)
    try:
        source_record = _external_file_record(system_path)
        measure_rows = []
        for measure_index, (left, right) in enumerate(pairwise(boundary_pixels), start=1):
            identity = {
                "slug": slug,
                "system_index": system_index,
                "system_measure_index": measure_index,
            }
            measure_dir = temp_dir / "measures" / f"measure_{measure_index:03d}"
            measure_dir.mkdir(parents=True)
            raw_path = measure_dir / "raw.png"
            raw_image = system_image.crop((left, 0, right, system_image.height))
            raw_image.save(raw_path)
            raw_record = _local_file_record(raw_path, temp_dir)

            candidates_path = measure_dir / "candidates.json"
            candidate_payload = _build_candidate_payload(
                raw_image,
                identity=identity,
                raw_record=raw_record,
                staff_lines=staff_lines,
                staff_spacing=staff_spacing,
                x_bounds=(left, right),
            )
            _write_json(candidates_path, candidate_payload)
            candidate_record = _local_file_record(candidates_path, temp_dir)

            proposals, considered_pairs = hollow.propose_hollow_notehead_centers(
                raw_image,
                candidate_payload,
            )
            proposal_path = measure_dir / "proposals.json"
            proposal_payload = {
                "schema_version": SCHEMA_VERSION,
                "kind": PROPOSAL_KIND,
                "split": "fresh_heldout_morphology",
                "status": "frozen_awaiting_human_review",
                "truth_accessed": False,
                "identity": identity,
                "inputs": {
                    "raw_image": raw_record,
                    "candidate_artifact": candidate_record,
                },
                "rule": {
                    "source_sha256": _sha256(RULE_SOURCE),
                    "entrypoint": (
                        "scripts.experiments.spike_consumed_hollow_notehead_proposals."
                        "propose_hollow_notehead_centers"
                    ),
                },
                "considered_pair_count": len(considered_pairs),
                "considered_pairs": considered_pairs,
                "proposal_count": len(proposals),
                "proposals": [hollow._proposal_json(proposal) for proposal in proposals],
                "provenance": {
                    "proposal_generation_is_gt_blind": True,
                    "proposal_function_inputs": ["raw_image", "candidate_artifact"],
                    "ground_truth_files_read": [],
                    "review_files_read": [],
                },
            }
            _write_json(proposal_path, proposal_payload)
            proposal_record = _local_file_record(proposal_path, temp_dir)
            measure_rows.append(
                {
                    "identity": identity,
                    "raw_image": raw_record,
                    "candidate_artifact": candidate_record,
                    "proposal_artifact": proposal_record,
                }
            )

        segmentation_path = temp_dir / "segmentation.json"
        segmentation_payload = {
            "schema_version": SCHEMA_VERSION,
            "kind": SEGMENTATION_KIND,
            "split": "fresh_heldout_morphology",
            "truth_accessed": False,
            "identity": {"slug": slug, "system_index": system_index},
            "source_system": source_record,
            "source_dimensions_px": {
                "width": system_image.width,
                "height": system_image.height,
            },
            "detected_barlines_x_fraction": [round(value, 12) for value in barlines],
            "measure_boundaries_x_fraction": boundaries,
            "measure_boundaries_x_px": boundary_pixels,
            "measure_count": len(measure_rows),
            "measures": [
                {
                    "identity": row["identity"],
                    "x_bounds_px": {
                        "left": boundary_pixels[index],
                        "right": boundary_pixels[index + 1],
                    },
                    "raw_image": row["raw_image"],
                }
                for index, row in enumerate(measure_rows)
            ],
            "provenance": {
                "segmentation_is_gt_blind": True,
                "ground_truth_files_read": [],
                "review_files_read": [],
            },
        }
        _write_json(segmentation_path, segmentation_payload)

        sealed_path = temp_dir / "sealed_manifest.json"
        sealed_payload = {
            "schema_version": SCHEMA_VERSION,
            "kind": SEALED_KIND,
            "status": "frozen_awaiting_human_review",
            "split": "fresh_heldout_morphology",
            "create_once": True,
            "truth_accessed": False,
            "target": {
                "slug": slug,
                "system_index": system_index,
                "expected_measure_count": expected_measure_count,
            },
            "gate": {
                "purpose": "heldout hollow-notehead morphology gate",
                "end_to_end_transcription_claim": False,
                "eligible_for_candidate_pipeline_integration": False,
                "next_step": "human review of hollow-notehead centers against frozen proposals",
            },
            "source_system": source_record,
            "selection": _external_file_record(selection_path),
            "segmentation": _local_file_record(segmentation_path, temp_dir),
            "rule": {
                "source": _external_file_record(RULE_SOURCE),
                "entrypoint": (
                    "scripts.experiments.spike_consumed_hollow_notehead_proposals."
                    "propose_hollow_notehead_centers"
                ),
                "fixed_configuration": _rule_configuration(),
            },
            "builders": {
                "freeze_orchestrator": _external_file_record(Path(__file__).resolve()),
                "candidate_detector": _external_file_record(CANDIDATE_DETECTOR_SOURCE),
                "staff_geometry": _external_file_record(STAFF_GEOMETRY_SOURCE),
                "barline_alignment": _external_file_record(ALIGNMENT_SOURCE),
            },
            "measure_count": len(measure_rows),
            "measures": measure_rows,
            "provenance": {
                "scope": "spike_only_unseen_score_morphology_gate",
                "proposal_generation_is_gt_blind": True,
                "candidate_generation_is_gt_blind": True,
                "ground_truth_files_read": [],
                "review_files_read": [],
                "musicxml_files_read": [],
            },
        }
        _write_json(sealed_path, sealed_payload)
        verify_sealed_manifest(sealed_path, artifact_root=temp_dir)
        temp_dir.rename(destination)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise

    published_manifest = destination / "sealed_manifest.json"
    verify_sealed_manifest(published_manifest)
    return published_manifest


def verify_sealed_manifest(
    manifest_path: Path,
    *,
    artifact_root: Path | None = None,
) -> dict[str, Any]:
    """Fail closed unless every sealed input and derived artifact is intact."""
    payload = _load_object(manifest_path, "Sealed manifest")
    root = (artifact_root or manifest_path.parent).resolve()
    if payload.get("kind") != SEALED_KIND:
        raise ValueError(f"Unexpected sealed-manifest kind: {manifest_path}")
    if payload.get("status") != "frozen_awaiting_human_review":
        raise ValueError("Sealed manifest is not frozen awaiting human review")
    if payload.get("split") != "fresh_heldout_morphology":
        raise ValueError("Sealed manifest is not a heldout morphology gate")
    if payload.get("create_once") is not True or payload.get("truth_accessed") is not False:
        raise ValueError("Sealed manifest violates create-once truth-blind contract")

    target = _required_object(payload, "target", "Sealed manifest")
    slug = str(target.get("slug"))
    system_index = _positive_int(target.get("system_index"), "target system_index")
    expected_count = _positive_int(
        target.get("expected_measure_count"),
        "target expected_measure_count",
    )
    if payload.get("measure_count") != expected_count:
        raise ValueError("Sealed manifest measure count does not match target")

    source_system_path = _validate_record(
        payload.get("source_system"),
        root=root,
        label="Source system",
    )
    selection_path = _validate_record(payload.get("selection"), root=root, label="Selection")
    selected_boundaries = _validate_selection_manifest(
        selection_path,
        slug=slug,
        system_index=system_index,
        expected_measure_count=expected_count,
        system_path=source_system_path,
    )
    segmentation_path = _validate_record(
        payload.get("segmentation"),
        root=root,
        label="Segmentation",
    )
    rule = _required_object(payload, "rule", "Sealed manifest")
    rule_source = _validate_record(rule.get("source"), root=root, label="Rule source")
    if rule_source.resolve() != RULE_SOURCE.resolve():
        raise ValueError(f"Unexpected hollow-notehead rule source: {rule_source}")
    builders = _required_object(payload, "builders", "Sealed manifest")
    for name in (
        "freeze_orchestrator",
        "candidate_detector",
        "staff_geometry",
        "barline_alignment",
    ):
        _validate_record(builders.get(name), root=root, label=f"Builder {name}")

    rows = payload.get("measures")
    if not isinstance(rows, list) or len(rows) != expected_count:
        raise ValueError(f"Expected {expected_count} ordered measure rows")
    expected_identities = [
        {
            "slug": slug,
            "system_index": system_index,
            "system_measure_index": index,
        }
        for index in range(1, expected_count + 1)
    ]
    identities = [_required_object(row, "identity", "Measure row") for row in rows]
    if identities != expected_identities:
        raise ValueError(
            f"Frozen measure identities changed: expected {expected_identities}, got {identities}"
        )

    for row, identity in zip(rows, expected_identities, strict=True):
        raw_path = _validate_record(
            row.get("raw_image"),
            root=root,
            label=f"Raw image {identity}",
        )
        candidate_path = _validate_record(
            row.get("candidate_artifact"),
            root=root,
            label=f"Candidate artifact {identity}",
        )
        proposal_path = _validate_record(
            row.get("proposal_artifact"),
            root=root,
            label=f"Proposal artifact {identity}",
        )
        _validate_candidate_artifact(
            candidate_path,
            identity=identity,
            raw_path=raw_path,
            raw_record=row["raw_image"],
        )
        _validate_proposal_artifact(
            proposal_path,
            identity=identity,
            raw_record=row["raw_image"],
            candidate_record=row["candidate_artifact"],
            candidate_path=candidate_path,
            rule_sha256=_sha256(rule_source),
        )

    segmentation = _load_object(segmentation_path, "Segmentation")
    if segmentation.get("kind") != SEGMENTATION_KIND:
        raise ValueError("Malformed segmentation artifact")
    if segmentation.get("measure_count") != expected_count:
        raise ValueError("Segmentation measure count changed")
    segmentation_rows = segmentation.get("measures")
    if not isinstance(segmentation_rows, list):
        raise ValueError("Segmentation measures must be a list")
    if [row.get("identity") for row in segmentation_rows] != expected_identities:
        raise ValueError("Segmentation identities do not match sealed manifest")
    boundaries = segmentation.get("measure_boundaries_x_fraction")
    pixels = segmentation.get("measure_boundaries_x_px")
    if not isinstance(boundaries, list) or len(boundaries) != expected_count + 1:
        raise ValueError("Malformed fractional measure boundaries")
    if boundaries != selected_boundaries:
        raise ValueError("Segmentation boundaries do not match preregistration")
    if not isinstance(pixels, list) or len(pixels) != expected_count + 1:
        raise ValueError("Malformed pixel measure boundaries")
    if any(right <= left for left, right in pairwise(pixels)):
        raise ValueError("Pixel measure boundaries are not strictly increasing")

    provenance = _required_object(payload, "provenance", "Sealed manifest")
    for key in ("ground_truth_files_read", "review_files_read", "musicxml_files_read"):
        if provenance.get(key) != []:
            raise ValueError(f"Forbidden access recorded in {key}")
    return payload


def _build_candidate_payload(
    image: Image.Image,
    *,
    identity: Mapping[str, Any],
    raw_record: Mapping[str, Any],
    staff_lines: Sequence[int],
    staff_spacing: float,
    x_bounds: tuple[int, int],
) -> dict[str, Any]:
    detected = detect_staff_grid_density_candidates(
        image,
        staff_lines=list(staff_lines),
        max_candidates=MAX_CANDIDATES,
    )
    candidates = [
        _candidate_to_json(index, candidate, image_size=image.size)
        for index, candidate in enumerate(detected[:MAX_CANDIDATES], start=1)
    ]
    return {
        "schema_version": 2,
        "kind": CANDIDATE_KIND,
        "strategy": "staff-grid-density",
        "strategy_version": 2,
        **identity,
        "source_image_path": str(raw_record["path"]),
        "source_image_sha256": str(raw_record["sha256"]),
        "source_image_size_px": {"width": image.width, "height": image.height},
        "source_system_x_bounds_px": {"left": x_bounds[0], "right": x_bounds[1]},
        "coordinate_space": "raw measure pixels, origin at top-left",
        "staff_lines_y_px": list(staff_lines),
        "staff_spacing_px": round(staff_spacing, 3),
        "max_candidates": MAX_CANDIDATES,
        "candidate_count": len(candidates),
        "candidates": candidates,
        "provenance": {
            "detector": (
                "scripts.build_vlm_notehead_candidates." "detect_staff_grid_density_candidates"
            ),
            "detector_rank_becomes_candidate_id": True,
            "candidate_generation_is_blind": True,
            "ground_truth_files_read": [],
            "review_files_read": [],
        },
    }


def _candidate_to_json(
    index: int,
    candidate: GridCandidate,
    *,
    image_size: tuple[int, int],
) -> dict[str, Any]:
    width, height = image_size
    center_x, center_y = candidate.center
    return {
        "id": f"c{index:03d}",
        "rank": index,
        "bbox": {
            "left": candidate.bbox[0],
            "top": candidate.bbox[1],
            "right": candidate.bbox[2] + 1,
            "bottom": candidate.bbox[3] + 1,
        },
        "center": {"x": round(center_x, 2), "y": round(center_y, 2)},
        "normalized_center": {
            "x": round(center_x / width, 6),
            "y": round(center_y / height, 6),
        },
        "score": candidate.score,
        "features": candidate.features,
    }


def _validate_candidate_artifact(
    path: Path,
    *,
    identity: Mapping[str, Any],
    raw_path: Path,
    raw_record: Mapping[str, Any],
) -> None:
    payload = _load_object(path, "Candidate artifact")
    if payload.get("kind") != CANDIDATE_KIND:
        raise ValueError(f"Malformed candidate artifact: {path}")
    if _artifact_identity(payload) != dict(identity):
        raise ValueError(f"Candidate identity changed: {path}")
    if payload.get("source_image_sha256") != raw_record.get("sha256"):
        raise ValueError(f"Candidate source-image hash changed: {path}")
    with Image.open(raw_path) as opened:
        expected_size = {"width": opened.width, "height": opened.height}
    if payload.get("source_image_size_px") != expected_size:
        raise ValueError(f"Candidate source-image dimensions changed: {path}")
    _validate_staff_lines(payload.get("staff_lines_y_px"))
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or payload.get("candidate_count") != len(candidates):
        raise ValueError(f"Malformed candidate rows: {path}")
    expected_ids = [f"c{index:03d}" for index in range(1, len(candidates) + 1)]
    if [candidate.get("id") for candidate in candidates] != expected_ids:
        raise ValueError(f"Candidate IDs are not deterministic: {path}")
    provenance = _required_object(payload, "provenance", "Candidate artifact")
    if provenance.get("ground_truth_files_read") != []:
        raise ValueError(f"Candidate artifact records ground-truth access: {path}")
    if provenance.get("review_files_read") != []:
        raise ValueError(f"Candidate artifact records review access: {path}")


def _validate_proposal_artifact(
    path: Path,
    *,
    identity: Mapping[str, Any],
    raw_record: Mapping[str, Any],
    candidate_record: Mapping[str, Any],
    candidate_path: Path,
    rule_sha256: str,
) -> None:
    payload = _load_object(path, "Proposal artifact")
    if payload.get("kind") != PROPOSAL_KIND:
        raise ValueError(f"Malformed proposal artifact: {path}")
    if payload.get("identity") != dict(identity):
        raise ValueError(f"Proposal identity changed: {path}")
    inputs = _required_object(payload, "inputs", "Proposal artifact")
    if inputs.get("raw_image") != dict(raw_record):
        raise ValueError(f"Proposal raw-image pin changed: {path}")
    if inputs.get("candidate_artifact") != dict(candidate_record):
        raise ValueError(f"Proposal candidate pin changed: {path}")
    rule = _required_object(payload, "rule", "Proposal artifact")
    if rule.get("source_sha256") != rule_sha256:
        raise ValueError(f"Proposal rule hash changed: {path}")
    candidate_payload = _load_object(candidate_path, "Candidate artifact")
    candidate_ids = {str(candidate["id"]) for candidate in candidate_payload["candidates"]}
    proposals = payload.get("proposals")
    if not isinstance(proposals, list) or payload.get("proposal_count") != len(proposals):
        raise ValueError(f"Malformed proposal rows: {path}")
    for proposal in proposals:
        support = proposal.get("support_candidate_ids")
        if (
            not isinstance(support, list)
            or len(support) != 2
            or any(candidate_id not in candidate_ids for candidate_id in support)
        ):
            raise ValueError(f"Proposal references unknown candidate IDs: {path}")
    provenance = _required_object(payload, "provenance", "Proposal artifact")
    if provenance.get("ground_truth_files_read") != []:
        raise ValueError(f"Proposal artifact records ground-truth access: {path}")
    if provenance.get("review_files_read") != []:
        raise ValueError(f"Proposal artifact records review access: {path}")


def _rule_configuration() -> dict[str, Any]:
    return {
        "min_candidate_score": hollow.MIN_CANDIDATE_SCORE,
        "dx_spacing": [hollow.MIN_DX_SPACING, hollow.MAX_DX_SPACING],
        "dy_spacing": [hollow.MIN_DY_SPACING, hollow.MAX_DY_SPACING],
        "min_ring_density": hollow.MIN_RING_DENSITY,
        "min_ring_coverage": hollow.MIN_RING_COVERAGE,
        "min_hole_area_spacing_squared": hollow.MIN_HOLE_AREA_SPACING_SQUARED,
        "max_hole_center_distance_spacing": hollow.MAX_HOLE_CENTER_DISTANCE_SPACING,
        "min_hole_support_alignment": hollow.MIN_HOLE_SUPPORT_ALIGNMENT,
        "max_hole_axis_ratio": hollow.MAX_HOLE_AXIS_RATIO,
        "min_open_ring_density": hollow.MIN_OPEN_RING_DENSITY,
        "max_open_inner_density": hollow.MAX_OPEN_INNER_DENSITY,
        "contour_close_radius_px": hollow.CONTOUR_CLOSE_RADIUS_PX,
    }


def _validate_selection_manifest(
    path: Path,
    *,
    slug: str,
    system_index: int,
    expected_measure_count: int,
    system_path: Path,
) -> list[float]:
    selection = _load_object(path, "Unseen-gate selection")
    if selection.get("kind") != SELECTION_KIND:
        raise ValueError("Unexpected unseen-gate selection kind")
    if selection.get("split_status") != "unseen_morphology_gate":
        raise ValueError("Selection is not an unseen morphology gate")
    if selection.get("eligible_for_end_to_end_transcription_claim") is not False:
        raise ValueError("Selection cannot support an end-to-end transcription claim")
    identity = _required_object(selection, "identity", "Unseen-gate selection")
    if identity != {"slug": slug, "system_index": system_index}:
        raise ValueError(
            "Selection target changed: "
            f"expected {{'slug': {slug!r}, 'system_index': {system_index}}}, got {identity}"
        )

    source = _required_object(selection, "source", "Unseen-gate selection")
    _validate_record(source.get("dataset_pdf"), root=REPO_ROOT, label="Selected dataset PDF")
    selected_system_record = _required_object(
        source,
        "system_image",
        "Unseen-gate selection source",
    )
    selected_system_path = _validate_record(
        selected_system_record,
        root=REPO_ROOT,
        label="Selected system image",
    )
    if selected_system_path != system_path.resolve():
        raise ValueError(
            f"Selection system path changed: expected {system_path.resolve()}, "
            f"got {selected_system_path}"
        )
    with Image.open(selected_system_path) as opened:
        expected_dimensions = {"width_px": opened.width, "height_px": opened.height}
    recorded_dimensions = {
        "width_px": selected_system_record.get("width_px"),
        "height_px": selected_system_record.get("height_px"),
    }
    if recorded_dimensions != expected_dimensions:
        raise ValueError(
            f"Selection system dimensions changed: expected {expected_dimensions}, "
            f"got {recorded_dimensions}"
        )

    segmentation = _required_object(selection, "segmentation", "Unseen-gate selection")
    if segmentation.get("expected_measure_count") != expected_measure_count:
        raise ValueError("Selection expected measure count changed")
    boundaries = segmentation.get("measure_boundaries_x_fraction")
    if (
        not isinstance(boundaries, list)
        or len(boundaries) != expected_measure_count + 1
        or any(
            isinstance(value, bool) or not isinstance(value, (int, float)) for value in boundaries
        )
    ):
        raise ValueError("Selection measure boundaries are malformed")
    normalized_boundaries = [float(value) for value in boundaries]
    if any(right <= left for left, right in pairwise(normalized_boundaries)):
        raise ValueError("Selection measure boundaries are not strictly increasing")
    alignment_path = _validate_record(
        segmentation.get("alignment_source"),
        root=REPO_ROOT,
        label="Selected alignment source",
    )
    if alignment_path != ALIGNMENT_SOURCE.resolve():
        raise ValueError(f"Selection alignment source changed: {alignment_path}")

    rule_path = _validate_record(
        selection.get("frozen_rule"),
        root=REPO_ROOT,
        label="Selected hollow-notehead rule",
    )
    if rule_path != RULE_SOURCE.resolve():
        raise ValueError(f"Selection hollow-notehead rule changed: {rule_path}")
    policy = _required_object(selection, "selection_policy", "Unseen-gate selection")
    for field in (
        "automatic_candidate_outputs_inspected",
        "automatic_hollow_proposals_inspected",
        "review_truth_available_at_selection",
    ):
        if policy.get(field) is not False:
            raise ValueError(f"Selection policy requires {field}=false")
    return normalized_boundaries


def _require_pipeline_work(out_root: Path, *, slug: str) -> None:
    manifest_path = out_root / "manifest.jsonl"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Pipeline manifest not found: {manifest_path}")
    selected = [item for item in load_manifest_jsonl(manifest_path) if item.slug == slug]
    if len(selected) != 1:
        raise ValueError(f"Expected exactly one pipeline manifest item for {slug!r}")


def _boundary_pixels(width: int, boundaries: Sequence[float]) -> list[int]:
    pixels = [min(width, max(0, int(round(value * width)))) for value in boundaries]
    if len(pixels) < 2 or any(right <= left for left, right in pairwise(pixels)):
        raise ValueError(f"Malformed measure boundaries: {boundaries}")
    return pixels


def _validate_staff_lines(values: Any) -> None:
    if not isinstance(values, (list, tuple)) or len(values) != 5:
        raise ValueError(f"Expected exactly five staff lines, got {values!r}")
    lines = [int(value) for value in values]
    if lines != sorted(lines) or len(set(lines)) != 5:
        raise ValueError(f"Malformed staff-line geometry: {values!r}")


def _staff_spacing(lines: Sequence[int]) -> float:
    gaps = [right - left for left, right in pairwise(lines)]
    spacing = sum(gaps) / len(gaps)
    if spacing <= 0:
        raise ValueError(f"Malformed staff-line spacing: {lines!r}")
    return spacing


def _artifact_identity(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "slug": str(payload.get("slug")),
        "system_index": _positive_int(payload.get("system_index"), "system_index"),
        "system_measure_index": _positive_int(
            payload.get("system_measure_index"),
            "system_measure_index",
        ),
    }


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a positive integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a positive integer") from exc
    if result <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return result


def _required_object(payload: Mapping[str, Any], key: str, label: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{label} {key} must be an object")
    return value


def _load_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is malformed JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise TypeError(f"{label} must be a JSON object: {path}")
    return payload


def _validate_record(record: Any, *, root: Path, label: str) -> Path:
    if not isinstance(record, dict):
        raise ValueError(f"{label} file record must be an object")
    value = record.get("path")
    expected_hash = record.get("sha256")
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} file record must provide a path")
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise ValueError(f"{label} file record must provide a SHA256")
    path = Path(value)
    resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    actual_hash = _sha256(resolved)
    if actual_hash != expected_hash:
        raise ValueError(f"{label} hash drift: expected {expected_hash}, got {actual_hash}")
    return resolved


def _external_file_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Expected source file not found: {resolved}")
    try:
        display_path = resolved.relative_to(REPO_ROOT).as_posix()
        path_value = str((REPO_ROOT / display_path).resolve())
    except ValueError:
        path_value = str(resolved)
    return {
        "path": path_value,
        "sha256": _sha256(resolved),
    }


def _local_file_record(path: Path, root: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Expected generated artifact not found: {resolved}")
    return {
        "path": resolved.relative_to(root.resolve()).as_posix(),
        "sha256": _sha256(resolved),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
