"""Build versioned, unreviewed Carrizal cross-score training inputs.

This spike command is intentionally create-once. It uses the corrected barline
segmentation and the existing GT-blind notehead candidate builder, but it never
reads MusicXML or writes into the original heldout/input/review namespaces.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from score2abc.chord_ocr.alignment import (  # noqa: E402
    detect_barlines,
    measure_boundaries_for_system,
)
from score2abc.manifest import load_manifest_jsonl  # noqa: E402
from score2abc.schemas import WorkItem  # noqa: E402
from scripts.build_vlm_melody_inputs import (  # noqa: E402
    build_measure_inputs_for_system,
)
from scripts.build_vlm_notehead_localization_inputs import (  # noqa: E402
    build_notehead_localization_artifacts_for_record,
)

DEFAULT_SLUG = "jaime-llanos_19_carrizal_pasillo_emilio-murillo"
DEFAULT_SYSTEM = 4
DEFAULT_NAMESPACE = "carrizal_system_004_seg_v2"
EXPECTED_MEASURE_COUNT = 8
MAX_CANDIDATES = 24
NAMESPACE_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
PROTECTED_NAMESPACES = {
    "fresh_heldout",
    "freeze",
    "notehead_reviews",
    "vlm_melody_fresh_heldout",
    "vlm_melody_inputs",
    "vlm_melody_reviews",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", type=Path, help="Pipeline output directory.")
    parser.add_argument("--slug", default=DEFAULT_SLUG, help="Consumed score slug.")
    parser.add_argument("--system", type=int, default=DEFAULT_SYSTEM, help="1-based system index.")
    parser.add_argument(
        "--namespace",
        default=DEFAULT_NAMESPACE,
        help="Create-once leaf under vlm_melody_training_inputs/.",
    )
    args = parser.parse_args(argv)
    try:
        destination = build_consumed_cross_score_training_inputs(
            args.out_dir,
            slug=args.slug,
            system_index=args.system,
            namespace=args.namespace,
        )
    except (FileExistsError, FileNotFoundError, OSError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(destination)
    return 0


def build_consumed_cross_score_training_inputs(
    out_dir: Path,
    *,
    slug: str = DEFAULT_SLUG,
    system_index: int = DEFAULT_SYSTEM,
    namespace: str = DEFAULT_NAMESPACE,
) -> Path:
    """Create the isolated eight-crop Carrizal training-input namespace."""
    out_root = out_dir.resolve()
    destination = _validated_destination(out_root, slug=slug, namespace=namespace)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing training namespace: {destination}")

    manifest_path = out_root / "manifest.jsonl"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Pipeline manifest not found: {manifest_path}")
    item = _selected_work_item(manifest_path, slug)
    system_path = out_root / slug / "systems" / f"system_{system_index:03d}.png"
    if not system_path.is_file():
        raise FileNotFoundError(f"System crop not found: {system_path}")

    barlines = sorted(detect_barlines(system_path))
    boundaries = measure_boundaries_for_system(system_path, barlines)
    if len(boundaries) - 1 != EXPECTED_MEASURE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_MEASURE_COUNT} corrected measures for {slug} system "
            f"{system_index}, found {len(boundaries) - 1}: {boundaries}"
        )

    destination.mkdir(parents=True)
    try:
        measure_root = destination / "measure_inputs"
        records = build_measure_inputs_for_system(
            item,
            system_path=system_path,
            output_root=measure_root,
            system_index=system_index,
            start_global_measure_index=_global_measure_offset(
                out_root, slug=slug, target_system_index=system_index
            ),
            barlines=barlines,
            boundaries=boundaries,
            overwrite=False,
        )
        _validate_records(records, slug=slug, system_index=system_index)

        inputs_manifest_path = destination / "inputs_manifest.jsonl"
        _write_jsonl(inputs_manifest_path, records)
        candidate_records = _build_candidate_records(
            out_root,
            records=records,
            input_manifest_path=inputs_manifest_path,
            output_root=destination / "candidates",
        )
        candidates_manifest_path = destination / "candidates_manifest.jsonl"
        _write_jsonl(candidates_manifest_path, candidate_records)

        segmentation_path = destination / "segmentation.json"
        _write_json(
            segmentation_path,
            _segmentation_payload(
                system_path=system_path,
                slug=slug,
                system_index=system_index,
                namespace=namespace,
                barlines=barlines,
                boundaries=boundaries,
                records=records,
            ),
        )
        mapping_path = destination / "mapping.json"
        _write_json(
            mapping_path,
            _mapping_payload(
                slug=slug,
                system_index=system_index,
                namespace=namespace,
                segmentation_path=segmentation_path,
                inputs_manifest_path=inputs_manifest_path,
                candidates_manifest_path=candidates_manifest_path,
                candidate_records=candidate_records,
            ),
        )
        _write_json(
            destination / "manifest.json",
            _namespace_manifest(
                slug=slug,
                system_index=system_index,
                namespace=namespace,
                system_path=system_path,
                segmentation_path=segmentation_path,
                inputs_manifest_path=inputs_manifest_path,
                candidates_manifest_path=candidates_manifest_path,
                mapping_path=mapping_path,
            ),
        )
    except Exception:
        shutil.rmtree(destination)
        raise
    return destination


def _build_candidate_records(
    out_dir: Path,
    *,
    records: Sequence[dict[str, Any]],
    input_manifest_path: Path,
    output_root: Path,
) -> list[dict[str, Any]]:
    result = []
    for record in records:
        artifacts = build_notehead_localization_artifacts_for_record(
            out_dir,
            input_manifest_path=input_manifest_path,
            base_record=record,
            system_records=records,
            max_candidates=MAX_CANDIDATES,
            overwrite=False,
            output_root=output_root,
            include_evaluation_metadata=False,
        )
        paths = artifacts["paths"]
        candidate_path = Path(paths["candidates"])
        candidate_payload = _load_json_object(candidate_path)
        source_path = Path(record["paths"]["measure_raw"])
        overlay_path = candidate_path.with_name("candidate_overlay.png")
        _write_candidate_overlay(source_path, candidate_payload, overlay_path)
        artifact_paths = {
            **{key: Path(value) for key, value in paths.items()},
            "candidate_overlay": overlay_path,
        }
        result.append(
            {
                "schema_version": 1,
                "kind": "vlm_melody_cross_score_candidate_record",
                "split_status": "consumed_training",
                "review_status": "unreviewed",
                "eligible_for_training": False,
                "eligible_for_promotion": False,
                "identity": {
                    "slug": record["slug"],
                    "system_index": int(record["system_index"]),
                    "system_measure_index": int(record["system_measure_index"]),
                },
                "source": {
                    "measure_raw": _file_record(source_path),
                    "context": _file_record(Path(record["paths"]["context"])),
                },
                "candidate_generation": {
                    "strategy": candidate_payload["strategy"],
                    "strategy_version": candidate_payload["strategy_version"],
                    "max_candidates": candidate_payload["max_candidates"],
                    "candidate_count": candidate_payload["candidate_count"],
                    "ground_truth_files_read": [],
                },
                "artifacts": {
                    key: _file_record(path) for key, path in sorted(artifact_paths.items())
                },
            }
        )
    return result


def _segmentation_payload(
    *,
    system_path: Path,
    slug: str,
    system_index: int,
    namespace: str,
    barlines: Sequence[float],
    boundaries: Sequence[float],
    records: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    alignment_path = REPO_ROOT / "score2abc" / "chord_ocr" / "alignment.py"
    return {
        "schema_version": 1,
        "kind": "vlm_melody_cross_score_segmentation",
        "split_status": "consumed_training",
        "review_status": "unreviewed",
        "eligible_for_training": False,
        "eligible_for_promotion": False,
        "identity": {"slug": slug, "system_index": system_index},
        "segmentation_namespace": namespace,
        "segmentation_version": 2,
        "segmentation_policy": (
            "detect_barlines with near-threshold staff-edge recovery, followed by "
            "measure_boundaries_for_system"
        ),
        "source_system": _file_record(system_path),
        "detected_barlines_x_fraction": list(barlines),
        "measure_boundaries_x_fraction": list(boundaries),
        "measure_count": len(records),
        "crops": [
            {
                "system_measure_index": int(record["system_measure_index"]),
                "x_bounds_px": record["x_bounds_px"],
                "x_fraction_bounds": record["x_fraction_bounds"],
                "measure_raw": _file_record(Path(record["paths"]["measure_raw"])),
                "measure_staff": _file_record(Path(record["paths"]["measure_staff"])),
                "measure_staff_overlay": _file_record(
                    Path(record["paths"]["measure_staff_overlay"])
                ),
                "context": _file_record(Path(record["paths"]["context"])),
            }
            for record in records
        ],
        "provenance": {
            "scope": "spike_only",
            "alignment_source": _file_record(alignment_path),
            "ground_truth_files_read": [],
        },
    }


def _mapping_payload(
    *,
    slug: str,
    system_index: int,
    namespace: str,
    segmentation_path: Path,
    inputs_manifest_path: Path,
    candidates_manifest_path: Path,
    candidate_records: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    by_measure = {
        int(record["identity"]["system_measure_index"]): record for record in candidate_records
    }
    return {
        "schema_version": 1,
        "kind": "vlm_melody_consumed_training_segmentation_mapping",
        "split_status": "consumed_training",
        "review_status": "unreviewed",
        "eligible_for_training": False,
        "eligible_for_promotion": False,
        "identity": {"slug": slug, "system_index": system_index},
        "segmentation_namespace": namespace,
        "mapping_basis": (
            "Corrected segmentation creates one crop per approved physical measure; "
            "MusicXML and proposal truth are intentionally absent from this artifact."
        ),
        "source": {
            "segmentation": _file_record(segmentation_path),
            "inputs_manifest": _file_record(inputs_manifest_path),
            "candidates_manifest": _file_record(candidates_manifest_path),
        },
        "crops": [
            {
                "system_measure_index": index,
                "physical_measure_numbers": [index],
                "candidate_artifact": by_measure[index]["artifacts"]["candidates"],
            }
            for index in range(1, EXPECTED_MEASURE_COUNT + 1)
        ],
    }


def _namespace_manifest(
    *,
    slug: str,
    system_index: int,
    namespace: str,
    system_path: Path,
    segmentation_path: Path,
    inputs_manifest_path: Path,
    candidates_manifest_path: Path,
    mapping_path: Path,
) -> dict[str, Any]:
    script_path = Path(__file__).resolve()
    candidate_builder_path = REPO_ROOT / "scripts" / "build_vlm_notehead_localization_inputs.py"
    crop_builder_path = REPO_ROOT / "scripts" / "build_vlm_melody_inputs.py"
    return {
        "schema_version": 1,
        "kind": "vlm_melody_consumed_cross_score_training_inputs",
        "split_status": "consumed_training",
        "review_status": "unreviewed",
        "eligible_for_training": False,
        "eligible_for_promotion": False,
        "identity": {"slug": slug, "system_index": system_index},
        "segmentation_namespace": namespace,
        "measure_count": EXPECTED_MEASURE_COUNT,
        "source_system": _file_record(system_path),
        "artifacts": {
            "segmentation": _file_record(segmentation_path),
            "inputs_manifest": _file_record(inputs_manifest_path),
            "candidates_manifest": _file_record(candidates_manifest_path),
            "mapping": _file_record(mapping_path),
        },
        "provenance": {
            "scope": "spike_only",
            "builder": _file_record(script_path),
            "measure_input_builder": _file_record(crop_builder_path),
            "candidate_builder": _file_record(candidate_builder_path),
            "candidate_generation_is_gt_blind": True,
            "ground_truth_files_read": [],
            "protected_namespaces_mutated": [],
        },
    }


def _write_candidate_overlay(
    source_path: Path, candidate_payload: Mapping[str, Any], output_path: Path
) -> None:
    with Image.open(source_path) as opened:
        overlay = opened.convert("RGB")
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default()
    for candidate in candidate_payload.get("candidates", []):
        bbox = candidate["bbox"]
        center = candidate["center"]
        box = (bbox["left"], bbox["top"], bbox["right"] - 1, bbox["bottom"] - 1)
        draw.rectangle(box, outline=(235, 30, 30), width=2)
        draw.ellipse(
            (
                center["x"] - 3,
                center["y"] - 3,
                center["x"] + 3,
                center["y"] + 3,
            ),
            outline=(20, 100, 220),
            width=2,
        )
        draw.text(
            (max(0, bbox["left"]), max(0, bbox["top"] - 10)),
            candidate["id"],
            fill=(235, 30, 30),
            font=font,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(output_path)


def _global_measure_offset(out_dir: Path, *, slug: str, target_system_index: int) -> int:
    offset = 0
    systems_dir = out_dir / slug / "systems"
    for system_path in sorted(systems_dir.glob("system_[0-9][0-9][0-9].png")):
        system_index = int(system_path.stem.rsplit("_", 1)[1])
        if system_index >= target_system_index:
            break
        barlines = sorted(detect_barlines(system_path))
        offset += max(0, len(measure_boundaries_for_system(system_path, barlines)) - 1)
    return offset


def _selected_work_item(manifest_path: Path, slug: str) -> WorkItem:
    selected = [item for item in load_manifest_jsonl(manifest_path) if item.slug == slug]
    if len(selected) != 1:
        raise ValueError(f"Expected exactly one manifest item for {slug!r}, found {len(selected)}")
    return selected[0]


def _validated_destination(out_dir: Path, *, slug: str, namespace: str) -> Path:
    if not NAMESPACE_RE.fullmatch(namespace):
        raise ValueError(f"Unsafe training namespace: {namespace!r}")
    if namespace.lower() in PROTECTED_NAMESPACES:
        raise ValueError(f"Refusing protected training namespace: {namespace!r}")
    root = (out_dir / slug / "vlm_melody_training_inputs").resolve()
    destination = (root / namespace).resolve()
    if destination.parent != root:
        raise ValueError(f"Training namespace escaped its versioned root: {namespace!r}")
    return destination


def _validate_records(
    records: Sequence[Mapping[str, Any]], *, slug: str, system_index: int
) -> None:
    identities = [
        (
            record.get("slug"),
            int(record.get("system_index", -1)),
            int(record.get("system_measure_index", -1)),
        )
        for record in records
    ]
    expected = [(slug, system_index, index) for index in range(1, EXPECTED_MEASURE_COUNT + 1)]
    if identities != expected:
        raise ValueError(
            f"Corrected crop identities changed: expected {expected}, got {identities}"
        )


def _file_record(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Expected artifact not found: {resolved}")
    return {"path": _display_path(resolved), "sha256": _sha256(resolved)}


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
