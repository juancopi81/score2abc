"""Prepare and freeze a truth-blind third-score melody gate.

This spike-only command deliberately separates layout-only candidate selection
from prediction freezing. ``prepare`` may inspect only system PNGs. ``freeze``
requires explicit prediction, model, and training artifacts and snapshots them
before any target truth or MusicXML may be opened.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import shutil
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from score2abc.chord_ocr.alignment import (  # noqa: E402
    detect_barlines,
    measure_boundaries_for_system,
)

SCHEMA_VERSION = 1
SPLIT_NAME = "fresh_heldout"
OUTPUT_SUBDIR = "vlm_melody_third_score_heldout"
DEFAULT_NAMESPACE = "v1"
EVALUATOR_VERSION = "third-score-melody-gate-v1"


@dataclass(frozen=True)
class Candidate:
    slug: str
    system_index: int
    label: str


@dataclass(frozen=True)
class LayoutPolicy:
    min_width_px: int = 1200
    min_height_px: int = 100
    min_measure_count: int = 4
    max_measure_count: int = 12
    min_crop_width_px: int = 80
    max_spacing_cv: float = 0.35


@dataclass(frozen=True)
class HeldoutGateSpec:
    key: str
    output_subdir: str
    evaluator_version: str
    implementation_path: Path

    @property
    def selection_kind(self) -> str:
        return f"{self.key}_layout_only_selection"

    @property
    def prepare_kind(self) -> str:
        return f"{self.key}_fresh_heldout_prepare"

    @property
    def freeze_kind(self) -> str:
        return f"{self.key}_fresh_heldout_freeze"

    @property
    def sealed_kind(self) -> str:
        return f"{self.key}_fresh_heldout_sealed_manifest"


THIRD_SCORE_GATE = HeldoutGateSpec(
    key="third_score",
    output_subdir=OUTPUT_SUBDIR,
    evaluator_version=EVALUATOR_VERSION,
    implementation_path=Path(__file__),
)


DEFAULT_CANDIDATE_POOL = (
    Candidate(
        slug="jaime-llanos_64_la-chata_pasillo_luis-a-calvo",
        system_index=7,
        label="primary_layout_only",
    ),
    Candidate(
        slug="jaime-llanos_49_gatoe-fique_pasillo_emilio-murillo",
        system_index=4,
        label="fallback_layout_only",
    ),
    Candidate(
        slug="jaime-llanos_25_chispazo_pasillo_pedro-morales-pino",
        system_index=3,
        label="harder_fallback_layout_only",
    ),
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare", help="Select and seal layout inputs")
    prepare_parser.add_argument("out_dir", nargs="?", type=Path, default=Path("out"))
    prepare_parser.add_argument("--namespace", default=DEFAULT_NAMESPACE)

    freeze_parser = subparsers.add_parser("freeze", help="Snapshot predictions and model inputs")
    freeze_parser.add_argument("prepared_manifest", type=Path)
    freeze_parser.add_argument("--predictions", type=Path, required=True)
    freeze_parser.add_argument("--model-artifact", type=Path, action="append", required=True)
    freeze_parser.add_argument("--training-artifact", type=Path, action="append", required=True)

    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            report = prepare_third_score(args.out_dir, namespace=args.namespace)
            print(report["prepared_manifest"])
        else:
            report = freeze_prepared_third_score(
                args.prepared_manifest,
                predictions_path=args.predictions,
                model_artifact_paths=args.model_artifact,
                training_artifact_paths=args.training_artifact,
            )
            print(report["sealed_manifest"])
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def prepare_third_score(
    out_dir: Path,
    *,
    namespace: str = DEFAULT_NAMESPACE,
    candidate_pool: Sequence[Candidate] = DEFAULT_CANDIDATE_POOL,
    policy: LayoutPolicy | None = None,
) -> dict[str, Any]:
    """Prepare one layout-selected candidate without target-score truth access."""
    return prepare_heldout_score(
        out_dir,
        namespace=namespace,
        candidate_pool=candidate_pool,
        policy=policy,
        gate=THIRD_SCORE_GATE,
    )


def prepare_heldout_score(
    out_dir: Path,
    *,
    namespace: str,
    candidate_pool: Sequence[Candidate],
    gate: HeldoutGateSpec,
    policy: LayoutPolicy | None = None,
) -> dict[str, Any]:
    """Prepare one configurable, layout-selected truth-blind score gate."""
    policy = policy or LayoutPolicy()
    _validate_namespace(namespace)
    if not candidate_pool:
        raise ValueError("Candidate pool must not be empty")

    analyses = [
        _analyze_candidate(out_dir, candidate, policy, priority=index)
        for index, candidate in enumerate(candidate_pool, start=1)
    ]
    selected = next((analysis for analysis in analyses if analysis["policy_passed"]), None)
    if selected is None:
        blockers = "; ".join(
            f"{row['candidate']['slug']}/system_{row['candidate']['system_index']:03d}: "
            f"{', '.join(row['rejection_reasons'])}"
            for row in analyses
        )
        raise ValueError(f"No {gate.key} candidate passed the layout-only policy: {blockers}")

    slug = str(selected["candidate"]["slug"])
    system_index = int(selected["candidate"]["system_index"])
    namespace_root = out_dir / slug / gate.output_subdir / namespace / f"system_{system_index:03d}"
    if namespace_root.exists():
        raise ValueError(f"Prepared {gate.key} namespace already exists: {namespace_root}")
    namespace_root.mkdir(parents=True, exist_ok=False)

    source_path = out_dir / str(selected["source_system_path"])
    crop_records = _write_measure_crops(
        source_path,
        namespace_root / "crops",
        selected["cleaned_boundaries"],
    )
    requests = _build_requests(
        slug=slug,
        system_index=system_index,
        namespace=namespace,
        source_sha256=str(selected["source_system_sha256"]),
        cleaned_boundaries=selected["cleaned_boundaries"],
        crops=crop_records,
    )
    requests_path = namespace_root / "requests.jsonl"
    _write_jsonl(requests_path, requests)
    request_hashes = [_hash_json(row) for row in requests]

    forbidden_truth_paths = _forbidden_truth_paths(slug)
    evaluator_path = namespace_root / "evaluator_spec.json"
    evaluator_spec = {
        "schema_version": SCHEMA_VERSION,
        "version": gate.evaluator_version,
        "split": SPLIT_NAME,
        "status": "preregistered_before_prediction_and_truth",
        "truth_gate": "canonical truth and physical-measure mapping may be added only after freeze",
        "required_checks": [
            "verify prepared_manifest and freeze hashes before loading truth",
            "preserve one-to-many automatic-crop to physical-measure mappings",
            "report ordered pitch, note, onset, duration, rest, and exact-measure metrics",
        ],
        "forbidden_before_freeze": forbidden_truth_paths,
    }
    _write_json(evaluator_path, evaluator_spec)

    selection_path = namespace_root / "selection.json"
    selection_payload = {
        "schema_version": SCHEMA_VERSION,
        "kind": gate.selection_kind,
        "status": "selected_before_prediction_and_truth",
        "split": SPLIT_NAME,
        "truth_accessed": False,
        "namespace": namespace,
        "candidate_pool": analyses,
        "policy": asdict(policy),
        "selected": {
            **selected,
            "crop_count": len(crop_records),
            "crops": crop_records,
            "crop_manifest_sha256": _hash_json(crop_records),
        },
        "forbidden_truth_paths": forbidden_truth_paths,
    }
    _write_json(selection_path, selection_payload)

    prepared_path = namespace_root / "prepared_manifest.json"
    prepared_payload = {
        "schema_version": SCHEMA_VERSION,
        "kind": gate.prepare_kind,
        "status": "prepared_awaiting_model_predictions",
        "split": SPLIT_NAME,
        "truth_accessed": False,
        "truth_must_be_added_only_after_freeze": True,
        "namespace": namespace,
        "target": {"slug": slug, "system_index": system_index},
        "forbidden_truth_paths": forbidden_truth_paths,
        "artifacts": {
            "selection": {"path": "selection.json", "sha256": _sha256(selection_path)},
            "requests": {
                "path": "requests.jsonl",
                "sha256": _sha256(requests_path),
                "row_sha256": request_hashes,
            },
            "evaluator": {
                "version": gate.evaluator_version,
                "path": "evaluator_spec.json",
                "sha256": _sha256(evaluator_path),
                "implementation_path": _repo_display_path(gate.implementation_path),
                "implementation_sha256": _sha256(gate.implementation_path),
            },
            "source_system": {
                "path_relative_to_out": str(selected["source_system_path"]),
                "sha256": str(selected["source_system_sha256"]),
            },
            "crops": crop_records,
        },
    }
    _write_json(prepared_path, prepared_payload)
    return {
        "prepared_manifest": str(prepared_path),
        "prepared_manifest_sha256": _sha256(prepared_path),
        "selection": str(selection_path),
        "selection_sha256": _sha256(selection_path),
        "requests_sha256": _sha256(requests_path),
        "evaluator_sha256": _sha256(evaluator_path),
        "target": prepared_payload["target"],
    }


def freeze_prepared_third_score(
    prepared_manifest_path: Path,
    *,
    predictions_path: Path,
    model_artifact_paths: Sequence[Path],
    training_artifact_paths: Sequence[Path],
) -> dict[str, Any]:
    """Snapshot explicit inference artifacts into an immutable frozen namespace."""
    return freeze_prepared_heldout_score(
        prepared_manifest_path,
        predictions_path=predictions_path,
        model_artifact_paths=model_artifact_paths,
        training_artifact_paths=training_artifact_paths,
        gate=THIRD_SCORE_GATE,
    )


def freeze_prepared_heldout_score(
    prepared_manifest_path: Path,
    *,
    predictions_path: Path,
    model_artifact_paths: Sequence[Path],
    training_artifact_paths: Sequence[Path],
    gate: HeldoutGateSpec,
) -> dict[str, Any]:
    """Snapshot inference artifacts for a configurable held-out score gate."""
    if not model_artifact_paths:
        raise ValueError("At least one model artifact is required")
    if not training_artifact_paths:
        raise ValueError("At least one training artifact is required")

    namespace_root = prepared_manifest_path.parent
    frozen_dir = namespace_root / "frozen"
    if frozen_dir.exists():
        raise ValueError(
            f"{gate.key} freeze already exists and cannot be overwritten: {frozen_dir}"
        )

    prepared = _read_json(prepared_manifest_path)
    _verify_prepared_manifest(
        namespace_root,
        prepared_manifest_path,
        prepared,
        expected_kind=gate.prepare_kind,
    )
    all_inputs = [predictions_path, *model_artifact_paths, *training_artifact_paths]
    target_slug = str(prepared["target"]["slug"])
    for path in all_inputs:
        _validate_external_artifact(path, target_slug=target_slug)

    frozen_dir.mkdir(parents=False, exist_ok=False)
    prediction_pin = _snapshot_artifact(
        predictions_path,
        frozen_dir=frozen_dir,
        role="predictions",
        index=1,
    )
    model_pins = [
        _snapshot_artifact(path, frozen_dir=frozen_dir, role="model", index=index)
        for index, path in enumerate(model_artifact_paths, start=1)
    ]
    training_pins = [
        _snapshot_artifact(path, frozen_dir=frozen_dir, role="training", index=index)
        for index, path in enumerate(training_artifact_paths, start=1)
    ]

    freeze_path = frozen_dir / "freeze.json"
    freeze_payload = {
        "schema_version": SCHEMA_VERSION,
        "kind": gate.freeze_kind,
        "status": "frozen_awaiting_truth",
        "split": SPLIT_NAME,
        "truth_accessed": False,
        "truth_must_be_added_only_after_freeze": True,
        "target": prepared["target"],
        "namespace": prepared["namespace"],
        "prepared_manifest": {
            "path": "../prepared_manifest.json",
            "sha256": _sha256(prepared_manifest_path),
        },
        "selection_sha256": prepared["artifacts"]["selection"]["sha256"],
        "requests": prepared["artifacts"]["requests"],
        "evaluator": prepared["artifacts"]["evaluator"],
        "forbidden_truth_paths": prepared["forbidden_truth_paths"],
        "predictions": prediction_pin,
        "model_artifacts": model_pins,
        "training_artifacts": training_pins,
    }
    _write_json(freeze_path, freeze_payload)

    sealed_path = frozen_dir / "sealed_manifest.json"
    sealed_payload = {
        "schema_version": SCHEMA_VERSION,
        "kind": gate.sealed_kind,
        "status": "frozen_awaiting_truth",
        "split": SPLIT_NAME,
        "truth_accessed": False,
        "target": prepared["target"],
        "freeze": {"path": "freeze.json", "sha256": _sha256(freeze_path)},
        "prepared_manifest_sha256": _sha256(prepared_manifest_path),
        "next_gate": "verify all hashes, then materialize canonical truth exactly once",
    }
    _write_json(sealed_path, sealed_payload)
    return {
        "freeze": str(freeze_path),
        "freeze_sha256": _sha256(freeze_path),
        "sealed_manifest": str(sealed_path),
        "sealed_manifest_sha256": _sha256(sealed_path),
    }


def _analyze_candidate(
    out_dir: Path,
    candidate: Candidate,
    policy: LayoutPolicy,
    *,
    priority: int,
) -> dict[str, Any]:
    source_relative = Path(candidate.slug) / "systems" / f"system_{candidate.system_index:03d}.png"
    source_path = out_dir / source_relative
    result: dict[str, Any] = {
        "priority": priority,
        "candidate": asdict(candidate),
        "source_system_path": source_relative.as_posix(),
        "exists": source_path.is_file(),
        "policy_passed": False,
        "rejection_reasons": [],
    }
    if not source_path.is_file():
        result["rejection_reasons"] = ["source system PNG missing"]
        return result

    with Image.open(source_path) as image:
        width, height = image.size
    detected = _stable_boundaries(detect_barlines(source_path))
    cleaned = _stable_boundaries(measure_boundaries_for_system(source_path, detected))
    crop_widths = _crop_widths(width, cleaned)
    spacing_cv = _coefficient_of_variation(crop_widths)
    reasons = []
    if width < policy.min_width_px:
        reasons.append(f"width {width} < {policy.min_width_px}")
    if height < policy.min_height_px:
        reasons.append(f"height {height} < {policy.min_height_px}")
    if not policy.min_measure_count <= len(crop_widths) <= policy.max_measure_count:
        reasons.append(
            f"measure count {len(crop_widths)} outside "
            f"[{policy.min_measure_count}, {policy.max_measure_count}]"
        )
    if crop_widths and min(crop_widths) < policy.min_crop_width_px:
        reasons.append(f"minimum crop width {min(crop_widths)} < {policy.min_crop_width_px}")
    if spacing_cv > policy.max_spacing_cv:
        reasons.append(f"spacing CV {spacing_cv:.6f} > {policy.max_spacing_cv:.6f}")

    result.update(
        {
            "dimensions_px": {"width": width, "height": height},
            "source_system_sha256": _sha256(source_path),
            "detected_barlines": detected,
            "detected_barlines_sha256": _hash_json(detected),
            "cleaned_boundaries": cleaned,
            "cleaned_boundaries_sha256": _hash_json(cleaned),
            "crop_widths_px": crop_widths,
            "spacing_cv": round(spacing_cv, 9),
            "policy_passed": not reasons,
            "rejection_reasons": reasons,
        }
    )
    return result


def _write_measure_crops(
    source_path: Path,
    crops_dir: Path,
    boundaries: Sequence[float],
) -> list[dict[str, Any]]:
    crops_dir.mkdir(parents=False, exist_ok=False)
    records = []
    with Image.open(source_path) as source:
        image = source.convert("RGB")
        width, height = image.size
        x_boundaries = _boundary_pixels(width, boundaries)
        for measure_index, (left, right) in enumerate(
            zip(x_boundaries, x_boundaries[1:], strict=False), start=1
        ):
            crop = image.crop((left, 0, right, height))
            buffer = io.BytesIO()
            crop.save(buffer, format="PNG")
            data = buffer.getvalue()
            relative_path = Path("crops") / f"measure_{measure_index:03d}.png"
            output_path = crops_dir.parent / relative_path
            output_path.write_bytes(data)
            records.append(
                {
                    "measure_index": measure_index,
                    "path": relative_path.as_posix(),
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "bbox_px": [left, 0, right, height],
                    "dimensions_px": {"width": right - left, "height": height},
                }
            )
    return records


def _build_requests(
    *,
    slug: str,
    system_index: int,
    namespace: str,
    source_sha256: str,
    cleaned_boundaries: Sequence[float],
    crops: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "schema_version": SCHEMA_VERSION,
            "split": SPLIT_NAME,
            "truth_accessed": False,
            "identity": {
                "slug": slug,
                "system_index": system_index,
                "automatic_measure_index": int(crop["measure_index"]),
            },
            "input": {
                "path_relative_to_namespace": crop["path"],
                "sha256": crop["sha256"],
                "bbox_px": crop["bbox_px"],
            },
            "layout_provenance": {
                "namespace": namespace,
                "source_system_sha256": source_sha256,
                "left_boundary": cleaned_boundaries[index],
                "right_boundary": cleaned_boundaries[index + 1],
            },
        }
        for index, crop in enumerate(crops)
    ]


def _verify_prepared_manifest(
    namespace_root: Path,
    manifest_path: Path,
    payload: Mapping[str, Any],
    *,
    expected_kind: str = THIRD_SCORE_GATE.prepare_kind,
) -> None:
    if payload.get("kind") != expected_kind:
        raise ValueError(f"Prepared manifest kind mismatch: {manifest_path}")
    if payload.get("status") != "prepared_awaiting_model_predictions":
        raise ValueError("Prepared manifest is not awaiting model predictions")
    if payload.get("split") != SPLIT_NAME or payload.get("truth_accessed") is not False:
        raise ValueError("Prepared manifest is not a truth-blind fresh_heldout artifact")
    artifacts = payload["artifacts"]
    for name in ("selection", "requests", "evaluator"):
        record = artifacts[name]
        path = namespace_root / str(record["path"])
        if _sha256(path) != str(record["sha256"]):
            raise ValueError(f"Prepared {name} hash drift: {path}")
    for name, record in artifacts.get("context", {}).items():
        path = namespace_root / str(record["path"])
        if _sha256(path) != str(record["sha256"]):
            raise ValueError(f"Prepared context artifact hash drift ({name}): {path}")
    source = artifacts["source_system"]
    source_path = _find_out_dir(namespace_root) / str(source["path_relative_to_out"])
    if _sha256(source_path) != str(source["sha256"]):
        raise ValueError(f"Prepared source-system hash drift: {source_path}")
    for crop in artifacts["crops"]:
        crop_path = namespace_root / str(crop["path"])
        if _sha256(crop_path) != str(crop["sha256"]):
            raise ValueError(f"Prepared crop hash drift: {crop_path}")


def _find_out_dir(namespace_root: Path) -> Path:
    # <out>/<slug>/<OUTPUT_SUBDIR>/<version>/system_NNN
    try:
        return namespace_root.parents[3]
    except IndexError as exc:
        raise ValueError(f"Unexpected third-score namespace path: {namespace_root}") from exc


def _snapshot_artifact(
    source: Path,
    *,
    frozen_dir: Path,
    role: str,
    index: int,
) -> dict[str, Any]:
    role_dir = frozen_dir / "artifacts" / role
    role_dir.mkdir(parents=True, exist_ok=True)
    snapshot = role_dir / f"{index:03d}_{source.name}"
    shutil.copyfile(source, snapshot)
    source_hash = _sha256(source)
    snapshot_hash = _sha256(snapshot)
    if source_hash != snapshot_hash:
        raise ValueError(f"Snapshot hash mismatch for {source}")
    return {
        "source_path": _repo_display_path(source),
        "source_sha256": source_hash,
        "snapshot_path_relative_to_namespace": snapshot.relative_to(frozen_dir.parent).as_posix(),
        "snapshot_sha256": snapshot_hash,
    }


def _validate_external_artifact(path: Path, *, target_slug: str) -> None:
    if not path.is_file():
        raise ValueError(f"Freeze artifact must be an existing file: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"Freeze artifact must not be empty: {path}")
    slug = target_slug.casefold()
    lexical_parts = tuple(part.casefold() for part in path.absolute().parts)
    resolved_parts = tuple(part.casefold() for part in path.resolve().parts)
    if any(
        _is_forbidden_target_truth_path(parts, target_slug=slug)
        for parts in (lexical_parts, resolved_parts)
    ):
        raise ValueError(f"Target truth/MusicXML paths are forbidden before freeze: {path}")


def _is_forbidden_target_truth_path(parts: tuple[str, ...], *, target_slug: str) -> bool:
    slug = target_slug.casefold()
    dataset_truth_names = {f"{slug}.json"}
    dataset_musicxml_names = {f"{slug}{suffix}" for suffix in (".musicxml", ".xml", ".mxl")}
    for index in range(len(parts) - 2):
        parent_pair = parts[index : index + 2]
        filename = parts[index + 2]
        if parent_pair == ("dataset", "ground_truth") and filename in dataset_truth_names:
            return True
        if parent_pair == ("dataset", "musicxml") and filename in dataset_musicxml_names:
            return True
        if parent_pair == ("out", slug):
            target_tail = parts[index + 2 :]
            if any("truth" in part for part in target_tail) or target_tail[-1] == "musicxml.xml":
                return True
    return False


def _forbidden_truth_paths(slug: str) -> list[str]:
    return [
        f"dataset/ground_truth/{slug}.json",
        f"dataset/musicxml/{slug}.musicxml",
        f"dataset/musicxml/{slug}.xml",
        f"dataset/musicxml/{slug}.mxl",
        f"out/{slug}/**/*truth*",
        f"out/{slug}/**/musicxml.xml",
    ]


def _crop_widths(width: int, boundaries: Sequence[float]) -> list[int]:
    pixels = _boundary_pixels(width, boundaries)
    return [right - left for left, right in zip(pixels, pixels[1:], strict=False)]


def _boundary_pixels(width: int, boundaries: Sequence[float]) -> list[int]:
    pixels = [min(width, max(0, int(round(boundary * width)))) for boundary in boundaries]
    if len(pixels) < 2 or any(
        right <= left for left, right in zip(pixels, pixels[1:], strict=False)
    ):
        raise ValueError(f"Invalid measure boundaries: {boundaries}")
    return pixels


def _coefficient_of_variation(values: Sequence[int]) -> float:
    if len(values) < 2:
        return 0.0
    mean = statistics.fmean(values)
    return statistics.pstdev(values) / mean if mean else float("inf")


def _stable_boundaries(values: Iterable[float]) -> list[float]:
    return [round(float(value), 9) for value in values]


def _validate_namespace(namespace: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", namespace):
        raise ValueError(f"Invalid versioned namespace: {namespace!r}")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"Prepared manifest does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def _hash_json(payload: Any) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(resolved)


if __name__ == "__main__":
    raise SystemExit(main())
