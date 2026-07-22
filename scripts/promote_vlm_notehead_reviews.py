"""Promote saved human notehead reviews into portable spike-only fixtures.

The reviewer writes machine-local source paths and hidden post-save evaluation
metrics under ``out/<slug>/vlm_melody_reviews``. This utility validates that the
review still points at the exact image and candidate artifact that were reviewed,
then writes a deterministic, GT-free fixture for later spike experiments.

Example:
    uv run python scripts/promote_vlm_notehead_reviews.py out \
        --slug jaime-llanos_12_aviador_pasillo_fulgencio-garcia \
        --system 1 --measure 1 --measure 2
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DESTINATION = REPO_ROOT / "tests/fixtures/vlm_melody/notehead_reviews"
REVIEW_KIND = "vlm_melody_notehead_candidate_review"
REVIEW_TOOL_PATH = "scripts/review_vlm_notehead_candidates.py"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SYSTEM_DIR_RE = re.compile(r"^system_(\d+)$")
MEASURE_DIR_RE = re.compile(r"^measure_(\d+)$")
HIDDEN_EVALUATION_KEYS = {
    "evaluation",
    "ground_truth_path",
    "ground_truth_paths",
    "metrics",
}


@dataclass(frozen=True)
class SavedReview:
    path: Path
    slug: str
    system_index: int
    measure_index: int

    @property
    def fixture_name(self) -> str:
        return f"{self.slug}_system_{self.system_index:03d}_measure_{self.measure_index:03d}.json"


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        promoted = promote_reviews(
            args.out_dir,
            destination=args.destination,
            selected_slugs=set(args.slug) if args.slug else None,
            selected_systems=set(args.system) if args.system else None,
            selected_measures=set(args.measure) if args.measure else None,
            force=args.force,
            repo_root=REPO_ROOT,
        )
    except (FileExistsError, FileNotFoundError, OSError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    for path in promoted:
        print(f"Promoted: {path}")
    print(f"Promoted {len(promoted)} saved notehead review(s).")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", type=Path, help="Pipeline output directory.")
    parser.add_argument("--slug", action="append", help="Limit to a work slug; repeatable.")
    parser.add_argument(
        "--system",
        action="append",
        type=_positive_index,
        help="Limit to a 1-based system index; repeatable.",
    )
    parser.add_argument(
        "--measure",
        action="append",
        type=_positive_index,
        help="Limit to a 1-based system-local measure index; repeatable.",
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=DEFAULT_DESTINATION,
        help=("Fixture destination. Defaults to " "tests/fixtures/vlm_melody/notehead_reviews."),
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing fixtures.")
    return parser


def promote_reviews(
    out_dir: Path,
    *,
    destination: Path = DEFAULT_DESTINATION,
    selected_slugs: Iterable[str] | None = None,
    selected_systems: Iterable[int] | None = None,
    selected_measures: Iterable[int] | None = None,
    force: bool = False,
    repo_root: Path | None = None,
) -> list[Path]:
    """Validate and promote all selected saved reviews as one preflighted batch."""
    root = (repo_root or REPO_ROOT).resolve()
    output_root = out_dir.resolve()
    fixture_root = destination.resolve()
    reviews = discover_saved_reviews(
        output_root,
        selected_slugs=set(selected_slugs) if selected_slugs is not None else None,
        selected_systems=set(selected_systems) if selected_systems is not None else None,
        selected_measures=set(selected_measures) if selected_measures is not None else None,
    )
    plans = [(review, fixture_root / review.fixture_name) for review in reviews]
    _validate_unique_targets(plans)

    existing = [target for _, target in plans if target.exists()]
    if existing and not force:
        raise FileExistsError(
            f"Refusing to overwrite existing fixture: {existing[0]}. Rerun with --force."
        )

    serialized = [
        (
            target,
            _serialize_fixture(
                _build_portable_fixture(review, out_dir=output_root, repo_root=root)
            ),
        )
        for review, target in plans
    ]

    fixture_root.mkdir(parents=True, exist_ok=True)
    for target, content in serialized:
        if force:
            target.write_bytes(content)
        else:
            with target.open("xb") as handle:
                handle.write(content)
    return [target for target, _ in serialized]


def discover_saved_reviews(
    out_dir: Path,
    *,
    selected_slugs: set[str] | None = None,
    selected_systems: set[int] | None = None,
    selected_measures: set[int] | None = None,
) -> list[SavedReview]:
    """Find saved review files and apply independent slug/system/measure selectors."""
    if not out_dir.is_dir():
        raise FileNotFoundError(f"Pipeline output directory not found: {out_dir}")

    reviews = []
    pattern = "*/vlm_melody_reviews/system_*/measure_*/review.json"
    for path in sorted(out_dir.glob(pattern)):
        review = _saved_review_from_path(path, out_dir)
        if selected_slugs is not None and review.slug not in selected_slugs:
            continue
        if selected_systems is not None and review.system_index not in selected_systems:
            continue
        if selected_measures is not None and review.measure_index not in selected_measures:
            continue
        reviews.append(review)

    if not reviews:
        selectors = _selector_description(
            selected_slugs=selected_slugs,
            selected_systems=selected_systems,
            selected_measures=selected_measures,
        )
        raise FileNotFoundError(f"No saved notehead reviews matched under {out_dir}{selectors}.")
    return reviews


def build_portable_fixture(
    review_path: Path,
    out_dir: Path,
    *,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Build one validated fixture payload without writing it."""
    output_root = out_dir.resolve()
    review = _saved_review_from_path(review_path.resolve(), output_root)
    return _build_portable_fixture(
        review,
        out_dir=output_root,
        repo_root=(repo_root or REPO_ROOT).resolve(),
    )


def _build_portable_fixture(
    review_record: SavedReview,
    *,
    out_dir: Path,
    repo_root: Path,
) -> dict[str, Any]:
    review = _load_json_object(review_record.path, "Saved review")
    schema_version = _required_int(review, "schema_version", "Saved review")
    kind = _required_string(review, "kind", "Saved review")
    if kind != REVIEW_KIND:
        raise ValueError(f"Unexpected saved review kind at {review_record.path}: {kind!r}")

    identity_payload = _required_object(review, "identity", "Saved review")
    identity = {
        "slug": _required_string(identity_payload, "slug", "Saved review identity"),
        "system_index": _required_int(identity_payload, "system_index", "Saved review identity"),
        "system_measure_index": _required_int(
            identity_payload, "system_measure_index", "Saved review identity"
        ),
        "global_measure_index": _required_int(
            identity_payload, "global_measure_index", "Saved review identity"
        ),
    }
    expected_identity = (
        review_record.slug,
        review_record.system_index,
        review_record.measure_index,
    )
    actual_identity = (
        identity["slug"],
        identity["system_index"],
        identity["system_measure_index"],
    )
    if actual_identity != expected_identity:
        raise ValueError(
            f"Saved review identity does not match its path at {review_record.path}: "
            f"expected {expected_identity}, got {actual_identity}"
        )

    source_payload = _required_object(review, "source", "Saved review")
    image_path = _resolve_recorded_path(
        _required_string(source_payload, "image_path", "Saved review source"),
        out_dir=out_dir,
        repo_root=repo_root,
    )
    candidate_path = _resolve_recorded_path(
        _required_string(
            source_payload,
            "candidate_artifact_path",
            "Saved review source",
        ),
        out_dir=out_dir,
        repo_root=repo_root,
    )
    image_sha256 = _required_string(source_payload, "image_sha256", "Saved review source")
    candidate_sha256 = _required_string(
        source_payload,
        "candidate_artifact_sha256",
        "Saved review source",
    )
    _validate_sha256(image_path, image_sha256, "Source image")
    _validate_sha256(candidate_path, candidate_sha256, "Candidate artifact")

    candidates = _required_list(review, "candidates", "Saved review")
    _validate_candidates(candidates, review_record.path)
    manual_noteheads = _required_list(review, "manual_noteheads", "Saved review")
    final_noteheads = _required_list(review, "final_noteheads", "Saved review")
    _validate_noteheads(manual_noteheads, "manual_noteheads", review_record.path)
    _validate_noteheads(final_noteheads, "final_noteheads", review_record.path)
    _validate_review_graph(candidates, manual_noteheads, final_noteheads, review_record.path)
    timing = _required_object(review, "timing", "Saved review")
    _validate_timing(timing, review_record.path)

    fixture = {
        "schema_version": schema_version,
        "kind": kind,
        "identity": identity,
        "source": {
            "image_path": _repo_relative_path(image_path, repo_root, "Source image"),
            "image_sha256": image_sha256,
            "candidate_artifact_path": _repo_relative_path(
                candidate_path, repo_root, "Candidate artifact"
            ),
            "candidate_artifact_sha256": candidate_sha256,
            "candidate_cap": _required_int(source_payload, "candidate_cap", "Saved review source"),
            "coordinate_space": _required_string(
                source_payload, "coordinate_space", "Saved review source"
            ),
        },
        "candidates": _without_hidden_evaluation(candidates),
        "manual_noteheads": _without_hidden_evaluation(manual_noteheads),
        "final_noteheads": _without_hidden_evaluation(final_noteheads),
        "timing": _without_hidden_evaluation(timing),
        "provenance": {
            "review_type": "human",
            "scope": "spike_only",
            "review_tool_path": REVIEW_TOOL_PATH,
            "source_review_path": _repo_relative_path(
                review_record.path, repo_root, "Saved review"
            ),
        },
    }
    _assert_no_absolute_paths(fixture)
    return fixture


def _saved_review_from_path(path: Path, out_dir: Path) -> SavedReview:
    try:
        relative = path.relative_to(out_dir)
    except ValueError as exc:
        raise ValueError(f"Saved review is outside the output directory: {path}") from exc
    if len(relative.parts) != 5 or relative.parts[1] != "vlm_melody_reviews":
        raise ValueError(f"Unexpected saved review path: {path}")
    system_match = SYSTEM_DIR_RE.fullmatch(relative.parts[2])
    measure_match = MEASURE_DIR_RE.fullmatch(relative.parts[3])
    if not system_match or not measure_match or relative.parts[4] != "review.json":
        raise ValueError(f"Unexpected saved review path: {path}")
    return SavedReview(
        path=path,
        slug=relative.parts[0],
        system_index=int(system_match.group(1)),
        measure_index=int(measure_match.group(1)),
    )


def _resolve_recorded_path(value: str, *, out_dir: Path, repo_root: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    candidates = (repo_root / path, out_dir / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def _validate_sha256(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    if not SHA256_RE.fullmatch(expected):
        raise ValueError(f"{label} SHA256 is invalid in the saved review: {expected!r}")
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(f"{label} SHA256 is stale for {path}: expected {expected}, got {actual}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_relative_path(path: Path, repo_root: Path, label: str) -> str:
    try:
        relative = path.resolve().relative_to(repo_root.resolve())
    except ValueError as exc:
        raise ValueError(
            f"{label} is outside the repository and cannot be made portable: {path}"
        ) from exc
    return relative.as_posix()


def _validate_candidates(candidates: list[Any], review_path: Path) -> None:
    if not candidates:
        raise ValueError(f"Saved review has no candidate labels: {review_path}")
    seen_ids = set()
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            raise ValueError(f"candidates[{index}] must be an object at {review_path}")
        candidate_id = candidate.get("id")
        if not isinstance(candidate_id, str) or not candidate_id:
            raise ValueError(f"candidates[{index}].id must be a non-empty string at {review_path}")
        if candidate_id in seen_ids:
            raise ValueError(f"Duplicate candidate id {candidate_id!r} at {review_path}")
        seen_ids.add(candidate_id)
        if candidate.get("label") not in {"accepted", "rejected"}:
            raise ValueError(
                f"candidates[{index}].label must be accepted or rejected at {review_path}"
            )


def _validate_noteheads(noteheads: list[Any], field: str, review_path: Path) -> None:
    for index, notehead in enumerate(noteheads):
        if not isinstance(notehead, dict):
            raise ValueError(f"{field}[{index}] must be an object at {review_path}")
        auto_pitch = notehead.get("auto_pitch")
        pitch = notehead.get("pitch")
        corrected = notehead.get("pitch_corrected")
        if not isinstance(auto_pitch, str) or not auto_pitch:
            raise ValueError(f"{field}[{index}].auto_pitch is required at {review_path}")
        if not isinstance(pitch, str) or not pitch:
            raise ValueError(f"{field}[{index}].pitch is required at {review_path}")
        if not isinstance(corrected, bool) or corrected != (pitch != auto_pitch):
            raise ValueError(f"{field}[{index}].pitch_corrected is inconsistent at {review_path}")


def _validate_review_graph(
    candidates: list[Any],
    manual_noteheads: list[Any],
    final_noteheads: list[Any],
    review_path: Path,
) -> None:
    candidate_by_id = {candidate["id"]: candidate for candidate in candidates}
    manual_by_id: dict[str, dict[str, Any]] = {}
    for index, manual in enumerate(manual_noteheads):
        manual_id = manual.get("id")
        if not isinstance(manual_id, str) or not manual_id:
            raise ValueError(
                f"manual_noteheads[{index}].id must be a non-empty string at {review_path}"
            )
        if manual_id in manual_by_id:
            raise ValueError(f"Duplicate manual notehead id {manual_id!r} at {review_path}")
        manual_by_id[manual_id] = manual

    seen_orders: set[int] = set()
    seen_references: set[tuple[str, str]] = set()
    referenced_candidates: set[str] = set()
    referenced_manual_heads: set[str] = set()
    for index, final in enumerate(final_noteheads):
        order = final.get("order")
        if isinstance(order, bool) or not isinstance(order, int) or order < 1:
            raise ValueError(
                f"final_noteheads[{index}].order must be a positive integer at {review_path}"
            )
        if order in seen_orders:
            raise ValueError(f"Duplicate final notehead order {order} at {review_path}")
        seen_orders.add(order)

        source = final.get("source")
        if not isinstance(source, dict):
            raise ValueError(f"final_noteheads[{index}].source must be an object at {review_path}")
        kind = source.get("kind")
        if kind == "candidate":
            if "manual_id" in source:
                raise ValueError(
                    f"final_noteheads[{index}].source has contradictory manual_id at {review_path}"
                )
            source_id = source.get("candidate_id")
            if not isinstance(source_id, str) or not source_id:
                raise ValueError(
                    f"final_noteheads[{index}].source.candidate_id is required at {review_path}"
                )
            candidate = candidate_by_id.get(source_id)
            if candidate is None:
                raise ValueError(
                    f"final_noteheads[{index}] references unknown candidate {source_id!r} "
                    f"at {review_path}"
                )
            if candidate["label"] != "accepted":
                raise ValueError(
                    f"final_noteheads[{index}] references rejected candidate {source_id!r} "
                    f"at {review_path}"
                )
            _validate_source_snapshot(final, candidate, index, "candidate", review_path)
            referenced_candidates.add(source_id)
        elif kind == "manual":
            if "candidate_id" in source:
                raise ValueError(
                    f"final_noteheads[{index}].source has contradictory candidate_id "
                    f"at {review_path}"
                )
            source_id = source.get("manual_id")
            if not isinstance(source_id, str) or not source_id:
                raise ValueError(
                    f"final_noteheads[{index}].source.manual_id is required at {review_path}"
                )
            manual = manual_by_id.get(source_id)
            if manual is None:
                raise ValueError(
                    f"final_noteheads[{index}] references unknown manual notehead {source_id!r} "
                    f"at {review_path}"
                )
            _validate_source_snapshot(final, manual, index, "manual notehead", review_path)
            if final["pitch"] != manual["pitch"]:
                raise ValueError(
                    f"final_noteheads[{index}].pitch contradicts manual notehead {source_id!r} "
                    f"at {review_path}"
                )
            referenced_manual_heads.add(source_id)
        else:
            raise ValueError(
                f"final_noteheads[{index}].source.kind must be candidate or manual "
                f"at {review_path}"
            )

        reference = (kind, source_id)
        if reference in seen_references:
            raise ValueError(
                f"Duplicate final notehead source reference {kind}:{source_id} at {review_path}"
            )
        seen_references.add(reference)

    accepted_candidates = {
        candidate_id
        for candidate_id, candidate in candidate_by_id.items()
        if candidate["label"] == "accepted"
    }
    if referenced_candidates != accepted_candidates:
        missing = sorted(accepted_candidates - referenced_candidates)
        raise ValueError(
            f"Accepted candidates missing from final_noteheads: {missing} at {review_path}"
        )
    if referenced_manual_heads != set(manual_by_id):
        missing = sorted(set(manual_by_id) - referenced_manual_heads)
        raise ValueError(
            f"Manual noteheads missing from final_noteheads: {missing} at {review_path}"
        )


def _validate_source_snapshot(
    final: dict[str, Any],
    source: dict[str, Any],
    final_index: int,
    source_label: str,
    review_path: Path,
) -> None:
    for field in ("center", "auto_pitch"):
        if final.get(field) != source.get(field):
            raise ValueError(
                f"final_noteheads[{final_index}].{field} contradicts its {source_label} "
                f"source at {review_path}"
            )


def _validate_timing(timing: dict[str, Any], review_path: Path) -> None:
    active_ms = timing.get("active_review_ms")
    if (
        isinstance(active_ms, bool)
        or not isinstance(active_ms, (int, float))
        or not math.isfinite(float(active_ms))
        or active_ms < 0
    ):
        raise ValueError(f"timing.active_review_ms must be non-negative at {review_path}")
    saved_at = timing.get("saved_at")
    if not isinstance(saved_at, str) or not saved_at:
        raise ValueError(f"timing.saved_at is required at {review_path}")
    timeout_ms = timing.get("inactivity_timeout_ms")
    if isinstance(timeout_ms, bool) or not isinstance(timeout_ms, int) or timeout_ms < 0:
        raise ValueError(f"timing.inactivity_timeout_ms must be non-negative at {review_path}")


def _without_hidden_evaluation(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _without_hidden_evaluation(item)
            for key, item in value.items()
            if key not in HIDDEN_EVALUATION_KEYS
        }
    if isinstance(value, list):
        return [_without_hidden_evaluation(item) for item in value]
    return value


def _assert_no_absolute_paths(value: Any, location: str = "fixture") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            child_location = f"{location}.{key}"
            if key.endswith("_path") and isinstance(item, str) and Path(item).is_absolute():
                raise ValueError(f"Portable {child_location} must be repository-relative")
            _assert_no_absolute_paths(item, child_location)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_no_absolute_paths(item, f"{location}[{index}]")


def _validate_unique_targets(plans: list[tuple[SavedReview, Path]]) -> None:
    seen: set[Path] = set()
    for _, target in plans:
        if target in seen:
            raise ValueError(f"Multiple saved reviews map to the same fixture: {target}")
        seen.add(target)


def _selector_description(
    *,
    selected_slugs: set[str] | None,
    selected_systems: set[int] | None,
    selected_measures: set[int] | None,
) -> str:
    parts = []
    if selected_slugs is not None:
        parts.append(f"slugs={sorted(selected_slugs)}")
    if selected_systems is not None:
        parts.append(f"systems={sorted(selected_systems)}")
    if selected_measures is not None:
        parts.append(f"measures={sorted(selected_measures)}")
    return f" for {', '.join(parts)}" if parts else ""


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return payload


def _required_object(payload: dict[str, Any], key: str, label: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{label}.{key} must be an object")
    return value


def _required_list(payload: dict[str, Any], key: str, label: str) -> list[Any]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise ValueError(f"{label}.{key} must be an array")
    return value


def _required_string(payload: dict[str, Any], key: str, label: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label}.{key} must be a non-empty string")
    return value


def _required_int(payload: dict[str, Any], key: str, label: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label}.{key} must be an integer")
    return value


def _serialize_fixture(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n").encode(
        "utf-8"
    )


def _positive_index(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
