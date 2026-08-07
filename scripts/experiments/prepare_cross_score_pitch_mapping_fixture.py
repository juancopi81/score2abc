"""Freeze portable consumed inputs for the cross-score pitch-mapping spike."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DESTINATION = REPO_ROOT / "tests/fixtures/vlm_melody/cross_score_pitch_mapping"
FIXTURE_KIND = "consumed_cross_score_pitch_mapping_fixture"


@dataclass(frozen=True)
class ScoreSource:
    score_id: str
    slug: str
    system_index: int
    inference: str
    truth: str
    frozen_fifths: int | None
    excluded_crops: tuple[int, ...] = ()
    carrizal_context: bool = False


SCORES = (
    ScoreSource(
        score_id="carrizal",
        slug="jaime-llanos_19_carrizal_pasillo_emilio-murillo",
        system_index=4,
        inference=(
            "jaime-llanos_19_carrizal_pasillo_emilio-murillo/"
            "vlm_melody_fresh_heldout/system_004/fresh_heldout/inference.jsonl"
        ),
        truth=(
            "jaime-llanos_19_carrizal_pasillo_emilio-murillo/"
            "vlm_melody_fresh_heldout/system_004/fresh_heldout/truth.jsonl"
        ),
        frozen_fifths=-1,
        excluded_crops=(2,),
        carrizal_context=True,
    ),
    ScoreSource(
        score_id="la_chata",
        slug="jaime-llanos_64_la-chata_pasillo_luis-a-calvo",
        system_index=7,
        inference=(
            "jaime-llanos_64_la-chata_pasillo_luis-a-calvo/"
            "vlm_melody_third_score_heldout/v2/system_007/inference_v2/inference.jsonl"
        ),
        truth=(
            "jaime-llanos_64_la-chata_pasillo_luis-a-calvo/"
            "vlm_melody_third_score_heldout/v2/system_007/evaluation_v1/truth.jsonl"
        ),
        frozen_fifths=None,
    ),
    ScoreSource(
        score_id="gatoe_fique",
        slug="jaime-llanos_49_gatoe-fique_pasillo_emilio-murillo",
        system_index=3,
        inference=(
            "jaime-llanos_49_gatoe-fique_pasillo_emilio-murillo/"
            "vlm_melody_fourth_score_heldout/v1/system_003/inference_v2/inference.jsonl"
        ),
        truth=(
            "jaime-llanos_49_gatoe-fique_pasillo_emilio-murillo/"
            "vlm_melody_fourth_score_heldout/v1/system_003/evaluation_v1/truth.jsonl"
        ),
        frozen_fifths=-1,
    ),
    ScoreSource(
        score_id="coqueteos",
        slug="jaime-llanos_22_coqueteos_pasillo_fulgencio-garcia",
        system_index=2,
        inference=(
            "jaime-llanos_22_coqueteos_pasillo_fulgencio-garcia/"
            "vlm_melody_fifth_score_heldout/v1/system_002/inference_v2/inference.jsonl"
        ),
        truth=(
            "jaime-llanos_22_coqueteos_pasillo_fulgencio-garcia/"
            "vlm_melody_fifth_score_heldout/v1/system_002/evaluation_v1/truth.jsonl"
        ),
        frozen_fifths=None,
        excluded_crops=(6,),
    ),
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", type=Path, nargs="?", default=Path("out"))
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    args = parser.parse_args(argv)
    try:
        destination = prepare_fixture(
            out_dir=args.out_dir.expanduser().resolve(),
            destination=args.destination.expanduser().resolve(),
        )
    except (FileExistsError, FileNotFoundError, OSError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(destination)
    return 0


def prepare_fixture(*, out_dir: Path, destination: Path) -> Path:
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Refusing to overwrite pitch fixture: {destination}")
    temporary = destination.with_name(f".{destination.name}.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(f"Refusing stale pitch fixture temp directory: {temporary}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    (temporary / "images").mkdir(parents=True)

    requests: list[dict[str, Any]] = []
    truth: list[dict[str, Any]] = []
    score_records: list[dict[str, Any]] = []
    source_records: dict[str, dict[str, Any]] = {}
    try:
        for score in SCORES:
            inference_path = out_dir / score.inference
            truth_path = out_dir / score.truth
            inference_rows = _load_jsonl(inference_path, f"{score.score_id} inference")
            truth_rows = _load_jsonl(truth_path, f"{score.score_id} truth")
            if len(inference_rows) != len(truth_rows):
                raise ValueError(f"{score.score_id} inference/truth row counts differ")
            clean_count = 0
            for position, (inference_row, truth_row) in enumerate(
                zip(inference_rows, truth_rows, strict=True), start=1
            ):
                crop = _crop_index(inference_row, fallback=position)
                truth_crop = _crop_index(truth_row, fallback=position)
                if crop != truth_crop:
                    raise ValueError(
                        f"{score.score_id} inference/truth crop mismatch: {crop} != {truth_crop}"
                    )
                if crop in score.excluded_crops:
                    continue
                request, truth_record = _freeze_measure(
                    score=score,
                    crop=crop,
                    inference_row=inference_row,
                    truth_row=truth_row,
                    temporary=temporary,
                    out_dir=out_dir,
                )
                requests.append(request)
                truth.append(truth_record)
                clean_count += 1
            score_records.append(
                {
                    "score_id": score.score_id,
                    "slug": score.slug,
                    "system_index": score.system_index,
                    "clean_measure_count": clean_count,
                    "excluded_segmentation_confounded_crops": list(score.excluded_crops),
                }
            )
            source_records[score.score_id] = {
                "inference": _external_record(inference_path),
                "truth": _external_record(truth_path),
            }

        requests_path = temporary / "requests.jsonl"
        truth_path = temporary / "truth.jsonl"
        _write_jsonl(requests_path, requests)
        _write_jsonl(truth_path, truth)
        key_events = _freeze_key_events(
            out_dir=out_dir,
            temporary=temporary,
            requests=requests,
        )
        manifest = {
            "schema_version": 1,
            "kind": FIXTURE_KIND,
            "evidence_tier": "consumed_postmortem",
            "eligible_for_heldout_claim": False,
            "selection_frozen": {
                "candidate_ids": True,
                "coordinates": True,
                "note_counts": True,
            },
            "requests": _local_record(requests_path, temporary),
            "truth": _local_record(truth_path, temporary),
            "scores": score_records,
            "key_events": key_events,
            "source_provenance": source_records,
        }
        _write_json(temporary / "manifest.json", manifest)
        temporary.rename(destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def _freeze_measure(
    *,
    score: ScoreSource,
    crop: int,
    inference_row: dict[str, Any],
    truth_row: dict[str, Any],
    temporary: Path,
    out_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source = inference_row.get("source")
    if not isinstance(source, dict):
        raise ValueError(f"{score.score_id} crop {crop} has no source object")
    raw_image = source.get("image")
    expected_hash = source.get("sha256", source.get("image_sha256"))
    if not isinstance(raw_image, str) or not isinstance(expected_hash, str):
        raise ValueError(f"{score.score_id} crop {crop} has no pinned source image")
    source_image = Path(raw_image)
    if not source_image.is_absolute():
        if source_image.parts and source_image.parts[0] == "out":
            source_image = out_dir.joinpath(*source_image.parts[1:]).resolve()
        else:
            source_image = (REPO_ROOT / source_image).resolve()
    if not source_image.is_file() or _sha256(source_image) != expected_hash:
        raise ValueError(f"{score.score_id} crop {crop} source image pin failed")
    image_name = f"{score.score_id}_measure_{crop:03d}{source_image.suffix.lower()}"
    target_image = temporary / "images" / image_name
    shutil.copyfile(source_image, target_image)

    lines = _staff_lines(
        score=score,
        crop=crop,
        inference_row=inference_row,
        out_dir=out_dir,
    )
    anchors = inference_row.get("automatic_anchors")
    if not isinstance(anchors, list):
        raise ValueError(f"{score.score_id} crop {crop} has no automatic anchors")
    notes = []
    for anchor in anchors:
        center = anchor.get("center") if isinstance(anchor, dict) else None
        origin = anchor.get("source") if isinstance(anchor, dict) else None
        if not isinstance(center, dict) or not isinstance(origin, dict):
            raise ValueError(f"{score.score_id} crop {crop} anchor is incomplete")
        notes.append(
            {
                "candidate_id": str(origin["candidate_id"]),
                "x": float(center["x"]),
                "y": float(center["y"]),
                "baseline_pitch": str(anchor["pitch"]),
            }
        )
    truth_notes = truth_row.get("notes")
    if not isinstance(truth_notes, list):
        raise ValueError(f"{score.score_id} crop {crop} truth has no notes")
    return (
        {
            "score_id": score.score_id,
            "measure_index": crop,
            "segmentation_confounded": False,
            "frozen_fifths": score.frozen_fifths,
            "staff_lines_y_px": lines,
            "image": _local_record(target_image, temporary),
            "notes": notes,
        },
        {
            "score_id": score.score_id,
            "measure_index": crop,
            "pitch_midi": [int(note["pitch_midi"]) for note in truth_notes],
        },
    )


def _staff_lines(
    *,
    score: ScoreSource,
    crop: int,
    inference_row: dict[str, Any],
    out_dir: Path,
) -> list[float]:
    if score.carrizal_context:
        context_path = (
            out_dir
            / score.slug
            / "vlm_melody_inputs"
            / f"system_{score.system_index:03d}"
            / f"measure_{crop:03d}_context.json"
        )
        context = _load_object(context_path, f"{score.score_id} crop {crop} context")
        raw_lines = context.get("staff_lines_y_px_in_system")
    else:
        geometry = inference_row.get("staff_geometry")
        raw_lines = geometry.get("raw_staff_lines_y_px") if isinstance(geometry, dict) else None
    if not isinstance(raw_lines, list) or len(raw_lines) != 5:
        raise ValueError(f"{score.score_id} crop {crop} has invalid staff geometry")
    return [float(value) for value in raw_lines]


def _freeze_key_events(
    *,
    out_dir: Path,
    temporary: Path,
    requests: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    events = []
    for score in SCORES:
        if score.score_id == "la_chata":
            source_record = next(
                row
                for row in requests
                if row["score_id"] == score.score_id and row["measure_index"] == 2
            )["image"]
            image_record = dict(source_record)
            mode = "change"
            start_measure = 2
        else:
            source_image = out_dir / score.slug / "systems" / "system_001.png"
            if not source_image.is_file():
                raise FileNotFoundError(f"Key-event source image is missing: {source_image}")
            target_image = temporary / "images" / f"{score.score_id}_key_initial.png"
            shutil.copyfile(source_image, target_image)
            image_record = _local_record(target_image, temporary)
            mode = "initial"
            start_measure = 1
        events.append(
            {
                "score_id": score.score_id,
                "mode": mode,
                "start_measure": start_measure,
                "fallback_fifths": score.frozen_fifths,
                "image": image_record,
            }
        )
    return events


def _crop_index(row: dict[str, Any], *, fallback: int) -> int:
    direct = row.get("automatic_crop_index")
    if isinstance(direct, int):
        return direct
    identity = row.get("identity")
    if isinstance(identity, dict):
        for key in ("automatic_measure_index", "system_measure_index"):
            value = identity.get(key)
            if isinstance(value, int):
                return value
    return fallback


def _load_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload


def _load_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{label} has invalid JSON on line {line_number}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{label} line {line_number} must be an object")
        rows.append(row)
    if not rows:
        raise ValueError(f"{label} is empty")
    return rows


def _external_record(path: Path) -> dict[str, Any]:
    try:
        display = path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        display = str(path.resolve())
    return {"path": display, "sha256": _sha256(path), "bytes": path.stat().st_size}


def _local_record(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": path.resolve().relative_to(root.resolve()).as_posix(),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as output:
        json.dump(payload, output, indent=2, sort_keys=True)
        output.write("\n")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
