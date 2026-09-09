# ruff: noqa: E501
"""Run a local candidate-confirming notehead review spike.

The reviewer is intentionally separate from the production pipeline. It builds
cap-24 blind proposal artifacts, serves a dependency-free localhost UI, and
writes review labels under ``out/<slug>/vlm_melody_reviews``. Independent
coordinate ground truth is never read while building browser state; when it is
available, it is used only after a review is submitted to attach hidden metrics.

Example:
    ./.venv/bin/python scripts/review_vlm_notehead_candidates.py out \
        --slug jaime-llanos_12_aviador_pasillo_fulgencio-garcia \
        --system 1 --measure 1 --measure 2 --measure 3 --measure 4
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import urlparse

from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import eval_vlm_notehead_proposal_baselines as evaluator  # noqa: E402
from scripts.build_vlm_notehead_localization_inputs import (  # noqa: E402
    build_vlm_notehead_localization_inputs,
)
from scripts.run_vlm_notehead_localization_spike import treble_pitch_for_y  # noqa: E402

CANDIDATE_CAP = 24
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
MAX_REQUEST_BYTES = 1_000_000
PITCH_RE = re.compile(r"^[A-G](?:#|b)?-?\d+$")
GT_DIR = REPO_ROOT / "tests/fixtures/vlm_melody/notehead_ground_truth"


@dataclass(frozen=True)
class ReviewMeasure:
    slug: str
    system_index: int
    measure_index: int
    global_measure_index: int
    title: str
    source_image_path: Path
    candidate_artifact_path: Path
    review_dir: Path
    image_sha256: str
    candidate_artifact_sha256: str
    image_width: int
    image_height: int
    staff_lines: tuple[float, ...]
    candidates: tuple[dict[str, Any], ...]

    @property
    def review_path(self) -> Path:
        return self.review_dir / "review.json"

    @property
    def overlay_path(self) -> Path:
        return self.review_dir / "review_overlay.png"


@dataclass
class ReviewApp:
    out_dir: Path
    slug: str
    system_index: int
    measures: dict[int, ReviewMeasure]

    def public_state(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "slug": self.slug,
            "system_index": self.system_index,
            "measures": [
                build_public_measure_state(self.measures[index]) for index in sorted(self.measures)
            ],
            "pitch_options": pitch_options(),
        }

    def save(self, measure_index: int, payload: dict[str, Any]) -> dict[str, Any]:
        measure = self.measures.get(measure_index)
        if measure is None:
            raise ValueError(f"Unknown measure: {measure_index}")
        return save_review(measure, payload)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        app = prepare_review_app(
            args.out_dir,
            slug=args.slug,
            system_index=args.system,
            measures=args.measure,
        )
    except (FileNotFoundError, KeyError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    server = create_server(app, host=args.host, port=args.port)
    host, port = server.server_address[:2]
    print(f"http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--slug", required=True)
    parser.add_argument("--system", type=int, required=True)
    parser.add_argument("--measure", action="append", type=int)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    return parser


def prepare_review_app(
    out_dir: Path,
    *,
    slug: str,
    system_index: int,
    measures: Sequence[int] | None = None,
) -> ReviewApp:
    selected_measures = set(measures) if measures else None
    records = build_vlm_notehead_localization_inputs(
        out_dir,
        selected_slugs={slug},
        selected_systems={system_index},
        selected_measures=selected_measures,
        task_kind="candidate-assisted-localization",
        max_candidates=CANDIDATE_CAP,
        overwrite=True,
    )
    review_measures = {}
    for record in records:
        measure = _review_measure_from_record(out_dir, record)
        if measure.measure_index in review_measures:
            raise ValueError(f"Duplicate review measure: {measure.measure_index}")
        review_measures[measure.measure_index] = measure
    if not review_measures:
        raise FileNotFoundError("No measure records matched the review selection")
    return ReviewApp(
        out_dir=out_dir,
        slug=slug,
        system_index=system_index,
        measures=review_measures,
    )


def _review_measure_from_record(out_dir: Path, record: dict[str, Any]) -> ReviewMeasure:
    candidate_path = _resolve_path(out_dir, record["candidate_artifact_path"])
    artifact = _load_json(candidate_path)
    candidates = artifact.get("candidates")
    if not isinstance(candidates, list) or not 0 < len(candidates) <= CANDIDATE_CAP:
        raise ValueError(
            f"Expected 1-{CANDIDATE_CAP} candidates at {candidate_path}, "
            f"got {len(candidates) if isinstance(candidates, list) else 'invalid'}"
        )
    source_path = _resolve_path(out_dir, artifact["source_image_path"])
    image_width = int(artifact["source_image_size_px"]["width"])
    image_height = int(artifact["source_image_size_px"]["height"])
    staff_lines = tuple(float(value) for value in artifact["staff_lines_y_px"])
    if len(staff_lines) != 5:
        raise ValueError(f"Expected five staff lines at {candidate_path}")
    measure_index = int(record["system_measure_index"])
    review_dir = (
        out_dir
        / str(record["slug"])
        / "vlm_melody_reviews"
        / f"system_{int(record['system_index']):03d}"
        / f"measure_{measure_index:03d}"
    )
    context = _load_json(_resolve_path(out_dir, record["context_path"]))
    return ReviewMeasure(
        slug=str(record["slug"]),
        system_index=int(record["system_index"]),
        measure_index=measure_index,
        global_measure_index=int(record["global_measure_index"]),
        title=str(context.get("title") or record["slug"]),
        source_image_path=source_path,
        candidate_artifact_path=candidate_path,
        review_dir=review_dir,
        image_sha256=_sha256(source_path),
        candidate_artifact_sha256=_sha256(candidate_path),
        image_width=image_width,
        image_height=image_height,
        staff_lines=staff_lines,
        candidates=tuple(candidates),
    )


def build_public_measure_state(measure: ReviewMeasure) -> dict[str, Any]:
    existing = _load_json(measure.review_path) if measure.review_path.exists() else None
    return {
        "measure_index": measure.measure_index,
        "global_measure_index": measure.global_measure_index,
        "title": measure.title,
        "image_url": f"/assets/measure_{measure.measure_index:03d}.png",
        "image_size": {"width": measure.image_width, "height": measure.image_height},
        "source_hashes": {
            "image_sha256": measure.image_sha256,
            "candidate_artifact_sha256": measure.candidate_artifact_sha256,
        },
        "staff_lines_y_px": list(measure.staff_lines),
        "candidates": [
            {
                "id": str(candidate["id"]),
                "rank": int(candidate["rank"]),
                "bbox": candidate["bbox"],
                "center": candidate["center"],
                "score": candidate["score"],
                "auto_pitch": treble_pitch_for_y(
                    float(candidate["center"]["y"]), measure.staff_lines
                ),
            }
            for candidate in measure.candidates
        ],
        "existing_review": _existing_review_state(existing),
    }


def _existing_review_state(review: dict[str, Any] | None) -> dict[str, Any] | None:
    if not review:
        return None
    return {
        "selected_candidate_ids": [
            note["source"]["candidate_id"]
            for note in review.get("final_noteheads", [])
            if note.get("source", {}).get("kind") == "candidate"
        ],
        "manual_noteheads": review.get("manual_noteheads", []),
        "pitch_overrides": {
            note["source"]["candidate_id"]: note["pitch"]
            for note in review.get("final_noteheads", [])
            if note.get("source", {}).get("kind") == "candidate"
            and note.get("pitch") != note.get("auto_pitch")
        },
        "active_review_ms": int(review.get("timing", {}).get("active_review_ms", 0)),
    }


def save_review(measure: ReviewMeasure, payload: dict[str, Any]) -> dict[str, Any]:
    validated = validate_review_payload(measure, payload)
    selected_ids = validated["selected_candidate_ids"]
    selected_set = set(selected_ids)
    overrides = validated["pitch_overrides"]
    candidate_snapshots = []
    final_notes = []
    for candidate in measure.candidates:
        candidate_id = str(candidate["id"])
        auto_pitch = treble_pitch_for_y(float(candidate["center"]["y"]), measure.staff_lines)
        snapshot = {
            **candidate,
            "label": "accepted" if candidate_id in selected_set else "rejected",
        }
        snapshot["auto_pitch"] = auto_pitch
        candidate_snapshots.append(snapshot)
        if candidate_id in selected_set:
            final_pitch = overrides.get(candidate_id, auto_pitch)
            final_notes.append(
                {
                    "source": {"kind": "candidate", "candidate_id": candidate_id},
                    "center": candidate["center"],
                    "auto_pitch": auto_pitch,
                    "pitch": final_pitch,
                    "pitch_corrected": final_pitch != auto_pitch,
                }
            )
    manual_notes = []
    for index, item in enumerate(validated["manual_noteheads"], start=1):
        auto_pitch = treble_pitch_for_y(float(item["center"]["y"]), measure.staff_lines)
        manual = {
            "id": f"manual_{index:03d}",
            "center": item["center"],
            "auto_pitch": auto_pitch,
            "pitch": item["pitch"],
            "pitch_corrected": item["pitch"] != auto_pitch,
        }
        manual_notes.append(manual)
        final_notes.append(
            {
                "source": {"kind": "manual", "manual_id": manual["id"]},
                "center": manual["center"],
                "auto_pitch": auto_pitch,
                "pitch": manual["pitch"],
                "pitch_corrected": manual["pitch_corrected"],
            }
        )
    final_notes.sort(key=lambda item: (float(item["center"]["x"]), float(item["center"]["y"])))
    for order, note in enumerate(final_notes, start=1):
        note["order"] = order

    review = {
        "schema_version": 1,
        "kind": "vlm_melody_notehead_candidate_review",
        "identity": {
            "slug": measure.slug,
            "system_index": measure.system_index,
            "system_measure_index": measure.measure_index,
            "global_measure_index": measure.global_measure_index,
        },
        "source": {
            "image_path": str(measure.source_image_path),
            "image_sha256": measure.image_sha256,
            "candidate_artifact_path": str(measure.candidate_artifact_path),
            "candidate_artifact_sha256": measure.candidate_artifact_sha256,
            "candidate_cap": CANDIDATE_CAP,
            "coordinate_space": "source image pixels, origin at top-left",
        },
        "candidates": candidate_snapshots,
        "manual_noteheads": manual_notes,
        "final_noteheads": final_notes,
        "timing": {
            "active_review_ms": validated["active_review_ms"],
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "inactivity_timeout_ms": 30_000,
        },
        "metrics": {},
    }
    review["metrics"] = evaluate_saved_review(measure, review)
    measure.review_dir.mkdir(parents=True, exist_ok=True)
    _write_json(measure.review_path, review)
    _write_review_overlay(measure, review)
    return {
        "review_path": str(measure.review_path),
        "overlay_path": str(measure.overlay_path),
        "final_notehead_count": len(final_notes),
    }


def validate_review_payload(measure: ReviewMeasure, payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Review payload must be a JSON object")
    hashes = payload.get("source_hashes")
    if not isinstance(hashes, dict):
        raise ValueError("source_hashes is required")
    if hashes.get("image_sha256") != measure.image_sha256:
        raise ValueError("Source image hash is stale")
    if hashes.get("candidate_artifact_sha256") != measure.candidate_artifact_sha256:
        raise ValueError("Candidate artifact hash is stale")
    selected = payload.get("selected_candidate_ids")
    if not isinstance(selected, list) or not all(isinstance(value, str) for value in selected):
        raise ValueError("selected_candidate_ids must be an array of strings")
    if len(selected) != len(set(selected)):
        raise ValueError("selected_candidate_ids contains duplicates")
    known_ids = {str(candidate["id"]) for candidate in measure.candidates}
    unknown = sorted(set(selected) - known_ids)
    if unknown:
        raise ValueError(f"Unknown candidate IDs: {unknown}")
    overrides = payload.get("pitch_overrides", {})
    if not isinstance(overrides, dict):
        raise ValueError("pitch_overrides must be an object")
    invalid_override_ids = sorted(set(overrides) - set(selected))
    if invalid_override_ids:
        raise ValueError(f"Pitch overrides require selected candidates: {invalid_override_ids}")
    for candidate_id, pitch in overrides.items():
        _validate_pitch(pitch, f"pitch_overrides.{candidate_id}")
    manual = payload.get("manual_noteheads", [])
    if not isinstance(manual, list):
        raise ValueError("manual_noteheads must be an array")
    normalized_manual = []
    for index, item in enumerate(manual):
        if not isinstance(item, dict) or set(item) != {"center", "pitch"}:
            raise ValueError(f"manual_noteheads[{index}] must contain center and pitch")
        center = item["center"]
        if not isinstance(center, dict) or set(center) != {"x", "y"}:
            raise ValueError(f"manual_noteheads[{index}].center must contain x and y")
        x = _finite_number(center["x"], f"manual_noteheads[{index}].center.x")
        y = _finite_number(center["y"], f"manual_noteheads[{index}].center.y")
        if not 0 <= x < measure.image_width or not 0 <= y < measure.image_height:
            raise ValueError(f"manual_noteheads[{index}] is outside the source image")
        _validate_pitch(item["pitch"], f"manual_noteheads[{index}].pitch")
        normalized_manual.append(
            {"center": {"x": round(x, 3), "y": round(y, 3)}, "pitch": item["pitch"]}
        )
    active_ms = payload.get("active_review_ms")
    if isinstance(active_ms, bool) or not isinstance(active_ms, (int, float)):
        raise ValueError("active_review_ms must be a non-negative number")
    if not math.isfinite(float(active_ms)) or not 0 <= float(active_ms) <= 86_400_000:
        raise ValueError("active_review_ms must be between 0 and 86400000")
    return {
        "selected_candidate_ids": selected,
        "pitch_overrides": overrides,
        "manual_noteheads": normalized_manual,
        "active_review_ms": round(float(active_ms)),
    }


def evaluate_saved_review(measure: ReviewMeasure, review: dict[str, Any]) -> dict[str, Any]:
    gt_path = evaluator._ground_truth_path(
        GT_DIR,
        slug=measure.slug,
        system_index=measure.system_index,
        measure=measure.measure_index,
    )
    if not gt_path.exists():
        return {"status": "ground_truth_missing"}
    ground_truth = evaluator._load_ground_truth_fixture(gt_path)
    points = [
        {"id": _note_id(note), "center": note["center"]}
        for note in review.get("final_noteheads", [])
    ]
    region = evaluator.match_region_points(
        points,
        ground_truth,
        staff_lines=measure.staff_lines,
        margin=evaluator.ANNOTATION_REGION_MARGIN,
    )
    note_by_id = {_note_id(note): note for note in review.get("final_noteheads", [])}
    auto_correct = 0
    final_correct = 0
    assignments = []
    for assignment in region["assignments"]:
        note = note_by_id[assignment["candidate_id"]]
        expected = evaluator._natural_pitch_name(
            str(ground_truth[assignment["ground_truth_index"]].get("pitch", ""))
        )
        auto_pitch = evaluator._natural_pitch_name(str(note["auto_pitch"]))
        final_pitch = evaluator._natural_pitch_name(str(note["pitch"]))
        auto_correct += auto_pitch == expected
        final_correct += final_pitch == expected
        assignments.append(
            {
                "review_note_id": assignment["candidate_id"],
                "ground_truth_id": assignment["ground_truth_id"],
                "auto_pitch": auto_pitch,
                "final_pitch": final_pitch,
                "expected_pitch": expected,
            }
        )
    matched = region["tp"]
    return {
        "status": "evaluated_after_save",
        "ground_truth_path": str(gt_path),
        "selection": {key: region[key] for key in ("tp", "fp", "fn", "precision", "recall", "f1")},
        "automatic_natural_pitch": {
            "correct": auto_correct,
            "total": matched,
            "accuracy": evaluator._ratio(auto_correct, matched),
        },
        "final_natural_pitch": {
            "correct": final_correct,
            "total": matched,
            "accuracy": evaluator._ratio(final_correct, matched),
        },
        "manual_addition_count": len(review.get("manual_noteheads", [])),
        "pitch_correction_count": sum(
            bool(note.get("pitch_corrected")) for note in review.get("final_noteheads", [])
        ),
        "assignments": assignments,
    }


def _note_id(note: dict[str, Any]) -> str:
    source = note["source"]
    return str(source.get("candidate_id") or source.get("manual_id"))


def _write_review_overlay(measure: ReviewMeasure, review: dict[str, Any]) -> None:
    with Image.open(measure.source_image_path) as opened:
        overlay = opened.convert("RGB")
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default()
    accepted = {
        note["source"]["candidate_id"]: note
        for note in review["final_noteheads"]
        if note["source"]["kind"] == "candidate"
    }
    for candidate in measure.candidates:
        candidate_id = str(candidate["id"])
        bbox = candidate["bbox"]
        color = (0, 155, 70) if candidate_id in accepted else (180, 180, 180)
        width = 2 if candidate_id in accepted else 1
        draw.rectangle(
            (bbox["left"], bbox["top"], bbox["right"], bbox["bottom"]),
            outline=color,
            width=width,
        )
        if candidate_id in accepted:
            center = candidate["center"]
            draw.text(
                (float(center["x"]) + 7, float(center["y"]) - 7),
                f"{candidate_id} {accepted[candidate_id]['pitch']}",
                fill=color,
                font=font,
            )
    for manual in review["manual_noteheads"]:
        x = float(manual["center"]["x"])
        y = float(manual["center"]["y"])
        draw.ellipse((x - 7, y - 7, x + 7, y + 7), outline=(25, 105, 210), width=2)
        draw.text((x + 8, y - 7), f"{manual['id']} {manual['pitch']}", fill=(25, 105, 210))
    measure.review_dir.mkdir(parents=True, exist_ok=True)
    overlay.save(measure.overlay_path)


def pitch_options() -> list[str]:
    options = []
    for octave in range(2, 8):
        for letter in "CDEFGAB":
            options.extend((f"{letter}b{octave}", f"{letter}{octave}", f"{letter}#{octave}"))
    return options


def create_server(app: ReviewApp, *, host: str, port: int) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        review_app = app

        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path == "/":
                self._send_bytes(_HTML.encode("utf-8"), "text/html; charset=utf-8")
                return
            if path == "/api/state":
                self._send_json(self.review_app.public_state())
                return
            match = re.fullmatch(r"/assets/measure_(\d{3})\.png", path)
            if match:
                measure = self.review_app.measures.get(int(match.group(1)))
                if measure is None:
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                self._send_bytes(measure.source_image_path.read_bytes(), "image/png")
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            match = re.fullmatch(r"/api/reviews/(\d+)", path)
            if not match:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= MAX_REQUEST_BYTES:
                    raise ValueError("Invalid request size")
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                result = self.review_app.save(int(match.group(1)), payload)
            except (json.JSONDecodeError, OSError, ValueError) as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return
            self._send_json(result)

        def _send_json(self, payload: dict[str, Any], status: int = HTTPStatus.OK) -> None:
            self._send_bytes(
                (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8"),
                "application/json; charset=utf-8",
                status=status,
            )

        def _send_bytes(self, data: bytes, content_type: str, status: int = HTTPStatus.OK) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, format: str, *args: Any) -> None:
            return

    return ThreadingHTTPServer((host, port), Handler)


def _validate_pitch(value: Any, field: str) -> None:
    if not isinstance(value, str) or not PITCH_RE.fullmatch(value):
        raise ValueError(f"{field} must be a pitch such as F4, Bb4, or C#5")


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be a finite number")
    return result


def _resolve_path(out_dir: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    repo_path = REPO_ROOT / path
    if repo_path.exists():
        return repo_path
    out_path = out_dir / path
    return out_path if out_path.exists() else repo_path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Notehead review</title>
<style>
:root{color-scheme:light;--ink:#151719;--muted:#667078;--line:#d9dde0;--panel:#f4f6f7;--green:#087a46;--red:#c53228;--blue:#1769c2}
*{box-sizing:border-box}body{margin:0;background:#fff;color:var(--ink);font:14px/1.4 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;letter-spacing:0}
header{height:56px;border-bottom:1px solid var(--line);display:flex;align-items:center;gap:12px;padding:0 16px;background:#fff}
h1{font-size:16px;margin:0;white-space:nowrap}.meta{color:var(--muted);min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.spacer{flex:1}button,select{height:34px;border:1px solid #b9c0c5;background:#fff;color:var(--ink);border-radius:6px;font:inherit}
button{padding:0 12px;cursor:pointer}button:hover{background:#eef1f2}button.primary{background:#1d252b;color:#fff;border-color:#1d252b}button.mode.active{border-color:var(--blue);color:var(--blue);background:#edf5ff}
main{height:calc(100vh - 56px);display:grid;grid-template-columns:minmax(0,1fr) 320px}.workspace{overflow:auto;background:#e8ecee;padding:20px;display:grid;place-items:center}.score{position:relative;width:min(100%,1200px);background:#fff;border:1px solid #bdc4c8}.score img{display:block;width:100%;height:auto}.markers{position:absolute;inset:0}.marker{position:absolute;width:26px;height:26px;padding:0;transform:translate(-50%,-50%);border:2px solid var(--red);border-radius:50%;background:rgba(255,255,255,.78);color:#7a1611;font-size:9px;line-height:22px;text-align:center}.marker.selected{border-color:var(--green);background:rgba(227,250,238,.9);color:#075c36}.marker.manual{border-color:var(--blue);color:#0e4f92}.sidebar{border-left:1px solid var(--line);background:var(--panel);overflow:auto;padding:14px}.sidebar h2{font-size:13px;margin:4px 0 10px}.note-row{display:grid;grid-template-columns:58px 1fr 32px;gap:8px;align-items:center;padding:8px 0;border-bottom:1px solid var(--line)}.note-id{font-variant-numeric:tabular-nums;color:var(--muted)}.remove{width:32px;padding:0;font-size:18px}.empty{color:var(--muted);padding:16px 0}.status{font-size:12px;color:var(--muted);margin-top:12px;min-height:34px}
@media(max-width:760px){header{height:auto;min-height:56px;flex-wrap:wrap;padding:10px}main{height:auto;grid-template-columns:1fr}.workspace{min-height:55vh;padding:8px}.sidebar{border-left:0;border-top:1px solid var(--line)}}
</style>
</head>
<body>
<header><h1>Notehead review</h1><span class="meta" id="meta"></span><span class="spacer"></span><button id="prev" title="Previous measure" aria-label="Previous measure">&#8592;</button><button id="add" class="mode" title="Add a missing notehead">+</button><button id="next" title="Next measure" aria-label="Next measure">&#8594;</button><button id="save" class="primary">Save &amp; next</button></header>
<main><section class="workspace"><div class="score" id="score"><img id="image" alt="Target measure"><div class="markers" id="markers"></div></div></section><aside class="sidebar"><h2>Selected notes</h2><div id="notes"></div><div class="status" id="status"></div></aside></main>
<script>
const state={data:null,index:0,selected:new Set(),manual:[],pitches:{},addMode:false,activeMs:0,lastTick:Date.now(),lastAction:Date.now()};
const $=id=>document.getElementById(id);const touch=()=>state.lastAction=Date.now();document.addEventListener('pointerdown',touch);document.addEventListener('keydown',touch);
setInterval(()=>{const now=Date.now();if(!document.hidden&&document.hasFocus()&&now-state.lastAction<=30000)state.activeMs+=now-state.lastTick;state.lastTick=now},1000);
const current=()=>state.data.measures[state.index];
function loadMeasure(){const m=current(),old=m.existing_review;state.selected=new Set(old?.selected_candidate_ids||[]);state.manual=(old?.manual_noteheads||[]).map((n,i)=>({id:n.id||`manual_${i+1}`,center:n.center,pitch:n.pitch,auto_pitch:n.auto_pitch||pitchForY(n.center.y,m.staff_lines_y_px)}));state.pitches={...(old?.pitch_overrides||{})};state.activeMs=old?.active_review_ms||0;state.addMode=false;$('add').classList.remove('active');$('image').src=m.image_url;$('meta').textContent=`${m.title} · system ${state.data.system_index} · measure ${m.measure_index}`;render()}
function roundHalfAway(value){return value<0?-Math.floor(-value+.5):Math.floor(value+.5)}
function pitchForY(y,lines){const gap=(lines[4]-lines[0])/4,step=roundHalfAway((y-lines[0])/(gap/2)),letters=['C','D','E','F','G','A','B'],top=letters.indexOf('F'),idx=((top-step)%7+7)%7,oct=5+Math.floor((top-step)/7);return `${letters[idx]}${oct}`}
function render(){const m=current(),layer=$('markers');layer.innerHTML='';for(const c of m.candidates){const b=document.createElement('button');b.className='marker'+(state.selected.has(c.id)?' selected':'');b.textContent=c.id.slice(1);b.title=`${c.id} · ${c.auto_pitch}`;b.style.left=`${c.center.x/m.image_size.width*100}%`;b.style.top=`${c.center.y/m.image_size.height*100}%`;b.onclick=e=>{e.stopPropagation();state.selected.has(c.id)?state.selected.delete(c.id):state.selected.add(c.id);render()};layer.appendChild(b)}for(const n of state.manual){const b=document.createElement('button');b.className='marker manual';b.textContent='+';b.title=n.pitch;b.style.left=`${n.center.x/m.image_size.width*100}%`;b.style.top=`${n.center.y/m.image_size.height*100}%`;layer.appendChild(b)}renderNotes()}
function renderNotes(){const m=current(),rows=[];for(const c of m.candidates)if(state.selected.has(c.id))rows.push({id:c.id,x:c.center.x,auto:c.auto_pitch,pitch:state.pitches[c.id]||c.auto_pitch,manual:false});for(const n of state.manual)rows.push({id:n.id,x:n.center.x,auto:n.auto_pitch,pitch:n.pitch,manual:true});rows.sort((a,b)=>a.x-b.x);const root=$('notes');root.innerHTML='';if(!rows.length){root.innerHTML='<div class="empty">No notes selected</div>';return}for(const n of rows){const row=document.createElement('div');row.className='note-row';const id=document.createElement('span');id.className='note-id';id.textContent=n.id;const sel=document.createElement('select');for(const p of state.data.pitch_options){const o=document.createElement('option');o.value=p;o.textContent=p;if(p===n.pitch)o.selected=true;sel.appendChild(o)}sel.onchange=()=>{if(n.manual){const hit=state.manual.find(x=>x.id===n.id);hit.pitch=sel.value}else state.pitches[n.id]=sel.value};const remove=document.createElement('button');remove.className='remove';remove.textContent='×';remove.title='Remove note';remove.onclick=()=>{if(n.manual)state.manual=state.manual.filter(x=>x.id!==n.id);else state.selected.delete(n.id);render()};row.append(id,sel,remove);root.appendChild(row)}}
$('score').onclick=e=>{if(!state.addMode)return;const img=$('image'),r=img.getBoundingClientRect(),m=current(),x=(e.clientX-r.left)/r.width*m.image_size.width,y=(e.clientY-r.top)/r.height*m.image_size.height;if(x<0||y<0||x>=m.image_size.width||y>=m.image_size.height)return;const auto=pitchForY(y,m.staff_lines_y_px);state.manual.push({id:`manual_${Date.now()}`,center:{x:+x.toFixed(3),y:+y.toFixed(3)},auto_pitch:auto,pitch:auto});render()};
$('add').onclick=()=>{state.addMode=!state.addMode;$('add').classList.toggle('active',state.addMode)};$('prev').onclick=()=>{if(state.index>0){state.index--;loadMeasure()}};$('next').onclick=()=>{if(state.index<state.data.measures.length-1){state.index++;loadMeasure()}};
$('save').onclick=async()=>{const m=current();$('status').textContent='Saving…';const payload={source_hashes:m.source_hashes,selected_candidate_ids:[...state.selected],manual_noteheads:state.manual.map(n=>({center:n.center,pitch:n.pitch})),pitch_overrides:Object.fromEntries(Object.entries(state.pitches).filter(([id,p])=>state.selected.has(id)&&p!==m.candidates.find(c=>c.id===id).auto_pitch)),active_review_ms:Math.round(state.activeMs)};const response=await fetch(`/api/reviews/${m.measure_index}`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});const result=await response.json();if(!response.ok){$('status').textContent=result.error||'Save failed';return}$('status').textContent=`Saved ${result.final_notehead_count} notes`;if(state.index<state.data.measures.length-1){state.index++;loadMeasure()}};
fetch('/api/state').then(r=>r.json()).then(data=>{state.data=data;loadMeasure()}).catch(e=>$('status').textContent=e.message);
</script>
</body></html>"""


if __name__ == "__main__":
    raise SystemExit(main())
