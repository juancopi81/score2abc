# ruff: noqa: E501
"""Review frozen unseen-gate measures using raw images only.

The sealed manifest must have kind
``vlm_melody_hollow_notehead_unseen_gate_sealed_manifest`` and an ordered
``measures`` array. Each row contains an ``identity`` object and these
hash-pinned records:

- ``raw_image``
- ``candidate_artifact``
- ``proposal_artifact``

Every record uses ``{"path": "...", "sha256": "..."}``, with relative paths
resolved from the sealed manifest directory. Candidate and proposal artifacts
are validated server-side but are never exposed to the browser.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import mimetypes
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse

from PIL import Image, ImageDraw

MANIFEST_KIND = "vlm_melody_hollow_notehead_unseen_gate_sealed_manifest"
REVIEW_KIND = "vlm_melody_hollow_notehead_unseen_gate_human_truth"
PUBLIC_STATE_KIND = "vlm_melody_hollow_notehead_unseen_gate_raw_review_state"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
MAX_REQUEST_BYTES = 1_000_000
MAX_CENTERS_PER_MEASURE = 256
MIN_CENTER_DISTANCE_PX = 4.0
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SLUG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
COORDINATE_SPACE = (
    "raw measure image pixels; origin (0, 0) is top-left; x increases right; "
    "y increases down; each center marks one visually hollow or open notehead"
)


@dataclass(frozen=True)
class GateMeasure:
    ordinal: int
    identity: dict[str, Any]
    manifest_path: Path
    manifest_sha256: str
    raw_record: dict[str, str]
    candidate_record: dict[str, str]
    proposal_record: dict[str, str]
    raw_path: Path
    candidate_path: Path
    proposal_path: Path
    review_root: Path
    image_width: int
    image_height: int
    image_content_type: str

    @property
    def review_dir(self) -> Path:
        return (
            self.review_root
            / self.identity["slug"]
            / f"system_{self.identity['system_index']:03d}"
            / f"measure_{self.identity['system_measure_index']:03d}"
        )

    @property
    def review_path(self) -> Path:
        return self.review_dir / "review.json"

    @property
    def overlay_path(self) -> Path:
        return self.review_dir / "truth_overlay.png"


@dataclass
class ReviewApp:
    manifest_path: Path
    manifest_sha256: str
    measures: tuple[GateMeasure, ...]

    def public_state(self) -> dict[str, Any]:
        self.assert_manifest_current()
        return {
            "schema_version": 1,
            "kind": PUBLIC_STATE_KIND,
            "review_mode": "raw_image_only",
            "measures": [build_public_measure_state(measure) for measure in self.measures],
        }

    def save(self, ordinal: int, payload: Any) -> dict[str, Any]:
        measure = self.measure_at(ordinal)
        self.assert_manifest_current()
        return save_review(
            measure, payload, manifest_path=self.manifest_path, manifest_sha256=self.manifest_sha256
        )

    def measure_at(self, ordinal: int) -> GateMeasure:
        if ordinal < 0 or ordinal >= len(self.measures):
            raise ValueError(f"Unknown measure ordinal: {ordinal}")
        return self.measures[ordinal]

    def assert_manifest_current(self) -> None:
        if _sha256(self.manifest_path) != self.manifest_sha256:
            raise ValueError("Sealed manifest hash is stale")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sealed_manifest", type=Path)
    parser.add_argument(
        "--review-root",
        type=Path,
        default=None,
        help="Defaults to <sealed-manifest-dir>/review/human_truth.",
    )
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args(argv)
    try:
        app = load_review_app(args.sealed_manifest, review_root=args.review_root)
        server = create_server(app, host=args.host, port=args.port)
    except (FileNotFoundError, OSError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    host, port = server.server_address[:2]
    print(f"http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


def load_review_app(
    sealed_manifest: Path,
    *,
    review_root: Path | None = None,
) -> ReviewApp:
    manifest_path = sealed_manifest.resolve()
    manifest = _load_object(manifest_path, "Sealed manifest")
    if manifest.get("kind") != MANIFEST_KIND:
        raise ValueError(f"Unexpected sealed manifest kind: {manifest.get('kind')!r}")
    if manifest.get("status") != "frozen_awaiting_human_review":
        raise ValueError("Sealed manifest is not frozen awaiting human review")
    if manifest.get("split") != "fresh_heldout_morphology":
        raise ValueError("Sealed manifest is not a fresh heldout morphology gate")
    if manifest.get("create_once") is not True or manifest.get("truth_accessed") is not False:
        raise ValueError("Sealed manifest violates its create-once truth-blind contract")
    rows = manifest.get("measures")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Sealed manifest measures must be a non-empty ordered array")
    if manifest.get("measure_count") != len(rows):
        raise ValueError("Sealed manifest measure count does not match its rows")
    provenance = _required_object(manifest, "provenance", "Sealed manifest")
    for field in ("ground_truth_files_read", "review_files_read", "musicxml_files_read"):
        if provenance.get(field) != []:
            raise ValueError(f"Sealed manifest records forbidden access in {field}")

    destination = (
        review_root.resolve()
        if review_root is not None
        else (manifest_path.parent / "review" / "human_truth").resolve()
    )
    measures: list[GateMeasure] = []
    seen: set[tuple[str, int, int]] = set()
    for ordinal, row in enumerate(rows):
        label = f"Sealed manifest measures[{ordinal}]"
        if not isinstance(row, Mapping):
            raise ValueError(f"{label} must be an object")
        identity = _identity(_required_object(row, "identity", label), label)
        identity_key = (
            identity["slug"],
            identity["system_index"],
            identity["system_measure_index"],
        )
        if identity_key in seen:
            raise ValueError(f"Duplicate measure identity in sealed manifest: {identity_key}")
        seen.add(identity_key)

        raw_record, raw_path = _file_record(
            row, "raw_image", manifest_path=manifest_path, label=label
        )
        candidate_record, candidate_path = _file_record(
            row, "candidate_artifact", manifest_path=manifest_path, label=label
        )
        proposal_record, proposal_path = _file_record(
            row, "proposal_artifact", manifest_path=manifest_path, label=label
        )
        candidate = _load_object(candidate_path, "Candidate artifact")
        proposal = _load_object(proposal_path, "Frozen proposal artifact")
        _validate_artifact_identity(candidate, identity, "Candidate artifact")
        _validate_artifact_identity(proposal, identity, "Frozen proposal artifact")

        try:
            with Image.open(raw_path) as image:
                image.verify()
            with Image.open(raw_path) as image:
                width, height = image.size
        except (OSError, ValueError) as exc:
            raise ValueError(f"Raw image is not a readable image: {raw_path}") from exc
        if width <= 0 or height <= 0:
            raise ValueError(f"Raw image has invalid dimensions: {raw_path}")
        content_type = mimetypes.guess_type(raw_path.name)[0] or "application/octet-stream"
        if not content_type.startswith("image/"):
            raise ValueError(f"Raw image path has no supported image type: {raw_path}")

        measure = GateMeasure(
            ordinal=ordinal,
            identity=identity,
            manifest_path=manifest_path,
            manifest_sha256=_sha256(manifest_path),
            raw_record=raw_record,
            candidate_record=candidate_record,
            proposal_record=proposal_record,
            raw_path=raw_path,
            candidate_path=candidate_path,
            proposal_path=proposal_path,
            review_root=destination,
            image_width=width,
            image_height=height,
            image_content_type=content_type,
        )
        _assert_sources_current(measure)
        _validate_existing_review(measure)
        measures.append(measure)
    return ReviewApp(
        manifest_path=manifest_path,
        manifest_sha256=_sha256(manifest_path),
        measures=tuple(measures),
    )


def build_public_measure_state(measure: GateMeasure) -> dict[str, Any]:
    _assert_sources_current(measure)
    existing = _validate_existing_review(measure)
    state = {
        "ordinal": measure.ordinal,
        "identity": measure.identity,
        "image_url": f"/assets/{measure.ordinal}/raw",
        "image_size": {"width": measure.image_width, "height": measure.image_height},
        "status": "completed" if existing is not None else "pending",
        "existing_review": None,
    }
    if existing is not None:
        state["image_url"] = f"/assets/{measure.ordinal}/truth-overlay"
        state["existing_review"] = {
            "centers": existing["centers"],
            "completed_at": existing["completed_at"],
            "completion_confirmed": True,
        }
    return state


def save_review(
    measure: GateMeasure,
    payload: Any,
    *,
    manifest_path: Path,
    manifest_sha256: str,
) -> dict[str, Any]:
    _assert_sources_current(measure)
    if measure.review_dir.exists():
        raise FileExistsError(f"Refusing to overwrite completed review: {measure.review_path}")
    centers = validate_review_payload(measure, payload)
    completed_at = datetime.now(timezone.utc).isoformat()
    review = {
        "schema_version": 1,
        "kind": REVIEW_KIND,
        "identity": measure.identity,
        "source": {
            "sealed_manifest": {
                "path": str(manifest_path),
                "sha256": manifest_sha256,
            },
            "raw_image": measure.raw_record,
            "candidate_artifact": measure.candidate_record,
            "proposal_artifact": measure.proposal_record,
            "coordinate_space": COORDINATE_SPACE,
        },
        "centers": centers,
        "completion_confirmed": True,
        "completed_at": completed_at,
    }
    overlay_bytes = _truth_overlay_bytes(measure, centers)

    measure.review_dir.parent.mkdir(parents=True, exist_ok=True)
    created_review_dir = False
    try:
        measure.review_dir.mkdir()
        created_review_dir = True
        _write_json_exclusive(measure.review_path, review)
        with measure.overlay_path.open("xb") as output:
            output.write(overlay_bytes)
    except Exception:
        if created_review_dir and measure.review_dir.exists():
            shutil.rmtree(measure.review_dir)
        raise
    return {
        "status": "completed",
        "review_path": str(measure.review_path),
        "overlay_path": str(measure.overlay_path),
        "center_count": len(centers),
    }


def validate_review_payload(measure: GateMeasure, payload: Any) -> list[dict[str, float]]:
    if not isinstance(payload, Mapping):
        raise ValueError("Review payload must be a JSON object")
    if set(payload) != {"centers", "completion_confirmed"}:
        raise ValueError("Review payload must contain only centers and completion_confirmed")
    if payload["completion_confirmed"] is not True:
        raise ValueError("completion_confirmed must be true")
    raw_centers = payload["centers"]
    if not isinstance(raw_centers, list):
        raise ValueError("centers must be an array")
    if len(raw_centers) > MAX_CENTERS_PER_MEASURE:
        raise ValueError(f"centers cannot exceed {MAX_CENTERS_PER_MEASURE} items")

    centers = []
    for index, center in enumerate(raw_centers):
        if not isinstance(center, Mapping) or set(center) != {"x", "y"}:
            raise ValueError(f"centers[{index}] must contain only x and y")
        x = _finite_number(center["x"], f"centers[{index}].x")
        y = _finite_number(center["y"], f"centers[{index}].y")
        if not 0 <= x < measure.image_width or not 0 <= y < measure.image_height:
            raise ValueError(f"centers[{index}] is outside the raw image")
        point = {"x": round(x, 3), "y": round(y, 3)}
        if any(
            math.dist((point["x"], point["y"]), (other["x"], other["y"])) < MIN_CENTER_DISTANCE_PX
            for other in centers
        ):
            raise ValueError(f"centers[{index}] duplicates or nearly duplicates an earlier center")
        centers.append(point)
    return centers


def create_server(app: ReviewApp, *, host: str, port: int) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        review_app = app

        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            try:
                if path == "/":
                    self._send_bytes(_HTML.encode("utf-8"), "text/html; charset=utf-8")
                    return
                if path == "/api/state":
                    self._send_json(self.review_app.public_state())
                    return
                match = re.fullmatch(r"/assets/(\d+)/(raw|truth-overlay)", path)
                if match:
                    measure = self.review_app.measure_at(int(match.group(1)))
                    self.review_app.assert_manifest_current()
                    _assert_sources_current(measure)
                    if match.group(2) == "raw":
                        self._send_bytes(measure.raw_path.read_bytes(), measure.image_content_type)
                        return
                    _validate_existing_review(measure)
                    self._send_bytes(measure.overlay_path.read_bytes(), "image/png")
                    return
            except (FileNotFoundError, OSError, TypeError, ValueError) as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.CONFLICT)
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:  # noqa: N802
            match = re.fullmatch(r"/api/reviews/(\d+)", urlparse(self.path).path)
            if not match:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= MAX_REQUEST_BYTES:
                    raise ValueError("Invalid request size")
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                result = self.review_app.save(int(match.group(1)), payload)
            except FileExistsError as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.CONFLICT)
                return
            except (json.JSONDecodeError, OSError, TypeError, ValueError) as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return
            self._send_json(result, status=HTTPStatus.CREATED)

        def _send_json(
            self,
            payload: dict[str, Any],
            status: int = HTTPStatus.OK,
        ) -> None:
            self._send_bytes(
                (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8"),
                "application/json; charset=utf-8",
                status=status,
            )

        def _send_bytes(
            self,
            data: bytes,
            content_type: str,
            status: int = HTTPStatus.OK,
        ) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, format: str, *args: Any) -> None:
            return

    return ThreadingHTTPServer((host, port), Handler)


def _validate_existing_review(measure: GateMeasure) -> dict[str, Any] | None:
    if not measure.review_path.exists():
        if measure.overlay_path.exists() or measure.review_dir.exists():
            raise ValueError(f"Incomplete review directory exists: {measure.review_dir}")
        return None
    review = _load_object(measure.review_path, "Existing human truth review")
    if review.get("kind") != REVIEW_KIND:
        raise ValueError(f"Unexpected existing review kind: {measure.review_path}")
    if (
        _identity(_required_object(review, "identity", "Existing review"), "Existing review")
        != measure.identity
    ):
        raise ValueError(f"Existing review identity mismatch: {measure.review_path}")
    source = _required_object(review, "source", "Existing review")
    expected_manifest_record = {
        "path": str(measure.manifest_path),
        "sha256": measure.manifest_sha256,
    }
    if _required_object(source, "sealed_manifest", "Existing review") != expected_manifest_record:
        raise ValueError("Existing review sealed_manifest source is stale")
    for key, expected in (
        ("raw_image", measure.raw_record),
        ("candidate_artifact", measure.candidate_record),
        ("proposal_artifact", measure.proposal_record),
    ):
        if _required_object(source, key, "Existing review") != expected:
            raise ValueError(f"Existing review {key} source is stale")
    if source.get("coordinate_space") != COORDINATE_SPACE:
        raise ValueError("Existing review coordinate space mismatch")
    if review.get("completion_confirmed") is not True:
        raise ValueError("Existing review is not completion-confirmed")
    completed_at = review.get("completed_at")
    if not isinstance(completed_at, str) or not completed_at:
        raise ValueError("Existing review completed_at must be a timestamp string")
    centers = validate_review_payload(
        measure,
        {"centers": review.get("centers"), "completion_confirmed": True},
    )
    if not measure.overlay_path.is_file():
        raise ValueError(f"Existing review is missing truth-only overlay: {measure.overlay_path}")
    return {**review, "centers": centers}


def _assert_sources_current(measure: GateMeasure) -> None:
    for label, path, record in (
        ("Raw image", measure.raw_path, measure.raw_record),
        ("Candidate artifact", measure.candidate_path, measure.candidate_record),
        ("Frozen proposal artifact", measure.proposal_path, measure.proposal_record),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} is missing: {path}")
        if _sha256(path) != record["sha256"]:
            raise ValueError(f"{label} hash is stale: {path}")


def _file_record(
    row: Mapping[str, Any],
    key: str,
    *,
    manifest_path: Path,
    label: str,
) -> tuple[dict[str, str], Path]:
    record = _required_object(row, key, label)
    if set(record) != {"path", "sha256"}:
        raise ValueError(f"{label}.{key} must contain only path and sha256")
    raw_path = record["path"]
    digest = record["sha256"]
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"{label}.{key}.path must be a non-empty string")
    if not isinstance(digest, str) or not SHA256_RE.fullmatch(digest):
        raise ValueError(f"{label}.{key}.sha256 must be a lowercase SHA256")
    path = Path(raw_path)
    if not path.is_absolute():
        path = manifest_path.parent / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label}.{key} is missing: {path}")
    if _sha256(path) != digest:
        raise ValueError(f"{label}.{key} hash mismatch: {path}")
    return {"path": raw_path, "sha256": digest}, path


def _validate_artifact_identity(
    artifact: Mapping[str, Any],
    expected: Mapping[str, Any],
    label: str,
) -> None:
    raw_identity = artifact.get("identity")
    if isinstance(raw_identity, Mapping):
        actual = _identity(raw_identity, label)
    else:
        actual = _identity(artifact, label)
    if actual != expected:
        raise ValueError(f"{label} identity mismatch: expected {expected}, got {actual}")


def _identity(value: Mapping[str, Any], label: str) -> dict[str, Any]:
    slug = value.get("slug")
    if not isinstance(slug, str) or not SLUG_RE.fullmatch(slug):
        raise ValueError(f"{label} identity slug is invalid")
    result: dict[str, Any] = {
        "slug": slug,
        "system_index": _positive_int(value.get("system_index"), f"{label}.system_index"),
        "system_measure_index": _positive_int(
            value.get("system_measure_index"),
            f"{label}.system_measure_index",
        ),
    }
    if "global_measure_index" in value:
        result["global_measure_index"] = _positive_int(
            value["global_measure_index"],
            f"{label}.global_measure_index",
        )
    return result


def _required_object(
    payload: Mapping[str, Any],
    key: str,
    label: str,
) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{label}.{key} must be an object")
    return value


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be a finite number")
    return result


def _load_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as output:
        json.dump(payload, output, indent=2, ensure_ascii=False)
        output.write("\n")


def _truth_overlay_bytes(
    measure: GateMeasure,
    centers: list[dict[str, float]],
) -> bytes:
    with Image.open(measure.raw_path) as source:
        overlay = source.convert("RGB")
    draw = ImageDraw.Draw(overlay)
    radius = max(4, round(min(measure.image_width, measure.image_height) * 0.015))
    for center in centers:
        x = center["x"]
        y = center["y"]
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            outline=(0, 150, 70),
            width=2,
        )
        draw.line((x - radius, y, x + radius, y), fill=(0, 150, 70), width=1)
        draw.line((x, y - radius, x, y + radius), fill=(0, 150, 70), width=1)
    buffer = io.BytesIO()
    overlay.save(buffer, format="PNG")
    return buffer.getvalue()


_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Hollow notehead truth review</title>
<style>
:root{color-scheme:light;--ink:#16191b;--muted:#667078;--line:#d7dcdf;--panel:#f4f6f7;--green:#087a46;--red:#bd3028}
*{box-sizing:border-box}body{margin:0;background:#fff;color:var(--ink);font:14px/1.4 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;letter-spacing:0}
header{min-height:56px;border-bottom:1px solid var(--line);display:flex;align-items:center;gap:10px;padding:10px 16px}
h1{font-size:16px;margin:0;white-space:nowrap}.meta{color:var(--muted);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.spacer{flex:1}
button{height:34px;min-width:36px;border:1px solid #b8c0c5;border-radius:6px;background:#fff;color:var(--ink);font:inherit;cursor:pointer}button:hover:not(:disabled){background:#eef1f2}button:disabled{cursor:not-allowed;opacity:.45}button.primary{padding:0 14px;background:#1d252b;color:#fff;border-color:#1d252b}
main{height:calc(100vh - 56px);display:grid;grid-template-columns:minmax(0,1fr) 280px}.workspace{overflow:auto;background:#e8ecee;padding:20px;display:grid;place-items:center}.score{position:relative;width:min(100%,1400px);background:#fff;border:1px solid #bdc4c8}.score img{display:block;width:100%;height:auto}.markers{position:absolute;inset:0}.marker{position:absolute;width:20px;height:20px;transform:translate(-50%,-50%);border:2px solid var(--green);border-radius:50%;background:rgba(255,255,255,.35);pointer-events:none}
aside{border-left:1px solid var(--line);background:var(--panel);padding:16px}.state{font-weight:600;margin-bottom:8px}.status{color:var(--muted);min-height:44px}.pending{color:#8a4a00}.completed{color:var(--green)}.error{color:var(--red)}
@media(max-width:760px){header{flex-wrap:wrap}main{height:auto;grid-template-columns:1fr}.workspace{min-height:60vh;padding:8px}aside{border-left:0;border-top:1px solid var(--line)}}
</style>
</head>
<body>
<header><h1>Hollow notehead truth</h1><span class="meta" id="meta"></span><span class="spacer"></span><button id="prev" title="Previous measure" aria-label="Previous measure">&#8592;</button><button id="undo" title="Undo last center" aria-label="Undo last center">&#8630;</button><button id="clear" title="Clear centers" aria-label="Clear centers">&#215;</button><button id="next" title="Next measure" aria-label="Next measure">&#8594;</button><button id="save" class="primary">Finalize</button></header>
<main><section class="workspace"><div class="score" id="score"><img id="image" alt="Raw target measure"><div class="markers" id="markers"></div></div></section><aside><div class="state" id="reviewState"></div><div class="status" id="status"></div></aside></main>
<script>
const state={data:null,index:0,centers:[]};const $=id=>document.getElementById(id);const current=()=>state.data.measures[state.index];
function loadMeasure(){const m=current(),done=m.status==='completed';state.centers=done?m.existing_review.centers.map(p=>({...p})):[];$('image').src=m.image_url;$('meta').textContent=`${m.identity.slug} · system ${m.identity.system_index} · measure ${m.identity.system_measure_index}`;$('reviewState').textContent=done?'Completed':'Pending';$('reviewState').className='state '+(done?'completed':'pending');$('status').textContent=done?`${state.centers.length} centers saved · read-only`:'';$('undo').disabled=done;$('clear').disabled=done;$('save').disabled=done;render()}
function render(){const m=current(),done=m.status==='completed',layer=$('markers');layer.innerHTML='';if(!done)for(const p of state.centers){const marker=document.createElement('span');marker.className='marker';marker.style.left=`${p.x/m.image_size.width*100}%`;marker.style.top=`${p.y/m.image_size.height*100}%`;layer.appendChild(marker)}$('prev').disabled=state.index===0;$('next').disabled=state.index===state.data.measures.length-1}
$('score').onclick=e=>{const m=current();if(m.status==='completed')return;const box=$('image').getBoundingClientRect(),x=(e.clientX-box.left)/box.width*m.image_size.width,y=(e.clientY-box.top)/box.height*m.image_size.height;if(x<0||y<0||x>=m.image_size.width||y>=m.image_size.height)return;state.centers.push({x:+x.toFixed(3),y:+y.toFixed(3)});render()};
$('undo').onclick=()=>{state.centers.pop();render()};$('clear').onclick=()=>{state.centers=[];render()};$('prev').onclick=()=>{if(state.index>0){state.index--;loadMeasure()}};$('next').onclick=()=>{if(state.index<state.data.measures.length-1){state.index++;loadMeasure()}};
$('save').onclick=async()=>{const m=current();if(!confirm('Finalize this measure? The review is create-once and cannot be edited.'))return;$('status').className='status';$('status').textContent='Saving…';const response=await fetch(`/api/reviews/${m.ordinal}`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({centers:state.centers,completion_confirmed:true})});const result=await response.json();if(!response.ok){$('status').className='status error';$('status').textContent=result.error||'Save failed';return}state.data=await fetch('/api/state').then(r=>r.json());loadMeasure();if(state.index<state.data.measures.length-1){state.index++;loadMeasure()}};
fetch('/api/state').then(r=>{if(!r.ok)throw new Error(`State request failed: ${r.status}`);return r.json()}).then(data=>{state.data=data;loadMeasure()}).catch(error=>{$('status').className='status error';$('status').textContent=error.message});
</script>
</body>
</html>"""


if __name__ == "__main__":
    raise SystemExit(main())
