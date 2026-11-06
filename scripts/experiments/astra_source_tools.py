"""Two bounded, source-only MCP tools for the Astra baseline reproduction spike.

This file is copied into an isolated directory before execution. It imports no
repository package, opens no caller-selected path and never invokes a network API.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

MAX_DRAFT_BYTES = 256_000
MAX_TOOL_CALLS = 128
MAX_DRAFTS = 64
ALLOWED_DPI = (160, 240, 320)
MAX_IMAGE_SIDE = 2000
SERVER_INSTRUCTIONS = (
    "Use inspect_source to inspect the supplied PDF and make your own crops. "
    "Use save_draft early, even for an incomplete draft, and after meaningful changes. "
    "It preserves your draft and returns notation/meter checks, never reference answers. "
    "Set complete=true only for your final complete best-effort transcription. "
    "Only the supplied PDF and your own work are available."
)
TOOLS = [
    {
        "name": "inspect_source",
        "description": (
            "Render a page or crop of the supplied source PDF. Crop coordinates are normalized "
            "[left, top, right, bottom], with origin at the page's top left. No external files."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "page": {"type": "integer", "minimum": 1, "maximum": 8},
                "crop": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 4,
                    "maxItems": 4,
                },
                "dpi": {"type": "integer", "enum": list(ALLOWED_DPI)},
            },
            "required": ["page"],
            "additionalProperties": False,
        },
        "annotations": {"readOnlyHint": True, "openWorldHint": False},
    },
    {
        "name": "save_draft",
        "description": (
            "Save a complete current ABC draft (partial progress is welcome) and editorial "
            "decisions. Returns deterministic notation/meter checks, not musical ground truth. "
            "Every call creates a new immutable checkpoint. Use complete=true on the final draft."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "abc": {"type": "string", "maxLength": MAX_DRAFT_BYTES},
                "decisions": {"type": "array", "items": {"type": "object"}, "maxItems": 128},
                "complete": {"type": "boolean"},
                "note": {"type": "string", "maxLength": 2000},
            },
            "required": ["abc", "decisions", "complete"],
            "additionalProperties": False,
        },
        "annotations": {"readOnlyHint": False, "destructiveHint": False, "openWorldHint": False},
    },
]


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def validate_crop(crop: object) -> tuple[float, float, float, float]:
    if (
        not isinstance(crop, list)
        or len(crop) != 4
        or any(type(x) not in (int, float) or not math.isfinite(x) for x in crop)
    ):
        raise ValueError("Crop must contain four finite numbers")
    left, top, right, bottom = crop
    if not (0 <= left < right <= 1 and 0 <= top < bottom <= 1):
        raise ValueError("Crop must lie within the source page")
    return left, top, right, bottom


class SourceTools:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.config = json.loads((self.root / "tool_config.json").read_text())
        if sha((self.root / "source.pdf").read_bytes()) != self.config["source_pdf_sha256"]:
            raise ValueError("Source PDF changed")
        self.calls = 0
        self.drafts = 0
        self.artifacts = self.root / "artifacts"
        self.artifacts.mkdir(exist_ok=True)
        (self.artifacts / "drafts").mkdir(exist_ok=True)
        (self.artifacts / "inspections").mkdir(exist_ok=True)

    def call(self, name: str, arguments: object) -> dict:
        if not isinstance(arguments, dict):
            raise ValueError("Arguments must be an object")
        if self.calls >= MAX_TOOL_CALLS:
            raise ValueError("Source tool call allowance exhausted")
        self.calls += 1
        started = datetime.now(timezone.utc).isoformat()
        if name == "inspect_source":
            content, evidence = self.inspect(arguments)
        elif name == "save_draft":
            content, evidence = self.save(arguments)
        else:
            raise ValueError("Unknown source-only tool")
        with (self.artifacts / "tool_events.jsonl").open("a") as stream:
            stream.write(
                json.dumps(
                    {
                        "call": self.calls,
                        "name": name,
                        "started_at": started,
                        "finished_at": datetime.now(timezone.utc).isoformat(),
                        "evidence": evidence,
                    }
                )
                + "\n"
            )
        return {"content": content, "isError": False}

    def inspect(self, args: dict) -> tuple[list, dict]:
        if set(args) - {"page", "crop", "dpi"}:
            raise ValueError("Only page, crop and dpi are supported")
        page, dpi = args.get("page"), args.get("dpi", 240)
        if type(page) is not int or not 1 <= page <= self.config["pages"]:
            raise ValueError("Page outside supplied PDF")
        if type(dpi) is not int or dpi not in ALLOWED_DPI:
            raise ValueError("Unsupported rendering resolution")
        crop = validate_crop(args.get("crop", [0, 0, 1, 1]))
        cache = self.root / f"render_p{page}_{dpi}.png"
        if not cache.exists():
            subprocess.run(
                [
                    self.config["pdftoppm"],
                    "-f",
                    str(page),
                    "-l",
                    str(page),
                    "-r",
                    str(dpi),
                    "-singlefile",
                    "-png",
                    str(self.root / "source.pdf"),
                    str(cache.with_suffix("")),
                ],
                capture_output=True,
                check=True,
                timeout=30,
            )
        with Image.open(cache) as source:
            width, height = source.size
            if width * height > 30_000_000:
                raise ValueError("Source page exceeds rendering bound")
            box = (
                int(width * crop[0]),
                int(height * crop[1]),
                math.ceil(width * crop[2]),
                math.ceil(height * crop[3]),
            )
            image = source.crop(box).convert("RGB")
            image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE), Image.Resampling.LANCZOS)
            output = io.BytesIO()
            image.save(output, format="PNG")
            pixels = output.getvalue()
            filename = f"inspect_{self.calls:03d}.png"
            (self.artifacts / "inspections" / filename).write_bytes(pixels)
            evidence = {
                "page": page,
                "dpi": dpi,
                "crop": list(crop),
                "pixel_box": list(box),
                "source_pixels": [width, height],
                "returned_pixels": list(image.size),
                "path": "inspections/" + filename,
                "sha256": sha(pixels),
            }
        content = [
            {"type": "text", "text": json.dumps(evidence)},
            {
                "type": "image",
                "mimeType": "image/png",
                "data": base64.b64encode(pixels).decode(),
                "_meta": {"codex/imageDetail": "original"},
            },
        ]
        return content, evidence

    def save(self, args: dict) -> tuple[list, dict]:
        if set(args) - {"abc", "decisions", "complete", "note"}:
            raise ValueError("Only ABC, decisions, complete and note are supported")
        abc, decisions, complete = args.get("abc"), args.get("decisions"), args.get("complete")
        if not isinstance(abc, str) or len(abc.encode()) > MAX_DRAFT_BYTES:
            raise ValueError("ABC must be bounded text")
        if (
            not isinstance(decisions, list)
            or len(decisions) > 128
            or any(not isinstance(item, dict) for item in decisions)
            or len(json.dumps(decisions).encode()) > MAX_DRAFT_BYTES
        ):
            raise ValueError("Decisions must be a bounded array of objects")
        if type(complete) is not bool:
            raise ValueError("Complete must be true or false")
        note = args.get("note", "")
        if not isinstance(note, str) or len(note) > 2000:
            raise ValueError("Checkpoint note must be bounded text")
        if self.drafts >= MAX_DRAFTS:
            raise ValueError("Draft allowance exhausted")
        self.drafts += 1
        directory = self.artifacts / "drafts" / f"draft_{self.drafts:03d}"
        directory.mkdir(exist_ok=False)
        (directory / "score.abc").write_text(abc)
        write_json(directory / "decisions.json", decisions)
        try:
            process = subprocess.run(
                [
                    self.config["node"],
                    "--max-old-space-size=256",
                    str(self.root / "score_eval_parse.js"),
                    str(self.root / "renderer"),
                ],
                input=json.dumps({"abc": abc}),
                text=True,
                capture_output=True,
                timeout=10,
            )
            if process.returncode:
                raise ValueError("Parser process failed")
            normalized = json.loads(process.stdout)
            if not isinstance(normalized, dict) or "valid" not in normalized:
                raise ValueError("Invalid parser response")
        except (subprocess.SubprocessError, ValueError, OSError):
            normalized = {
                "valid": False,
                "errors": ["Local parser failed or timed out"],
                "measures": [],
                "route_status": "unavailable",
                "counts": {},
            }
        write_json(directory / "normalized.json", normalized)
        issues = [{"kind": "parse", "detail": e} for e in normalized.get("errors", [])]
        for bar in normalized.get("measures", []):
            if bar.get("meter_valid") is not True:
                issues.append(
                    {
                        "kind": "meter",
                        "measure": bar["number"],
                        "observed_ticks": bar["duration_ticks"],
                        "expected_ticks": bar["expected_ticks"],
                        "partial_reason": bar.get("partial_reason"),
                        "detail": "Inspect the source; do not pad rests automatically.",
                    }
                )
        if normalized.get("route_status") != "resolved":
            issues.append({"kind": "form", "detail": "Playback route is unresolved."})
        required = ("id", "location", "category", "adopted_choice", "reason", "uncertainty")
        if any(any(not item.get(key) for key in required) for item in decisions):
            issues.append({"kind": "decisions", "detail": "Decisions need " + ", ".join(required)})
        evidence = {
            "checkpoint": f"drafts/draft_{self.drafts:03d}",
            "complete_claimed": complete,
            "abc_sha256": sha(abc.encode()),
            "decisions_sha256": sha((directory / "decisions.json").read_bytes()),
            "note": note,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        write_json(directory / "checkpoint.json", evidence)
        feedback = {
            **evidence,
            "valid": normalized.get("valid"),
            "counts": normalized.get("counts"),
            "issues": issues,
            "musical_accuracy": "not_measured_by_validation",
        }
        write_json(directory / "validation.json", feedback)
        return [{"type": "text", "text": json.dumps(feedback)}], evidence


def serve(root: Path) -> None:
    tools = SourceTools(root)
    for line in sys.stdin:
        request = None
        try:
            if len(line.encode()) > 600_000:
                raise ValueError("Request too large")
            request = json.loads(line)
            method = request.get("method")
            if method.startswith("notifications/"):
                continue
            if method == "initialize":
                result = {
                    "protocolVersion": request.get("params", {}).get(
                        "protocolVersion", "2025-06-18"
                    ),
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "score_source", "version": "1.0"},
                    "instructions": SERVER_INSTRUCTIONS,
                }
            elif method == "tools/list":
                result = {"tools": TOOLS}
            elif method == "tools/call":
                params = request.get("params", {})
                try:
                    result = tools.call(params.get("name"), params.get("arguments", {}))
                except Exception as exc:
                    # Never expose tracebacks, filesystem paths or arbitrary process stderr.
                    result = {
                        "isError": True,
                        "content": [
                            {
                                "type": "text",
                                "text": "Source tool failed: "
                                + type(exc).__name__
                                + ". Check bounded arguments.",
                            }
                        ],
                    }
            elif method in ("resources/list", "resources/templates/list", "prompts/list"):
                key = {
                    "resources/list": "resources",
                    "resources/templates/list": "resourceTemplates",
                    "prompts/list": "prompts",
                }[method]
                result = {key: []}
            elif method == "ping":
                result = {}
            else:
                raise ValueError("Unsupported MCP method")
            response = {"jsonrpc": "2.0", "id": request.get("id"), "result": result}
        except Exception:
            response = {
                "jsonrpc": "2.0",
                "id": request.get("id") if isinstance(request, dict) else None,
                "error": {"code": -32602, "message": "Invalid bounded source-tool request"},
            }
        sys.stdout.write(json.dumps(response) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    serve(Path(sys.argv[1]))
