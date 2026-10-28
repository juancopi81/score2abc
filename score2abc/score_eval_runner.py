"""Audited PDF-only Codex subscription execution; never reads musical references.

Each generation uses a fresh CLI process, no exposed tools, a copied source packet,
and a restrictive outer read sandbox. A local mock request capture checks the exact
model-visible context before every subscription request. This is macOS-only v1.
"""

from __future__ import annotations

import base64
import hashlib
import http.server
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image
from PIL import __version__ as PILLOW_VERSION

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
RUNTIME_ROOT = PACKAGE_ROOT / "out/eval/runtime/codex_subscription_v2"
CALL_TIMEOUT_SECONDS = 900
RENDER_POLICY = {"dpi": 160, "vertical_tiles": 3, "overlap_fraction": 0.10, "max_pages": 8}
BASE_INSTRUCTIONS = (
    "You are a music transcription model. Use only the supplied images and task instructions. "
    "Return the requested JSON. No external information or tools are permitted."
)
PERMISSIONS_INSTRUCTIONS = (
    "<permissions instructions>\n"
    "Filesystem sandboxing defines which files can be read or written. `sandbox_mode` is "
    "`read-only`: The sandbox only permits reading files. Network access is restricted.\n"
    "Approval policy is currently never. Do not provide the `sandbox_permissions` for any "
    "reason, commands will be rejected.\n</permissions instructions>"
)
DECISION_FIELDS = ("id", "location", "category", "adopted_choice", "reason", "uncertainty")
OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "abc": {"type": "string"},
        "decisions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    **{key: {"type": "string"} for key in DECISION_FIELDS},
                    "alternatives": {"type": "array", "items": {"type": "string"}},
                },
                "required": [*DECISION_FIELDS, "alternatives"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["abc", "decisions"],
    "additionalProperties": False,
}
# This reduces capabilities; it does not bypass approvals or managed policy.
DISABLED_FEATURES = (
    "shell_tool",
    "apps",
    "plugins",
    "memories",
    "multi_agent",
    "multi_agent_v2",
    "browser_use",
    "computer_use",
    "in_app_browser",
    "image_generation",
    "view_image",
    "code_mode",
    "code_mode_only",
    "code_mode_host",
    "workspace_dependencies",
    "skill_search",
    "hooks",
    "goals",
    "request_permissions_tool",
    "tool_suggest",
    "enable_request_compression",
    "sleep_tool",
    "send_message_to_user_async",
)
CONTROLS = {
    **{f"features.{key}": False for key in DISABLED_FEATURES},
    "tools.update_plan.enabled": False,
    "tools.experimental_request_user_input.enabled": False,
    "include_environment_context": False,
    "include_collaboration_mode_instructions": False,
    "web_search": "disabled",
    "skills.include_instructions": False,
    "skills.bundled.enabled": False,
    "project_doc_max_bytes": 0,
    "developer_instructions": "",
    "history.persistence": "none",
    "mcp_servers": {},
}
CATALOG_OVERRIDES = {
    "tool_mode": "none",
    "multi_agent_version": "none",
    "experimental_supported_tools": [],
    "apply_patch_tool_type": None,
}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _write(path: Path, value: object) -> None:
    path.write_bytes(_json(value))


def _environment() -> dict:
    # Keep the user's existing auth location. Do not copy API keys, endpoints,
    # agent IDs, inherited session controls or arbitrary shell startup variables.
    names = {"HOME", "USER", "LOGNAME", "PATH", "TMPDIR", "CODEX_HOME", "LANG", "LC_ALL"}
    return {key: value for key, value in os.environ.items() if key in names}


def _version(command: list[str]) -> str:
    run = subprocess.run(command, capture_output=True, text=True, timeout=15, check=True)
    return (run.stdout or run.stderr).strip().splitlines()[0]


def prepare_runtime(destination: Path = RUNTIME_ROOT, executable: Path | None = None) -> dict:
    """Freeze installed capabilities and auth mode without making an inference call."""
    destination = Path(destination).resolve()
    if destination.exists():
        raise ValueError("Runtime is create-once; select a new directory")
    app = Path("/Applications/ChatGPT.app/Contents/Resources/codex-cli/bin/codex")
    binary = Path(executable or (app if app.is_file() else shutil.which("codex") or ""))
    if (
        not binary.is_file()
        or sys.platform != "darwin"
        or not Path("/usr/bin/sandbox-exec").is_file()
    ):
        raise ValueError("Runner v1 requires an installed Codex CLI and macOS sandbox-exec")
    binary = binary.resolve()
    auth = subprocess.run(
        [str(binary), "login", "status"],
        capture_output=True,
        text=True,
        timeout=20,
        env=_environment(),
    )
    if auth.returncode or "Logged in using ChatGPT" not in auth.stdout + auth.stderr:
        raise ValueError("ChatGPT subscription login is required; API-key execution is forbidden")
    catalog_run = subprocess.run(
        [str(binary), "debug", "models"],
        capture_output=True,
        text=True,
        timeout=40,
        check=True,
        env=_environment(),
    )
    catalog = json.loads(catalog_run.stdout)
    if not isinstance(catalog.get("models"), list) or not catalog["models"]:
        raise ValueError("Installed CLI did not return a model catalog")
    # This changes client tool exposure, never the requested model or effort.
    filtered = {"models": [{**model, **CATALOG_OVERRIDES} for model in catalog["models"]]}
    poppler = shutil.which("pdftoppm")
    if not poppler:
        raise ValueError("Poppler pdftoppm is required")
    binary_files = {str(binary): _sha(binary.read_bytes())}
    app_binary = binary.parent.parent / "CodexCLI.app/Contents/MacOS/codex"
    if app_binary.is_file():
        binary_files[str(app_binary.resolve())] = _sha(app_binary.read_bytes())
    runtime = {
        "schema_version": 1,
        "status": "prepared",
        "adapter": "codex_subscription",
        "auth_mode": "chatgpt_subscription",
        "codex_path": str(binary),
        "codex_version": _version([str(binary), "--version"]),
        "binary_files": binary_files,
        "poppler_path": str(Path(poppler).resolve()),
        "poppler_version": _version([poppler, "-v"]),
        "pillow_version": PILLOW_VERSION,
        "render_policy": RENDER_POLICY,
        "call_timeout_seconds": CALL_TIMEOUT_SECONDS,
        "controls_sha256": _sha(_json(CONTROLS)),
        "catalog_sha256": _sha(_json(filtered)),
        "catalog_overrides": CATALOG_OVERRIDES,
        "source_catalog_sha256": _sha(_json(catalog)),
        "output_schema_sha256": _sha(_json(OUTPUT_SCHEMA)),
        "base_instructions_sha256": _sha(BASE_INSTRUCTIONS.encode()),
    }
    destination.mkdir(parents=True)
    _write(destination / "source_catalog.json", catalog)
    _write(destination / "model_catalog.json", filtered)
    _write(destination / "runtime.json", runtime)
    (destination / "runtime.sha256").write_text(_sha(_json(runtime)) + "\n")
    return runtime


def runtime_identity(root: Path = RUNTIME_ROOT) -> dict:
    """Read the frozen runtime fingerprint without refreshing models or authentication."""
    root = Path(root)
    if not (root / "runtime.json").is_file():
        return {"status": "unavailable", "adapter": "codex_subscription"}
    data = (root / "runtime.json").read_bytes()
    if _sha(data) != (root / "runtime.sha256").read_text().strip():
        raise ValueError("Runtime manifest changed")
    runtime = json.loads(data)
    if _sha((root / "model_catalog.json").read_bytes()) != runtime["catalog_sha256"]:
        raise ValueError("Runtime model catalog changed")
    return runtime


def verify_runtime(root: Path = RUNTIME_ROOT) -> dict:
    runtime = runtime_identity(root)
    if runtime.get("status") != "prepared":
        raise ValueError("Prepare the subscription runtime first")
    for path, expected in runtime["binary_files"].items():
        if _sha(Path(path).read_bytes()) != expected:
            raise ValueError("Codex binary changed; prepare a new runtime version")
    if (
        runtime["controls_sha256"] != _sha(_json(CONTROLS))
        or runtime["output_schema_sha256"] != _sha(_json(OUTPUT_SCHEMA))
        or runtime["base_instructions_sha256"] != _sha(BASE_INSTRUCTIONS.encode())
        or runtime["call_timeout_seconds"] != CALL_TIMEOUT_SECONDS
        or runtime["catalog_overrides"] != CATALOG_OVERRIDES
        or runtime["render_policy"] != RENDER_POLICY
        or runtime["pillow_version"] != PILLOW_VERSION
        or runtime["poppler_version"] != _version([runtime["poppler_path"], "-v"])
    ):
        raise ValueError("Runner capabilities changed; prepare a new runtime version")
    return runtime


def sandbox_policy(workspace: Path) -> str:
    """Add read-denials to the CLI's normal read-only execution sandbox."""
    codex_home = Path(os.environ.get("CODEX_HOME", str(Path.home() / ".codex"))).resolve()
    literal = [
        codex_home / name
        for name in ("AGENTS.md", "AGENTS.override.md", "CODEX_WORKFLOW.md", "CODEX_SCRATCHPAD.md")
    ]
    subpaths = [
        PACKAGE_ROOT,
        codex_home / "memories",
        codex_home / "sessions",
        codex_home / "archived_sessions",
    ]
    denied = [f"(literal {json.dumps(str(path))})" for path in literal]
    denied += [f"(subpath {json.dumps(str(path))})" for path in subpaths]
    if any(workspace.is_relative_to(path) for path in subpaths):
        raise ValueError("Isolated workspace must be outside repository and history")
    return "(version 1)(allow default)(deny file-read-data " + " ".join(denied) + ")"


def command(
    runtime: dict,
    workspace: Path,
    model: str,
    effort: str,
    images: list[Path],
    *,
    audit_url: str | None = None,
) -> list[str]:
    controls = {
        **CONTROLS,
        "model_reasoning_effort": effort,
        "model_instructions_file": str(workspace / "base.txt"),
        "model_catalog_json": str(workspace / "model_catalog.json"),
    }
    if audit_url:
        controls.update(
            {
                "model_provider": "audit",
                "model_providers.audit.name": "local audit",
                "model_providers.audit.base_url": audit_url,
                "model_providers.audit.wire_api": "responses",
                "model_providers.audit.requires_openai_auth": False,
                "model_providers.audit.request_max_retries": 0,
                "model_providers.audit.stream_max_retries": 0,
            }
        )
    else:
        controls.update({"model_provider": "openai", "forced_login_method": "chatgpt"})
    args = [
        "/usr/bin/sandbox-exec",
        "-p",
        sandbox_policy(workspace),
        runtime["codex_path"],
        "exec",
        "--ignore-user-config",
        "--ephemeral",
        "--skip-git-repo-check",
        "--sandbox",
        "read-only",
        "--cd",
        str(workspace),
        "--model",
        model,
        "--json",
        "--output-schema",
        str(workspace / "schema.json"),
        "--output-last-message",
        str(workspace / "last.json"),
    ]
    for key, value in controls.items():
        args += ["-c", f"{key}=" + ("{}" if value == {} else json.dumps(value))]
    for path in images:
        args += ["--image", str(path)]
    return args + ["--", "-"]


def verify_request(
    payload: dict,
    prompt: str,
    image_hashes: list[str],
    model: str,
    effort: str,
    image_paths: list[str] | None = None,
) -> dict:
    """Fail closed on tools, unsolicited context, foreign images or changed settings."""
    if payload.get("model") != model or payload.get("reasoning", {}).get("effort") != effort:
        raise ValueError("Captured model/settings differ from request")
    if payload.get("tools"):
        raise ValueError("Captured request exposes tools")
    allowed_text = {prompt, BASE_INSTRUCTIONS}
    allowed_text.update(
        f'<image name=[Image #{index + 1}] path="{path}">'
        for index, path in enumerate(image_paths or [])
    )
    if image_paths:
        allowed_text.add("</image>")
    seen_images, seen_prompt, permission_hashes = [], False, []
    if payload.get("instructions") not in (None, BASE_INSTRUCTIONS):
        raise ValueError("Unexpected base instructions")
    for item in payload.get("input", []):
        if item.get("type") == "additional_tools":
            if item.get("tools"):
                raise ValueError("Captured incremental request exposes tools")
            continue
        if item.get("type") != "message" or item.get("role") not in ("user", "developer"):
            raise ValueError("Unexpected context or conversation history")
        for part in item.get("content", []):
            if part.get("type") == "input_image":
                url = part.get("image_url", "")
                if not url.startswith("data:image/png;base64,"):
                    raise ValueError("Nonlocal/non-PNG image in request")
                seen_images.append(_sha(base64.b64decode(url.split(",", 1)[1], validate=True)))
            elif part.get("type") == "input_text":
                text = part.get("text", "")
                if text == prompt:
                    seen_prompt = True
                elif text in allowed_text or re.fullmatch(r"\[Image #\d+\]", text):
                    pass
                elif text == PERMISSIONS_INSTRUCTIONS:
                    permission_hashes.append(_sha(text.encode()))
                else:
                    raise ValueError("Captured request includes unapproved textual context")
            else:
                raise ValueError("Unapproved input modality")
    if not seen_prompt or seen_images != image_hashes:
        raise ValueError("Captured prompt/images differ from supplied inputs")
    return {
        "status": "passed",
        "exposed_tools": [],
        "input_image_sha256": seen_images,
        "prompt_sha256": _sha(prompt.encode()),
        "permissions_sha256": permission_hashes,
        "request_sha256": _sha(_json(payload)),
        "model": model,
        "reasoning_effort": effort,
    }


def _process(args: list[str], prompt: str, timeout: int) -> tuple[int, str, str, float]:
    start = time.monotonic()
    proc = subprocess.Popen(
        args,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=_environment(),
        start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(prompt, timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            stdout, stderr = proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            stdout, stderr = proc.communicate()
        return 124, stdout, stderr + "\nRunner deadline exceeded", time.monotonic() - start
    return proc.returncode, stdout, stderr, time.monotonic() - start


def capture_request(
    runtime: dict, workspace: Path, model: str, effort: str, images: list[Path], prompt: str
) -> dict:
    captures = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 80_000_000:
                self.send_error(413)
                return
            captures.append(json.loads(self.rfile.read(length)))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            text = json.dumps({"abc": "X:1\nM:3/4\nL:1/4\nK:C\nC D E|]", "decisions": []})
            events = [
                {"type": "response.created", "response": {"id": "audit", "status": "in_progress"}},
                {
                    "type": "response.output_item.done",
                    "item": {
                        "type": "message",
                        "id": "audit-message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": text, "annotations": []}],
                    },
                },
                {
                    "type": "response.completed",
                    "response": {
                        "id": "audit",
                        "status": "completed",
                        "output": [],
                        "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                    },
                },
            ]
            for event in events:
                self.wfile.write(("data: " + json.dumps(event) + "\n\n").encode())

        def do_GET(self):
            self.send_error(404)

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        args = command(
            runtime,
            workspace,
            model,
            effort,
            images,
            audit_url=f"http://127.0.0.1:{server.server_port}/v1",
        )
        code, _stdout, stderr, _elapsed = _process(args, prompt, 60)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
    if code or len(captures) != 1:
        raise ValueError(f"Local capability audit failed ({code}): {stderr[-1200:]}")
    result = verify_request(
        captures[0],
        prompt,
        [_sha(p.read_bytes()) for p in images],
        model,
        effort,
        [str(p) for p in images],
    )
    result["transport"] = "localhost_mock_no_auth_no_inference"
    (workspace / "last.json").unlink(missing_ok=True)
    return result


def _read_events(stdout: str) -> list[dict]:
    events = [json.loads(line) for line in stdout.splitlines() if line.strip()]
    allowed_items = {"agent_message", "reasoning", "error"}
    allowed_events = {
        "thread.started",
        "turn.started",
        "turn.completed",
        "turn.failed",
        "error",
        "item.started",
        "item.completed",
        "item.updated",
    }
    for event in events:
        if not isinstance(event, dict) or event.get("type") not in allowed_events:
            raise ValueError("Unknown event type; quarantine this attempt")
        if "item" in event and event["item"].get("type") not in allowed_items:
            raise ValueError("Unknown/tool item; quarantine this attempt")
    return events


def _read_response(path: Path) -> dict:
    response = json.loads(path.read_text())
    if (
        not isinstance(response, dict)
        or set(response) != {"abc", "decisions"}
        or not isinstance(response["abc"], str)
        or not isinstance(response["decisions"], list)
    ):
        raise ValueError("Invalid structured author response")
    if len(response["abc"].encode()) > 256_000:
        raise ValueError("ABC response exceeds grader bound")
    for decision in response["decisions"]:
        if (
            not isinstance(decision, dict)
            or set(decision) != {*DECISION_FIELDS, "alternatives"}
            or not all(isinstance(decision[key], str) for key in DECISION_FIELDS)
            or not isinstance(decision["alternatives"], list)
            or not all(isinstance(value, str) for value in decision["alternatives"])
        ):
            raise ValueError("Invalid structured editorial decision")
    return response


def subscription_call(
    runtime: dict,
    runtime_root: Path,
    model: str,
    effort: str,
    images: list[Path],
    prompt: str,
    destination: Path,
) -> dict:
    """Audit then execute one bounded fresh-context call; retain every raw event."""
    destination.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    result = {
        "returncode": None,
        "status": "failed",
        "error": None,
        "usage": None,
        "requested_model": model,
        "resolved_model": None,
        "reasoning_effort": effort,
        "inference_started": False,
        "quarantined": False,
    }
    phase = "isolation_audit"
    try:
        with tempfile.TemporaryDirectory(prefix="score2abc-isolated-") as tmp:
            workspace = Path(tmp).resolve()
            (workspace / "base.txt").write_text(BASE_INSTRUCTIONS)
            _write(workspace / "schema.json", OUTPUT_SCHEMA)
            shutil.copyfile(runtime_root / "model_catalog.json", workspace / "model_catalog.json")
            copied = []
            for i, image in enumerate(images):
                path = workspace / f"input_{i:03d}.png"
                shutil.copyfile(image, path)
                copied.append(path)
            audit = capture_request(runtime, workspace, model, effort, copied, prompt)
            _write(destination / "isolation_audit.json", audit)
            result["audit_sha256"] = _sha((destination / "isolation_audit.json").read_bytes())
            (destination / "prompt.txt").write_text(prompt)
            phase = "inference"
            result["inference_started"] = True
            code, stdout, stderr, elapsed = _process(
                command(runtime, workspace, model, effort, copied), prompt, CALL_TIMEOUT_SECONDS
            )
            result.update(returncode=code, inference_elapsed_seconds=elapsed)
            (destination / "events.jsonl").write_text(stdout)
            (destination / "stderr.txt").write_text(stderr)
            result["events_sha256"] = _sha((destination / "events.jsonl").read_bytes())
            if (workspace / "last.json").is_file():
                shutil.copyfile(workspace / "last.json", destination / "response.json")
        phase = "event_verification"
        events = _read_events(stdout)
        result["usage"] = next(
            (
                event.get("usage")
                for event in reversed(events)
                if event.get("type") == "turn.completed"
            ),
            None,
        )
        errors = [
            event.get("message") or event.get("error", {}).get("message")
            for event in events
            if event.get("type") in ("error", "turn.failed")
        ]
        result["error"] = "; ".join(error for error in errors if error)
        if code != 0 or result["usage"] is None:
            result["error"] = result["error"] or stderr[-1000:] or "No completed model turn"
        else:
            phase = "response_validation"
            result["response"] = _read_response(destination / "response.json")
            result["status"] = "complete"
    except Exception as exc:
        result["error"] = f"{phase}: {type(exc).__name__}: {exc}"
        result["quarantined"] = phase in {"isolation_audit", "event_verification"}
    result["phase"] = phase
    result["elapsed_seconds"] = time.monotonic() - started
    _write(destination / "call.json", result)
    return result


def render_inputs(pdf: Path, destination: Path, runtime: dict) -> list[Path]:
    """Fixed, source-only page and overlapping third-page images; no note hints."""
    destination.mkdir(parents=True, exist_ok=False)
    subprocess.run(
        [
            runtime["poppler_path"],
            "-png",
            "-r",
            str(RENDER_POLICY["dpi"]),
            "-f",
            "1",
            "-l",
            str(RENDER_POLICY["max_pages"] + 1),
            str(pdf),
            str(destination / "page"),
        ],
        capture_output=True,
        check=True,
        timeout=90,
    )
    pages = sorted(destination.glob("page-*.png"))
    if not pages or len(pages) > RENDER_POLICY["max_pages"]:
        raise ValueError("PDF page count outside bounded render policy")
    result = []
    for page in pages:
        result.append(page)
        with Image.open(page) as image:
            width, height = image.size
            for index in range(3):
                start = max(0, round(height * (index / 3 - 0.05)))
                stop = min(height, round(height * ((index + 1) / 3 + 0.05)))
                path = page.with_name(f"{page.stem}-tile-{index + 1}.png")
                image.crop((0, start, width, stop)).save(path)
                result.append(path)
    _write(
        destination / "images.json",
        [{"path": p.name, "sha256": _sha(p.read_bytes())} for p in result],
    )
    return result


def run_task(
    packet: Path, task_identifier: str, destination: Path, runtime_root: Path = RUNTIME_ROOT
) -> dict:
    """Run one prepared schema2 task. No frozen answer/label path is an argument."""
    from score2abc.score_eval import validate_candidate
    from score2abc.score_eval_config import digest, task_id, verify_generation, verify_prepared_task

    packet, destination, runtime_root = map(
        lambda p: Path(p).resolve(), (packet, destination, runtime_root)
    )
    task_bytes = (packet / "tasks.json").read_bytes()
    if _sha(task_bytes) != (packet / "tasks.sha256").read_text().strip():
        raise ValueError("Input task packet changed")
    contract = json.loads(task_bytes)
    if contract.get("schema_version") != 2:
        raise ValueError("Runner requires a versioned schema2 input packet")
    verify_generation(contract["generation"], contract["config"])
    tasks = [t for t in contract["tasks"] if t["task_id"] == task_identifier]
    if len(tasks) != 1 or task_id(tasks[0]) != task_identifier:
        raise ValueError("Unknown or changed task")
    task = verify_prepared_task(tasks[0], contract)
    if task.get("adapter") != "codex_subscription" or task.get("provider") != "openai":
        raise ValueError("This runner supports only Codex subscription systems")
    if task.get("settings") != {}:
        raise ValueError("Unsupported generation settings; no silent settings omissions")
    runtime = verify_runtime(runtime_root)
    if runtime != contract["generation"]["harness"]["manifest"]["runner_runtime"]:
        raise ValueError("Prepared runtime differs from packet")
    for name, expected in contract["generation"]["harness"]["manifest"]["implementation"].items():
        if _sha((PACKAGE_ROOT / "score2abc" / name).read_bytes()) != expected:
            raise ValueError("Authoring implementation changed after preparation")
    catalog = json.loads((runtime_root / "model_catalog.json").read_text())["models"]
    model = next((m for m in catalog if m["slug"] == task["model"]), None)
    if not model or task["reasoning_effort"] not in [
        x["effort"] for x in model["supported_reasoning_levels"]
    ]:
        raise ValueError("Requested model/effort absent from frozen installed catalog")
    source = (packet / task["input_pdf_path"]).resolve()
    if not source.is_relative_to(packet) or _sha(source.read_bytes()) != task["input_pdf_sha256"]:
        raise ValueError("Source PDF changed or escaped packet")
    instructions = (packet / "instructions.txt").read_text()
    if _sha(instructions.encode()) != task["instructions_sha256"]:
        raise ValueError("Instructions changed")
    destination.mkdir(parents=True, exist_ok=False)
    started_at = datetime.now(timezone.utc).isoformat()
    start = time.monotonic()
    attempts, response, images = [], None, []
    status, error, phase = "failed", None, "rendering"
    try:
        images = render_inputs(source, destination / "images", runtime)
        prompt = instructions + (
            "\nThe images are ordered by page: full-page overview, then three overlapping crops "
            "from top to bottom. Crops repeat material; do not transcribe repeated pixels twice. "
            "Return a JSON object with abc and decisions, including an alternatives array "
            "(empty if none) for every decision. Do not leave musical placeholders.\n"
        )
        phase = "generation"
        for index in range(task["max_revision_passes"] + 1):
            phase = "generation"
            call = subscription_call(
                runtime,
                runtime_root,
                task["model"],
                task["reasoning_effort"],
                images,
                prompt,
                destination / f"attempt_{index:02d}",
            )
            attempts.append({"attempt": index, **call})
            if call["status"] == "failed":
                if call.get("quarantined"):
                    response = None
                status, error = ("partial" if response else "failed"), call["error"]
                break
            response = call["response"]
            phase = "validation"
            feedback = validate_candidate(response["abc"], response["decisions"])
            _write(destination / f"attempt_{index:02d}/validation.json", feedback)
            if not feedback["issues"]:
                status = "complete"
                break
            status = "partial"
            prompt = instructions + (
                "\nRevise the complete draft using only the attached PDF images and this "
                "truth-free validation feedback. Preserve uncertainty decisions. Return the entire "
                "score in the same JSON shape. This is repair pass "
                + str(index + 1)
                + ".\n"
                + json.dumps({"draft": response, "validation": feedback}, ensure_ascii=False)
            )
    except Exception as exc:
        status = "partial" if response else "failed"
        error = f"{phase}: {type(exc).__name__}: {exc}"
        _write(destination / "failure.json", {"phase": phase, "error": error})
    elapsed = time.monotonic() - start
    record = {
        key: task[key]
        for key in (
            "slug",
            "model_id",
            "model",
            "reasoning_effort",
            "provider",
            "adapter",
            "settings",
            "task_id",
            "trial",
            "system_sha256",
            "generation_sha256",
            "input_pdf_sha256",
        )
    }
    record.update(
        {
            "run_id": destination.name,
            "status": status,
            "authorship": "model",
            "error": error,
            "elapsed_seconds": elapsed,
            "cost_usd": None,
            "human_review_minutes": None,
        }
    )
    if response:
        (destination / "score.abc").write_text(response["abc"])
        _write(destination / "decisions.json", response["decisions"])
        record.update(
            {
                "abc_path": "score.abc",
                "abc_sha256": _sha((destination / "score.abc").read_bytes()),
                "decision_log_path": "decisions.json",
                "decision_log_sha256": _sha((destination / "decisions.json").read_bytes()),
            }
        )
    trace = {
        **record,
        "schema_version": 2,
        "instructions_sha256": task["instructions_sha256"],
        "isolation": "input_only",
        "fresh_context": True,
        "network_access": False,
        "network_scope": "subscription_transport_only_no_model_network_tools",
        "runner_name": "codex_subscription_audited_v1",
        "revision_passes": max(0, len(attempts) - 1),
        "runtime_sha256": digest(runtime),
        "auth_mode": "chatgpt_subscription",
        "requested_model": task["model"],
        "resolved_model": None,
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "last_phase": phase,
        "attempts": attempts,
        "images": (
            json.loads((destination / "images/images.json").read_text())
            if (destination / "images/images.json").is_file()
            else []
        ),
        "cost_basis": "not_exposed_by_subscription",
        "usage": [a["usage"] for a in attempts],
    }
    _write(destination / "run_record.json", trace)
    _write(
        destination / "measurements.json",
        {
            "elapsed_seconds": {
                "value": elapsed,
                "basis": "monotonic_clock",
                "evidence": "Monotonic interval covering rendering, audits, calls and validation",
            }
        },
    )
    record.update(
        {
            "run_record_path": "run_record.json",
            "run_record_sha256": _sha((destination / "run_record.json").read_bytes()),
            "measurement_path": "measurements.json",
            "measurement_sha256": _sha((destination / "measurements.json").read_bytes()),
        }
    )
    _write(
        destination / "registry.json",
        {
            "schema_version": 2,
            "evidence_kind": "actual",
            "config": contract["config"],
            "generation": contract["generation"],
            "runs": [record],
        },
    )
    return record
