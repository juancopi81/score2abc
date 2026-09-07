"""Local human review. Canonical pipeline and experimental artifacts are read-only."""

from __future__ import annotations

import hashlib
import json
import mimetypes
import os
import re
import secrets
import shutil
import subprocess
import tempfile
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

from score2abc.schemas import WorkItem

MAX_BODY = 256_000
# Cap accounting per save at 24 hours without preventing long-session notation saves.
MAX_REVIEW_INTERVAL_MS = 24 * 60 * 60 * 1000
PACKAGE = Path(__file__).resolve().parent


class ReviewError(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


def _safe(root: Path, path: Path) -> Path:
    """Do not follow symlinks, including those inside otherwise allowed roots."""
    path = Path(os.path.abspath(path))
    if not path.is_relative_to(root) or any(p.is_symlink() for p in (path, *path.parents)):
        raise ReviewError("Unsafe source or review path", 400)
    return path


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class ReviewApp:
    def __init__(self, out_dir: Path, renderer_dir: Path | None = None):
        self.out_dir = Path(out_dir).resolve()
        self.token = secrets.token_urlsafe(32)
        self.lock = threading.RLock()
        self.initial_slug = None
        renderer = shutil.which("abc2svg")
        self.renderer_dir = (
            Path(renderer_dir).resolve()
            if renderer_dir is not None
            else Path(renderer).resolve().parent if renderer else None
        )
        self.node = shutil.which("node")
        self.works = {}
        manifest = _safe(self.out_dir, self.out_dir / "manifest.jsonl")
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = WorkItem.model_validate_json(line)
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", item.slug):
                raise ReviewError("Unsafe manifest slug")
            _safe(self.out_dir, self.out_dir / item.slug)
            if item.slug in self.works:
                raise ReviewError("Duplicate manifest slug")
            self.works[item.slug] = item

    @property
    def renderer_available(self) -> bool:
        return bool(
            self.node and self.renderer_dir and (self.renderer_dir / "abc2svg-1.js").is_file()
        )

    def path(self, slug: str, suffix: str = "") -> Path:
        if slug not in self.works:
            raise ReviewError("Unknown work", 404)
        return _safe(self.out_dir, self.out_dir / slug / suffix)

    def validate(self, abc: str) -> dict:
        failure = {"valid": False, "errors": [], "warnings": [], "note_count": 0}
        if not self.renderer_available:
            return {**failure, "errors": ["Local abc2svg and Node renderer unavailable."]}
        try:
            result = subprocess.run(
                [self.node, str(PACKAGE / "review_render.js"), str(self.renderer_dir)],
                input=json.dumps({"abc": abc}),
                text=True,
                capture_output=True,
                timeout=8,
                check=True,
            )
            return json.loads(result.stdout)
        except (subprocess.SubprocessError, ValueError, OSError):
            return {**failure, "errors": ["Notation validation failed or timed out."]}

    def base(self, slug: str) -> tuple[str, str]:
        stage = _json(self.path(slug, "stages/extract_melody.json"))
        abc_path = self.path(slug, "final/melody_with_chords.abc")
        if stage.get("status") == "success" and abc_path.is_file():
            backend = _json(self.path(slug, "stages/extract_musicxml.json"))
            kind = backend.get("params", {}).get("backend")
            status = "fixture_manual" if kind == "fixture" else "recognized"
            return abc_path.read_text(encoding="utf-8"), status
        title = self.works[slug].metadata.title.replace("\n", " ").replace("\r", " ")
        metadata = self.works[slug].metadata
        meter = metadata.time_signature or "?"
        if not re.fullmatch(r"(?:[0-9]+/[0-9]+|C\|?)", meter):
            meter = "?"
        key = metadata.key_hint or "?"
        if not re.fullmatch(r"[A-G][#b]?(?:m|maj|min|dor|phr|lyd|mix|aeo|loc)?", key):
            key = "?"
        return (
            f"X:1\nT:{title}\nM:{meter}\nL:1/8\nK:{key}\n"
            "% Confirm meter and key from manuscript before review\n",
            "no_recognition",
        )

    def assets(self, slug: str) -> dict:
        result = {}
        for folder, pattern, kind in (
            ("pages", "page_*.png", "page"),
            ("systems", "system_*.png", "system"),
        ):
            directory = self.path(slug, folder)
            for path in sorted(directory.glob(pattern)):
                if not re.fullmatch(rf"{kind}_\d+\.png", path.name):
                    continue
                path = _safe(self.out_dir, path)
                result[f"{kind}-{path.stem}"] = (path, kind, path.stem.replace("_", " "))
        # A manifest PDF may live in the sibling dataset directory; never serve arbitrary paths.
        local_pdf = self.path(slug, "source.pdf")
        if local_pdf.is_file():
            result["source-pdf"] = (local_pdf, "pdf", "Original PDF")
            return result
        pdf = self.out_dir / self.works[slug].pdf_path
        try:
            pdf = _safe(self.out_dir.parent, pdf)
            if pdf.is_file() and pdf.suffix.lower() == ".pdf":
                result["source-pdf"] = (pdf, "pdf", "Original PDF")
        except ReviewError:
            pass
        return result

    def work(self, slug: str) -> dict:
        with self.lock:
            base, status = self.base(slug)
            draft = _json(self.path(slug, "overrides/review.json"))
            abc = draft.get("abc", base)
            return {
                "slug": slug,
                "metadata": self.works[slug].metadata.model_dump(),
                "abc": abc,
                "revision": draft.get("revision", 0),
                "review_state": draft.get("review_state", "unreviewed"),
                "unresolved": draft.get("unresolved", []),
                "active_review_ms": draft.get("active_review_ms", 0),
                "source_status": status,
                "base_changed": bool(draft and draft.get("base_abc_sha256") != _hash(base)),
                "sources": [
                    {"id": id_, "label": label, "kind": kind, "url": f"/assets/{slug}/{id_}"}
                    for id_, (_, kind, label) in self.assets(slug).items()
                ],
                "validation": self.validate(abc),
            }

    def state(self) -> dict:
        works = []
        for slug, item in self.works.items():
            saved = _json(self.path(slug, "overrides/review.json"))
            works.append(
                {
                    "slug": slug,
                    **item.metadata.model_dump(),
                    "review_state": saved.get("review_state", "unreviewed"),
                    "has_draft": bool(saved),
                }
            )
        return {
            "csrf_token": self.token,
            "renderer_available": self.renderer_available,
            "initial_slug": self.initial_slug,
            "works": works,
        }

    def save(self, slug: str, body: dict) -> dict:
        if not isinstance(body, dict):
            raise ReviewError("Expected JSON object")
        abc, state = body.get("abc"), body.get("review_state")
        unresolved, elapsed = body.get("unresolved", []), body.get("review_ms", 0)
        if not isinstance(abc, str) or len(abc.encode("utf-8")) > MAX_BODY:
            raise ReviewError("ABC must be bounded text")
        if state not in ("draft", "reviewed"):
            raise ReviewError("Review state must be draft or reviewed")
        if (
            not isinstance(unresolved, list)
            or len(unresolved) > 100
            or any(not isinstance(v, str) or len(v) > 2000 for v in unresolved)
        ):
            raise ReviewError("Invalid unresolved items")
        if type(elapsed) is not int or elapsed < 0:
            raise ReviewError("review_ms must be a nonnegative integer")
        elapsed = min(elapsed, MAX_REVIEW_INTERVAL_MS)
        with self.lock:
            target = self.path(slug, "overrides/review.json")
            previous = _json(target)
            if type(body.get("revision")) is not int or body["revision"] != previous.get(
                "revision", 0
            ):
                raise ReviewError("Stale revision; reopen the saved work before saving", 409)
            validation = self.validate(abc)
            if state == "reviewed" and (
                not validation["valid"] or not validation["note_count"] or unresolved
            ):
                raise ReviewError(
                    "Reviewed requires valid notation with notes and no unresolved items", 422
                )
            base, _ = self.base(slug)
            payload = {
                "version": 1,
                "abc": abc,
                "revision": body["revision"] + 1,
                "review_state": state,
                "unresolved": unresolved,
                "active_review_ms": previous.get("active_review_ms", 0) + elapsed,
                "base_abc_sha256": previous.get("base_abc_sha256", _hash(base)),
                "base_abc": previous.get("base_abc", base),
            }
            target.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=target.parent, delete=False
            ) as handle:
                temp = Path(handle.name)
                try:
                    json.dump(payload, handle, ensure_ascii=False, indent=2)
                    handle.flush()
                    os.fsync(handle.fileno())
                    os.replace(temp, target)
                finally:
                    temp.unlink(missing_ok=True)
            return self.work(slug)

    def export(self, slug: str) -> bytes:
        with self.lock:
            saved = _json(self.path(slug, "overrides/review.json"))
            if not saved:
                raise ReviewError("Save a valid draft before exporting", 422)
            validation = self.validate(saved["abc"])
            if not validation["valid"] or not validation["note_count"]:
                raise ReviewError(
                    "Saved notation cannot be exported: "
                    + "; ".join(validation["errors"] or ["No notes found"]),
                    422,
                )
            return saved["abc"].encode("utf-8")


def create_server(app: ReviewApp, host: str = "127.0.0.1", port: int = 0):
    if host != "127.0.0.1":
        raise ValueError("Review server must bind to 127.0.0.1")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def respond(self, data, content_type="application/json", status=200, filename=None):
            if not isinstance(data, bytes):
                data = json.dumps(data, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
                "img-src 'self' data:; font-src 'self' data:; connect-src 'self'; "
                "object-src 'none'; base-uri 'none'; frame-ancestors 'none'",
            )
            if filename:
                self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
            self.end_headers()
            self.wfile.write(data)

        def dispatch(self, mutation=False):
            try:
                authority = f"127.0.0.1:{self.server.server_port}"
                if self.headers.get("Host") != authority:
                    raise ReviewError("Invalid Host", 403)
                origin = self.headers.get("Origin")
                if origin and origin != f"http://{authority}":
                    raise ReviewError("Cross-origin request rejected", 403)
                path = unquote(urlsplit(self.path).path)
                if mutation:
                    if self.headers.get("X-Review-Token") != app.token:
                        raise ReviewError("Invalid review token", 403)
                    if self.headers.get_content_type() != "application/json":
                        raise ReviewError("Expected application/json", 415)
                    size = int(self.headers.get("Content-Length", "0"))
                    if not 0 < size <= MAX_BODY:
                        raise ReviewError("Request too large or empty", 413)
                    match = re.fullmatch(r"/api/work/([^/]+)", path)
                    if not match:
                        raise ReviewError("Unknown endpoint", 404)
                    return self.respond(app.save(match[1], json.loads(self.rfile.read(size))))
                if path == "/api/state":
                    return self.respond(app.state())
                match = re.fullmatch(r"/api/work/([^/]+)(/export\.abc)?", path)
                if match:
                    if match[2]:
                        with app.lock:
                            exported = app.export(match[1])
                            saved = _json(app.path(match[1], "overrides/review.json"))
                            suffix = "reviewed" if saved["review_state"] == "reviewed" else "draft"
                            return self.respond(
                                exported,
                                "text/vnd.abc; charset=utf-8",
                                filename=f"{match[1]}-{suffix}.abc",
                            )
                    return self.respond(app.work(match[1]))
                match = re.fullmatch(r"/assets/([^/]+)/([^/]+)", path)
                if match:
                    asset = app.assets(match[1]).get(match[2])
                    if not asset:
                        raise ReviewError("Unknown asset", 404)
                    return self.respond(
                        asset[0].read_bytes(),
                        mimetypes.guess_type(asset[0])[0] or "application/octet-stream",
                    )
                if path in ("/", "/review_ui.js", "/review_ui.css"):
                    file = PACKAGE / ("review_ui.html" if path == "/" else path[1:])
                elif path in ("/renderer/abc2svg-1.js", "/renderer/toaudio-1.js"):
                    if not app.renderer_dir:
                        raise ReviewError("Renderer unavailable", 404)
                    file = app.renderer_dir / path.rsplit("/", 1)[1]
                else:
                    raise ReviewError("Unknown endpoint", 404)
                if not file.is_file():
                    raise ReviewError("Asset unavailable", 404)
                return self.respond(
                    file.read_bytes(), mimetypes.guess_type(file)[0] or "text/plain"
                )
            except ReviewError as exc:
                self.respond({"error": str(exc)}, status=exc.status)
            except (ValueError, OSError, KeyError):
                self.respond({"error": "Invalid request or unavailable local artifact"}, status=400)

        def do_GET(self):
            self.dispatch()

        def do_POST(self):
            self.dispatch(mutation=True)

    return ThreadingHTTPServer((host, port), Handler)


def run_review(
    out_dir: Path, slug: str | None = None, port: int = 8766, open_browser: bool = False
) -> int:
    app = ReviewApp(out_dir)
    if slug is not None:
        app.path(slug)
    app.initial_slug = slug
    server = create_server(app, port=port)
    url = f"http://127.0.0.1:{server.server_port}/"
    print(f"Local collection review: {url} (Ctrl-C to stop)")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0
