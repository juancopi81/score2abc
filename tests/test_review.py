"""Review persistence, isolation, and local HTTP boundaries."""

import json
import threading
import urllib.error
import urllib.request

import pytest

from score2abc.review import ReviewApp, ReviewError, create_server

ABC = 'X:1\nT:Edited\nM:3/4\nL:1/8\nK:G\n"Am"(3ABC [df]2- [df]2 | z2 G4 |]\n'


@pytest.fixture
def app(tmp_path, monkeypatch):
    (tmp_path / "manifest.jsonl").write_text(
        json.dumps(
            {
                "slug": "sample",
                "pdf_path": "source.pdf",
                "metadata": {"title": "Sample", "composer": "Composer", "rhythm": "Pasillo"},
            }
        )
    )
    work = tmp_path / "sample"
    (work / "final").mkdir(parents=True)
    (work / "final/melody_with_chords.abc").write_text("X:1\nM:4/4\nK:C\nCDEF|\n")
    app = ReviewApp(tmp_path, renderer_dir=tmp_path / "missing-renderer")
    monkeypatch.setattr(
        app,
        "validate",
        lambda abc: {
            "valid": abc == ABC,
            "errors": [] if abc == ABC else ["Invalid notation"],
            "warnings": [],
            "note_count": 8 if abc == ABC else 0,
        },
    )
    return app


def save(app, abc=ABC, revision=0, state="draft", **extra):
    return app.save("sample", {"abc": abc, "revision": revision, "review_state": state, **extra})


def test_placeholder_is_not_recognition_and_reads_do_not_write(app):
    before = sorted(app.out_dir.rglob("*"))
    work = app.work("sample")
    assert work["source_status"] == "no_recognition"
    assert "CDEF" not in work["abc"]
    assert "M:?\n" in work["abc"] and "K:?\n" in work["abc"]
    assert work["review_state"] == "unreviewed"
    assert sorted(app.out_dir.rglob("*")) == before
    with pytest.raises(ReviewError, match="Save a valid draft"):
        app.export("sample")


def test_roundtrip_reopen_exact_notation_and_regenerated_base(app, monkeypatch):
    original = app.path("sample", "final/melody_with_chords.abc").read_bytes()
    result = save(app, review_ms=1234)
    assert result["revision"] == 1
    assert result["active_review_ms"] == 1234
    assert app.export("sample") == ABC.encode()
    assert app.path("sample", "final/melody_with_chords.abc").read_bytes() == original
    reopened = ReviewApp(app.out_dir)
    monkeypatch.setattr(reopened, "validate", app.validate)
    assert reopened.work("sample")["abc"] == ABC
    stage = app.path("sample", "stages")
    stage.mkdir()
    (stage / "extract_melody.json").write_text('{"status":"success"}')
    assert reopened.work("sample")["base_changed"]
    assert reopened.work("sample")["abc"] == ABC
    save(reopened, revision=1, state="reviewed", review_ms=100)
    assert reopened.work("sample")["active_review_ms"] == 1334
    with pytest.raises(ReviewError) as exc:
        save(app, revision=1)
    assert exc.value.status == 409


def test_invalid_drafts_allowed_but_review_and_export_require_validation(app):
    saved = save(app, abc="unfinished", unresolved=["Missing bass voice"])
    assert not saved["validation"]["valid"]
    with pytest.raises(ReviewError, match="cannot be exported"):
        app.export("sample")
    with pytest.raises(ReviewError) as exc:
        save(app, abc="unfinished", revision=1, state="reviewed")
    assert exc.value.status == 422
    with pytest.raises(ReviewError):
        save(app, revision=1, state="reviewed", unresolved=["Check accidentals"])


def test_missing_renderer_fails_closed(app, monkeypatch):
    monkeypatch.delattr(app, "validate")
    assert not app.renderer_available
    assert not app.validate(ABC)["valid"]
    save(app)
    with pytest.raises(ReviewError, match="renderer unavailable"):
        app.export("sample")
    with pytest.raises(ReviewError):
        save(app, revision=1, state="reviewed")


def test_assets_exclude_chord_bands_and_symlinks(app):
    systems = app.path("sample", "systems")
    systems.mkdir()
    (systems / "system_001.png").write_bytes(b"png")
    (systems / "chord_region_above_001.png").write_bytes(b"png")
    assert list(app.assets("sample")) == ["system-system_001"]
    (systems / "system_002.png").symlink_to(systems / "system_001.png")
    with pytest.raises(ReviewError, match="Unsafe"):
        app.assets("sample")
    with pytest.raises(ReviewError):
        app.path("../sample")


def test_http_csrf_host_traversal_and_export(app):
    try:
        server = create_server(app)
    except PermissionError:
        pytest.skip("Sandbox does not allow loopback binding")
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    base = f"http://127.0.0.1:{server.server_port}"

    def request(path, body=None, headers=None):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(base + path, data=data, headers=headers or {})
        try:
            with urllib.request.urlopen(req) as response:
                return response.status, response.read()
        except urllib.error.HTTPError as error:
            return error.code, error.read()

    try:
        assert request("/api/state")[0] == 200
        body = {"abc": ABC, "revision": 0, "review_state": "draft"}
        headers = {"Content-Type": "application/json", "X-Review-Token": app.token}
        assert request("/api/work/sample", body)[0] == 403
        assert (
            request("/api/work/sample", body, {**headers, "Origin": "https://evil.example"})[0]
            == 403
        )
        assert request("/api/state", headers={"Host": "evil.example"})[0] == 403
        assert request("/assets/sample/%2e%2e")[0] == 404
        assert request("/api/work/sample", body, headers)[0] == 200
        assert request("/api/work/sample/export.abc") == (200, ABC.encode())
        assert request("/api/work/sample", body, headers)[0] == 409
        with urllib.request.urlopen(base + "/api/work/sample/export.abc") as response:
            assert 'filename="sample-draft.abc"' in response.headers["Content-Disposition"]
            assert "object-src 'none'" in response.headers["Content-Security-Policy"]
        body.update(revision=1, review_state="reviewed")
        assert request("/api/work/sample", body, headers)[0] == 200
        with urllib.request.urlopen(base + "/api/work/sample/export.abc") as response:
            assert 'filename="sample-reviewed.abc"' in response.headers["Content-Disposition"]
            assert response.read() == ABC.encode()
    finally:
        server.shutdown()
        server.server_close()
        worker.join()


def test_installed_renderer_smoke(tmp_path):
    (tmp_path / "manifest.jsonl").write_text("")
    app = ReviewApp(tmp_path)
    if not app.renderer_available:
        pytest.skip("Optional local abc2svg/Node unavailable")
    assert app.validate(ABC)["valid"]
    assert app.validate(ABC)["note_count"] > 0
    assert not app.validate(ABC + "\n%%beginjs\nvar harmless = 1;\n%%endjs")["valid"]
    assert not app.validate("X:1\nK:C\nC|\n")["valid"]
    assert not app.validate(ABC + "\nX:2\nM:3/4\nK:C\nC|")["valid"]
    assert not app.validate("X:1\nM:3/4\nK:C\n[CE|")["valid"]
    assert not app.validate("X:1\nM:?\nK:?\nC|")["valid"]


def test_manifest_rejects_escaping_slug(tmp_path):
    (tmp_path / "manifest.jsonl").write_text(
        json.dumps(
            {
                "slug": "../sample",
                "pdf_path": "source.pdf",
                "metadata": {"title": "Sample", "composer": "Composer", "rhythm": "Pasillo"},
            }
        )
    )
    with pytest.raises(ReviewError, match="Unsafe manifest slug"):
        ReviewApp(tmp_path)


def test_local_source_pdf_preferred_and_metadata_context(app):
    local_pdf = app.path("sample", "source.pdf")
    local_pdf.write_bytes(b"%PDF-local")
    (app.out_dir / "source.pdf").write_bytes(b"%PDF-fallback")
    assert app.assets("sample")["source-pdf"][0] == local_pdf
    app.works["sample"].metadata.time_signature = "3/4"
    app.works["sample"].metadata.key_hint = "Bb"
    assert "M:3/4\n" in app.base("sample")[0]
    assert "K:Bb\n" in app.base("sample")[0]
    app.works["sample"].metadata.key_hint = "one flat"
    assert "K:?\n" in app.base("sample")[0]


def test_long_review_session_saves_and_caps_accounting(app):
    two_hours = 2 * 60 * 60 * 1000
    first = save(app, review_ms=two_hours)
    assert first["abc"] == ABC
    assert first["active_review_ms"] == two_hours
    second = save(app, revision=1, review_ms=48 * 60 * 60 * 1000)
    assert second["active_review_ms"] == two_hours + 24 * 60 * 60 * 1000
    assert app.export("sample") == ABC.encode()


@pytest.mark.parametrize("interval", [-1, True, 1.5, "100"])
def test_invalid_review_accounting_rejected(app, interval):
    with pytest.raises(ReviewError, match="nonnegative integer"):
        save(app, review_ms=interval)
