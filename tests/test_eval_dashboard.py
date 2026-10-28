import hashlib
from pathlib import Path

import pytest

from score2abc import eval_dashboard


def _manifest(root: Path) -> dict:
    data = b"fixture"
    (root / "manifest.json").write_bytes(data)
    (root / "manifest.sha256").write_text(hashlib.sha256(data).hexdigest() + "\n", encoding="ascii")
    works = []
    for index in range(6):
        slug = f"work-{index}"
        files = {
            "source_pdf_path": f"inputs/{slug}/source.pdf",
            "answer_abc_path": f"answers/{slug}/review.abc",
            "answer_review_path": f"answers/{slug}/review.json",
        }
        for name in files.values():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"verified")
        works.append(
            {
                "title": "<script>alert(1)</script>" if index == 0 else f"Work {index}",
                "composer": "Composer & arranger",
                "slug": slug,
                "revision": index + 1,
                "review_state": "reviewed",
                "annotation_status": "pending",
                "history_revisions": [1] if index else [],
                "review_sha256": "a" * 64,
                "abc_sha256": "b" * 64,
                "source_manifest_line_sha256": "c" * 64,
                **files,
            }
        )
    return {"created_at": "2026-10-28T12:00:00Z", "works": works}


def test_renders_verified_snapshot_without_imaginary_results(tmp_path: Path, monkeypatch) -> None:
    manifest = tmp_path / "freeze" / "manifest.json"
    manifest.parent.mkdir()
    snapshot = _manifest(manifest.parent)
    monkeypatch.setattr(eval_dashboard, "verify_freeze", lambda path: snapshot)

    html = eval_dashboard.render_dashboard(manifest, tmp_path / "dashboard.html")

    assert "References preserved; comparison not run." in html
    assert "Comparison results: unavailable." in html
    assert html.count("Source labels pending") == 6
    assert "before measuring how accurately a model reads the PDF" in html
    assert html.count('<details class="score">') == 6
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "<script>alert(1)</script>" not in html
    assert "Composer &amp; arranger" in html
    assert "https://" not in html
    assert 'href="freeze/inputs/work-0/source.pdf"' in html


def test_refuses_unverified_or_unsupported_snapshot(tmp_path: Path, monkeypatch) -> None:
    manifest = tmp_path / "freeze" / "manifest.json"
    manifest.parent.mkdir()

    def failed_verification(_path: Path) -> dict:
        raise ValueError("Source changed")

    monkeypatch.setattr(eval_dashboard, "verify_freeze", failed_verification)
    with pytest.raises(ValueError, match="Source changed"):
        eval_dashboard.render_dashboard(manifest, tmp_path / "dashboard.html")

    snapshot = _manifest(manifest.parent)
    snapshot["works"][0]["annotation_status"] = "complete"
    monkeypatch.setattr(eval_dashboard, "verify_freeze", lambda path: snapshot)
    with pytest.raises(ValueError, match="Unsupported annotation status"):
        eval_dashboard.render_dashboard(manifest, tmp_path / "dashboard.html")

    manifest.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="changed after verification"):
        eval_dashboard.render_dashboard(manifest, tmp_path / "dashboard.html")


def test_refuses_path_escape_even_after_verifier(tmp_path: Path, monkeypatch) -> None:
    freeze = tmp_path / "freeze"
    freeze.mkdir()
    snapshot = _manifest(freeze)
    snapshot["works"][0]["source_pdf_path"] = "../../outside.pdf"
    monkeypatch.setattr(eval_dashboard, "verify_freeze", lambda path: snapshot)
    with pytest.raises(ValueError, match="Unsafe freeze path"):
        eval_dashboard.render_dashboard(freeze / "manifest.json", tmp_path / "dashboard.html")


def test_output_cannot_invalidate_the_freeze(tmp_path: Path, monkeypatch) -> None:
    freeze = tmp_path / "freeze"
    freeze.mkdir()
    monkeypatch.setattr(eval_dashboard, "verify_freeze", lambda path: _manifest(freeze))

    with pytest.raises(ValueError, match="outside the immutable freeze"):
        eval_dashboard.render_dashboard(freeze / "manifest.json", freeze / "dashboard.html")
