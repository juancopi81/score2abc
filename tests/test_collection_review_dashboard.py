"""The queue is a current read-only view, not a review-state mutation."""

import hashlib
import json
import sys

import pytest

from scripts import build_collection_review_dashboard as dashboard


def collection(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    slugs = [*reversed(dashboard.PRIORITY), "another_work", "approved_work"]
    titles = {
        dashboard.PRIORITY[0]: "La Chata",
        dashboard.PRIORITY[1]: "Estrella del Caribe",
        dashboard.PRIORITY[2]: "Celina",
        "another_work": "Other score",
        "approved_work": "Approved score",
    }
    records = []
    for index, slug in enumerate(slugs):
        records.append({"slug": slug, "metadata": {"title": titles[slug]}})
        directory = out / slug / "overrides"
        directory.mkdir(parents=True)
        (directory / "review.json").write_text(
            json.dumps(
                {
                    "review_state": "reviewed" if slug == "approved_work" else "draft",
                    "revision": index + 1,
                    "unresolved": [] if slug == "approved_work" else ["A saved question?"],
                    "review_notes": "Keep the saved notes intact.",
                    "save_metadata": {"reported_actor": "human"},
                }
            )
        )
        history = directory / "history"
        history.mkdir()
        (history / "older.json").write_text("historical bytes")
    (out / "manifest.jsonl").write_text("\n".join(json.dumps(row) for row in records))
    preview = out / "collection/preview/another_work/overrides"
    preview.mkdir(parents=True)
    (preview / "review.json").write_text("not live JSON; must not be read")
    return out


def hashes(out):
    return {
        str(path.relative_to(out)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in out.rglob("*.json")
    }


def test_current_states_and_priority_advance_after_rebuild(tmp_path):
    out = collection(tmp_path)
    reviews = dashboard.load_reviews(out)
    assert [row["slug"] for row in dashboard.review_order(reviews)[:3]] == list(dashboard.PRIORITY)
    assert len(reviews) == 5
    html = dashboard.render_dashboard(reviews, built_at="synthetic snapshot")
    assert "<strong>1</strong>reviewed" in html
    assert "<strong>4</strong>drafts to review" in html
    assert "<strong>5</strong>prepared scores" in html
    grid = html.split('<div class="grid">')[1].split('<section class="routine">')[0]
    assert grid.index("La Chata") < grid.index("Estrella del Caribe") < grid.index("Celina")
    review_path = out / dashboard.PRIORITY[0] / "overrides/review.json"
    review = json.loads(review_path.read_text())
    review.update(review_state="reviewed", revision=20, unresolved=[])
    review_path.write_text(json.dumps(review))
    updated = dashboard.load_reviews(out)
    assert dashboard.review_order(updated)[0]["slug"] == dashboard.PRIORITY[1]
    refreshed = dashboard.render_dashboard(updated)
    assert "<strong>2</strong>reviewed" in refreshed
    assert "<strong>3</strong>drafts to review" in refreshed
    assert "La Chata <span class='muted'>· revision 20" in refreshed


def test_builder_writes_only_dashboard_and_preserves_reviews_and_history(tmp_path, monkeypatch):
    out = collection(tmp_path)
    before = hashes(out)
    monkeypatch.setattr(sys, "argv", ["build_collection_review_dashboard.py", str(out)])
    dashboard.main()
    assert hashes(out) == before
    html = (out / "eval/collection_review_20261118_v1/index.html").read_text()
    assert "http://127.0.0.1:8766/?slug=" + dashboard.PRIORITY[0] in html
    assert "Review notes and resolutions" in html
    assert "Save, reopen, and check" in html
    assert "AI reconstruction uses only the supplied PDF" in html
    assert "That classification is closed" in html
    assert "source correction" in html and "editorial reconstruction" in html
    assert "editorial interpretation" in html and "revision [number]" in html
    assert "<script" not in html


def test_saved_questions_are_escaped_and_empty_queue_is_not_a_new_task(tmp_path):
    reviews = dashboard.load_reviews(collection(tmp_path))
    reviews[0]["questions"] = ["<script>alert('bad')</script>"]
    html = dashboard.render_dashboard(reviews)
    assert "&lt;script&gt;" in html and "<script>" not in html
    for review in reviews:
        review.update(state="reviewed", questions=[])
    finished = dashboard.render_dashboard(reviews)
    assert "Every currently prepared score is marked reviewed" in finished
    assert "Your next score" not in finished


def test_la_chata_focused_guidance_tracks_its_current_open_question(tmp_path):
    reviews = dashboard.load_reviews(collection(tmp_path))
    la_chata = next(review for review in reviews if review["slug"] == dashboard.PRIORITY[0])
    la_chata["questions"] = ["S2m1: resolve the crowded span.", "S7m6: check the chord."]
    assert "Compare the focused auditions" in dashboard.render_dashboard(reviews)
    la_chata["questions"] = ["S7m6: check the chord.", "S2m10: a different bar."]
    html = dashboard.render_dashboard(reviews)
    assert "Compare the focused auditions" not in html
    assert "For the crowded row 2 passage" not in html
    assert "S7m6: check the chord" in html


@pytest.mark.parametrize("slug", ["../outside", "duplicate"])
def test_unsafe_or_duplicate_manifest_slug_is_rejected(tmp_path, slug):
    out = collection(tmp_path)
    rows = [{"slug": slug, "metadata": {"title": "Wrong"}}]
    if slug == "duplicate":
        rows.append(rows[0])
    (out / "manifest.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    with pytest.raises(ValueError, match="unique safe"):
        dashboard.load_reviews(out)


def test_remote_dashboard_links_and_override_output_are_rejected(tmp_path, monkeypatch):
    out = collection(tmp_path)
    with pytest.raises(ValueError, match="local HTTP"):
        dashboard.render_dashboard(dashboard.load_reviews(out), desk_url="https://example.com")
    forbidden = out / dashboard.PRIORITY[0] / "overrides/score.html"
    monkeypatch.setattr(
        sys,
        "argv",
        ["build_collection_review_dashboard.py", str(out), "--output", str(forbidden)],
    )
    with pytest.raises(ValueError, match="outside score overrides"):
        dashboard.main()
    assert not forbidden.exists()
