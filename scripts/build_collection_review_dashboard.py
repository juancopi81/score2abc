"""Build a read-only review queue from the live collection; never save a score."""

# ruff: noqa: E501  # Keep the self-contained HTML beside its presentation code.

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from urllib.parse import urlencode, urlsplit

PRIORITY = (
    "jaime-llanos_64_la-chata_pasillo_luis-a-calvo",
    "jaime-llanos_41_estrella-del-caribe_danza_luis-a-calvo",
    "jaime-llanos_20_celina_pasillo_rodrigo-mantilla",
)
FOCUSED_URL = "http://127.0.0.1:8777/diagnostics/focused_review_sprint_20261106_v1/index.html"
MODEL_URL = "http://127.0.0.1:8777/reproduction/astra62_api_rumichaca_20261118_v1/index.html"


def local_url(value: str) -> str:
    parsed = urlsplit(value)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "localhost"}
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("Dashboard links must use a local HTTP URL without credentials or query")
    return value


def load_reviews(out: Path) -> list[dict]:
    """Read registered current reviews only, not history or preview-desk copies."""
    out = Path(out).resolve()
    manifest = out / "manifest.jsonl"
    if manifest.is_symlink():
        raise ValueError("Collection manifest must not be a symlink")
    reviews, seen = [], set()
    for line in manifest.read_text().splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        slug = item.get("slug")
        if (
            not isinstance(slug, str)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", slug)
            or slug in seen
        ):
            raise ValueError("Collection requires unique safe work slugs")
        seen.add(slug)
        path = out / slug / "overrides/review.json"
        if path.is_symlink() or not path.resolve().is_relative_to(out):
            raise ValueError("Current review must be a local regular file")
        if not path.is_file():
            continue
        raw = path.read_bytes()
        review = json.loads(raw)
        state, revision = review.get("review_state"), review.get("revision")
        questions = review.get("unresolved")
        title = item.get("metadata", {}).get("title")
        if (
            state not in {"draft", "reviewed"}
            or type(revision) is not int
            or revision < 1
            or not isinstance(questions, list)
            or any(not isinstance(question, str) for question in questions)
            or not isinstance(title, str)
            or not title.strip()
        ):
            raise ValueError(f"Unsupported saved review record: {slug}")
        metadata = review.get("save_metadata") or {}
        reviews.append(
            {
                "slug": slug,
                "title": title.strip(),
                "state": state,
                "revision": revision,
                "questions": [question.strip() for question in questions if question.strip()],
                "human_save": metadata.get("reported_actor") == "human",
                "review_sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    return reviews


def review_order(reviews: list[dict]) -> list[dict]:
    priority = {slug: position for position, slug in enumerate(PRIORITY)}
    return sorted(
        (review for review in reviews if review["state"] == "draft"),
        key=lambda review: (
            priority.get(review["slug"], len(priority)),
            review["title"].casefold(),
            review["slug"],
        ),
    )


def desk_link(review: dict, desk_url: str) -> str:
    return local_url(desk_url).rstrip("/") + "/?" + urlencode({"slug": review["slug"]})


def anchor(url: str, label: str, css: str = "") -> str:
    return (
        f'<a class="{escape(css)}" href="{escape(url, quote=True)}" '
        f'target="_blank" rel="noopener">{escape(label)}</a>'
    )


def question_brief(review: dict) -> str:
    questions = review["questions"]
    if not questions:
        return "<p>No open questions saved. Finish the full-score pass before marking reviewed.</p>"
    first = "".join(f"<li>{escape(question)}</li>" for question in questions[:2])
    rest = "".join(f"<li>{escape(question)}</li>" for question in questions[2:])
    return f"<ul>{first}</ul>" + (
        f"<details><summary>{len(questions) - 2} more saved questions</summary><ul>{rest}</ul></details>"
        if rest
        else ""
    )


def score_card(review: dict, index: int, desk_url: str, focused_url: str) -> str:
    position = "Your next score" if index == 0 else f"Then · {index + 1}"
    progress = (
        "Resume your saved progress" if review["human_save"] else "Begin your full-score pass"
    )
    focus = (
        '<p class="focus">For the crowded row 2 passage, decide rhythm/voice structure and '
        "accidentals separately. The focused packet has three auditions; its alternatives "
        "are proposals, not installed changes.</p>"
        + anchor(focused_url, "Compare the focused auditions")
        if review["slug"] == PRIORITY[0]
        and any(re.search(r"\bS2M1\b", question, re.IGNORECASE) for question in review["questions"])
        else ""
    )
    return f"""<article class="score {'next' if index == 0 else ''}">
<div class="eyebrow">{position}</div><h2>{escape(review['title'])}</h2>
<p class="muted">{progress} · revision {review['revision']} · {len(review['questions'])} open questions</p>
{anchor(desk_link(review, desk_url), 'Open editable score', 'button')}
{focus}<h3>Start with these saved questions</h3>{question_brief(review)}</article>"""


def render_dashboard(
    reviews: list[dict],
    *,
    desk_url: str = "http://127.0.0.1:8766",
    focused_url: str = FOCUSED_URL,
    model_url: str = MODEL_URL,
    built_at: str | None = None,
) -> str:
    for url in (desk_url, focused_url, model_url):
        local_url(url)
    pending = review_order(reviews)
    reviewed = sorted(
        (review for review in reviews if review["state"] == "reviewed"),
        key=lambda review: review["title"].casefold(),
    )
    cards = "".join(
        score_card(review, index, desk_url, focused_url) for index, review in enumerate(pending[:3])
    )
    backlog = "".join(
        f"<tr><td>{anchor(desk_link(review, desk_url), review['title'])}</td>"
        f"<td>{review['revision']}</td><td>{len(review['questions'])}</td></tr>"
        for review in pending[3:]
    )
    approved = "".join(
        f"<li>{escape(review['title'])} <span class='muted'>· revision {review['revision']}</span></li>"
        for review in reviewed
    )
    timestamp = built_at or datetime.now(timezone.utc).isoformat(timespec="seconds")
    summary = (
        "One score at a time. Resume the saved work, settle its questions, then listen through the complete route."
        if pending
        else "Every currently prepared score is marked reviewed. Tell the coordinator when you want the next draft."
    )
    next_score = escape(pending[0]["title"]) if pending else "the score you just finished"
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Your score review queue</title>
<style>
:root{{--ink:#16384a;--muted:#506778;--line:#d8e4e9;--teal:#087e82;--paper:#f6f9fa}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font:16px/1.6 -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif}}
main,header,footer{{max-width:1100px;margin:auto;padding:28px}}h1,h2{{font-family:Georgia,serif;line-height:1.2;font-weight:400}}h1{{font-size:clamp(2.1rem,5vw,3.4rem);margin:10px 0}}h2{{font-size:1.8rem;margin:8px 0}}h3{{font-size:1rem;margin:18px 0 8px}}p{{margin:8px 0 16px}}a{{color:var(--teal);text-underline-offset:3px}}a:focus-visible,summary:focus-visible{{outline:3px solid #d49d36;outline-offset:4px}}
.eyebrow{{text-transform:uppercase;letter-spacing:.12em;font-size:.76rem;font-weight:700}}.muted{{color:var(--muted)}}.counts{{display:flex;gap:12px;flex-wrap:wrap;margin-top:20px}}.count{{background:white;border:1px solid var(--line);border-radius:10px;padding:12px 20px}}.count strong{{font-size:1.8rem;padding-right:8px}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:18px}}.score,.routine,.backlog{{background:white;border:1px solid var(--line);border-radius:14px;padding:24px}}.score.next{{grid-column:1/-1;border-top:5px solid var(--teal)}}.button{{display:inline-block;background:var(--teal);color:white;border-radius:8px;padding:10px 16px;text-decoration:none;font-weight:600;margin:4px 0 12px}}ul,ol{{padding-left:22px}}li{{margin:9px 0}}.focus{{background:#eaf5f5;border-radius:8px;padding:14px}}details{{margin-top:14px}}summary{{cursor:pointer;font-weight:600}}.routine,.backlog{{margin-top:24px}}.finish{{background:#eaf5f5;padding:16px;border-radius:8px}}table{{width:100%;border-collapse:collapse}}th,td{{text-align:left;padding:9px;border-bottom:1px solid var(--line)}}nav{{display:flex;gap:24px;flex-wrap:wrap}}footer{{font-size:.85rem;color:var(--muted)}}@media(max-width:720px){{main,header,footer{{padding:18px}}.grid{{grid-template-columns:1fr}}.score,.routine,.backlog{{padding:18px}}}}
</style></head><body><header><div class="eyebrow">Jaime Llanos · A Puño y Letra</div>
<h1>Your next listening session</h1><p>{summary}</p><div class="counts">
<div class="count"><strong>{len(reviewed)}</strong>reviewed</div>
<div class="count"><strong>{len(pending)}</strong>drafts to review</div>
<div class="count"><strong>{len(reviews)}</strong>prepared scores</div></div></header>
<main><div class="grid">{cards}</div><section class="routine"><h2>A complete review, step by step</h2>
<ol><li><strong>Compare every manuscript row.</strong> Check melody, lower voices, accidentals, rhythm, chords, ties and endings—not just the flagged passages.</li>
<li><strong>Listen through the complete route.</strong> Check repeats, casillas, transitions, returns and the final stopping point. Slow down where needed.</li>
<li><strong>Record each decision in Review notes and resolutions.</strong> Give the source row/bar, your chosen reading and why. Label it <em>source correction</em> (visible ink), <em>editorial reconstruction</em> (damaged or missing music), or <em>editorial interpretation</em> (for example, a return route). Record any uncertainty.</li>
<li><strong>Move resolved questions into those notes.</strong> Remove only the answered item from Questions to resolve. If you cannot settle it, leave it open and send the location to the coordinator.</li>
<li><strong>Save, reopen, and check.</strong> Confirm the saved correction and playback survived reopening. Keep the score as Draft while the full pass is unfinished.</li>
<li><strong>Mark reviewed after the whole-score pass.</strong> All questions must be resolved or explicitly accepted as documented editorial choices, with no item left open. A plausible sound or a valid bar alone is not source proof.</li></ol>
<p class="finish">When finished, tell the coordinator: <strong>“Done — {next_score}, revision [number], marked reviewed.”</strong> Mention any editorial choices or remaining issue; you do not need to copy the whole score.</p>
<details><summary>If the manuscript is damaged</summary><p>A complete performing edition may include a documented best guess. AI reconstruction uses only the supplied PDF. If you use a recording, another edition or your musical knowledge, record that explicitly, including the source link and location when available. Your chosen performance version stays separate from what the PDF visibly proves.</p></details>
</section><section class="backlog"><h2>The rest can wait</h2>
{('<table><thead><tr><th>Score</th><th>Revision</th><th>Open questions</th></tr></thead><tbody>' + backlog + '</tbody></table>') if backlog else '<p>No additional prepared drafts are waiting.</p>'}
<p class="muted">Entusiasmo’s terminal A/B/C lettering has already been identified as navigation. That classification is closed; only its current saved questions need attention.</p>
<details><summary>{len(reviewed)} reviewed scores · preserved</summary><ul>{approved}</ul></details></section>
<section class="routine"><h2>Useful companions</h2><nav>{anchor(focused_url, 'Focused passage comparisons')}{anchor(model_url, 'Latest model comparison')}</nav>
<p class="muted">This page is a read-only snapshot. Edit and save only in the score desk. Rebuilding this page refreshes the queue, revisions and saved questions.</p></section></main>
<footer>Built {escape(timestamp)} from the current local saved reviews. No score or review history was changed.</footer></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--desk-url", default="http://127.0.0.1:8766")
    parser.add_argument("--focused-url", default=FOCUSED_URL)
    parser.add_argument("--model-url", default=MODEL_URL)
    args = parser.parse_args()
    output = args.output or args.out / "eval/collection_review_20261118_v1/index.html"
    if output.suffix != ".html" or "overrides" in output.parts or output.is_symlink():
        raise ValueError("Output must be an HTML file outside score overrides")
    reviews = load_reviews(args.out)
    html = render_dashboard(
        reviews, desk_url=args.desk_url, focused_url=args.focused_url, model_url=args.model_url
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html)
    print(json.dumps({"output": str(output.resolve()), "prepared": len(reviews)}))


if __name__ == "__main__":
    main()
