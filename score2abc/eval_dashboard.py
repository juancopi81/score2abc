"""Render a verified reference freeze as a standalone, offline readiness page."""

# ruff: noqa: E501  # Readable inline CSS and HTML are kept in this standalone renderer.

from __future__ import annotations

import hashlib
import os
from datetime import datetime, timezone
from html import escape
from pathlib import Path, PurePosixPath
from urllib.parse import quote

from score2abc.reference_freeze import verify_freeze

_CSS = """
:root{color-scheme:light;--ink:#153F61;--paper:#FAFCFF;--teal:#227C78;--gold:#9B681F;
  --line:#C9D8E1;--muted:#4E6271;--soft:#EDF4F6}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--paper);
  color:#183547;font:16px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Arial,sans-serif}
a{color:var(--ink);text-decoration-thickness:1px;text-underline-offset:3px}
a:hover{color:var(--teal)}a:focus-visible,summary:focus-visible{outline:3px solid var(--gold);
  outline-offset:3px;border-radius:2px}
.wrap{max-width:1100px;margin:auto;padding:0 32px}.mast{border-bottom:1px solid var(--line);
  background:#F3F8FB}.mast .wrap{padding-top:26px;padding-bottom:30px}
.folio{display:flex;justify-content:space-between;align-items:baseline;gap:16px;
  color:var(--muted);font-size:.88rem}.folio strong{color:var(--ink);font-weight:700}
h1,h2,h3,p{margin-top:0}h1{font:normal clamp(2rem,4vw,2.8rem)/1.09 Georgia,
  "Times New Roman",serif;letter-spacing:-.035em;color:var(--ink);max-width:850px;margin:24px 0 12px}
.lede{max-width:680px;font-size:1.12rem;color:var(--muted);margin:0}
.decision{border-top:4px solid var(--teal);border-bottom:1px solid var(--line);
  display:grid;grid-template-columns:minmax(180px,1fr) minmax(0,2.4fr);gap:20px;
  padding:23px 0 25px;margin:31px 0 0}.decision .label{font-weight:700;color:var(--teal)}
.decision strong{display:block;font:normal 1.55rem/1.18 Georgia,"Times New Roman",serif;
  color:var(--ink);margin-bottom:8px}.decision p{margin:0;color:var(--muted);max-width:65ch}
main{padding-top:38px!important;padding-bottom:65px!important}
.section-head{display:flex;align-items:baseline;justify-content:space-between;gap:16px;
  border-bottom:2px solid var(--ink);padding-bottom:10px;margin:0 0 7px}
h2{font:normal 1.7rem/1.2 Georgia,"Times New Roman",serif;color:var(--ink);margin:0}
.section-head p{color:var(--muted);margin:0;font-size:.92rem}
.register{margin-bottom:37px}.score{border-bottom:1px solid var(--line)}
.score summary{list-style:none;cursor:pointer;display:grid;grid-template-columns:minmax(0,2.1fr) 90px 140px 150px 24px;
  align-items:center;gap:14px;padding:13px 5px}.score summary::-webkit-details-marker{display:none}
.score summary:hover{background:var(--soft)}.score[open] summary{background:var(--soft)}
.work{font-weight:700;color:var(--ink);font-size:1.06rem;overflow-wrap:anywhere}
.work small{display:block;font-size:.78rem;font-weight:400;color:var(--muted);margin-top:3px}
.cell{font-size:.91rem}.pending{color:var(--gold);font-weight:700}.verified{color:var(--teal);font-weight:700}
.caret{font-size:1.45rem;line-height:1;color:var(--teal);text-align:center}
.score[open] .caret{transform:rotate(45deg)}.detail{padding:8px 20px 23px 20px;
  border-left:3px solid var(--teal);margin-left:5px}.detail p{max-width:72ch;margin-bottom:16px}
.detail-grid{display:grid;grid-template-columns:1fr 1fr;gap:15px 32px;margin:0}
.detail-grid div{min-width:0}.detail-grid dt{font-size:.82rem;color:var(--muted)}
.detail-grid dd{margin:3px 0 0;overflow-wrap:anywhere;font-size:.93rem}
.hash{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;font-size:.82rem}
.note{border-left:4px solid var(--gold);padding:15px 20px;background:#FBF7EE;
  margin:28px 0 37px}.note h3{font-size:1rem;color:#6E4A17;margin:0 0 5px}.note p{margin:0;max-width:80ch}
.guide{display:grid;grid-template-columns:1fr 1fr;gap:32px;margin-top:30px}
.guide h2{border-bottom:2px solid var(--ink);padding-bottom:10px;margin-bottom:15px}
.guide p,.guide li{max-width:68ch}.guide ol{margin:0;padding-left:22px}.guide li{margin-bottom:12px}
.empty{border:1px solid var(--line);padding:17px 19px;background:#F3F8FB}.empty strong{color:var(--ink)}
.seal{margin-top:34px;border-top:1px solid var(--line);padding-top:15px;color:var(--muted);
  font-size:.84rem;overflow-wrap:anywhere}.seal strong{color:var(--ink)}
footer{border-top:1px solid var(--line);color:var(--muted);font-size:.83rem;padding:21px 0 30px}
@media(max-width:760px){.wrap{padding-left:19px;padding-right:19px}.decision,.guide{grid-template-columns:1fr}
  .score summary{grid-template-columns:minmax(0,1fr) 24px;gap:8px}
  .score summary .cell{grid-column:1}.score summary .caret{grid-column:2;grid-row:1}
  .detail-grid{grid-template-columns:1fr}.section-head{display:block}.section-head p{margin-top:5px}}
@media(prefers-reduced-motion:reduce){html{scroll-behavior:auto}}
"""


def _text(value: object) -> str:
    return escape(str(value), quote=True)


def _safe_file_link(freeze_root: Path, output: Path, relative: str, label: str) -> str:
    """Link only to a file within the already verified freeze."""
    path = PurePosixPath(relative)
    if path.is_absolute() or not path.parts or any(part in (".", "..") for part in path.parts):
        raise ValueError(f"Unsafe freeze path: {relative}")
    target = (freeze_root / Path(*path.parts)).resolve()
    if not target.is_relative_to(freeze_root.resolve()) or not target.is_file():
        raise ValueError(f"Invalid freeze file: {relative}")
    href = quote(Path(os.path.relpath(target, output.parent.resolve())).as_posix(), safe="/.-_~")
    return f'<a href="{escape(href, quote=True)}">{_text(label)}</a>'


def _date(value: object) -> str:
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("Freeze has an invalid created_at") from exc
    if dt.tzinfo is None:
        raise ValueError("Freeze created_at needs a timezone")
    return dt.astimezone(timezone.utc).strftime("%d %B %Y, %H:%M UTC")


def _score_markup(work: dict, root: Path, output: Path) -> str:
    title = _text(work["title"])
    composer = _text(work["composer"])
    slug = _text(work["slug"])
    revision = _text(work["revision"])
    state = _text(work["review_state"])
    history = work["history_revisions"]
    history_text = (
        ", ".join(str(int(item)) for item in history) if history else "No archived revisions"
    )
    omitted = work.get("omitted_context_references", [])
    context_notice = (
        f"<p><strong>Context coverage:</strong> {len(omitted)} referenced item(s) were not preserved: "
        f"{_text('; '.join(str(item) for item in omitted))}</p>"
        if omitted
        else ""
    )
    links = [
        ("source_pdf_path", "Original source PDF"),
        ("answer_abc_path", "Reviewed ABC export"),
        ("answer_review_path", "Saved review record"),
    ]
    link_markup = "".join(
        f"<div><dt>{_text(label)}</dt><dd>{_safe_file_link(root, output, work[key], 'Open file')}</dd></div>"
        for key, label in links
        if work.get(key)
    )
    return f"""<details class="score"><summary>
      <span class="work">{title}<small>{composer}</small></span>
      <span class="cell">Revision {revision}</span>
      <span class="cell verified">{state}</span>
      <span class="cell pending">Source labels pending</span><span class="caret" aria-hidden="true">+</span>
    </summary><div class="detail">
      <p>The saved edition and its history are preserved and hash checked. Source versus editorial
      reconstruction has not been annotated for this score.</p>
      {context_notice}
      <dl class="detail-grid">{link_markup}
        <div><dt>Work ID</dt><dd>{slug}</dd></div>
        <div><dt>Review SHA-256</dt><dd class="hash">{_text(work['review_sha256'])}</dd></div>
        <div><dt>ABC SHA-256</dt><dd class="hash">{_text(work['abc_sha256'])}</dd></div>
        <div><dt>Preserved history revisions</dt><dd>{_text(history_text)}</dd></div>
        <div><dt>Source manifest line SHA-256</dt><dd class="hash">{_text(work['source_manifest_line_sha256'])}</dd></div>
      </dl>
    </div></details>"""


def render_dashboard(manifest_path: Path, output: Path) -> str:
    """Verify the sealed snapshot, then render its current readiness state."""
    manifest_path = Path(manifest_path).resolve()
    output = Path(output).resolve()
    if manifest_path.name != "manifest.json":
        raise ValueError("Expected a freeze manifest.json")
    if output.is_relative_to(manifest_path.parent):
        raise ValueError("Dashboard output must be outside the immutable freeze")
    manifest = verify_freeze(manifest_path.parent)
    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if (manifest_path.parent / "manifest.sha256").read_text(
        encoding="ascii"
    ).strip() != manifest_sha256:
        raise ValueError("Freeze manifest changed after verification")
    works = manifest["works"]
    if not isinstance(works, list) or len(works) != 6:
        raise ValueError("Readiness dashboard requires six verified references")
    if any(work.get("annotation_status") != "pending" for work in works):
        raise ValueError("Unsupported annotation status in freeze")
    score_rows = "\n".join(_score_markup(work, manifest_path.parent, output) for work in works)
    snapshot_date = _date(manifest["created_at"])
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Reference readiness | score2abc</title><style>{_CSS}</style></head><body>
<header class="mast"><div class="wrap">
  <div class="folio"><strong>score2abc · evaluation desk</strong><span>Reference freeze · {snapshot_date}</span></div>
  <h1>Six reviewed scores, preserved.</h1>
  <p class="lede">A five-minute view of the evidence available before an Astra–Terra score-reading evaluation.</p>
  <div class="decision" role="status"><div class="label">Current decision</div><div>
    <strong>References preserved; comparison not run.</strong>
    <p>The six saved editions, their histories, and source identities passed snapshot verification.
    There are no model outputs or accuracy results in this dashboard.</p>
  </div></div>
</div></header>
<main class="wrap">
  <section class="register" aria-labelledby="register-heading"><div class="section-head">
    <h2 id="register-heading">Reference register</h2><p>Open a score for files and provenance</p>
  </div>{score_rows}</section>
  <aside class="note"><h3>What the reference means</h3><p>These are human-reviewed editions.
  Some passages may include editorial reconstruction rather than notes directly supported by the source PDF.
  We still need to label copied notation separately from editorial choices before measuring how accurately a model reads the PDF.
  History includes the revisions preserved at freeze time; older overwritten saves may not be recoverable.</p></aside>
  <div class="guide"><section aria-labelledby="guide-heading"><h2 id="guide-heading">Five-minute guide</h2>
    <ol><li>Confirm the current decision above. A preserved reference is a starting point, not a model result.</li>
    <li>Open each score to inspect its saved revision, ABC export, source PDF, and history.</li>
    <li>The coordinator will identify source readings and editorial choices. You will only be asked to settle musical ambiguities.</li></ol>
  </section><section aria-labelledby="next-heading"><h2 id="next-heading">What comes next</h2>
    <div class="empty"><strong>Comparison results: unavailable.</strong><p>No Astra or Terra run is attached.</p></div>
    <p>The coordinator will label the references and prepare the same bounded score-reading task for both models,
    then run and review their outputs before adding any comparison metrics.</p>
  </section></div>
  <div class="seal"><strong>Verified manifest SHA-256:</strong> <span class="hash">{manifest_sha256}</span></div>
</main><footer><div class="wrap">Built from a verified, local reference freeze. Keep this HTML beside its snapshot to open file links offline.</div></footer>
</body></html>"""
