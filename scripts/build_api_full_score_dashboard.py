"""Add a reader-first summary to a sealed full-score API report, without rescoring."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_api_full_score_pilot import grader, render, runner, text  # noqa: E402


def summary(report: dict) -> str:
    final = report["stages"]["reviewed"]
    prior = report["subscription_baseline"]["reviewed"]
    issues = final["validation"]["issues"]
    meter = [str(i["measure"]) for i in issues if i["kind"] == "meter"]
    worse = final["metrics"]["edition_note_f1"] < prior["metrics"]["edition_note_f1"]
    title = "Pause before scaling" if issues or worse else "Pilot complete; review before scaling"
    form = "matches" if final["metrics"]["form_exact_match"] else "differs from"
    numbers = ", ".join(meter) or "none"
    return f"""<section class="section" aria-label="Five-minute findings">
<p class="eyebrow">Decision</p><h2>{title}</h2>
<p>Both API sessions completed normally. The reader repaired the author's bar grouping,
and the final playback route {form} the adopted edition. Musical quality still needs review.</p>
<ul><li><strong>Notes:</strong> {final['metrics']['edition_note_f1']:.1%} agreement with the
reviewed edition versus {prior['metrics']['edition_note_f1']:.1%} for the saved subscription result.
This metric checks pitch, onset and duration together.</li>
<li><strong>Rhythm warnings:</strong> {len(meter)} bars ({text(numbers)}).
These are flagged for inspection; the model's “finished” claim does not clear them.</li>
<li><strong>Next step:</strong> inspect these saved traces and disagreements offline before
another paid experiment. Keep this candidate separate from the musician-reviewed score.</li></ul>
<p class="scope">“Partial” below is the evaluator's musical-validation status, not a timeout.
The author used 67 divisions; review regrouped them into 45 written bars.
Chord agreement measures the adopted edition, which can contain editorial additions.</p>
</section>"""


def dashboard(report: dict) -> str:
    page = render(report)
    # The historical renderer's percent helper returns markup, while its table
    # escapes cell values. Unwrap only that known numeric markup for display.
    page = re.sub(r"&lt;strong class=&quot;metric&quot;&gt;([0-9.]+%)&lt;/strong&gt;", r"\1", page)
    return page.replace('<main class="wrap">', '<main class="wrap">' + summary(report), 1)


def build(root: Path) -> None:
    root = root.resolve()
    path = root / "evaluation/report.json"
    report = json.loads(path.read_text())
    for pin in report["pins"]:
        grader.verify_pin(ROOT, pin, "sealed evidence")
    manifest = root / "dashboard_manifest.json"
    if manifest.exists():
        raise FileExistsError("Dashboard publication is create-once")
    before = root / "evaluation/initial_dashboard.html"
    if before.exists():
        raise FileExistsError("First dashboard is preserved create-once")
    before.write_bytes((root / "index.html").read_bytes())
    page = dashboard(report)
    (root / "index.html").write_text(page)
    runner.write(
        manifest,
        {
            "kind": "api_full_score_dashboard_publication",
            "report_unchanged": grader.pin(path),
            "builder": grader.pin(Path(__file__)),
            "first_dashboard": grader.pin(before),
            "dashboard": grader.pin(root / "index.html"),
            "purpose": "Reader-first interpretation only; no metric or candidate changes.",
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    build(parser.parse_args().root)
