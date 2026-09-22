# Model-assisted drafts and review decisions

The local correction desk can review an ABC draft authored directly from a
manuscript, even when the generated pipeline output is only a placeholder.
This is an assisted transcription workflow; it does not promote an automatic
recognition backend or make a draft into canonical dataset truth.

## Seed a complete draft locally

Use `ReviewApp.seed_model_draft()` from a trusted local preparation script:

```python
from pathlib import Path
from score2abc.review import ReviewApp

app = ReviewApp(Path("out"))
app.seed_model_draft(
    slug,
    Path("draft.abc").read_text(encoding="utf-8"),
    ["System 2, measure 2: confirm the damaged final note."],
    model_name="Astra",
    provenance={"source_pdf": "source.pdf", "preparation": "local manuscript review"},
)
```

Provide the actual source hashes and preparation metadata in `provenance` when
available. `model_name` describes the author; it does not invoke a model.
Seeding requires valid rendered notation with notes and refuses any existing
review file, including malformed files. It creates revision 1 in draft state,
with zero human review time, while retaining the current generated-base ABC/hash.
No import endpoint is exposed over HTTP.

The saved record retains `draft_source`, `model_name` and `provenance`. Chords
present in a seeded draft are labeled as model-assisted too. The browser displays
these origins, and ordinary saves preserve them. Model provenance is established
by local seeding; client-supplied origin fields cannot replace it.

## Resolve questions without losing explanations

Keep only unanswered items in **Questions to resolve**. Move each answered
question and its explanation into **Review notes and resolutions**, and remove
it from the questions field. Notes are stored verbatim in `review_notes`, remain
available after saving and reopening, and do not block **Mark reviewed**.
The button still requires valid notation with notes and no open questions.

Document an accepted tentative reading explicitly in the notes. Completing a
human review does not make uncertain source ink certain. In particular, do not
silently correct a suspected manuscript mistake into a preferred musical reading.

Notes are optional; older clients that omit the field preserve existing notes.
An explicit empty string clears them. Review notes are limited to 64,000 UTF-8
bytes, within the server's existing overall request limit. Revision conflict
checks continue to protect newer saves.

ABC export contains the exact saved notation only. The full review record,
including decisions, is under `out/<slug>/overrides/review.json`. Canonical events,
MusicXML, generated ABC and frozen experimental truth remain unchanged. Back up
review records alongside exported ABC; ignored local `out/` files are not included
in Git commits.

## Prepared manuscript images and pending drafts

The sidebar counts saved pending drafts separately from human-reviewed scores.
A prepared work may provide `out/<slug>/review_sources/system_001.png` and
subsequent three-digit PNG names. When present, these verified manuscript-numbered
context crops replace generated system crops in the source selector. Full pages
and the PDF remain available. Empty or invalidly named folders fall back to the
generated crops; symlink paths remain forbidden. Preparation should cover every
musical row and retain enough context for chords and ending brackets.

## Repeat playback safeguard

A backward repeat following a closed explicit repeat without a new opening repeat
can make the installed audio renderer expand indefinitely. The desk detects this
known route before all three audio paths, shows a warning, and disables playback.
The notation remains visible, editable and savable. Correct a repeat route only
from source evidence; do not invent a repeat start just to satisfy the player.
Ordinary matched repeats and first/second endings remain playable. D.C./D.S./Fine
navigation remains an annotated source instruction, not automatic playback.

The [September 22 Sol calibration](SOL_CALIBRATION.md) uses consumed reviewed
passages and supports retaining Astra for the next manuscript-drafting batch.
