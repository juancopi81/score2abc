# Five additional review drafts — October 3, 2026

This batch adds Acuata, Amalia, Atardecer, Entusiasmo and Estrella del Caribe to
the local correction desk. The existing six pending drafts and the reviewed
Aviador and Carrizal records are preserved. The queue now contains **11 pending
drafts and 2 reviewed scores**. These remain model-assisted drafts; human source
review is the final correctness check.

| Score | Musical rows | Written spans | Open questions | First review priorities |
| --- | ---: | ---: | ---: | --- |
| Amalia | 7 | 48 | 17 | High ledger notes, two short half-note bars, segno/Fine route |
| Acuata | 9 | 54 | 20 | Chromatic signs versus grace notes, held chord heads, row 7 cancellation |
| Atardecer | 10 | 57 | 16 | Opening polyphony/fermata, low arpeggios, closing chromatic notes |
| Estrella del Caribe | 11 | 52 including pickup | 16 | Dense chromatic chords, independent holds, numbered endings |
| Entusiasmo | 12 | 85 timed spans | 18 | Sustained lower voices, first-ending rhythm, unresolved terminal glyphs |

Counts describe delivery before human edits; they are not accuracy scores.

## Review workflow

Start with [Amalia](http://127.0.0.1:8766/?slug=jaime-llanos_6_amalia_pasillo_joaquin-ariza),
then choose the other scores from the sidebar. Amalia is a useful first pass
because it has fewer simultaneous notes than the denser new scores.

1. Compare every manuscript row with the transcription, including unflagged
   passages. Listening and the BPM control help, but cannot establish accuracy.
2. In these five scores, **manuscript system 001** is the first musical row.
   Question numbers and `% System` comments refer to these source rows, not the
   automatically reflowed preview. Count measures from one within each row.
3. Correct the ABC and save the draft. Preserve suspected manuscript mistakes
   in the source transcription and describe any proposed musical correction in
   the review notes.
4. Move answered questions into **Review notes and resolutions**, then remove
   them from **Questions to resolve**. Answers left in the question field still
   block **Mark reviewed**.
5. Mark reviewed after checking the complete score and resolving or explicitly
   accepting each tentative reading. You can review one score at a time.

The [previous batch guide](COLLECTION_BATCH_20260922.md) still applies to the six
earlier pending drafts. Rumichaca retains its older numbering: system 001 is
title material; systems 002–008 are its seven musical rows.

## Reading and playback limits

Astra at Extra High reasoning read the manuscript images. Each draft also
received a separate source check. The prior small Sol calibration was not rerun
or promoted into a collection-wide accuracy claim.

Source chord labels are provisional readings. Their accompaniment ends at each
barline and does not carry into an unmarked bar. Long notated melody or chord
heads can have their own written durations.

Complex D.C./segno/Fine routes remain annotations where the source reading is
unresolved. Ordinary repeats and numbered endings are played only when encoded
in the draft. Short or long bars are not repaired by inventing notes or rests.

Entusiasmo's final post-double-bar glyph cluster remains an annotation: two
readers could not establish which strokes, if any, are timed music rather than
navigation. Its first question explicitly preserves that coverage gap. Atardecer
also flags an isolated possible lower-voice rest in S1m6, and small articulation
marks are not exhaustively encoded. These are review tasks, not settled omissions
from an approved transcription.

The full page and every musical row remain available for comparison. In Acuata,
the automatic detector missed real music rows and included an autograph staff;
the review images use a manually checked nine-row inventory instead. Short final
rows in Amalia, Entusiasmo and Estrella del Caribe are included.

## Local evidence and restart

Source images, preserved original model submissions, source audit notes,
assembled drafts and delivery records live under
`out/collection/astra_batch_20261003_v1/`. The saved working drafts are
`out/<slug>/overrides/review.json`. Keep those JSON files with your ABC exports:
they also contain current questions, resolutions, revision and accumulated review
time. They do not contain an automatic history of every save.

This batch does not promote drafts to canonical events, training truth or
experimental benchmarks. The manuscripts and drafts remain local and are not
part of a code PR.

All five drafts passed bounded rendering and finite-audio checks and
disposable-copy save/reopen/exact-export checks. Their manuscript images,
playback and blocked approval state were checked in the live browser. Existing
review/playback tests passed (28 passed, 1 skipped). Eight prior review records
and 112 existing source/generated assets were verified unchanged; existing
manifest rows were preserved while three new works were appended. Each new
draft is revision 2 with zero recorded human review time. The batch's
`delivery_manifest.json` records these checks and file hashes.

To restart the desk from the repository directory:

```sh
uv run python main.py review out --port 8766 --open-ui
```

Open <http://127.0.0.1:8766/>. Closing the browser preserves saved drafts; stop
the local server with `Ctrl+C` when finished.
