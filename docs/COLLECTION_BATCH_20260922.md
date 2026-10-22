# Six-score review batch — September 22, 2026

Five new complete provisional drafts are saved in the local desk. Together with
Rumichaca, there are **six pending drafts**. Aviador and Carrizal remain reviewed;
their saved records and Rumichaca's existing draft are unchanged.

## Open and review

Start with [Coqueteos](http://127.0.0.1:8766/?slug=jaime-llanos_22_coqueteos_pasillo_fulgencio-garcia).
The sidebar provides all six. Suggested order is about review effort, not measured
accuracy. Question counts below describe delivery, before human edits.

| Order | Score | Musical rows | Open questions | Start by checking |
| --- | --- | ---: | ---: | --- |
| 1 | Coqueteos | 9 | 8 | Chromatic signs, lower chord heads, S8m5's five eighth notes |
| 2 | Gato'e fique | 7 | 14 | Independent lower voices in S3–4, the 8a span, closing route |
| 3 | Chispazo | 8 | 16 | S2m7 sharp versus grace, missing/weak dots, S7m1 rest |
| 4 | Sobre el Humo | 9 | 9 | Lower chord heads in S5–6, crowded accidentals, S3m5 rest |
| 5 | Rumichaca | 7 | 17 | Existing questions on pitches, written durations and form |
| 6 | La Chata | 8 | 16 | Crowded S2m1, lower voices, S4m2 dot, final cadence |

1. Read the full page, then compare every musical row against the ABC. Questions
   identify known doubts; unflagged notes still need review. Adjust BPM for listening.
2. For the five new scores, **manuscript system 001** means the first musical row;
   this matches the ABC system comments and question numbering. Count measures
   from one within each source row, regardless of reflowed preview lines. Crops
   retain some neighboring context for chord labels and ending brackets.
3. Correct notes, rhythms and chord symbols in the ABC and **Save draft**. Keep
   suspected manuscript mistakes faithful to the source; explain a proposed
   musical correction separately in the notes. If the ink cannot be resolved,
   leave a question or explicitly document an accepted tentative reading.
4. Move answered questions and explanations to **Review notes and resolutions**,
   then remove those questions from **Questions to resolve**. Adding an answer
   inside the questions field still leaves it open and blocks **Mark reviewed**.
5. Mark reviewed only after the whole score is checked and no unresolved questions
   remain. You can pause and return without completing all six together.

Rumichaca retains its earlier source numbering: `system 001` is the title band;
`system 002` through `system 008` correspond to musical rows 1–7. Its **page 001**
view is the safest reference. Nothing in its saved draft was changed for this batch.

## Playback and source limits

Chispazo and Gato'e fique play their encoded ordinary repeats; Gato'e fique's
visible 8a direction is still an annotation until its span is confirmed.
Coqueteos, Sobre el Humo, La Chata and Rumichaca play the written measures once,
including both written endings where present. Complex segno/D.C. return routes
remain annotated, so playback is not yet a complete performance interpretation.

The visible source leaves these duration exceptions open. They were not filled
with invented rests or shortened to fit the printed meter:

| Score | Location | Current written extent in quarter-note beats |
| --- | --- | --- |
| Coqueteos | S8m5 | 2.5 |
| Chispazo | S3m6; S4m5; S7m1 | 2; 2; 3.5 |
| La Chata | S2m1; S4m2 | 5.5; 2 |
| Rumichaca | S2m1; S3m3; S7m3 | 2.5; 2.5; 4 |

Chord symbols were read by models from the manuscript and remain reviewable.
Accompaniment follows the current ABC symbols, stops at each barline and does
not carry into an unmarked bar. Listening alone cannot certify chord accuracy.

## How the batch was made

[Sol High calibration](SOL_CALIBRATION.md) against two already reviewed passages
produced only one completely matching bar out of fourteen. This small consumed
comparison did not qualify Sol as the primary reader for this batch. It is not a
collection-wide accuracy estimate or a blind comparison between models.

Astra at Extra High reasoning authored the five drafts from page/system images.
Each received a separate source audit and coordinator checks. These passes caught
omitted lower heads and a sharp mistaken for a grace note. After La Chata's
source-only submissions were preserved, an older consumed human fragment was
also reconciled with current row 6 against the manuscript. Its first bass B3
remains explicitly provisional. No experimental evaluator was rerun.

All 41 new musical rows are represented. Each draft passed bounded rendering and
finite-audio checks, plus disposable-copy save/reopen/exact-ABC-export checks.
New live reviews are revision 2, draft state, with zero recorded human time at
delivery. These are complete drafts, not approved transcriptions or accuracy results.
Canonical events, generated ABC, frozen experiments and existing review records
were preserved. Human correction time and remaining error patterns will determine
the next batch size.

Local evidence and immutable author copies are in
`out/collection/sol_batch_20260922_v1/`; `delivery_manifest.json` records draft
hashes, source coverage, model provenance, validation and preservation checks.
The drafts and manuscripts remain local and are not included in a code PR.
Back up `out/<slug>/overrides/review.json` with your ABC exports: the JSON also
contains your current questions, decisions, revision and accumulated review time.
It does not contain an automatic history of every save.

## Restart after closing the server

From the repository directory:

```sh
uv run python main.py review out --port 8766 --open-ui
```

Then open <http://127.0.0.1:8766/>. Stop the server with `Ctrl+C` when finished;
closing the browser does not delete saved reviews.
