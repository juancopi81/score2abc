# First collection pilot

This is a delivery pilot: finish three complete scores and measure the human
correction effort. It is not a new heldout recognition benchmark. The original
experimental freezes and their evaluations remain separate.

## Selected scores

| Work | Book PDF page | Manuscript page | Role and current evidence |
| --- | --- | --- | --- |
| 12, Aviador | 27 | 12 | Existing complete supplied-MusicXML draft; Juan's saved correction remains in the original review desk. |
| 19, Carrizal | 34 | 19 | Human review completed on 2026-09-14, revision 4: eight musical systems, 48 written bars, no open questions and exact resolutions retained. Its earlier system-4 reference remains separate. |
| 86, Rumichaca | 99 | 87 | Complete provisional Astra draft delivered on 2026-09-14: seven musical rows, 45 written spans, 17 questions, revision 2. Poor scan and three unresolved meter exceptions require broader review. |

The scan categories are relative visual judgments from the full pages, not
measured accuracy or legibility scores. Carrizal and Aviador already contributed
development evidence. Rumichaca becomes pilot development material from this
selection onward.

## Juan's review

Update, 2026-09-12: Juan confirmed the B-natural after B-flat in Aviador system 8,
measure 6, removed the question and saved revision 7 as reviewed. The saved ABC
matches the repaired candidate exactly and retains his earlier C-sharp correction.
There are no open questions in that review. No further note check is requested.
The exact prior review, repair diff and playback audit are retained under
`out/collection/aviador_accidentals_v1/`; the confirmation receipt is under
`out/collection/aviador_review_confirmation_v1/`. Repeats/endings still need a
source-based form pass before this becomes a publication-ready score.

Open Aviador at <http://127.0.0.1:8766/?slug=jaime-llanos_12_aviador_pasillo_fulgencio-garcia>.
If the desk is stopped, run this from the repository:

```sh
uv run python main.py review out --slug jaime-llanos_12_aviador_pasillo_fulgencio-garcia --port 8766 --open-ui
```

For subsequent complete-score drafts:

1. Compare every musical staff with the manuscript. Check pitches, accidentals,
   durations, rests, ties, repeats, chord symbols and key changes. Playback helps
   locate errors; the source page decides what is written.
2. Save as you go. Put unresolved readings in the questions field with a system
   and measure location; leaving a draft unresolved is useful progress.
3. Use any comfortable BPM. It is a listening preference. An unmarked bar has no
   chord accompaniment, even when the previous bar contains a chord symbol.
4. Mark reviewed only after the whole piece is checked and its questions are
   resolved. Download the saved ABC. This export preserves the reviewed text;
   it is not yet a MusicXML export or canonical training label.

The desk records active review time automatically; no separate stopwatch is
needed. Carrizal is reviewed. Rumichaca is now available for correction in the
same desk; start with its first three questions, then review the whole source
as legibility permits. Keep unresolved ink explicit instead of forcing a reading.

When sources become available, the inventory still needs El vallenato (37),
Entre amigos (38), Fita chiquita (44), and the identities of unnamed rows 99–100.
These gaps do not block the available-score pilot.

## Coordinator's preparation

All new preparation lives under ignored `out/collection/`. Aviador remains in
its existing `out/<slug>/overrides/review.json`; do not copy its draft into a
second editable desk. The source packet is a separate input set, not a count of
completed transcriptions.

Prepare source pages from the explicit catalog mappings. Retain source and
output hashes, the two-page El Tato, the missing rows and unknown musical
context. Validate the extracted pages against the original renders.

The local packet is now prepared at `out/collection/source_packet_v1/`:
95 work inputs covering 96 pages. `out/collection/source_packet_qa_v1.json`
records pixel equality for every extracted page against its original book page
at 48 DPI, plus unchanged source and matching output hashes. The three pilot
pages and both El Tato pages were also visually inspected. The replay command
and packet contract are in the [catalog guide](../dataset/catalog/README.md).

Before the direct-model comparison, finish physical measure mapping and key/
meter checks for each system. The first bounded offline Audiveris probe uses
complete pages with no manually supplied musical context. It checks whether the
installed backend can produce a useful candidate; it does not count as a
completed pilot transcription or a musical accuracy evaluation. Full-page
input also avoids inheriting the earlier Rumichaca crop omission (six detected
systems versus seven visible musical staves).

Retain raw XML, input hashes, configuration, logs and an explicit usability
decision. A parser accepting a file is insufficient. Keep incomplete or
misaligned output out of Juan's correction queue. No paid API calls are
authorized by this preparation; agree a concrete cap before a paid comparison.

## Preparation result — 2026-09-08

`out/collection/pilot_v1/structural_preflight_v1.json` records visual written-span
counts of `[7,7,7,8,6,6,6,2]` for Carrizal (49) and `[7,8,7,6,7,6,4]` for
Rumichaca (45). Both openings visibly specify 3/4. Carrizal starts with one flat
and changes to two flats at system 5, span 5. Rumichaca has no visible initial
accidental signature. These observations do not establish tonic/mode or certify
note durations. Verify the span counts while drafting; repeats are not expanded.

`out/collection/pilot_v1/direct_visual_inputs_v1/` contains all 15 manually
selected system context images, their full raw pages, crop coordinates, hashes
and per-work preflight records. The crops retain nearby context; transcribe only
the main system. All crops were visually checked. In particular, Rumichaca's
first/second-ending brackets above system 5 are retained. No direct-model draft
or paid request has been generated yet.

The bounded Audiveris 5.10.2 probe is recorded in
`out/collection/pilot_v1/offline_probe_report_v1.json`:

| Attempt | Carrizal | Rumichaca |
| --- | --- | --- |
| Default macOS batch launch | Aborted before recognition | Aborted before recognition |
| Same full-page input with `JAVA_TOOL_OPTIONS=-Djava.awt.headless=true` | Returned exit 1; no accepted XML; accuracy unassessed | Exported 45 measures but only 3 pitched notes, 7 rests and no harmonies; read 2/4 instead of visible 3/4 |

The default crash trace enters macOS AWT application registration. Headless mode
avoids that startup abort; no application or project code was changed for the
probe. Both attempts and their logs are preserved in separate namespaces.
The Rumichaca result is grossly incomplete, and neither output enters Juan's
correction queue. The next engineering step is direct visual drafting from the
prepared systems, with uncertainty recorded and checked before handoff.

The source utility and catalog checks pass 10 focused tests, including real
Poppler extraction of distinguishable synthetic pages to verify order and gaps.
Repository Ruff and Black checks pass. The six original Aviador draft/source/
generated artifacts remain byte-for-byte unchanged.

## Stronger-model checkpoint — 2026-09-09

A separate, sealed Astra experiment on Aviador system 8 matched all seven
reference measures from the original PDF, while the historical-crop condition
matched four and the frozen pipeline matched one. This is a consumed excerpt
comparison, not a complete pilot score or fresh generalization result. It
supports prioritizing the planned direct visual drafts with full-page context.
See [the benchmark report](ASTRA_BENCHMARK.md) for protocol, errors, costs and
remaining validation. Carrizal and Rumichaca still have no direct-model draft
in the collection pilot packet.

The subsequent [three-score diagnostic](ASTRA_COLLECTION_PILOT.md) reached 6/8
exact Carrizal measures and 5/7 exact Coqueteos measures. Gato'e Fique improved
pitch sequences but needs a polyphonic full-event scoring contract. An audit
found inherited repository instructions in the subagent contexts, so these are
assisted diagnostics, not a clean source-only test. No excerpt replaces a
complete pilot score or enters Juan's correction queue. The next delivery step
was a complete Carrizal draft with musical form and focused questions, delivered
below.

## Complete Carrizal draft — 2026-09-12

Update, 2026-09-14: Juan saved his pitch edits and responses to the seven questions.
The responses had been entered in the unresolved field, so the desk correctly
blocked approval but offered no place to retain the answers. A separate review
notes field now preserves the exact explanations. Juan completed the final approval:
Carrizal is revision 4, `reviewed`, with no blocking questions, unchanged user ABC
and 797,544 ms of recorded human review time. The verified receipt and exact review
backup are under `out/collection/shipping_20260914_v1/`.
The accepted A4 at system 2 measure 2 remains explicitly tentative in his notes.
His confirmed B-flat4 at system 4 measure 2 supersedes the earlier partial A4
reading for this delivery; historical experimental truth has not been rewritten.
The exact before/after reviews and migration receipt are retained under
`out/collection/carrizal_review_resolutions_v1/`.

Open [Carrizal in the review desk](http://127.0.0.1:8766/?slug=jaime-llanos_19_carrizal_pasillo_emilio-murillo).
The complete ABC was initially saved as revision 1, state `draft`, with seven questions
and zero human review time; the September 14 update above records the completed review. Both the melody and chord symbols are explicitly labeled
as a model-assisted manuscript draft. This is delivery work on consumed material,
not a new blind benchmark or a canonical training label.

Two Astra section drafts were independently checked against the opposite source
sections, then assembled and adjudicated by the coordinator. The source audit
recovered two stacked D-minor triads in system 3 measure 2 and the left repeat
markers at that section boundary. Original author drafts are retained. Ordinary
repeats, chord symbols, triplets, explicit naturals, the two-flat change and the
cross-system A4 tie are present. No first/second endings were securely observed;
no D.C./Fine routing was inferred from the cursive inscription. Fingerings and
precise engraving remain outside this lead-sheet draft.

**Preflight erratum:** Carrizal's written counts are now
`[7,7,7,8,5,6,6,2]` (48), rather than the earlier 49. The natural sign around
page x=420 inside system 5's first bar was mistaken for a barline. The two-flat
signature begins at system 5 measure 4, previously called span 5. The earlier
preflight record was preserved; the corrected delivery map is separate.

Review the complete score, using the questions as priorities rather than an
exhaustive error list. **System numbers refer to the manuscript's first eight
musical rows**, not the responsive preview's wrapped staff lines. Measures are
counted from the left within that manuscript row. The most uncertain readings
are system 2 measure 2's final pitch and system 7 measure 5's final low note.
System 4 measure 2's second eighth also needs adjudication: the source-based
draft reads B-flat4, while the existing partial reference says A4. Neither was
silently replaced. Resolve each question in the desk, then mark reviewed after
checking the whole piece. No separate stopwatch is needed.

Delivery evidence is under `out/collection/carrizal_delivery_v1/`. Its
`assembly/README.md` links the written event ledger, source corrections and
validation. Independent ABC parsing matches all 48 measures of that ledger,
including 197 written heads, 18 rests, durations, onsets and ties. These are
representation checks, not measured source accuracy. The local browser rendered
the draft, started playback with visible note highlighting, and stopped cleanly.
Exact save/reopen/export and origin preservation were verified with the actual
Carrizal ABC in an isolated review copy, leaving live human review time at zero.
All 56 pinned pre-existing Aviador/Carrizal review, source and generated files
remain unchanged. Repository checks: 774 tests passed, 3 skipped; Ruff and Black
passed. No commit, push or merge was performed for this delivery.

## Shipping and Rumichaca delivery — 2026-09-14

PR #10 (explicit naturals) and PR #11 (model-assisted seeding and review notes)
are merged. CI passed on both PRs and merged main `2d48ac5`. The active runtime
files match shipped main; the original preparation branch still retains unrelated
local source-preparation and experimental changes. Shipping and review backups are
under `out/collection/shipping_20260914_v1/`.

Rumichaca is in the existing desk at
<http://127.0.0.1:8766/?slug=jaime-llanos_86_rumichaca_bambuco_emilio-murillo>.
Two Astra/xhigh section drafts plus source cross-review produced a complete
provisional 45-span ABC. Independent parsing matches its 197 written heads, two
rests and all written events. This does not establish source accuracy.

Seventeen questions prioritize damaged low notes, accidentals, durations and form.
S2m1 and S3m3 remain 2.5 beats; S7m3 remains four beats. Repeat/ending/D.C.-like
marks are currently text annotations, with one-pass written-order playback:
the unresolved original form exposed a reproducible abc2svg repeat-loop bug.
Repeated-harmony glyphs are visible but their accompaniment is not expanded.
A narrow playback guard is the next engineering task before executable form work.

Choose `page 001` for the full source; the old automatic source selector includes
a title band, so its systems 002–008 correspond to musical rows 1–7. High ledger
notes and ending inscriptions are clearer on the full page. The live browser
rendered, played, advanced its note highlight and stopped successfully. Exact
seed/save/reopen/export and preparation notes were verified separately. The draft
is revision 2 with zero recorded human time; all 325 pre-existing protected files,
including Aviador/Carrizal reviews, remained unchanged. Full evidence is under
`out/collection/rumichaca_delivery_v1/assembly/`.
