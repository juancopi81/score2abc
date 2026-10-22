# A Puño y Letra: collection delivery

Goal: finish the 100-melody collection through assisted transcription and human
musical review. A complete, source-checked score is the unit of delivery.

Current handoff: [audited editorial drafts, October 22](EDITORIAL_BATCH_20261022.md).
At that delivery the local queue contained ten pending drafts and three reviewed
scores: Aviador, Carrizal, and Amalia. The subsequent
[form check](FORM_REVIEW_20261022.md) preserves Carrizal and adds an Aviador form-only
draft with two targeted questions; the desk shows the current review counts.
The [October 3](COLLECTION_BATCH_20261003.md) and
[September 22](COLLECTION_BATCH_20260922.md) submissions remain preserved.
The [editorial protocol](EDITORIAL_TRANSCRIPTION_PROTOCOL.md) separates source
readings from provisional reconstructions and requires an independent source audit.

## Verified starting point — 2026-09-07

- The supplied 111-page PDF contains 95 identifiable works on 96 music pages.
  Its table has 98 named entries plus unnamed rows 99–100. Named entries
  37 (El vallenato), 38 (Entre amigos), and 44 (Fita chiquita) are absent.
  El Tato spans two pages. The portable [catalog](../dataset/catalog/README.md)
  records explicit page ownership, literal table text, source hash and uncertainty.
- Sixteen individual PDFs already exist locally. MusicXML references cover
  twelve works, often only one staff; these are not twelve completed scores.
- Default runs replay fixture MusicXML or produce stub notes. Melody recognition
  improvements remain experimental and do not establish whole-book accuracy.
- The [spike integration review](SPIKE_INTEGRATION_REVIEW.md) supports source
  integration for the original research branch. Both
  [PR #8](https://github.com/juancopi81/score2abc/pull/8) and the correction desk in
  [PR #9](https://github.com/juancopi81/score2abc/pull/9) are now merged into `main`.
  Source integration and default recognizer promotion are separate decisions.

## Work division

| Coordinator and delegated engineering | Juan |
| --- | --- |
| Maintain catalog, provenance and progress; split source pages using verified mappings | Obtain the three missing scores and identify rows 99–100 when possible |
| Prepare focused branches/PRs, run checks, keep frozen experiments intact | Approve merge/promotion when a concrete result is ready |
| Generate transcription drafts, flag uncertainty, build and maintain the correction interface | Resolve ambiguous musical readings and approve complete scores |
| Compare recognizers with fixed inputs and measured correction effort | Try the interface briefly and report friction; agree a spend cap before paid model calls |

The coordinator keeps decisions and milestones; bounded subagents handle
implementation and independent review with compact evidence reports. No paid
recognition calls or automatic recognition promotion are part of the initial UI
and inventory work.

## First correction loop

Run `uv run python main.py review out --open-ui`. Start with Aviador, whose
supplied MusicXML makes the edit/save/reopen/export workflow testable now.
Other works without recognized melody open as explicit incomplete drafts.
Use `out/local_restricted` as a separate output root for the six restricted PDFs.

The library separately counts saved pending drafts and reviewed scores. Prepared
drafts may supply `out/<slug>/review_sources/system_001.png` and subsequent
three-digit files: these manuscript-numbered context images replace generated
system crops in the source selector, while the full page and PDF remain available.
An empty folder falls back to generated crops. Preparation must verify every
musical row; the optional folder is not a new segmentation result.

Compare page/system images with notation, edit ABC and chord symbols, listen,
record unresolved questions, save, reopen, and export. Only mark a score reviewed
after checking its complete music. The first UI has text editing and selectable
rendered symbols; it is not a graphical engraving editor.

Playback highlights each written note/group and rest, including tied
continuations, and follows it in the notation when “Follow playback” is checked.
Quiet, sustained chords follow the current ABC symbols without adding visible
notation. Each chord begins at its written position and ends at the next chord
change, barline or the end. Unmarked bars have no accompaniment and never inherit
the previous bar's chord, including on repeats. A pickup before the first symbol
has no accompaniment.
`N.C.` silences accompaniment. Unsupported or conflicting symbols produce a
warning and silence that passage. Editing or stopping cancels both voices.
The audio clock controls melody, accompaniment, highlighting and completion. Melody and chord
origins are displayed separately: XML harmony markings take precedence over OCR
proposals when supplied, and later saved corrections retain that origin.

The Tempo control uses quarter-note BPM and scales melody, chords and highlighting
together, retaining relative tempo changes within the score. Changing it stops
playback; press Play to restart at the new speed. The browser remembers the
preference per melody. Score tempo resets to the ABC's initial numeric tempo,
or 120 BPM if unspecified. Neither action edits ABC or marks the draft dirty.

The save contract is `out/<slug>/overrides/review.json`: exact ABC, original-ABC
snapshot/hash, revision, draft/reviewed status, unresolved questions and active
review time. Atomic saves reject stale revisions; invalid drafts remain savable.
Downloads preserve saved ABC bytes and identify draft/reviewed status. Generated
files and frozen research evidence are untouched. This override is not yet an
event-level training label, a MusicXML export, or part of `score2abc export`.

### What human saves preserve

Each Save replaces the current `review.json`; its increasing revision is a
concurrency check, not a revision archive. Notes and questions are saved as their
current contents. There is no automatic per-save history, timestamp, reviewer
identity or event-level correction log. Active review time is an accumulated
UI estimate: it counts focused, visible activity within 30 seconds of input and
is submitted when saving, rather than measuring every musical decision.

The retained `base_abc` is the original generated/supplied notation. For a
model-seeded score it may be a placeholder, so it must not be treated as the AI
draft that the human corrected. The editorial batches separately preserve the
delivered AI ABC, source reading, proposals, source hashes and save receipts under
`out/collection/editorial_pilot_20261022_v1/` and
`out/collection/editorial_batch_20261022_02/`. These snapshots support a later
before/after comparison with the completed review, but do not recover every
intermediate save or automatically train the recognizer.

Before using corrections for evaluation or training, align changes to source
measures and distinguish source corrections from accepted reconstructions and
performance choices. A future save journal should preserve each revision and
its provenance; that journal and label promotion are not implemented yet.
All `out/` artifacts, including human saves and these snapshots, remain local
and are excluded from Git. Committing and pushing application code does not
back up the score-review data to GitHub.

## Source fidelity and musical form

The archival transcription records what the manuscript says, including a
confirmed source mistake. Do not silently replace it with a musically preferred
reading. Record a suspected mistake as a question with its system/measure and
source evidence. A later editorial correction must retain the original reading,
proposed replacement, reason, reviewer and decision separately. An unreadable
symbol remains uncertain; it is not automatically a manuscript mistake.

Complete-score delivery includes ordinary repeats and first/second endings
(casillas, or volta brackets). Preserve their written structure instead of
duplicating bars in the archival ABC. The current renderer/player accepts ABC
repeat and ending syntax. Aviador's October 22 form proposal uses an explicit
ABAC route pending two source/editorial decisions; Carrizal's existing repeats
were independently verified. Confirm the form before publication. Retain
D.C., D.S., Fine and related directions as source annotations; automatic complex
navigation and an editorial-version UI are deferred. Listening through a linear
draft does not verify the performed form.

An explicitly chosen performance route can use a header such as
`P:AA BB CC A B CC` with body labels `P:A`, `P:B`, and `P:C`. The desk renders
each section once and follows the stated order for melody, chord accompaniment
and highlighting, retaining each section's written pitches and tempo. This is a
bounded route chosen by the reviewer; the desk does not interpret handwritten
D.C./D.S./Fine instructions automatically. Record an editorial route and its
evidence in review notes when the source remains ambiguous. The current player
accepts up to 128 uppercase section letters and 30 minutes, and rejects missing
or duplicate labels, combinations with local repeat bars, and held notes crossing
section boundaries. It does not expand numbered groups such as `(AB)2`.

Amalia's October 21 handoff selects `AA–BB–CC–A–B–CC` from Juan's reported
recording comparison. The 48 written measures and all reviewed note events are
preserved, while playback visits 160 measures in 180 seconds at the written
160 BPM. Revision 16 is backed up under `out/collection/amalia_form_20261021_v1/`;
revision 17 was the draft handoff. Juan subsequently reviewed Amalia; the current
saved revision is 23 and reviewed, with his ABC and decisions preserved. The
October 22 notes refresh did not change that music or status. This does not turn
the ambiguous manuscript return signs into a confirmed source reading.

The desk now blocks the known unsafe audio route where a backward repeat follows
a closed explicit repeat without a new opening repeat. It shows a warning and
keeps rendering, editing and saving available. Correct the route only from source
evidence; adding a repeat start solely to make the player run invents musical
meaning. Ordinary supported repeat routes remain playable. This bounded guard
does not implement D.C./D.S./Fine navigation.

Explicit natural signs now survive MusicXML import and ABC export, including
cancellation after a flat or sharp in the same bar. This technical repair does
not infer new pitches from musical plausibility.

Complete model-assisted drafts can be seeded locally with
`ReviewApp.seed_model_draft()`. This validates notation, refuses any existing
review file, records the model and source provenance, and begins at revision 1
in draft state with zero human review time. Human saves preserve that origin and
the generated-base hash. The UI labels melody and chord origins explicitly;
seeding never rewrites canonical events, MusicXML, generated ABC or experimental
truth. Carrizal is the first complete draft delivered through this path.

Keep only unanswered items in **Questions to resolve**. Move an answered question
and its explanation to **Review notes and resolutions**, then remove it from the
questions field. Notes are saved verbatim with the review and do not block
approval; open questions or invalid notation still do. Record tentative accepted
readings explicitly in the notes rather than silently presenting them as certain.
Notes survive later saves and reopening; the ABC download remains exact notation
only. Existing clients that omit the notes field preserve already saved notes.

## Next milestone and acceptance

1. The first Aviador edit/play/save/reopen, playback-following, chord
   accompaniment and BPM trials are complete. The user accepted the final
   barline cutoff behavior. Juan confirmed Aviador's repaired B-natural after
   B-flat and saved revision 7 as reviewed on 2026-09-12, with no open questions.
   The earlier review and correction remain backed up. Full musical form remains
   a separate publication check; no complete-score rereview is requested.
2. Prepare three complete scores spanning clear, intermediate and poor scans.
   Check physical measure mapping and key/meter context before inference.
   Keep prior consumed examples separate from any fresh evaluation set.
3. Compare a stronger direct-model draft with the existing recognizer on fixed
   inputs. Record model/configuration, cost, remaining errors and active human
   correction time. Agree the API budget before paid calls; no new accuracy
   claim comes from merely selecting a stronger model.
4. Complete and export all three scores, including repeats, ties, chords,
   accidentals, meter/key changes and uncertain passages. Judge success by
   faithful output and reduced human effort, not syntax validity alone.
5. Expand to small batches with a visible review queue and resumable progress.
   Build lossless event/MusicXML integration before treating ABC overrides as
   canonical dataset truth. Inventory all 95 available works first; unavailable
   items stay explicitly missing until their sources are supplied.

Human review should begin during the pilot, before a full-book automatic run.
The pilot determines how much automation is useful and where human time helps most.

The current pilot preparation and review instructions are in
[COLLECTION_PILOT.md](COLLECTION_PILOT.md).

The September 22 [Sol calibration](SOL_CALIBRATION.md) supports retaining Astra
for the next five manuscript drafts. The comparison uses consumed human-reviewed
passages; it is not a new blind benchmark or recognizer promotion.

## Initial engineering validation

The collection-review branch passed 730 tests with loopback access, Ruff, Black,
and JavaScript syntax checks. Source distribution and wheel build successfully;
the wheel contains all four UI/renderer assets with bytes matching the source.
A disposable browser copy verified ABC editing, rendering, playback controls,
draft save/reopen, question-gated approval and reviewed download. Exact saved
text includes a tuplet, dyad, tie, rest and chord symbol. Real source PDF and ABC
bytes remain unchanged. An independent review found and verified fixes for
editing during work loading and saving after more than an hour of active review.
Human musical accuracy and usability acceptance are still pending.

The first feedback follow-up passed 751 tests plus Ruff, Black, JavaScript checks,
and packaged-asset verification. Browser checks covered live highlights, scrolling,
stop cleanup and editing during playback. Aviador's draft was explicitly upgraded
to revision 2 with 26 supplied chord markings, an exact revision-1 backup, and
every non-chord character preserved, including the user's C-sharp correction.
Source PDF, XML, generated ABC and canonical events remain unchanged on disk.

The accompaniment follow-up also passed 751 tests, including 15 JavaScript
timing and audio-lifecycle checks, plus Ruff, Black and JavaScript syntax checks.
Source distribution and wheel build successfully with matching review assets.
Disposable browser checks covered chord edits, unsupported-symbol warnings,
playback highlighting and stop cleanup. All six pinned Aviador draft, backup,
source and generated artifacts remained byte-for-byte unchanged. The user's
subsequent listening review accepted chord timing, balance and harmony.

The BPM follow-up passed 751 tests, including 20 JavaScript timing/audio checks,
plus Ruff, Black, syntax checks and matching packaged assets. Browser checks
verified 160/80 BPM playback, cancellation, persisted tempo after reopening,
score-tempo reset and unchanged ABC. All six real Aviador artifacts remained
byte-for-byte unchanged.
