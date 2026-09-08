# A Puño y Letra: collection delivery

Goal: finish the 100-melody collection through assisted transcription and human
musical review. A complete, source-checked score is the unit of delivery.

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
  integration for the original research branch, subject to PR review. Draft
  [PR #8](https://github.com/juancopi81/score2abc/pull/8) preserves that history.
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

Compare page/system images with notation, edit ABC and chord symbols, listen,
record unresolved questions, save, reopen, and export. Only mark a score reviewed
after checking its complete music. The first UI has text editing and selectable
rendered symbols; it is not a graphical engraving editor. Reference tones do not
provide chord-symbol accompaniment.

Playback highlights each written note/group and rest, including tied
continuations, and follows it in the notation when “Follow playback” is checked.
The audio clock controls both highlighting and completion. Melody and chord
origins are displayed separately: XML harmony markings take precedence over OCR
proposals when supplied, and later saved corrections retain that origin.

The save contract is `out/<slug>/overrides/review.json`: exact ABC, original-ABC
snapshot/hash, revision, draft/reviewed status, unresolved questions and active
review time. Atomic saves reject stale revisions; invalid drafts remain savable.
Downloads preserve saved ABC bytes and identify draft/reviewed status. Generated
files and frozen research evidence are untouched. This override is not yet an
event-level training label, a MusicXML export, or part of `score2abc export`.

## Next milestone and acceptance

1. The first Aviador edit/play/save/reopen trial is complete. Recheck playback
   following and restored supplied chords before asking for substantial
   transcription time.
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
