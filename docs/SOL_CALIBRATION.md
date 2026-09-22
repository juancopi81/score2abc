# Sol manuscript calibration — 2026-09-22

**Decision: retain Astra for manuscript drafting.** Sol High remains useful for
bounded implementation and validation tasks, but these two readings do not
support using it as the primary transcriber for the next collection batch.

## Method

Two separate `gpt-6-sol` agents at High reasoning read source PNGs: Carrizal's
first musical system and Aviador's eighth. Each received a full page for context,
a target crop, and instructions to produce ABC, a written-event ledger, and
uncertainty notes. They were instructed not to consult saved transcriptions,
MusicXML, ground truth, or previous experiments. Each submitted one reading.

The coordinator snapshotted the existing human-reviewed references before the
submissions, then compared independently parsed ABC events after submission.
The agent JSON ledgers agree with their ABC in all 14 bars. Both ABC files render
without syntax errors; neither fact establishes musical accuracy.

This is **consumed practical calibration against saved human-reviewed ABC**, not
a blind benchmark, a new held-out score, or an estimate of accuracy across the
collection. The saved references may themselves contain unresolved source
interpretations. No recognizer or canonical dataset was changed.

## Results

| Source passage | Exact events / reference events | Exact note events / reference notes | Completely matching bars |
| --- | ---: | ---: | ---: |
| Carrizal, system 1 | 10 / 28 (35.7%) | 10 / 26 (38.5%) | 1 / 7 |
| Aviador, system 8 | 8 / 32 (25.0%) | 5 / 30 (16.7%) | 0 / 7 |

An exact event matches its physical bar, onset, written duration, pitches or rest,
and tie flags. A chord is one event containing its simultaneous pitches. Repeats
are not unfolded. The submitted event totals were 28 and 33 respectively, so
exact-event precision was 35.7% and 24.2%. Chord labels, articulations, engraving,
and performed form were not scored.

Errors include staff-position readings, rests interpreted as notes, written
rhythms, and inherited key/accidental context. Sol explicitly reported low
confidence in the dense handwriting. The gap is substantial enough that handing
these readings directly to Juan would create unnecessary correction work.

## Consequence for this batch

- Use `gpt-6-astra` at Extra High reasoning for the five full manuscript drafts.
- Keep source-reading uncertainty visible and perform a separate source audit.
- Use Sol for bounded software work where executable checks can assess its output.
- Preserve Aviador, Carrizal, and Rumichaca's existing review records unchanged.
- Judge the batch by human correction effort and source fidelity, not rendering.

The local evidence is under `out/collection/sol_batch_20260922_v1/`: source
requests, author submissions, protected reference hashes, independent parses,
per-bar differences, and `calibration_results.json`. These local manuscripts and
drafts are not published by this document.
