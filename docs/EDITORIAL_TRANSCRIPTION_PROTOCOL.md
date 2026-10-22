# Assisted transcription and editorial proposals

The October 22, 2026 workflow applies lessons from Juan's Amalia review to other
manuscripts. GPT-6 Astra at Extra High reasoning reads each complete score. A
different reader audits the source and proposed changes before coordinator
validation and delivery. This is assisted collection work on consumed material,
not a blind benchmark, model training, or promotion of the experimental recognizer.

## Preserve evidence while making a playable proposal

Retain the previous draft and a source reading alongside the proposed edition.
Classify each material decision:

- **Source correction:** visible evidence supports a different note, rhythm,
  rest, chord, accidental, tie, or other mark.
- **Editorial reconstruction:** damaged or missing material is provisionally
  completed using stated evidence such as a repeated phrase or rhythmic pattern.
- **Editorial interpretation:** a performance route or other musical decision
  goes beyond a settled reading of the manuscript.
- **Unresolved:** evidence does not support a sufficiently specific proposal.

Each decision records its manuscript row and measure, previous/source reading,
proposal, evidence, uncertainty or alternatives, and approval status. Model
proposals remain unapproved until the musician accepts them. A valid measure or
plausible sound is supporting context, not proof of what the manuscript contains.

Amalia's correction patterns inform the checking instructions, not automatic
substitutions: inspect faint dots, ledger positions, accidental scope, ties versus
slurs, dynamics versus chord letters, and repeated phrases. Retain independent
lower voices. Do not invent harmony in an unmarked bar or transfer Amalia's
performance order to a different composition. Keep uncertain navigation explicit.

## Review and delivery

The independent reader checks all musical rows, including unflagged passages,
and specifically audits every changed event and proposed reconstruction. Any
disagreement is resolved from source evidence or retained as a human question.
Agreement between model readers is not an accuracy score or human approval.

Before delivery, validate rendering, bounded playback, section/repeat handling,
and all exceptional measure lengths. Verify save/reopen and exact ABC export in
a disposable copy. Preserve all prior source, canonical, experimental, and review
artifacts. Install a replacement only for an untouched model draft whose saved
revision and hash still match the snapshot. If Juan has edited it in the meantime,
retain a separate proposal for reconciliation instead of overwriting his work.

For tied chords and independent voices, inspect audible events as well as parsed
note durations. A finite audio stream can still contain an unintended reattack.
Distinguish positive-volume attacks from silent cursor/timing markers. Preserve
old validation reports when a player fix changes their interpretation, and add
new evidence with the player and ABC hashes rather than rewriting the old audit.

The current review desk has one working ABC and free-text review notes. The source
reading and structured decisions remain in the local delivery packet; the notes
identify that packet and summarize proposals. Agent work adds no human review time
and never marks a score Reviewed.

## Batch decision

Begin with Coqueteos, Rumichaca, and Atardecer to exercise ordinary handwriting,
scan damage, and polyphony. Scale only after complete source coverage, independent
audit, traceable editorial decisions, and delivery checks pass without unresolved
blocking regressions. Record the coordinator's decision and its limits. Source
damage or failed notation support should trigger bounded remediation rather than
unattended expansion. Human correction effort remains the practical measure of
whether this process is improving.

The pilot's local records are in
`out/collection/editorial_pilot_20261022_v1/`. Original manuscripts, draft ABCs,
and review records remain local; this document contains no manuscript content.
