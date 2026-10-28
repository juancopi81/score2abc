# Reviewed six: evidence labels v3

V3 adds a **partial source-note audit: 67 of 312 physical source bars, containing
268 noteheads**, across all six pieces. The complete 314-bar performing editions
remain available for edition-agreement scoring. These are different metrics.
The two additional edition bars are Aviador's editorial cadences.

The source audit was performed by a model visually comparing the frozen PDF
pages and system crops with the already reviewed editions, before seeing the new
Astra/Terra outputs. It is **not new human approval, a blind audit, or complete
source ground truth**. The purposeful sample includes polyphonic intro holds,
later key contexts, accidentals, dotted rhythms and cross-bar ties. It is not a
random or representative sample of all 100 scores. Unlisted regions remain
unscored, including the documented reconstructions and unclear source readings.

| Work | Audited edition bars | Source bars audited | Noteheads |
| --- | --- | ---: | ---: |
| Amalia | 1-8, 41-47 | 15 / 48 | 62 |
| Atardecer | 1-3, 5-6 | 5 / 57 | 42 |
| Aviador | 2-16 | 15 / 62 | 47 |
| Carrizal | 1-4, 6-7, 22-24, 26-29 | 13 / 48 | 36 |
| Coqueteos | 1-5, 42-45, 47 | 10 / 52 | 45 |
| Rumichaca | 2-7, 42-43, 45 | 9 / 45 | 36 |

The versioned inventory is `score2abc/eval_labels_reviewed_six_v3.json`; its
create-once sidecar is `out/eval/labels/reviewed_six_v3/`. V1 and V2 remain
verifiable and retain their original empty source masks. V3 preserves all 30
decision labels, including the V2 Aviador corrections. It additionally pins the
source PDF hash, source/edition bar identity, page/system/measure location,
audited event-field mask, notehead count and canonical event hash for every
eligible bar. Changing the reference normalization fails closed until its audit
is updated in a new version. Here `system` means the musical row: Rumichaca's
musical row 1 corresponds to the review desk's source selector 002. The explicit
source bar number provides a second location independent of that UI selector.

```sh
uv run python scripts/build_eval_labels.py create \
  out/eval/references/reviewed_six_20261028_v1 out/eval/labels/reviewed_six_v3
uv run python scripts/build_eval_labels.py verify \
  out/eval/references/reviewed_six_20261028_v1 out/eval/labels/reviewed_six_v3
```

`score2abc.score_eval_source.score_source` evaluates pitch, onset, written
duration and tie endpoints only within these explicitly audited bars. It
preserves simultaneous duplicate noteheads. A missing audited bar contributes
false negatives; an added head within an audited bar contributes false positives.
Candidate-only bars cannot be located in the PDF by the current edition-based
alignment, so they are reported by the full edition grader rather than charged
to an arbitrary audited bar. Source coverage and alignment limitations accompany
the score. Rests, chord labels, dynamics, articulation and form are **not**
certified by this note mask. An editorial route differing from the selected
edition is not automatically a source error.

V3 also freezes `editorial_rubric_v1`: preservation of readable evidence,
complete playable choices, minimal defensible reconstruction, coherent form and
transparent decision records. Its grades are `acceptable`,
`acceptable_alternative`, `needs_revision` and `not_assessable`. Musical
adjudication must record the run, decision, criterion, grade, reason, reviewer and
date. No alternate reconstruction or route is automatically approved by exact
edition agreement or a structurally valid decision log. Until adjudicated,
`editorial_acceptance` stays null. Evaluate an alternative as a coherent whole,
not a combination of independently convenient bar choices.

## Preserved v1/v2 decision evidence

The create-once sidecar at `out/eval/labels/reviewed_six_v2/` annotates the
October 28 frozen **performing editions**. It pins the immutable reference
manifest SHA-256 `ad1995ca51050202770f8ceeb6fc6fe03ee2d622cbbf042b5ad84e3ca27f722f`
and each answer ABC/review hash. The decision inventory is versioned in
`score2abc/eval_labels_reviewed_six_v2.json`; the sidecar includes a byte-for-byte
inventory copy and pins its hash. V1 remains at `out/eval/labels/reviewed_six_v1/`
with its separate versioned inventory; both versions remain verifiable. No
reference, live review, history, or source PDF is changed.

```sh
uv run python scripts/build_eval_labels.py create \
  out/eval/references/reviewed_six_20261028_v1 out/eval/labels/reviewed_six_v2
uv run python scripts/build_eval_labels.py verify \
  out/eval/references/reviewed_six_20261028_v1 out/eval/labels/reviewed_six_v2
```

Creation refuses an existing destination. Verification checks the sidecar
checksum, frozen manifest, pinned ABC/review bytes, every quoted evidence file,
and the version-specific hand-authored inventory. The sidecar references paths within the frozen
answer tree, with hashes and short supporting excerpts. It contains decision
metadata rather than copies of complete scores. The `evidence_origin` field
distinguishes PDF review, human musical judgment, historical recording-informed
choices, and unclear basis. No external recording was opened for these labels.
`status=supported_by_review_notes` means the saved notes document the choice;
it is not a score of source certainty. `confidence=not_assessed` is deliberately
separate from the frozen review's musician approval state. Historical Draft
language in preserved notes does not override the frozen `review_state=reviewed`.
`status=supported_by_current_abc_and_history` marks a later human ABC edit that
superseded older model prose without a matching prose update.

## V2 correction from preserved human history

V1 repeated two Aviador model handoff claims that the frozen music disproves.
The model's revision 15 had D6 in inserted bar 45 and Gm on the first Trío bar.
Human revision 16 changed bar 45 to D5 dyads and removed the first Trío Gm;
human revision 17 restored the D→E7 chord sequence while retaining D5. Human
revision 18 marked Reviewed without changing that ABC. V2 labels the actual
bar 45 as quarter rest, F#5–D5 quarter dyad, G#5–D5 quarter dyad, with D then
E7. The Trío has 15 explicit editorial chord labels on edition bars 50–64;
bar 49 is unlabelled. V2 cites the frozen revision 15–17 ABC strings and current
ABC for both corrections. It preserves v1 as the initial annotation audit trail.

## V1/v2 source scoring boundary

Every V1/V2 work has `source_metric_eligible_measures=[]`. The preserved
notes establish particular decisions, but do not independently certify every
event and exact source-to-edition alignment in a whole bar. Unmentioned bars are
**pending source audit**, not confirmed. `source_metric_exclusions` identify
known editorial or uncertain dimensions for later use; they do not imply that
other dimensions or bars are already eligible. In particular, Aviador's Trío
chord exclusion applies to chords only, leaving its melody undecided. Edition
agreement can be shown descriptively; it is not headline PDF transcription
accuracy. The only exact numeric source-to-edition map presently documented is
Aviador's 62 original bars to 64 edition bars, with inserted bars 29 and 45.

## Evidence captured and remaining source coverage

| Work | Documented decision boundaries | Source audit still needed |
| --- | --- | --- |
| Amalia | S2M4 faint dot confirmed; S5M8 dotted-half D is contextual reconstruction; AA–BB–CC–A–B–CC is the accepted recording-informed edition route. | Unlisted bars, especially the contextual S5M8 duration, remain outside the V3 source mask. |
| Atardecer | S7M2 rhythm was clarified; the 147-visit route, opening recording-informed G7 arpeggio and numeric tempi are edition choices. | Only five intro/opening bars have V3 note masks; the recording-informed arpeggio is excluded. |
| Aviador | Opening eighth rest, B cadence bars 29/45, human-revised D5 dyads in bar 45, 15 Trío chord labels on bars 50–64 and closing harmony are editorial; the 62→64 map is pinned. | Bar 45 and Trío cannot be PDF note/chord truth merely because the edition was approved. |
| Carrizal | Named damaged pitches, cross-bar tie, and AA–BB–CC route are documented. | Thirteen bars audited, including the tied C; uncertain accidental glyphs in S1M5/S4M4 remain outside this audit. |
| Coqueteos | Ending lead-ins depend on destination; returned A and final B bypass endings when entering C; final stop is S9M4. | Clarify whether S8M5's saved B5-quarter duration reflects source ink or an editorial reconstruction if source duration scoring is needed. |
| Rumichaca | R2M1 rest and R3M1 changes are expressly nonvisible; Fdim7 was accepted; final D.C. includes intro provisionally; numeric tempi are editorial. | Establish the basis of R3M1 notes before source pitch scoring there; the damaged D.C. suffix cannot certify that return. |

The frozen source PDFs and reviewed editions are consumed material. These
labels do not make them heldout test data or establish accuracy on the full
collection. Expand source coverage or correct adjudications in a later sidecar
version and regrade both retained model outputs under that same version.
