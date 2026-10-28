# Reviewed six: evidence labels v2

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

## Source scoring boundary

Every work currently has `source_metric_eligible_measures=[]`. The preserved
notes establish particular decisions, but do not independently certify every
event and exact source-to-edition alignment in a whole bar. Unmentioned bars are
**pending source audit**, not confirmed. `source_metric_exclusions` identify
known editorial or uncertain dimensions for later use; they do not imply that
other dimensions or bars are already eligible. In particular, Aviador's Trío
chord exclusion applies to chords only, leaving its melody undecided. Edition
agreement can be shown descriptively; it is not headline PDF transcription
accuracy. The only exact numeric source-to-edition map presently documented is
Aviador's 62 original bars to 64 edition bars, with inserted bars 29 and 45.

## Evidence captured and remaining narrow questions

| Work | Documented decision boundaries | Source audit still needed |
| --- | --- | --- |
| Amalia | S2M4 faint dot confirmed; S5M8 dotted-half D is contextual reconstruction; AA–BB–CC–A–B–CC is the accepted recording-informed edition route. | Exact event-level source alignment beyond named decisions. |
| Atardecer | S7M2 rhythm was clarified; the 147-visit route, opening recording-informed G7 arpeggio and numeric tempi are edition choices. | Full source audit before whole-bar metrics. |
| Aviador | Opening eighth rest, B cadence bars 29/45, human-revised D5 dyads in bar 45, 15 Trío chord labels on bars 50–64 and closing harmony are editorial; the 62→64 map is pinned. | Bar 45 and Trío cannot be PDF note/chord truth merely because the edition was approved. |
| Carrizal | Named damaged pitches, cross-bar tie, and AA–BB–CC route are documented. | Whole-bar source coverage remains unproven. |
| Coqueteos | Ending lead-ins depend on destination; returned A and final B bypass endings when entering C; final stop is S9M4. | Clarify whether S8M5's saved B5-quarter duration reflects source ink or an editorial reconstruction if source duration scoring is needed. |
| Rumichaca | R2M1 rest and R3M1 changes are expressly nonvisible; Fdim7 was accepted; final D.C. includes intro provisionally; numeric tempi are editorial. | Establish the basis of R3M1 notes before source pitch scoring there; the damaged D.C. suffix cannot certify that return. |

The frozen source PDFs and reviewed editions are consumed material. These
labels do not make them heldout test data or establish model accuracy. New
adjudications belong in a later sidecar version, preserving this one.
