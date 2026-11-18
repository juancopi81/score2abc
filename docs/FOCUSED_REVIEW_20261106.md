# Gato labels and two difficult passages — November 6 checkpoint

Gato’s separate source-note label set is complete. Entusiasmo and La Chata have
focused, independently checked evidence and short playable comparisons. No live
draft, reviewed edition, review history or existing benchmark was changed.
Estrella remains Juan’s current review.

Open the [five-minute review packet](http://127.0.0.1:8777/diagnostics/focused_review_sprint_20261106_v1/index.html).
The detailed local evidence is under
`out/collection/focused_review_sprint_20261106_v1/`.

## Gato: source evidence and accepted choices stay separate

Two different Astra `xhigh` readers inspected all seven source rows and all 42
mapped bars. Both support **36 bars / 191 ordinary noteheads**. Six whole bars
remain excluded: bar 17 contains the accepted final F-sharp quarter reconstruction;
bars 38–42 involve the adopted octave span and its endpoint. These exclusions do
not reopen the musician’s approval or imply that the performing edition is wrong.

The readers saw the frozen edition being audited, but no candidate model outputs
or each other’s verdict before recording their own. This is consumed,
model-assisted visual evidence, not blind transcription or fresh human ground
truth. The mask covers note pitch, onset, duration, ties and duplicate-head
multiplicity. Rest events, chord symbols, grace notes, articulation, dynamics and
form remain outside the mask.

Twenty-three recorded decisions retain exact frozen-history evidence: 18 human
source-correction/articulation records, one reconstruction, two octave/form
choices, and two playback/encoding fixes. A separate reader checked their
classifications after freezing its own source audit. Human-save provenance is
evidence of the edit, not proof of its source accuracy. The full history, including
superseded octave endpoints, remains intact.

The separate form contract passes: 42 source identities, 42 written bars, and
112 visits following `AABBCXCYAB`, ending with an explicit stop. The expected
section ranges were declared from the adopted edition, not derived from the
parser’s actual route. This validates encoded form and mapping bookkeeping, not
the manuscript’s exact intended return policy.

`gato_source_v1` is a separately sealed sidecar under
`out/eval/labels/gato_source_v1/`. Gato is **not** added to the six-score comparison.
The original six references, v3/v4 label defaults and historical results remain
unchanged. [Portable counts and hashes](eval/gato_labels_20261106_summary.json)
are tracked; complete music and audit evidence stay in local artifacts/backup.

## Entusiasmo: navigation is the strongest reading

**November 13 human update:** Juan Carlos inspected groups A/B/C and classified
them as navigation lettering. The terminal-cluster concern about omitted timed
notes/rests is closed for the current draft. Exact wording, return destinations
and the no-repeats policy remain open. The ABC and playback route are unchanged;
the score remains Draft. The saved review notes retain this decision and all
earlier assessments. See the [human adjudication](eval/entusiasmo_human_adjudication_20261113.json).

The following records the original November 6 investigation.

Both readers favor a navigation/lettering cluster after S12’s final double bar,
probably a return to the earlier Fine. The opening segno and earlier cursive
Fine support that reading. The exact terminal lettering, whether any B/C stroke
is timed music, and the final no-repeats policy remain uncertain.

No extra pitches or durations can be recovered confidently. The existing draft
already follows the recommended return, so no full-score replacement was made.
At that checkpoint, the possible-omitted-events concern remained open; the human
classification above now resolves that specific concern.

The packet labels the three ink regions and compares them with the opening sign,
earlier Fine and ordinary noteheads. Its 24-second return audition preserves
baseline bars 85, then 2–16, exactly. Independent checks confirm notes, rests,
durations, harmony labels, route suffix and the complete final chord hold.

The original focused question was whether B/C were lettering/signs or contained
a timed event. The November 13 answer resolves it without granting whole-score
approval.

## La Chata: retain the unresolved reading

Neither reader finds a secure cancellation or internal barline in S2m1’s crowded
span. The current 11-eighth / 5.5-beat encoding cannot be replaced with a three-beat
bar merely because that fits the meter. A complete alternative voice alignment
also remains unsupported.

There is a definite encoding fact: `[C=G=c]` sounds low C-sharp4, G4 and C-natural5.
The upper natural does not change the lower octave in the current renderer. The
scan does not securely assign an additional natural to low C, and C7 harmony is
context rather than proof.

Three short auditions isolate the choices:

| Audition | Target span | Difference |
| --- | --- | --- |
| A | 5.5 beats | Exact retained draft |
| B | 3 beats | Hypothetical later-cell replacement; removes the early rest and nine sounding heads |
| C | 5.5 beats | Retains every event; changes only the two low C-sharps to C-naturals |

All include neighboring bars at 96 BPM. Each retains the final D4/C5 tie as a
2.5-beat hold without continuation reattack. B and C are explicitly unapproved
editorial auditions. Neither was installed. The musician can decide structure
and pitch separately, or leave both unresolved. “Possible overwrite” is the
appropriate description; cancellation itself is not established.

## Reproduction and validation

The new compiler accepts only complete, input-pinned reports from two distinct
readers. A bar enters the mask only when both support all five note fields and
their observed-note multisets equal the frozen edition. Known editorial exclusions
win even if both readers say supported. Decision evidence must name a file in the
verified freeze and quote exact content with a matching hash. Semantic editorial
classification remains the separate visual/history audit’s responsibility.

Publication is create-once, outside the immutable reference; verification checks
the exact file inventory, hashes and deterministic reconstruction of the labels.
It does not register or modify a benchmark default.

```sh
uv run python -m scripts.build_selected_reference_labels verify \
  out/eval/references/gato_reviewed_20261106_v1 \
  out/eval/labels/gato_source_v1

# Replay into a NEW destination only; the delivered sidecar must not be overwritten.
uv run python -m scripts.build_selected_reference_labels build \
  out/eval/references/gato_reviewed_20261106_v1 out/eval/labels/gato_source_replay \
  --label-set gato_source_v1 \
  --packet out/collection/focused_review_sprint_20261106_v1/gato/input/audit_packet.json \
  --author out/collection/focused_review_sprint_20261106_v1/gato/author/audit.json \
  --review out/collection/focused_review_sprint_20261106_v1/gato/review/audit.json \
  --decisions out/collection/focused_review_sprint_20261106_v1/gato/coordinator/decisions.json
```

The coordinator decision copy only exposes the octave-history record’s existing
three evidence quotes at the compiler’s top-level evidence field; the original
author report and all history records are preserved.

Validation: 1,380 tests passed, three socket-dependent tests skipped; all 18 new
integrity tests pass. Ruff and Black pass. Independent code review found no
actionable issue. Synthetic grader controls confirm an audited pitch mutation
lowers source F1 and an excluded-region mutation does not; these are adapter
checks, not new model performance results. The local dashboard loads both crops
and all four audio files. Four Astra agents completed the six source-review passes;
a Sol High agent reviewed the compiler. No paid API call was made.

All 78 pre-pinned live review/history files are byte-identical, including Estrella.
Both prior freezes verify: Gato’s 134 files and the original six’s 903 files.

## Checkpoint and resume

Local branch: `codex/gato-labels-passage-audit`, based on merged PR #18. This is a
local checkpoint; no push or new PR is part of this task. The verified archive and
receipt are saved separately under
`/Users/juanpineros/Downloads/jaime-llanos/backups/focused-review-20261106-v1.tar.gz`.
It is another local copy on the same computer, not an off-machine backup.

Continue Estrella at Juan’s pace. Then use the focused Entusiasmo and La Chata
packets when convenient. No transcription rerun is required to resume, and no
newly uncertain bar should silently enter source truth. To serve the saved page
again, run `python3 -m http.server 8777 --bind 127.0.0.1 --directory out/eval`.
