# Astra workflow reproduction pilot

The pilot approved on 2026-10-28 reproduces the earlier visual-reading workflow
before another model comparison. It runs Aviador system 8 first and advances to
the complete Rumichaca manuscript only after a preregistered gate passes. Terra
and the full six-piece comparison were deferred until these findings were reported.
The subsequent bounded comparison is recorded in
[Assisted Astra–Terra comparison](ASSISTED_MODEL_COMPARISON.md).

## What is being reproduced

The successful historical Aviador attempt could inspect the original page,
choose close-ups, infer inherited context and check its own notation. Its task
suggested a 15-minute budget; it was not the hard 900-second process deadline
introduced by the later tool-free runner. The recorded historical task took
13 minutes 9 seconds, including work on a preview-font problem.

This pilot restores adaptive source inspection and notation validation through
bounded local tools. It is a capability-level reproduction, not a byte-for-byte
replay of the historical desktop task. Runtime/client versions, instructions,
tool definitions, resource limits and input hashes are retained as provenance.
An unknown underlying model revision remains unknown.

One historical capability is deliberately narrower here: the earlier task could
run arbitrary local image-processing code, including a hand-drawn staff guide.
The bounded pilot offers PDF crops at 160/240/320 DPI and notation feedback, not
arbitrary shell or drawing tools. A changed result cannot by itself establish a
model regression.

## Stages and boundaries

Each case has one fresh Astra Extra High author followed by one fresh Astra
source reader. The reader receives only the same PDF and the sealed initial
candidate. Both use supplied-source evidence; neither receives reference answers,
historical model outputs or evaluator feedback. The coordinator does not repair
musical content using the references. Initial and reviewed outputs are separate.

The author may choose crops and resolution and save provisional drafts. The
notation checker supplies structural feedback, not musical answers. Unsupported
readings and editorial choices remain explicit, and no musician approval is
created. Existing review files and the earlier timeout cohort remain intact.

| Target | Author allowance | Independent reader allowance |
| --- | ---: | ---: |
| Aviador system 8 | 30 minutes | 20 minutes |
| Rumichaca, complete score | 60 minutes | 30 minutes |

These are fixed stage limits, not predictions of expected latency. Partial
checkpoints survive failure, but a saved `complete` flag cannot turn an execution
timeout into a successful stage. No retries are selected using reference scores.

## Evaluation fixed before inference

Local preregistration is stored at
`out/eval/reproduction/astra_workflow_20261028_v1/evaluation_plan.json`
with a separate `execution_contract.json`. Those grading-only documents are
withheld from the readers.

The historical Aviador event scorer is reused and its frozen automatic baseline
must reproduce before interpreting new scores. Exact historical reproduction
requires the initial author to match all seven target bars and achieve strict
note F1 of 1.0. That metric covers written pitches, onsets, durations and rests;
it does not certify every tie, articulation, chord symbol or performance route.

Advancement requires a successfully completed, structurally usable initial
submission covering the target, followed by a successfully completed independent
review with at least six exact bars and strict note F1 of 0.95. Any case where
review makes that gate pass is disclosed. Both outputs are sealed before either
is scored. Failing the gate stops the full-score stage.

Rumichaca is the shortest of the six reviewed references by physical bar count,
not a claim that it is the easiest. Its full-edition agreement and the existing
nine-bar source-note audit are reported separately for initial and reviewed
drafts. Editorial acceptance and human correction time remain unmeasured.

This is consumed development evidence. A successful reproduction establishes a
baseline for a later same-workflow model comparison; it does not establish
collection-wide accuracy or permit pooling results from different harnesses.

## Results

Aviador's two stages completed successfully and were sealed before grading.

| Aviador system 8 | Initial AI draft | After independent AI review |
| --- | ---: | ---: |
| Exact note/rest bars | 6 / 7 | 7 / 7 |
| Strict historical note F1 | 0.933333 | 1.0 |
| Rest F1 | 1.0 | 1.0 |
| Written bars / meter validity | 7 / 7, valid | 7 / 7, valid |
| Stage elapsed time | 423.85 seconds | 461.94 seconds |
| Source inspections | 25 | 23 |

The initial author did not reproduce the historical perfect result. Independent
AI review recovered it and made the preregistered advancement gate pass. The
two-stage total was 14 minutes 46 seconds of model-run time, excluding setup and
coordinator gaps. This is one workflow trial, not two independent trials.

The initial error was confined to bar 7: its third and sixth eighth notes were
F-natural instead of F-sharp. Adding the accidental corrects both under normal
bar-local accidental carry, raising exact note matches from 28/30 to 30/30.
Rhythm and rests already matched. A separate audit replayed the original scorer
and checked the captured author/reader context without finding reference leakage.

The frozen older automatic pipeline control reproduced exactly: one of seven
bars and strict note F1 0.42623. It is context, not another new Astra attempt.

The complete Rumichaca stage started only after the passing Aviador report was
written. Both stages finished successfully with valid notation and all 45 bars
aligned. No reference feedback or coordinator musical repairs were supplied.

| Rumichaca, complete score | Initial AI draft | After independent AI review |
| --- | ---: | ---: |
| Audited-source note F1, nine bars only | 0.972222 | 0.972222 |
| Audited note events matched | 35 / 36 | 35 / 36 |
| Complete-edition note F1 | 0.881443 | 0.935401 |
| Exact edition bars, including chords | 20 / 45 | 23 / 45 |
| Edition chord-event F1 | 0.702703 | 0.702703 |
| Exact edition performance route | No | No |
| Encoded performance length | 88 bars | 117 bars |
| Stage elapsed time | 40m 26s | 17m 16s |
| Source inspections | 66 | 51 |

The combined full-score model-run time was 57m 43s, excluding setup and
coordinator gaps. Its first complete 45-bar draft still needed syntax, ending
notation and musical revisions. Both stages used ChatGPT subscription
authentication; token counts are retained, but no API price or subscription
cost per score is inferred. Most reported input tokens are cached prefix tokens,
not new input; raw and cached counts are preserved separately.

### What the remaining differences mean

- The one remaining audited-note mismatch is bar 42 (row 7, measure 1): the
  candidates use B-flat for the last eighth, while the frozen audit and Juan's
  recorded correction use B-natural. This audit covers only nine of 45 bars and
  is model-assisted, not a new complete musician audit.
- The reviewed candidate has 36/45 exact note-event bars. Full exactness also
  requires matching rests, chord symbols/onsets and meter: 13 additional bars
  differ only in chord events. Fourteen `sim.` annotations were not expanded
  into chord events. Other harmony differences concern the introductory F,
  Dm-to-C7 timing in bar 6, F versus Dm in bar 30, and the literal `F7dim` versus
  the adopted `Fdim7` in bar 42. These are not all established visual-reading
  errors; some are encoding differences or existing editorial choices.
- The author's prose described a 117-bar route, but its ABC encoded only 88
  because D.C. remained text. The reader encoded 117 bars, using ending 1 on the
  return and stopping at written bar 29. The approved edition uses ending 2 into
  B and ends at bar 45, producing 132 performed bars. The damaged manuscript's
  closing suffix remains ambiguous. Edition-route agreement is therefore
  reported separately from source accuracy and editorial acceptance.

The existing human harmony and form decisions already define the performing
edition. This experiment does not replace them or require Juan to answer them
again. All 13 live review files were hash-checked and remained unchanged.

### Next comparison

Run Terra on these same two targets with the same supplied-source policy,
inspection/validation capabilities, stage allowances and independent-reader
protocol before expanding to all six pieces. Preserve first-draft and reviewed
results separately. This pilot shows Astra can finish the work under the restored
capabilities and longer limits; it does not isolate the causal effect of tools
versus time, establish a model winner or establish collection-wide accuracy.

After the matched model comparison, two bounded tool improvements are supported
by the traces: expand explicitly marked harmony repeats into real chord events,
and compare an agent's declared performance sequence against actual ABC playback.
Neither should infer harmony in unmarked bars or choose an editorial route on the
musician's behalf. Version any such change and compare both systems under it.

No Terra inference was run in this pilot. Implementation checks passed: 47
targeted tests, Ruff and Black. The setup's successful synthetic canary is
separate from the musical results.

Local evidence: `out/eval/reproduction/astra_workflow_20261028_v1/`, including the
preregistered plan, execution contract, packet/context audits, immutable draft
checkpoints, tool traces, submission seal and `aviador_s8/evaluation/report.json`.
The separate offline `index.html` presents the initial and AI-reviewed columns.
Portable results without scores or private paths are in
[`eval/astra_workflow_reproduction_20261028_v1.json`](eval/astra_workflow_reproduction_20261028_v1.json).
