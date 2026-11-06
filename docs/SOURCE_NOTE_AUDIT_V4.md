# Expanded source-note audit and offline regrading

PR #17 is merged. The follow-up uses the same frozen six performing editions,
original candidate bytes, ABC parser, alignment and numerical metrics. No new
transcription, API call, live review edit or human approval is part of this audit.

## Findings

Coverage expands **67 → 122 of 312 source bars**, **268 → 547 noteheads**. Of 60
newly selected bars, 55 pass both readers. None of the original 67 labels or 30
editorial decisions changes. V4 remains explicitly selected; defaults stay v3.

| Piece | V3 bars | V4 bars | V4 noteheads |
| --- | ---: | ---: | ---: |
| Amalia | 15 | 24 | 101 |
| Atardecer | 5 | 14 | 97 |
| Aviador | 15 | 25 | 97 |
| Carrizal | 13 | 22 | 77 |
| Coqueteos | 10 | 19 | 95 |
| Rumichaca | 9 | 18 | 80 |

Five selected bars are withheld: Amalia 12 (dot), Atardecer 51 (A accidental),
Carrizal 11 (dot and resulting onset), Coqueteos 15 (incoming tie), Rumichaca 44
(detached heads/stems and duration). These exclusions do not establish that the
approved edition is wrong and do not edit or reopen its review state.

| Rumichaca, after independent model review | Old source F1 (9 bars) | Expanded source F1 (18 bars) | Edition note F1 (unchanged) | Adopted form |
| --- | ---: | ---: | ---: | --- |
| Astra subscription | 97.22% | **98.75%** | 93.54% | Differs |
| Terra subscription | 58.23% | 60.71% | 63.13% | Differs |
| Astra API, review v2 | 69.44% | 77.50% | 80.21% | Matches |
| Astra API, completion gate | 83.33% | 87.50% | 88.14% | Differs |

The broader mask exposes a previously hidden review gain for Astra subscription:
**90.0% author → 98.75% reviewed**, whereas the old mask gave both 97.22%.
The reviewed result matches 79 of 80 audited noteheads; the remaining mismatch is
source bar 42. This supports retaining the independent review step. It does not
establish complete-score fidelity or a clean causal API/subscription difference.

The original tool-free Terra six-piece macro source F1 becomes **23.88% → 27.04%**
under v4. All six Astra timeouts still have no score; this remains a separate,
incomplete model comparison. All 14 available terminal full-score outputs were
regraded; all their original v3 metrics reproduced exactly.

**Recommendation:** keep assisted Astra subscription as the current transcription
baseline, continue the musician review queue, and use these retained outputs for
further offline error analysis. Do not spend on another API run just to confirm
the same ranking. Gato’e Fique remains the user's current review; there is no need
to interrupt it to re-review these five ambiguous reference regions now.

[Portable results](eval/source_note_regrade_v4_20261106.json) retain every stage.
The new [local dashboard](http://127.0.0.1:8777/source_audit/reviewed_six_v4_20261106/evaluation/index.html)
shows old/new source metrics beside unchanged edition metrics, grouped by
experiment. Prior dashboards and reports remain preserved.

## Method

Sixty previously unaudited bars were selected before consulting candidate
metrics: ten evenly distributed eligible bars per piece. The selection excludes
existing v3 masks and documented note, duration and rest reconstructions.
Rejected selections are not replaced with easier bars.

Two different Astra `xhigh` readers examined each selection using the supplied
PDF, whole-page context and closeups. Each could see the frozen approved edition
being checked, but was instructed not to inspect candidate outputs, historical
metrics or the other reader's report. The second pass crossed work assignments.
This is a partial, model-assisted audit of consumed material, not blind reading,
new musician certification or an estimate of unseen-score accuracy.

A whole bar enters v4 only when both readers support every pitch, onset,
duration, tie endpoint and duplicate notehead. Exact event hashes bind the mask
to the frozen edition. Ambiguous marks or a reader disagreement exclude the
whole bar. Old v3 masks and all 30 editorial decisions remain unchanged; v3
continues to be the default label set for existing workflows.

The exact audit plan, six input packets, twelve reader reports and reader
assignments are preserved in a checksum-verified evidence bundle. Local page
renders and source closeups remain under
`out/eval/source_audit/reviewed_six_v4_20261106/`. The expansion does not certify
chord symbols, rest events, form or unlisted source regions.

## Offline comparison

The regrader verifies terminal ABC hashes from their original reports or registry
before opening frozen reference content. It recomputes v3 and v4 using identical
candidate bytes and alignment, and refuses to publish if any old source, edition,
chord, meter, bar-coverage or form metric fails to reproduce.

Scope: author and reviewed outputs from four Rumichaca experiments, plus all six
Terra outputs from the original tool-free suite. Six tool-free Astra failures
still have no saved score. Intermediate tool drafts, the interrupted Terra
attempt and Aviador's separate seven-bar experiments are not silently turned
into full-score runs. Historical reports remain intact.

Generation conditions remain separate: assisted subscription, API review v2,
API completion gate and tool-free 900-second runs. Rumichaca's Astra/Terra
subscription pair also has the previously disclosed runtime and partial-draft
review-policy differences. Regrading cannot remove these confounds.

## Reproduce

With the local frozen data and completed audit reports available:

```sh
uv run python -m scripts.regrade_source_audit \
  out/eval/source_audit/reviewed_six_v4_20261106 \
  out/eval/source_audit/regrade_v4_replay
```

The output directory must be new. The command verifies the committed v4 evidence
and inventory; it does not regenerate model outputs. `--materialize-labels` is a
create-once authoring option and must not be used to overwrite the delivered v4
inventory or sidecar. V4 is selected explicitly; subscription and API generation
instructions and their defaults are unchanged.

## Checkpoint and next action

Branch: `codex/source-note-audit-v4`, based on merged PR #17. Code, evidence
inventory, portable results and documentation are saved locally; no new PR or
push is included in this task. The user continues Gato’e Fique review. Further
API generation awaits discussion of these findings.

Validation: 1,353 tests passed, three skipped; 17 new focused integrity tests
passed. Ruff, Black and the rendered dashboard pass. All 1,025 pre-pinned
reference/label/historical-report files are byte-identical. Every graded ABC
matches its original report or registry hash. No live review was edited. A
separate code review found and verified fixes for coercible null ties and
publication before failed grading; no remaining blocker was reported.
