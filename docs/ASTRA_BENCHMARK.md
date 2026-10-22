# Astra transcription experiment — 2026-09-09

**The original-PDF Astra attempt matched all seven reference measures.** It
recovered all 30 notes and three rests, including pitches, onsets, durations and
the two-note dotted-half chord. This supports a broader Astra transcription
pilot; it does not establish accuracy across the collection.

**Context-audit erratum, later on 2026-09-09:** A's separate projectless task
did not receive this repository's instructions. B's no-history subagent did
receive the repository `AGENTS.md` in its initial context, although it never
read a repository file through a tool. Those instructions contain historical
benchmark context. B therefore fails the intended total-context isolation
contract and must be treated as a diagnostic with repository context. The
scored outputs and numerical results are unchanged; the earlier tool-access
audit was too narrow to establish source-only isolation. This further limits
any causal interpretation of the A/B difference. The separate audit record is
`out/collection/astra_context_audit_erratum_v1.json`; original frozen artifacts
remain untouched.

## Fixed comparison

The target was Aviador system 8, selected before inference because it was our
strongest existing automatic full-event benchmark. Both new attempts used
`gpt-6-astra` at `xhigh`, with fresh contexts and one submission each:

- **A — original PDF:** a separate Codex task received the original single-page
  score, identified the requested staff itself, made its own crops and inferred
  clef, meter and inherited key. No measure or note count was supplied.
- **B — historical crops:** a separate subagent received the seven original
  raw/staff crops and their fixed historical context. It could inspect and
  enlarge those images but could not access the full PDF or A's work.
- **Baseline:** the existing frozen meter-gap-resolver predictions. Its complete
  metric summary was reproduced exactly using the original historical scorer.

| Metric | Frozen pipeline | Astra: original PDF | Astra: historical crops |
| --- | ---: | ---: | ---: |
| Strict note F1 | 42.6% | **100%** | **86.7%** |
| Exact measures, notes and rests | 1/7 | **7/7** | **4/7** |
| Ordered pitch accuracy | 46.9% | 100% | 86.7% |
| Ordered onset accuracy | 62.5% | 100% | 100% |
| Ordered duration accuracy | 75.0% | 100% | 100% |
| Rest F1 | 33.3% | 100% | 100% |

Strict note F1 requires pitch, onset and duration to agree together. It combines
precision and recall, so it is not generally the percentage of printed notes
read correctly. In A, all 30 reference notes match and none are added or missing.

## What explains the differences

The two Astra drafts have identical note counts, timing, rests and simultaneous
note grouping. B's four pitch errors are:

| Target measure | Event, counting rests | B output | Reference and A |
| --- | ---: | --- | --- |
| 3 | 4 | E4 | Eb4 |
| 6 | 3 | Bb4 | B4 |
| 6 | 6 | Eb5 | E5 |
| 7 | 4 | Bb4 | B4 |

The historical context supplied to B says **one flat, Bb**. The preceding
manuscript staff visibly carries **two flats, Bb and Eb**, which A independently
identified and inherited. This accounts for the E4/Eb4 difference. B also read
three handwritten natural signs as flats. A interpreted them correctly.

The primary benchmark was not repaired or rescored with improved context.
B remains a comparison with the historical input contract, including its defect.
The A/B gap cannot establish a general causal effect of full-page context:
resolution, inspection choices and run variation also differ.

Both models highlighted uncertainty around degraded notation, particularly
measure 3 and the two-note chord in measure 5. A nevertheless matched the
existing reference throughout. Their uncertainty statements are useful review
notes, not calibrated probabilities of error.

## Integrity and output checks

- The protocol, source inputs and existing reference identities were hash-pinned
  before either new inference attempt. Reference bytes were hashed for identity;
  their musical event content was not opened until after both submissions were
  sealed at **2026-09-09 14:51:32 UTC**.
- The seal covers 94 archived input, inspection, output and access-audit files.
  Neither draft received reference answers, evaluator feedback or the other's
  transcription. One procedural message asked A to finish rather than delay for
  an optional preview-font problem.
- Visible tool calls were audited. No reference, repository-content, sibling
  packet, memory-file or prior-task reads were found in either inference run.
  This checks tool access only. The later initial-context audit above found
  repository instructions already present in B; this was not an OS jail.
- Independent parsing with the same installed abc2svg library used by the review
  desk confirmed that each ABC matches its JSON in all seven measures, including
  chromatic pitches, onsets, durations, rests and simultaneous chord notes.
  Neither ABC produced parser errors. A's rendered notation was visually checked.
- A independently selected the correct staff and seven physical measures.
  The reference and both submissions were preserved after scoring. The live
  Aviador review draft was not used or overwritten by this experiment.

## Limits and next decision

This is **consumed development evidence**, not a fresh unseen-score accuracy
claim. It covers one system of one score, not a complete page or all 100 works.
There were no letter chord symbols or repeat-navigation examples in this target;
their transcription and playback remain untested here. Ties, slurs, articulations
and engraving fidelity are outside the historical event metric. The written
two-note chord in measure 5 is included in the scored notes.

Prioritize a bounded Astra-assisted transcription pilot using full-page context,
system crops, deterministic validation and the existing human review desk.
First test a fixed selection of excerpts from other scores with existing truth,
covering degradation, polyphony, key changes, rests, chord symbols and repeats.
Then freeze an attempt on a complete new sheet before human review and measure
active correction time. Keep runtime promotion separate from this diagnostic
win. Do not spend more effort on special-case recognition rules until that pilot
identifies the errors Astra actually leaves behind.

Juan can continue reviewing Aviador. This experiment creates no additional
ground-truth assignment for him; the next complete-sheet draft should come with
specific uncertainty locations and a timed review task.

## Runtime, artifacts and validation

A took **13 minutes 9 seconds**; B took **7 minutes 9 seconds**, running in
parallel. A spent part of its time solving a local preview-font problem. These
are agent task times, not human correction times or production latency estimates.
Both used Codex account usage; no external paid API calls were made.

| Logged usage | A | B |
| --- | ---: | ---: |
| Input tokens, cumulative across calls | 1,955,337 | 494,387 |
| Cached input tokens, included above | 1,884,288 | 449,024 |
| Output tokens | 21,645 | 13,062 |
| Reasoning output tokens, included above | 9,836 | 7,285 |

These are runtime-reported counters, not unique prompt sizes or a dollar-cost
estimate. Coordinator, evaluator implementation and audit usage are additional.
Future pilot runs should use the established renderer and constrain inspection
effort; lower reasoning levels can be compared after quality is characterized.

Local evidence is in `out/collection/astra_benchmark_v1/`:

- `protocol.json`, `dispatch.json`, `submission_seal.json` — fixed inputs,
  assignment and submission provenance.
- `arms/a/result/` and `arms/b/result/` — original ABC, JSON and uncertainties.
- `evaluation_v1/report.json` — complete three-lane metrics and hashes.
- `a_abc_consistency.json`, `b_abc_consistency.json`, `source_audit.json` —
  independent output and source checks.
- `access_audit/` — inspected tool-call records and runtime usage.
- `arms/a/transcription-verified-preview.png` — readable rendered A submission.

The new adapter is `scripts/experiments/evaluate_astra_music_benchmark.py`, with
focused synthetic tests in `tests/test_evaluate_astra_music_benchmark.py` for
event scoring, omitted/extra measures, invalid intervals, altered seals, reused
arm paths and changes to input hashes during evaluation. Existing output
directories are never overwritten. All six focused tests, focused Ruff and Black
checks passed. Reproduction depends on the pinned local
reference artifacts and original packet paths; this is not yet a portable public
benchmark bundle.
