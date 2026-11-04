# Assisted Astra–Terra comparison

This comparison tests the two targets from the completed Astra reproduction:
Aviador system 8 and the complete Rumichaca score. Both targets were selected
before Terra inference. Rumichaca is attempted regardless of Terra's Aviador
result; the earlier Astra advancement gate does not filter the comparison.

## Findings, October 29

Keep Astra as the transcription baseline. Terra produced a complete Rumichaca
draft after independent review, but its note agreement remained substantially
lower. Its faster execution does not establish lower cost per acceptable score.
No larger batch was run and the 13 live musician-review files remain unchanged.

| Metric | Astra initial → reviewed | Terra initial → reviewed |
| --- | ---: | ---: |
| Aviador exact note/rest bars | 6/7 → 7/7 | 4/7 → 2/7 |
| Aviador strict note F1 | 0.9333 → 1.0000 | 0.7541 → 0.6557 |
| Rumichaca written bars | 45 → 45 | 17 → 45 |
| Rumichaca edition note F1 | 0.8814 → 0.9354 | 0.4516 → 0.6313 |
| Rumichaca audited-source note F1, nine bars only | 0.9722 → 0.9722 | 0.5600 → 0.5823 |
| Rumichaca edition chord F1 | 0.7027 → 0.7027 | 0.2593 → 0.6301 |
| Rumichaca exact edition performance route | No → No | No → No |

Terra's Aviador reviewer changed three naturals to flats, worsening two previously
exact bars. The Rumichaca reviewer recovered missing coverage, but neither model
matched the adopted performing edition's complete route. Ambiguous form choices
are edition disagreements, not automatically proven source-reading errors.

Terra's reviewed Rumichaca has **17/45 identical note-event bars**. The broader
exact-edition-bar count is **0/45** because Terra encoded `6/8` while the edition
uses `3/4`; the comparator requires literal meter equality. Seven bars agree in
notes, rests and chords and differ only in meter. The zero must not be read as
zero note accuracy. Note F1 is calculated independently of that meter-label
mismatch. Both candidates fill their declared meters, which alone does not
certify correct musical content.

| Scored author + reader | Astra time / output tokens | Terra time / output tokens |
| --- | ---: | ---: |
| Aviador | 14m 46s / 20,279 | 4m 53s / 18,562 |
| Rumichaca | 57m 43s / 76,382 | 6m 11s / 19,116 |

Terra used about 8% fewer output tokens for Aviador, but 11% more input tokens.
For Rumichaca it used about 75% fewer output tokens and 83% fewer input tokens
in the scored pair. These are observed resource counts, not dollar savings or
equal-quality comparisons. The interrupted earlier attempt adds 123.72 seconds;
its token usage is unavailable and is not silently treated as zero. Setup,
synthetic canaries, coordinator work and idle gaps are excluded from model-run
times. Cached-input and reasoning-output subsets are shown in the dashboard.

The compatibility audit verified 17/20 dimensions. All ten Aviador checks match.
Rumichaca differs in the runtime for both stages and in eligibility to review a
partial author draft. The source, generic prompts, tool implementations,
allowances and numerical scoring were retained. These differences prevent a
clean model-only causal conclusion for Rumichaca.

Open `out/eval/comparison/index.html` for the updated dashboard. It retains a link
to the separate tool-free experiment. [Portable metrics and provenance](eval/assisted_astra_terra_20261029_v1.json)
contain no source scores or private paths. Validation: 143 focused tests passed;
Ruff, Black and patch whitespace checks passed.

## Shared workflow

Each model writes one source-only draft and then independently reviews that
draft in a fresh session using the same model and Extra High reasoning. The
reader receives its own model's sealed draft, the supplied PDF and the same
generic review instructions. Neither session receives the reference edition,
evaluation labels, the other model's answer or coordinator musical corrections.

The model can inspect PDF pages and choose crops at 160, 240 or 320 DPI. It can
save provisional ABC drafts and receive deterministic notation, meter and
playback feedback. Initial and reviewed outputs are sealed before either is
graded. All intermediate drafts and uncertain editorial decisions are retained.

| Target | Author allowance | Independent reader allowance |
| --- | ---: | ---: |
| Aviador system 8 | 30 minutes | 20 minutes |
| Rumichaca, complete score | 60 minutes | 30 minutes |

The requested models are `gpt-6-astra` and `gpt-6.1-terra`, both at `xhigh`.
The runner uses ChatGPT subscription authentication. It records reported usage;
it does not infer API charges or a dollar cost from the subscription.

This is the same kind of assisted author-and-reader process used to prepare the
collection for musician review. It is narrower than the historical desktop
workflow: it provides bounded source-inspection tools, excludes arbitrary shell
access and does not include the musician's subsequent corrections. Passing this
experiment does not grant human approval to a score.

## Why retain the earlier tool-free experiment?

The earlier runner supplied fixed PDF-derived images and performed validation
between model calls. That design made isolation and fixed inputs easier to
audit, but removed the models' ability to choose close-ups and interact with
the checker while drafting. It also used shorter deadlines. Its results remain
a separate experiment, not evidence of a model-only change from the successful
production workflow.

This comparison holds the assisted workflow fixed. A compatibility audit binds
the reports to the actual data, instructions, tools, allowances and grader.
The runner's model-selection wrapper and the evaluator's identity checks were
generalized; numerical scoring and source tools remain unchanged. Reviewer
candidate text legitimately differs because each reader reviews its own author.

### Interrupted attempt and runtime change

Juan paused the experiment on October 28 after subscription tokens ran out.
Terra's Aviador author and reader had finished; Rumichaca's author ended with a
partial checkpoint after 123.72 seconds, a broken pipe and no final usage total.
That interrupted attempt is retained and is not scored as transcription quality.

On October 29, Juan authorized a fresh Rumichaca author and independent reader.
The new author receives no partial draft or earlier answers. Its source, generic
instructions, bounded tools, effort and allowances remain fixed. The app had
updated from Codex CLI `0.164.0-alpha.11.1` to `0.165.0-alpha.1.2`, so the new
runtime is separately pinned, audited and tested with a synthetic source task.
Historical runtime manifests and attempts are preserved. Aviador's matching
conditions can still be verified from its sealed evidence; Rumichaca's comparison
must disclose the changed runtime and cannot isolate the model's causal effect.

The runner's `prepare --runtime-dir <prepared-runtime>` option selects a new
version explicitly and verifies its binaries before preparing an input packet.
The default remains the original runtime. No existing runtime pin is refreshed
in place.

The fresh Terra author ended its turn after 134.81 seconds with a valid 17-bar
excerpt and an explicit statement that it could not complete the manuscript.
It did not reach its deadline. This incomplete result is preserved without an
author retry or reference-guided correction.

To perform the requested independent review, an explicit addendum allows the
fresh Terra reader to receive this partial draft. The original generic review
instructions still request a complete best-effort reading. The opt-in
`--allow-partial-author` flag accepts only a cleanly completed, non-quarantined
author turn; it does not admit interrupted or timed-out runs. This is a disclosed
review-of-partial-draft variant of the original successful-author protocol.
Both initial and reviewed results remain visible, and the compatibility audit
records this reviewer-eligibility change as well as the runtime change.

## Reading the results

Aviador uses the historical seven-bar note/rest scorer, which excludes ties,
chord labels and form. Rumichaca reports two different quantities: agreement
with the complete musician-reviewed performing edition, and note accuracy on
the existing nine-bar source audit. The latter is a partial model-assisted audit,
not whole-score source accuracy. Form and reconstructed passages can differ
from the adopted edition without establishing a definite visual-reading error.
Terra’s resolved Rumichaca route contains 88 performed bars, an exact prefix of
the edition’s 132: its quoted D.C. text does not execute the final return. This
playback omission is distinct from choosing a different editorial route.

The dashboard separates first drafts from independently reviewed drafts and
keeps note, chord and form metrics distinct. Time and usage include both stages
when reported as a workflow total. Cached input is a subset of input tokens;
reasoning output is a subset of output tokens. Neither subset is added again.

There is one scored author/reader pair per model and target, with Terra's
interrupted Rumichaca attempt disclosed separately. These are consumed development cases,
not unseen-score generalization evidence. The next decision is whether the
observed quality and resource use justify expanding the comparison; no larger
batch is authorized by this experiment.

Local artifacts are under
`out/eval/reproduction/assisted_astra_terra_20261028_v1/`.
The immutable Astra predecessor remains under
`out/eval/reproduction/astra_workflow_20261028_v1/`.
