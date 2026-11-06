# Versioned assisted review instructions

These instructions apply to the independent reader on either execution backend.
They do not change the author, source tools, validator, evaluator, or frozen truth.
The runner already seals the complete task's `prompt_template_sha256` and the
candidate-appended `prompt_sha256`; no new runtime dependency is needed.

- `assisted_review_v1.txt` preserves the generic reader instructions from the
  original Aviador subscription/API tasks. Scope and time allowance are separate.
- `assisted_review_v2.txt` adds visual comparison of ambiguous accidentals,
  source-bounded musical context, and explicit separation of visible evidence,
  editorial inference, alternatives, and uncertainty. This is a development
  change motivated by an adjudicated error, not a demonstrated accuracy gain.
- `versions.json` records content hashes and provenance. Once used in a scored
  run, retain these versions unchanged and add a new version for later edits.
  Its status describes version creation; later measured outcomes are recorded in
  the separate experiment summaries, without rewriting that history.

To compose a task, use the same source-scope paragraph as the comparison run,
then the selected generic instructions, then the same time-allowance paragraph.
Pass the resulting file to `prepare --prompt` for a reviewed stage, or to
`pair --review-prompt`. Both API and subscription accept the same composed file.
The version is selected explicitly: old sealed packets and historical tasks
retain their old instructions. For new collection reviews, use v2.

The Aviador v2 task, original proposal, authorized execution and results are under
`out/eval/reproduction/astra_api_review_context_20261029_v3/`. The completed
reviewer-only replay corrected bar 7, reached 7/7 exact bars, and left the other
six bars unchanged. See [the findings](../API_WORKFLOW_PILOT.md). The task includes
no human answer, target accidental,
target-note count, expected key, or hypothesis specific to the disputed measure.
Its author draft and whole seven-bar source scope are unchanged.

## Reviewer-only regression protocol (completed October 29)

Run one fresh API reader on the sealed, completed v2 API author's draft, with
these v2 instructions, the same PDF/tools, Astra at xhigh, and the same 20-minute
allowance. Select an explicit spending policy before preparing the execution
packet. No paid execution is authorized by this document or by task composition.
Do not append the human adjudication, this README, the manifest, or old grading
feedback to the model's task. The runner appends only the saved author's draft
and decisions. Save to a new reviewed-stage directory; never replace old runs.

Score all seven bars after sealing the new reviewer output, reporting both the
targeted correction and regressions elsewhere. Report uncertainty/decision quality
separately from exact musical matches. Compare against the old reader on this
same author draft. This is a reviewer-only consumed regression test, not a new
end-to-end API reproduction, a blind accuracy estimate, or causal proof of an
instruction effect from one stochastic attempt. No automatic retry-to-pass.

A pass supports trying the instructions on another score; a failure should be
preserved and diagnosed before spending on another attempt. Fresh-score results
are needed before claiming general improvement. Keep prompt optimization such as
GEPA deferred until there are enough diverse development cases and separately
held-out scores to assess transfer rather than repeatedly optimize this example.
