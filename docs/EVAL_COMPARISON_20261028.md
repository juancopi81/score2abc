# Initial versioned Astra–Terra comparison

This is a consumed-development experiment on six reviewed pieces, not a
collection-wide accuracy estimate. It tests a shared, fixed-image authoring
harness. It does not reproduce the existing assisted drafting process with
interactive image inspection, an independent reader and musician corrections.

## Experiment

- Requested models: `gpt-6-astra` and `gpt-6.1-terra`, both `xhigh`.
- Authentication: ChatGPT subscription through the already-installed app-bundled
  Codex CLI `0.164.0-alpha.11.1`; no API-key billing. The provider did not expose
  a resolved weight revision or per-run USD charge.
- Each piece gets a fresh context, the PDF rendered at 160 DPI as a full-page
  overview plus three overlapping vertical crops, and the same instructions.
- Models have no tools, web access, repository access, reviewed answers or
  correction histories. A local request capture verifies prompt/image bytes,
  model settings and absent tools before each subscription call.
- One initial draft and at most two truth-free validation/repair passes.
  Each generation call has a 900-second deadline. Raw failures and repairs stay
  in the record; no best-attempt selection or human correction is performed.
- Label version `reviewed_six_v3` scores 67 of 312 physical source bars, containing
  268 noteheads. The partial visual audit is model-assisted, not new human approval.
  Full performing-edition agreement covers 314 bars, including two Aviador
  editorial insertions that are excluded from the source-completeness gate.

The frozen generation identity is
`c12eccee0c12f85b77bcc277227f7cf84330e100488b1f5488ad5e8106042569`.
Its code provenance is commit `7b786668b7579aa1a5df7aad54c5256abe55d23d`.
Later dashboard and documentation edits do not change those authoring files.
After all calls finished, a legacy-registry compatibility fix changed the grader;
the report records its exact code hashes separately from the frozen generation
identity. No candidate was rerun or edited.

## Results

| Piece | Astra | Terra draft | Audited-source note F1 | Edition note F1 |
| --- | --- | --- | ---: | ---: |
| Amalia | timed out | structurally complete | 0.339 | 0.333 |
| Atardecer | timed out | structurally complete | 0.260 | 0.352 |
| Aviador | timed out | partial | 0.000 | 0.052 |
| Carrizal | timed out | structurally complete | 0.138 | 0.170 |
| Coqueteos | timed out | partial | 0.286 | 0.266 |
| Rumichaca | timed out | partial | 0.411 | 0.414 |

Terra's macro mean is **0.239 audited-source note F1** and **0.265 edition
note F1**, including all three partial drafts. Astra's note metrics are unmeasured.
Terra used 13 reported calls (140,893 input + output tokens) and 9.5 minutes
of cumulative run time. Astra used six deadline-exhausted calls (90.2 cumulative
minutes); token usage was not returned. These are sums of run durations, not batch
wall time. USD costs remain unknown.

[Portable metrics and version identities](eval/astra_terra_20261028_v1.json)

Note F1 requires exact pitch, onset, written duration and tie matches after bar
alignment. It is stricter than pitch accuracy alone. A zero can reflect omitted
audited material; it does not necessarily mean every supplied pitch was wrong.
An absent score after a timeout is **unmeasured**, never zero musical accuracy.

Edition chord and route agreement compare the chosen human edition, including
its editorial choices. A different coherent route may still be acceptable.
Editorial acceptance and human correction time remain unmeasured.

## What the failure evidence tells us

Astra's deadline-exhausted calls returned no final score or token-usage receipt.
Retained logs show no authentication, rate-limit or transport error. The same
client passed small synthetic image-and-repair tasks. The full-size request audit
also verified input bytes and tool isolation. These observations do not identify
whether the full-score wait came from extended reasoning, service queueing or a
remote stall; they do not establish poor musical accuracy.

Terra illustrates why structural checks and musical checks must remain separate:

- **Amalia:** its first response was header-only. Repairs produced 48 bars and
  fixed the final bar's duration, but many pitches and ties still differed.
- **Atardecer:** it explicitly chose a simplified reduction, with 68 written
  bars versus the edition's 57. Some sustained lower voices became short block
  chords. Passing meter checks did not certify faithful polyphony.
- **Aviador:** its first response was a one-bar fragment. Repair expanded it to
  two valid bars, still explicitly labelled an excerpt. The local validator then
  stopped, with one repair unused; the complete-score grader correctly marked it
  partial and measured the missing coverage. This is an instruction-following
  failure and a completion-check gap, not a successful whole-score transcription.
- **Rumichaca:** all three calls returned a draft, but the opening short bar
  remained a possible pickup requiring inspection. The runner preserved this as
  partial after its repair budget. The displayed meter rate excludes that
  unresolved pickup candidate; 100% meter validity does not certify every bar.

These findings do not support replacing the current assisted collection workflow
with this Terra configuration. They also do not support a musical winner against
Astra while Astra's source scores are unavailable.

## Next bounded experiments

1. Diagnose one Astra full-score call under a separately frozen, longer deadline
   before scheduling another large batch. Preserve this 900-second cohort. Do
   not mix a longer-budget attempt into its leaderboard or attribute a timeout
   to bad note recognition.
2. Add PDF-row coverage evidence to a new shared authoring harness: source-only
   row identifiers, each mapped to an ABC span or explicitly untranscribed.
   Missing rows must reach repair feedback; valid meter alone must not stop a
   whole-score task. Retain readable polyphony rather than accepting a declared
   reduction as faithful transcription.
3. Compare both configurations again under the same new harness. Broaden source
   labels and adjudicate coherent editorial alternatives in new label versions,
   regrading all saved outputs together. Then repeat finalists and test fresh
   scores before extrapolating to the collection.

## Local evidence and reproduction

Original PDFs, reviewed editions, raw responses and generated scores remain
local. The portable metrics summary contains no source music or review histories.

- Input packet: `out/eval/run_inputs/versioned_astra_terra_20261028_v1/`
- Attempts: `out/eval/runs/astra_terra_20261028_v1/`
- Labels: `out/eval/labels/reviewed_six_v3/`
- Dashboard: `out/eval/comparison/index.html`

Run and regrade commands are documented in [the runner guide](EVAL_RUNNER.md).
Use a new destination for every run, label version and report. Configuration and
grading identities are described in [the versioning guide](EVAL_VERSIONING.md).
