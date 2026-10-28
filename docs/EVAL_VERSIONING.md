# Evaluation configurations and versioning — proposed next slice

Status: design recorded on 2026-10-28; not implemented by the current two-model
report schema. PR #15 delivers the offline baseline. No new model runs or provider
integrations are authorized by this document itself.

The useful comparison unit is a **system configuration**: provider, exact model
identifier, reasoning/generation settings, authoring instructions, tools, repair
policy and resource limits. A model name alone cannot identify what was tested.
The same model with a new rhythm checker is a different system configuration.
The existing deterministic pipeline can also be a system, with model/effort
recorded as not applicable rather than assigned a fictional model name.

## Separate the experiment's identities

| Identity | What it pins | What changes it |
| --- | --- | --- |
| Dataset | Exact PDFs, selected pieces, task targets and split | Different inputs, pieces or target scope |
| Labels | Source-supported events, editorial exclusions, accepted alternatives and mappings | A corrected reference or new adjudication |
| Grader | Normalization, alignment, metrics, denominators and acceptance rubric | A scoring fix or changed definition of success |
| Authoring harness | Prompt, image preparation, available tools, environment and feedback/repair policy | A new deterministic tool, prompt or tool behavior |
| System configuration | Harness plus provider/model/settings and attempt budget | Model, effort, decoding settings or allowed resources change |
| Run / trial | One attempted piece under a pinned system; artifacts, usage, timestamps and outcome | Every new attempt, including failures |

Keep readable version names, exact content hashes and code commits. Preserve
previous manifests and results. Record the requested model identifier and the
provider-resolved revision when available; an alias and a date are not proof of
identical underlying weights. File `schema_version` describes a serialization
format, not the musical benchmark version.

The current freeze and V1/V2 label sidecars already provide independent data and
annotation identities. Current code does **not** yet pin a complete grader/tool
release or permit arbitrary configured providers/models. Extend the manifest and
registry before spending on a broader comparison; retain the existing reports
as historical artifacts rather than silently rewriting their contracts.

## Compare one intended change at a time

Examples below are proposed experiment shapes, not measurements or claims that
a provider/model is currently available:

| Question | Change | Hold fixed |
| --- | --- | --- |
| Which model reads better? | Model/provider | Data, labels, grader, shared instructions/tool capabilities, repair limits |
| Is more reasoning worthwhile? | One model's reasoning setting | Data, labels, grader, tools and other settings |
| Does a beat-checking tool help? | Same model with tool off/on | Data, labels, grader and the rest of the authoring harness |
| Can code replace a model substep? | Implementation of that substep | Inputs, output contract and final musical grading; record total system cost |

A tool experiment uses a new authoring-harness version and a fixed grader. The
validator may help the author inspect a rhythm, but cannot supply hidden labels.
When the grader or labels change, regrade retained outputs for **all** compared
systems under the same new version and keep the original reports. Re-inference
is necessary if the changed validator was visible to the author, if inputs change,
or if a corrected answer was used during generation; rescoring cannot reconstruct
that counterfactual run. Mixed grader/label versions do not form one leaderboard.

Reasoning levels are provider-specific settings, not equal units of computation.
Record exact controls and measured tokens, time, failures and cost when available.
A shared-harness comparison measures models with common capabilities; a comparison
of separately optimized systems measures the finished products. Label these
experiments separately. Do not attribute a joint model/prompt/tool change solely
to the model. If a model lacks an input modality or tool capability, report that
constraint rather than quietly giving it a different task.

## Runs, uncertainty and the dashboard

Replace the hard-coded two-model list with validated configuration records. Derive
rows and denominators from the selected configurations and tasks, so Luna at two
effort settings becomes two separate rows and an Anthropic adapter can be added
without changing the grader. Actual provider identifiers and supported settings
must be checked before scheduling; adding a dashboard row does not implement an
inference adapter.

The dashboard should first select a compatible dataset/label/grader version, then
show selected configurations side by side, with model, effort and harness version
visible. Add paired per-piece deltas and expandable attempts. Keep failed and
partial attempts, measurement coverage, source fidelity, edition agreement,
editorial adjudication, time and cost distinct. Differences in accepted coherent
form routes or reconstructions require musical adjudication, not a single
reference-string match. Show Pareto tradeoffs rather than assuming the highest
single accuracy score is always the best production choice.

Start with one attempt per piece for the initial Astra–Terra comparison. Repeat
promising configurations before making reliability claims; a provisional three
trials per piece is a small diagnostic, not a precise estimate. Retain every
trial and average per piece before averaging across pieces. Never report only
the best attempt unless the production process has a defined selection mechanism
and its full cost is included. More trials on the same six works do not create
more independent pieces.

Use the six reviewed pieces for development/regression. Freeze the selected
harness and reserve fresh, uninspected scores for testing generalization. This
guards against tuning tools and prompts to the six references. Do not promote
their consumed results into claims about all 100 sheets.

## Publishing

Keep a short README status and a compact result summary per evaluated release.
Each measured row should link to the report and identify data, labels, grader,
system, trial count, scope, failure coverage and limitations. Publish historical
excerpt experiments separately from the new complete-score benchmark. Preserve
old results when a new version supersedes them; describe corrections in a changelog.
Until the first complete-score comparison runs, the README must say so explicitly.
Do not publish source PDFs or review histories as a side effect of publishing metrics.

## Recommended order

1. Ship the offline baseline in #15 after main synchronization and fresh CI.
2. Add configuration records and independent version identities, with dynamic
   dashboard rows. Keep the first scheduled model set to Astra and Terra.
3. Finish source labels and enforced isolated execution; run the initial twelve
   attempts, publish their scoped results, then repeat finalists.
4. Add another provider or effort setting and one bounded tool ablation; expand
   according to observed errors and cost, not an exhaustive Cartesian product.
5. Freeze the selected system and evaluate fresh scores.

## Practice informing this design

There is no single mandatory version-numbering standard. This proposal follows
the separation of tasks, trials, graders and agent harnesses in
[Anthropic's agent evaluation guidance](https://www.anthropic.com/research/demystifying-evals-for-ai-agents),
and the explicit task versions and changelog practice documented by
[EleutherAI's evaluation harness](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/new_task_guide.md#versioning).
Their [reproducibility paper](https://arxiv.org/abs/2405.14782) explains why evaluation
setup and reporting details matter. These sources inform the design; they do not
validate this repository's musical labels or results.
