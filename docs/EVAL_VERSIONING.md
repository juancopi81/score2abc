# Evaluation configurations and versioning

The schema-2 harness implements configurable systems and independently pinned
generation and grading identities. PR #15's schema-1 reports remain readable as
legacy evidence; they cannot be mixed into a versioned comparison.

The initial configuration is [`configs/eval/astra_terra_v1.json`](../configs/eval/astra_terra_v1.json).
It schedules Astra and Terra at `xhigh`, with the same PDF inputs and a maximum of
two deterministic-feedback revisions. Adding another configuration does not
establish that its model is available or that a provider adapter is implemented.

```sh
uv run python scripts/evaluate_complete_scores.py config /tmp/my-eval.json
# Edit model, effort and display name in this copy, using distinct system ids.
uv run python scripts/evaluate_complete_scores.py prepare \
  out/eval/references/reviewed_six_20261028_v1 /tmp/new-eval-inputs \
  --config /tmp/my-eval.json
```

Preparation makes no model calls. See [the isolated runner](EVAL_RUNNER.md) for
subscription authentication, enforced input isolation, availability checks and
bounded execution. Each trial uses a new destination. Source PDFs and review
answers remain local.

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

The task packet pins PDF inputs, generic instructions, authoring code, validator,
renderer and runner runtime policy. Its generation identity contains no reviewed
answers or label data. The grader separately adds the frozen answer/label hashes
and grading implementation. Exact file hashes identify working changes; Git
commit provenance is recorded separately so a documentation-only commit does not
change an otherwise identical implementation identity.

Each system pins its exact model identifier, effort, provider, adapter, settings
and harness hash. Each run pins a piece, system, trial and retained output files.
Requested and provider-resolved model identities are distinct; an unavailable
resolved revision remains unknown. A `local` / `deterministic` configuration may
use `null` model and effort, without pretending to be a language model. Its
execution adapter is not supplied by that declaration.

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

Configuration records drive dashboard rows and denominators. Luna at two effort
settings can be two separate systems, and another provider's retained outputs can
use the same grading contract once its adapter is implemented. Actual provider
identifiers and supported settings must be checked before scheduling; adding a
dashboard row does not implement an inference adapter.

Each dashboard shows one compatible dataset/label/grader group, with expandable
version hashes, exact model/effort/harness settings, and per-piece evidence. Open
a different versioned report to change that group; incompatible results are never
pooled. Paired differences use only pieces measured for both configurations,
rather than subtracting averages with different coverage. Keep failed and
partial attempts, measurement coverage, source fidelity, edition agreement,
editorial adjudication, time and cost distinct. Differences in accepted coherent
form routes or reconstructions require musical adjudication, not a single
reference-string match. Show Pareto tradeoffs rather than assuming the highest
single accuracy score is always the best production choice.

The current registry permits one trial per piece and configuration; repeat trials
require separate immutable registries/reports. It does not select the best run.
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

## Delivery sequence

1. The offline baseline shipped in merged PR #15.
2. Configuration records and independent version identities drive dynamic
   dashboard rows. The first scheduled model set remains Astra and Terra.
3. Audit source labels and enforce isolated execution; run the initial twelve
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
