# Run the assisted workflow through API or subscription

`scripts/run_assisted_transcription.py` selects the execution backend explicitly.
Both paths use the original PDF, adaptive close-ups, immutable ABC/decision
checkpoints, deterministic feedback, and a separate reader of the author's draft.
There is no automatic fallback between billing methods and no reference feedback
between stages. The existing musical tools and graders are unchanged.

An opt-in [completion gate](COMPLETION_GATE.md) is now available through
`prepare`/`pair --completion-policy completion_gate_v1`. It requires structural
warnings to be corrected or explicitly supported as source exceptions before a
stage can complete. Omitting the flag preserves the current subscription and API
defaults. Offline replay catches the saved Rumichaca API warnings; it is not yet
evidence of improved model transcription.

The subsequent [fresh gated Rumichaca pilot](API_COMPLETION_GATE_PILOT.md) improved
note agreement from 80.21% to 88.14% in one run, with one remaining source exception
and a different performance route. It does not establish a causal quality gain
or authorize collection-wide API scaling.

## One command for an author and independent reader

For new reviews, compose the source scope and time allowance with
[assisted review v2](prompts/assisted_review_v2.txt), then pass that file as
`--review-prompt` (pair) or `--prompt` (reviewed-stage prepare). See the
[version and regression protocol](prompts/README.md). Historical tasks remain
unchanged; the runner does not silently insert the new instructions.

Use generic task files describing the requested source span. For the Aviador pilot:

```sh
uv run python scripts/run_assisted_transcription.py pair \
  --backend api \
  --pdf out/eval/references/reviewed_six_20261028_v1/inputs/jaime-llanos_12_aviador_pasillo_fulgencio-garcia/source.pdf \
  --prompt out/eval/reproduction/astra_api_workflow_20261029_v1/author_tasks/aviador_s8.txt \
  --review-prompt out/eval/reproduction/astra_api_workflow_20261029_v1/author_tasks/aviador_s8_review.txt \
  --output out/eval/reproduction/NEW_TRIAL/aviador_s8 \
  --model gpt-6-astra --author-seconds 1800 --review-seconds 1200 \
  --budget-usd 5
```

For subscription execution, change `--backend api` to `--backend subscription`
and omit `--budget-usd`. The wrapper reuses the newest prepared runtime whose
binary pins still match the installed app. If the app changed and no matching
runtime exists, it stops and requires a newly prepared runtime; it never rewrites
historical evidence. `--runtime-dir` selects a particular verified runtime.

For an explicitly authorized API run without a dollar ceiling, replace
`--budget-usd 5` with `--no-dollar-cap`. This keeps the full 25,000-token response
allowance instead of shrinking it with remaining budget. Time, source-tool and
context limits still apply; cost is recorded but not capped. Omitting both
spending flags is an error, as is supplying both. The chosen policy is sealed in
the input packet before any paid request. The approved October 29 v2 Aviador
trial uses this mode for exactly one fresh author/reader pair, with no retries.

Output directories are create-once. The pair command allocates 60% of the API
budget to the author and 40% to the reader. It stops if the author does not finish
successfully with a complete claimed draft; partial checkpoints remain available.
It does not grade music or automatically start another score. Use `prepare`,
`audit`, and `run` subcommands when separate stage control is needed.

For an explicitly chosen diagnostic review, API `prepare --stage reviewed` also
accepts `--review-incomplete-checkpoint`. This is limited to a sealed, valid saved
checkpoint from an otherwise matching API author whose last response stopped at
`max_output_tokens`. It rejects quarantined, mismatched or unverified outputs.
The flag records the original interruption and does not retry the author or turn
it into a completed run. The pair command never enables it automatically; any
budget reassignment and protocol change must be recorded before the review.

## Credentials and accounting

The API path loads `OPENAI_API_KEY` from the process environment, or the explicit
`--env-file` (default: this project's `.env`). It parses only that variable and
does not execute shell configuration. The secret is not copied into input packets
or model messages. The subscription path does not load the API key.

API pricing is currently pinned for `gpt-6-astra`, Standard service, short context,
using the October 29, 2026 [official rate table](https://developers.openai.com/api/docs/pricing).
Only the API path has a dollar budget; subscription token usage is not converted
to fictional dollar charges. The wrapper's dollar limit governs this local run,
not the account's balance or other applications.

Before each generation, the adapter obtains an exact input count from the
[token-count endpoint](https://developers.openai.com/api/docs/guides/token-counting),
reserves a conservative input charge and bounded output charge, and writes the
reservation before sending the request. It restricts context to the priced tier.
Reported usage releases unused reservation; uncertain outcomes retain the full
reservation and stop without retrying. Explicit cached tokens use the cached rate;
uncached input is conservatively reserved at the highest applicable cache-write
rate. The resulting estimate is an upper estimate under the pinned rate table,
not an invoice. Actual credit consumption needs billing reconciliation.

The reservation can constrain output before the estimated cost reaches the dollar
limit: it must allow for the next full input being uncached. A long conversation
therefore needs headroom beyond its likely cached cost. If a response stops at its
output limit, saved checkpoints are preserved and the run remains incomplete.

Response receipts and encrypted reasoning are kept privately in the local run.
Normal request evidence contains image hashes, not duplicate image data or keys.
Cached-input and reasoning-output tokens remain subsets of the corresponding
input/output totals. Failures and budget stops remain separate from music quality.

## What is and is not identical

### Cache accounting and future score batches

The current adapter uses the model's default caching behavior; the reviewer-v2
experiment changes no cache settings or request ordering. Measure
`usage.input_tokens_details.cached_tokens` as a subset of input tokens, not as
additional tokens. Cache reads can reduce input processing cost, but a fresh run
still generates and pays for output, and additional inspection steps can raise
total usage. Report cached input, output, time, conservative estimates and actual
billing separately; a high cache fraction is not proof of a cheaper whole run.

For future batches, a candidate harness improvement is to place stable shared
instructions and tool definitions before per-score scope, PDF images and draft
content. Different PDFs do not share their image prefixes merely because they
belong to one collection. Keep each score's independent context isolated. Prompt
changes, cache lifetime and request settings affect reuse; do not assume a
previous run's cache is available. Reordering the current scoped review prompt
would be another harness version and should be measured separately.

These considerations follow the [official prompt-caching guide](https://developers.openai.com/api/docs/guides/prompt-caching).

### Execution controls

PDF bytes, task templates, base instructions, source-tool implementation, renderer,
validator, source-only policy, reasoning effort and stage time allowances can be
matched. The API uses native Responses function calling; subscription execution
uses Codex's code-mode MCP wrapper. Instruction placement, request envelopes and
client behavior therefore differ. API dollar and per-response output caps are
additional limits. Both paths use the separately sandboxed source-tool process.

Each API reviewer starts with only its own sealed author's candidate and the
generic review prompt. It never receives the old subscription draft or graded
reference. Both stages must be sealed before invoking the unchanged evaluator.

The seven-bar API pilot is a consumed development reproduction. It does not prove
collection-wide performance, approve a musician's edition, or authorize Rumichaca
or the six-score batch. Its outcome belongs in a separate transport-comparison
dashboard; the earlier Astra–Terra comparison remains intact.

The user subsequently authorized one fresh Rumichaca API pair with the original
60/30-minute limits and independent review v2. That pair is complete; see
[its separate report](API_RUMICHACA_PILOT.md). It used the same tools, fixed
25,000-token response allowance, no dollar cap and no automatic paid retry.
The saved traces support offline diagnosis. The six-score batch remains unrun.
