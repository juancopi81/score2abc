# Isolated subscription runner

The complete-score runner uses `codex exec` authenticated with the user's ChatGPT
subscription. It does not use an OpenAI API key or assign a fictional dollar cost
to subscription usage. Programmatic model requests still travel over the network;
`network_access: false` in a run record means the **author has no network tools**,
not that the authentication/inference transport is offline. Subscription limits apply.

## Isolation boundary

The v1 runner currently requires macOS and `sandbox-exec`. It creates a fresh temporary
working directory and fresh ephemeral CLI process for every draft and repair. It
copies only source-derived PNGs, generic instructions, a response schema and a pinned
model catalog into that directory. It does not pass the repository, labels, review
histories, prior conversations or a previous CLI session to the model.

`--ignore-user-config` preserves the existing login while excluding personal runtime
configuration. The environment excludes API keys, custom endpoints and inherited agent
session IDs. `forced_login_method=chatgpt` prevents switching to API-key billing.
The runner leaves `HOME` and `CODEX_HOME` unchanged and never reads credentials.

Feature switches alone are insufficient: current model metadata can re-enable tool
interfaces, and global `AGENTS.md` can load despite `project_doc_max_bytes=0`. Therefore:

- The frozen installed model catalog preserves model IDs, supported efforts and other
  metadata, but sets `tool_mode` and `multi_agent_version` to `none`, removes experimental
  tools, and disables the patch tool. These are explicit authoring-harness controls.
- An additional read sandbox denies the repository, Codex memory/session directories,
  global instruction files and archived guidance. The ordinary read-only sandbox and
  approval restrictions remain in force; no bypass option is used.
- Before **every** subscription call, the same CLI, model metadata, input images, prompt
  and controls are sent to a localhost mock provider with authentication disabled. The
  captured request must expose no tools, including incremental tool declarations. Its
  text must exactly match supplied instructions, image wrappers, and the inspected
  generic permission notice; PNG bytes and order must match exactly.
- The actual invocation changes only provider/transport authentication settings. Raw CLI
  events are retained and checked against a small allowlist. Any tool or unknown event
  quarantines the entire run; an earlier draft is not published from a contaminated run.

This is a local capability/prompt audit plus enforcement by fixed CLI controls and the
read sandbox. It is not interception of authenticated production traffic, and does not
prove anything about the provider's training data or undisclosed model revision.
Requested model IDs are retained; resolved model revisions remain unknown unless exposed.

## Inputs, repair policy and evidence

Poppler renders each PDF at 160 DPI. The model receives each page overview followed by
three overlapping vertical thirds. Overlap prevents staff rows at crop boundaries being
lost; the instructions explain that duplicate pixels are not repeated musical passages.
The runner supports at most eight PDF pages and uses no notehead guesses or reference hints.

Each initial draft may receive at most two deterministic validation/repair passes. Each
pass is a fresh context containing the same images, generic instructions, its previous
draft and truth-free parse/meter/route feedback. Every response includes complete ABC and
structured editorial decisions with uncertainty and alternative readings.

A call has a 900-second wall-clock deadline; timeout terminates its process group. A failed
repair retains the last usable draft as partial. Rendering, transport, response and
validation failures remain visible in a failed/partial registry rather than disappearing.
A parseable/meter-valid result is only structural completion, not musical approval.

The run directory retains images, prompts, per-call local audit hashes, raw events,
responses, validation feedback, decisions, timestamps, token usage when supplied, and
hash-pinned run/measurement records. System elapsed time includes rendering, audits,
generation and validation. Human review time and dollar cost stay null without evidence.
Frozen files are create-once. Code, renderer, CLI binary, catalog or input drift requires a
new preparation; the runner refuses a changed prepared contract before inference.

## Commands

Prepare the runtime once (currently the working installation is the CLI bundled with
ChatGPT.app, version `0.164.0-alpha.11.1`; the older standalone `0.147.0` cannot run Astra):

```bash
uv run python scripts/run_complete_score_eval.py bootstrap
```

The default prepared runtime is `out/eval/runtime/codex_subscription_v2/`. The earlier
runtime and failed preflight canaries remain intact as setup evidence, not benchmark runs.
To support another installation, pass `--codex /absolute/path/to/codex` when bootstrapping
a new runtime directory, and pass that runtime explicitly to subsequent commands. The
packet must pin the same runtime fingerprint. Do not silently substitute another model.

Before an experiment, run the synthetic image/repair check for each selected system:

```bash
uv run python scripts/run_complete_score_eval.py canary <new-output-directory> \
  --model gpt-6-astra --effort xhigh
```

After preparing a schema2 source-only packet with `evaluate_complete_scores.py prepare`,
run one task by its exact `task_id`:

```bash
uv run python scripts/run_complete_score_eval.py run <packet> <new-run-directory> \
  --task-id <task-id>
```

Collect compatible per-task registries without rerunning inference:

```bash
uv run python scripts/run_complete_score_eval.py collect <new-registry.json> \
  <run-directory-1/registry.json> <run-directory-2/registry.json>
```

The six development references are never arguments to the author runner. Only the
separate offline grader receives the answer and label roots. A new provider adapter must
meet the same input/output and provenance contracts; configuring a dashboard row alone
does not provide inference support.
