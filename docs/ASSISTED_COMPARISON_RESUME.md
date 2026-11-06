# Checkpoint: assisted Astra–Terra comparison

Later follow-up: the bounded API Aviador pilot and its resume instructions are in
[API_WORKFLOW_PILOT.md](API_WORKFLOW_PILOT.md). That separate experiment does not
replace the subscription comparison below.

Completed October 29, 2026, after Juan resumed the paused experiment. No inference
is pending. Stop here and report findings before expanding to more scores.

## Decision and results

Keep Astra as the transcription baseline for now. Terra is faster but has lower
observed musical agreement. These are two consumed development cases, one scored
author/reader pair per model and target, not collection-wide accuracy estimates.

| After independent review | Astra | Terra |
| --- | ---: | ---: |
| Aviador exact note/rest bars | 7/7 | 2/7 |
| Rumichaca written bars | 45/45 | 45/45 |
| Rumichaca performing-edition note F1 | 93.54% | 63.13% |
| Rumichaca source note F1, nine audited bars only | 97.22% | 58.23% |
| Rumichaca author + reader time | 57m 43s | 6m 11s |
| Rumichaca author + reader output tokens | 76,382 | 19,116 |

Terra's Aviador review worsened accidental readings (4/7 to 2/7 exact bars).
Its fresh Rumichaca author ended normally after 134.81 seconds with 17 bars;
the independent reader completed all 45. Neither hit its deadline.

Terra's Rumichaca has 17 bars with identical note events. Its zero exact-edition
bars is forced by literal 6/8 versus 3/4 meter comparison; seven bars differ
only in meter. Note F1 is unaffected by the meter label. Actual note differences
remain in 28 bars, chord differences in 21, and the executed route stops at
88 performed bars rather than the edition's 132. Neither model matches the
complete adopted edition route. Complete coverage and meter validity alone
do not establish a correct transcription.

## Matching and disclosed deviations

- Aviador: all 10 compatibility checks match.
- Rumichaca: source, generic prompts, bounded source tools, xhigh effort,
  3600/1800-second allowances and numerical graders retained.
- The app updated CLI from `0.164.0-alpha.11.1` to `0.165.0-alpha.1.2` overnight.
  A new runtime was pinned and passed a synthetic canary; old evidence remains.
- A prereview addendum explicitly allowed a cleanly completed partial author
  draft to go to the independent reviewer (`--allow-partial-author`). No author
  retry, reference answers or coordinator musical hints were supplied.
- The audit records 17/20 matching dimensions. Runtime and partial-author review
  eligibility prevent a clean model-only causal conclusion for Rumichaca.
- The October 28 interrupted attempt remains unscored and unchanged: 123.72
  seconds, return code -9, broken pipe, no final usage. It is excluded from the
  scored pair totals and disclosed separately; missing usage is not zero.
- Runs used subscription authentication. No dollar cost is inferred. Cached
  input and reasoning output are subsets, not additional usage.

## Where to resume

User dashboard: `out/eval/comparison/index.html`, served locally at
`http://127.0.0.1:8777/comparison/index.html`. It separates initial/reviewed stages,
models, pieces, musical metrics and resources; it links the older tool-free
experiment separately. Restart with
`uv run python -m http.server 8777 --bind 127.0.0.1 --directory out/eval` if needed.

Evidence root: `out/eval/reproduction/assisted_astra_terra_20261028_v1/`.

- `terra/aviador_s8/`: completed Terra Aviador pair and evaluation.
- `terra/rumichaca_full/initial/`: interrupted October 28 attempt; preserve.
- `retries/rumichaca_20261029_v1/`: fresh sealed author/reader pair, report,
  addenda and independent interpretation audit. Do not rerun or overwrite.
- `compatibility_audit_20261029.json`: seals, isolation and matching audit.
- `comparison_manifest_20261029.json`: dashboard inputs and caveats.
- `setup/implementation_snapshots/`: exact historical runner implementations.
- `live_review_preservation.json`: all 13 live review files verified unchanged.
- `completion_validation_20261029.json`: final implementation checks.

Astra predecessor: `out/eval/reproduction/astra_workflow_20261028_v1/`.
Fresh runtime: `out/eval/runtime/codex_subscription_20261029_v1/`.
Portable public summary: `docs/eval/assisted_astra_terra_20261029_v1.json`.
Interpretation and method: `docs/ASSISTED_MODEL_COMPARISON.md`.

Next discussion: retain Astra for transcription; prioritize checks for complete
coverage and executable form before scaling. A future controlled model comparison
should use one runtime and the same partial-draft policy for both models. Any
harness change needs a new version and matched reruns, not silent rescoring of
these submissions. No new batch, model trial or musical approval is implied.

## Implementation and preservation

143 focused tests, Ruff and Black pass. Dashboard rendered and local evidence
links checked. Both models' submissions were sealed before grading; independent
audits verified report pins and distinguished actual differences from metric
limitations. No reviewed score or frozen reference was changed.

Branch: `codex/astra-baseline-reproduction`; prior local commit `892256d`.
Comparison code, tests, README and notes are saved locally, not pushed or in a
new PR. Preserve unrelated `.playwright-mcp/` artifacts. Local experiment backup:
`out/eval/checkpoints/assisted_astra_terra_20261029_v1.tar.gz`, with a companion
`.backup.json` containing member hashes and archive verification. The earlier
Astra backup remains separate and is pinned by that manifest.
