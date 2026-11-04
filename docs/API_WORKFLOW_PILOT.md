# Aviador through the API: October 29, 2026

The later authorized [Rumichaca full-score pilot](API_RUMICHACA_PILOT.md) is now
complete. It matched the adopted route but did not reproduce subscription note
quality; scaling remains paused. The sections below preserve the Aviador results
and their original experiment boundaries.

## Latest: reviewer v2 fixes the accidental

The authorized reviewer-only replay completed in **5m26s**, within the unchanged
20-minute limit. The same sealed API author draft was reused; only the general
review instructions changed. The reviewer changed the two F-naturals to F-sharps.
All seven bars now match the historical note/rest reference, with **100% strict
note F1**. Bar 7 is the only bar with changed scored events; bars 1–6 are unchanged.
No additional author, Rumichaca run, or larger batch was launched.

The decision record compares nearby sharp/natural handwriting and cites local
musical context separately. It retains moderate visual uncertainty for the
overlapped sign. The reader received neither Juan's adjudication nor a hint about
the target accidental. This is a successful consumed regression, not proof of
general improvement or a new end-to-end API pair.

| Independent reviewer only | Original v1 instructions | New v2 instructions |
| --- | ---: | ---: |
| Exact note/rest bars | 6/7 | 7/7 |
| Strict note F1 | 93.33% | 100% |
| Time | 3m45s | 5m26s |
| Input tokens | 208,434 | 366,450 |
| Cached input, included above | 185,536 (89.0%) | 338,670 (92.4%) |
| Output tokens | 10,563 | 12,565 |
| Conservative estimate | $1.286136 | $1.661420 |

Actual per-run billing remains unverified. More input was cached, but the new
review used more requests/output and had a higher total estimate. Cache reuse
does not guarantee a cheaper run. The earlier $1.819979 Platform readback below
covered the two earlier trials, before this reviewer replay; it excludes this run.

- Dashboard: `http://127.0.0.1:8777/reproduction/astra_api_review_context_20261029_v3/index.html`.
- Evidence: `out/eval/reproduction/astra_api_review_context_20261029_v3/`.
- [Portable reviewer-only metrics](eval/astra_api_review_context_20261029_v3.json).
- The first launcher startup was blocked locally before any API request; its
  zero-cost failure and the permission amendment are preserved. One paid model
  review ran with the original fixed 25,000-token response allowance, no dollar
  truncation and no automatic paid retry. The source process remained sandboxed.
- An elapsed-time HTML formatting error was fixed offline; the first report and
  implementation were archived, and identical scoring was verified after the fix.
- All 13 live human review files and the historical model-stage seals remain
  unchanged. The new dashboard was visually checked in the browser.

The subsequently authorized complete-score test using v2 is recorded in the
[Rumichaca report](API_RUMICHACA_PILOT.md). No batch expansion has been launched.

## Human confirmation and review instruction v2

Juan inspected the disputed source crop and explicitly confirmed a sharp before
the third note of system 8, measure 7 (global measure 59). The sixth note inherits
that accidental. The subscription review and existing F-sharp reference agree
with his reading. This is a confirmed source reading, not a reconstructed note.
The [structured adjudication](eval/aviador_s8_accidental_adjudication_20261029.json)
records his confirmation and pins the source and results. The original API result
remains 6/7; no reference, saved transcription, or historical report was changed.

[Review instruction v2](prompts/assisted_review_v2.txt) requires comparison with
nearby accidental handwriting and separates visible evidence from musical-context
support. General theory and local harmony may help interpret ambiguous ink using
only the supplied PDF; an expected harmony must not replace clear notation.
The prompt contains no answer, key, or disputed location specific to this case.
Historical v1 instructions and both versions' hashes are preserved.

The proposed reviewer-only test was authorized and completed as recorded above.
Its original proposal, scoped task, execution contract and results are preserved
under `out/eval/reproduction/astra_api_review_context_20261029_v3/`.
See [versioning and experiment rules](prompts/README.md).

## Platform cost readback

After both runs, the user-requested Platform usage check with project filter
`score2abc` showed **$1.82**. Its daily cost CSV records **$1.819979 USD** for
October 29. The displayed 89 requests and 1,412,239 input tokens exactly match
the combined receipts from v1 and v2. This is the currently recorded project-day
usage cost, not a settled invoice or an exact per-experiment allocation.

The earlier $3.855968 and $3.372103 figures are conservative estimates, not
observed charges; their combined $7.228071 substantially exceeds this readback.
The cause of that difference has not been established. Keep the original
estimates as historical evidence, use the Platform figure for the observed
aggregate, and do not invent a per-run split. Private export and reconciliation:
`out/eval/billing/platform_cost_20261029/`.

## Earlier: completed pair with fixed output allowance

The user approved one fresh author/reader pair with no dollar-driven truncation,
the same generic tasks and source tools, and the original 30/20-minute deadlines.
Every response retained 25,000 output tokens. Both stages completed normally,
with no retry, partial-author salvage, deadline or output-limit interruption.

The result remains **6/7 exact bars and 93.33% strict note F1**, before and after
independent review. Both preserve the same two F4 readings instead of reference
F-sharp4 in the final bar. All note timing and rests match. The unchanged gate
correctly rejects advancement. This establishes a completed negative result for
this API attempt; it does not establish that the underlying model is worse through
the API generally. Removing dollar-driven truncation did not resolve these errors.

| Completed v2 run | Author | Independent reader | Total |
| --- | ---: | ---: | ---: |
| Time | 7m03s | 3m45s | 10m48s |
| Output tokens | 17,176 | 10,563 | 27,739 |
| Conservative USD estimate | $2.085967 | $1.286136 | **$3.372103** |

Exact per-run billing remains unverified; the aggregate readback is above.
The subscription predecessor used 20,279 output
tokens and reached 7/7 after review. One case with different client orchestration
does not establish a general speed or cost advantage.

The script prepared both independent sessions, sealed the outputs, graded them
and built the dashboard automatically. No additional coordination agents were
used. 130 focused tests, Ruff and Black passed; all 13 live review files remain
unchanged. No Rumichaca run or additional paid attempt was launched.

- Dashboard: `http://127.0.0.1:8777/reproduction/astra_api_fixed_output_20261029_v2/index.html`.
- Evidence: `out/eval/reproduction/astra_api_fixed_output_20261029_v2/`.
- Report SHA256: `97173ca3dfa9d88236113d130b8cf75577a93a45d4d738539a12b3b2c17eedac`.
- [Portable v2 metrics](eval/astra_api_fixed_output_20261029_v2.json).
- Follow-up: the reviewer-only v2 instruction test above resolves this error.
  Keep this original 6/7 pair result unchanged. Do not retry merely to obtain a
  passing score.

## Earlier interrupted pilot, preserved

API authentication, generation, adaptive source inspection and draft validation
work. The reusable [execution command](ASSISTED_EXECUTION.md) selects `--backend
api` or `--backend subscription` explicitly, with no automatic billing fallback.
This pilot did **not** reproduce the subscription's final 7/7 result and does not
advance to Rumichaca.

## Findings

| Stage | Execution | Exact note/rest bars | Strict note F1 | Time |
| --- | --- | ---: | ---: | ---: |
| Subscription author | Complete | 6/7 | 93.33% | 7m04s |
| Subscription independent review | Complete | 7/7 | 100% | 7m42s |
| API author checkpoint | Interrupted | 6/7 | 93.33% | 8m00s |
| API diagnostic independent review | Complete | 6/7 | 93.33% | 5m09s |

Both API candidates contain the same scored events: 28/30 matching pitches,
all matching note onsets and durations, and all three matching rests. In system
8, bar 7 (global bar 59), notes 3 and 6 are F4 instead of the frozen reference's
F-sharp4. The independent reviewer did not resolve those differences. Both
candidates contain seven valid 3/4 bars; structural validity is not musical truth.
Ties, slurs, chord symbols, form and engraving are outside this passage metric.

## Execution interruption and diagnostic review

The original combined budget was $5, allocated $3 to the author and $2 to the
reader. Before every API generation, the runner counts the input and reserves
enough money for an entirely uncached input plus bounded output. As context grew,
that reservation reduced the author's final response allowance to 1,424 tokens.
The response hit `max_output_tokens`; neither the 30-minute deadline nor the
entire $3 author allocation was exhausted. Two valid checkpoints survived, both
explicitly marked unfinished. The automatic pair command correctly stopped.

Before any reference scoring, an amendment authorized diagnostic review of the
last saved checkpoint. There was no author retry. An explicit preparation flag
validated its sealed source, model, checkpoint and output-limit evidence. The
reviewer received the original generic task and only this author's candidate.
$0.75 of the unused author allocation was reassigned: the author's conservative
estimate plus the $2.75 reviewer limit was $4.989626, still below $5. The reviewer
finished normally within its unchanged 20-minute allowance.

Both stages were sealed before opening the grading reference. The original gate
remains false: the author did not finish successfully and reviewed note F1 is
below the preregistered 95% threshold. A completed diagnostic review does not
convert the interrupted attempt into a clean workflow reproduction.

## Resources and cost

| Author plus reader | Subscription | API |
| --- | ---: | ---: |
| Stage time sum | 14m46s | 13m09s |
| Input tokens | 414,954 | 745,670 |
| Cached input, included above | 358,272 | 685,218 |
| Output tokens | 20,279 | 33,189 |
| Reasoning output, included above | 13,341 | 23,639 |
| Conservative API estimate | Not applicable | $3.855968 |
| Actual billed charge | Not measured | Not verified |

API author and reviewer estimates were $2.239626 and $1.616342, across 27 and
19 generation requests. All reported usage reconciled. The conservative estimate
uses the pinned official rates and the highest applicable uncached/cache-write
input rate; it is not an invoice or an account-balance readback. The API used more
tokens in this example. Its shorter stage-time sum includes an interrupted author
and diagnostic review, so it is not evidence of faster successful reproduction.

## What matched

Source PDF, author prompt, generic reviewer task, initial overview, source-tool
implementation, renderer, validator, model, effort and time allowances match the
subscription predecessor. Both use the separately sandboxed source process.
The API uses native Responses function calls rather than Codex's code-mode MCP
wrapper; instruction placement, request envelopes and budget caps differ. Full
reader prompts necessarily differ because each contains its own new candidate.
This is one consumed development case, not an unseen-score accuracy estimate or
a causal comparison of API versus subscription model quality.

## Resume and evidence

- Local dashboard: `out/eval/reproduction/astra_api_workflow_20261029_v1/index.html`.
- Local URL: `http://127.0.0.1:8777/reproduction/astra_api_workflow_20261029_v1/index.html`.
- Evidence root: `out/eval/reproduction/astra_api_workflow_20261029_v1/`.
- Graded report: `aviador_s8/evaluation/report.json`, SHA256
  `2370a5596ccb3b345b86962799adeba07feadd66d1edb13344fd4a629e0afd42`.
- `partial_checkpoint_review_amendment.json` preserves the protocol change.
- `comparison_manifest.json` pins both reports and the protocol-match audit.
- `setup/implementation_v1/` preserves the original entrypoint before the
  diagnostic-review flag was added. Private response receipts remain local.
- [Portable metrics](eval/astra_api_aviador_20261029_v1.json) contain no score or
  credential content. All 13 live review files remain unchanged.

143 focused tests, Ruff and Black pass. No new production dependency was added.
Code and findings are local on `codex/astra-baseline-reproduction`; no push or new
PR was made for this pilot. Existing Astra–Terra evidence and dashboard remain.

The initial proposal below was superseded by the approved fixed-output v2 run above.
Initial proposal: one fresh, isolated Aviador pair with enough reservation
headroom, using the same source tools and generic prompts. A $10 combined ceiling
would allocate $6/$4 with the current command; this is a proposed maximum, not
expected spend, and needs authorization before another paid run. Keep the present
failed attempt and diagnostic result separate. Do not scale to Rumichaca or change
the frozen reference to make this case pass.
