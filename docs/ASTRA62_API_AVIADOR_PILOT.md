# Astra 6.2 API reproduction: Aviador, November 18, 2026

This experiment runs one fresh `gpt-6.2-astra` author and one independent reader
at `xhigh` through the Responses API. It compares Aviador system 8 (seven bars)
with the saved November 17 Astra 6.2 subscription pair. Rumichaca and the larger
batch are outside this run.

The subsequently authorized [Rumichaca API test](ASTRA62_API_RUMICHACA_PILOT.md)
is now complete and recorded separately; this passage's evidence remains unchanged.

## Results

The fresh API author and independent reviewer both matched **7/7 note/rest bars**,
including the previously disputed sharp. The reviewer retained the author's ABC
byte for byte. Both stages passed the completion gate with no unresolved warnings
or accepted source exceptions. This reproduces the saved Astra 6.2 subscription
result on this passage.

| Aviador, author plus independent review | Subscription, Nov 17 | API, Nov 18 |
| --- | ---: | ---: |
| Author exact note/rest bars | 7/7 | 7/7 |
| Reviewed exact note/rest bars | 7/7 | 7/7 |
| Reviewed strict note F1 / rest F1 | 100% / 100% | 100% / 100% |
| Total stage time | 19m39s | 16m01s |
| Output tokens | 16,568 | 21,942 |
| Input tokens | 756,338 | 1,016,709 |
| Cached input, included above | 661,888 | 940,286 |
| Reasoning output, included above | 10,684 | 14,738 |
| Conservative API estimate | Not applicable | $2.7822544 |
| Actual per-run dollar cost | Unknown | Unverified |

The API author took 7m00s, used 9,270 output tokens and had a conservative estimate
of $1.2480048. Review took 9m01s, used 12,672 output tokens and had a conservative
estimate of $1.5342496. All 59 responses retained the full 25,000-token allowance;
none was interrupted, retried or constrained by the budget. Approximately 92.5%
of API input was reported cached. The API was faster in this one trial but used
about 32% more output tokens; this is not evidence of subscription-token savings.

The [dashboard](http://127.0.0.1:8777/reproduction/astra62_api_aviador_20261118_v1/index.html)
shows both stages and links to the previous model comparisons.
[Portable metrics](eval/astra62_api_aviador_20261118_v1.json) retain the exact
counts, report hashes and limits. All 87 protected review/history files remain
unchanged. The implementation passes 1,446 tests (3 skipped), Ruff and Black.

This is a successful fresh API reproduction, not just a repair of an earlier
author draft. It supports one separately authorized Rumichaca test next, with
full-score coverage and executable form checked before a larger batch. The older
Astra 6 API experiments used different review/context controls, so their lower
results cannot be attributed solely to the model version.

## Fixed conditions

The source PDF, generic author prompt, v2 review template, adaptive source tools,
renderer, completion gate and historical note/rest grader match the subscription
baseline. Both transports request `all_turns` reasoning context. Author and reader
have 30 and 20 minutes respectively. Each reader receives only its own author's
sealed ABC and decisions. Neither receives reference answers or prior human
adjudications. The API retains encrypted reasoning history between calls.

The API ceiling is $25 ($15 author, $10 reader), with no automatic retry. Each
response has 25,000 output tokens available unless the budget reserve constrains
it; the result must report whether that happened. Input context stays within the
272,000-token priced tier. Original and reviewed outputs are sealed before truth
is opened. The exact-model access check performed no generation.

API native function calls and subscription Codex code-mode MCP wrappers differ.
Instruction envelopes, client behavior and API spending/response caps remain
transport differences. A single pair cannot establish a causal transport effect.
The passage is consumed development evidence; ties, chord labels, articulation,
form and engraving are outside its note/rest metric.

## Evidence and accounting

The local create-once experiment is
`out/eval/reproduction/astra62_api_aviador_20261118_v1/`. It retains prospective
plans, pinned input packets, audits, immutable drafts, validation feedback,
private API receipts, sanitized request traces, usage ledgers and both-stage
reports. Credentials never enter the source tools or model context.

Reported cached input is a subset of input; reasoning output is a subset of
output. The adapter conservatively prices uncached input at the maximum verified
cache-write rate. Its USD estimate is not an invoice; actual per-run billing and
subscription dollars remain unknown. Coordinator and setup usage are excluded
from the model-pair totals.

The API adapter preserves the legacy Astra 6 profile. The new profile seals exact
model pricing and reasoning settings and rejects unsupported or altered profiles
before requests. The evaluator adds a single-target transport plan and strict API
stage linkage; historical numerical graders and saved outputs remain unchanged.
