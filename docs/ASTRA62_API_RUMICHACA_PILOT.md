# Astra 6.2 API full-score reproduction: Rumichaca, November 18, 2026

One fresh `gpt-6.2-astra` author and an independent same-model reviewer run at
`xhigh` through the Responses API. The comparison uses the completed November 17
Astra 6.2 subscription pair. It follows the successful fresh seven-bar
[Aviador API reproduction](ASTRA62_API_AVIADOR_PILOT.md).

## Results

The API reviewer reaches **100% edition note F1**, with all 194 note events matching
in pitch, onset and duration. Both stages match all **80 audited source noteheads**
under V4, covering 18 of 45 source bars. All 45 written bars pass meter checks;
both completion gates are ready with no warnings or accepted source exceptions.
This is strong note agreement, not a claim that the entire transcription is perfect.

| After independent review | Astra 6.2 subscription | Astra 6.2 API |
| --- | ---: | ---: |
| Edition note F1 | 98.97% | **100%** |
| Audited source notes, V4 (18/45 bars) | 80/80 | **80/80** |
| Edition chord F1 | 86.36% | **88.64%** |
| Exact edition bars, including chords | 38/45 | **40/45** |
| Written / meter-valid bars | 45/45 | **45/45** |
| Performed bars versus adopted 132-bar route | 117; different | **117; same difference** |
| Author + reviewer time | 76m34s | **35m31s** |
| Output tokens | 57,178 | **57,810** |
| Conservative API estimate | Not applicable | **$6.1715600** |
| Actual per-run dollars | Unknown | Unverified |

The API pair took about 54% less stage time in this trial, with almost the same
output tokens (1.1% more). No response allowance was reduced; all 97 responses
retained 25,000 output tokens. There were no time/budget interruptions or retries.
The $60 ceiling supplied headroom; it was not the cost of the run.

| API stage | Author | Independent reviewer |
| --- | ---: | ---: |
| Edition note F1 | 99.48% | 100% |
| Edition chord F1 | 86.36% | 88.64% |
| Exact edition bars | 38/45 | 40/45 |
| Executed route length | 88 | 117 |
| Stage time | 21m03s | 14m28s |
| Output tokens | 32,169 | 25,641 |
| Conservative estimate | $3.5200292 | $2.6515308 |

Review corrected C-natural to C-sharp in bar 23 and `F#dim` to `Fdim7` in bar 42.
It made the author's narrated 117-bar plan executable through explicit part
ordering; the original ABC actually performed 88 bars. No note regression was
introduced. Notes in the previously discussed bars 2, 16 and 44 now agree with the
edition. Bar 16 remains a documented PDF-only reconstruction, not certain visible
ink merely because it matches the reference.

## Remaining form and chord decisions

The reviewed API and subscription routes are **identical**, not merely the same
length. They agree with the edition for the first 116 visits. On the final D.C.,
they then take first-ending bar 29 and stop. The edition instead takes bars 30–45.
Replacing those 16 bars with one explains the net 15-bar difference. Both models
record a form interpretation; executable playback alone cannot settle which
interpretation the project should adopt.

All five remaining written-bar mismatches are chord symbols or timing:

| Written bar | Adopted edition | Reviewed API |
| --- | --- | --- |
| 1 | F | No chord supplied |
| 6 | Dm, then C7 later in the bar | C7 from the start |
| 30–32 | Dm | F |

These are edition disagreements, not proven PDF-reading errors. The API reader
records visible F and chord-ditto signs at bars 30–32 and no source chord at bar 1.
The source-note masks do not adjudicate source chords. Preserve both readings;
resolve this small set of chord/form decisions offline rather than launching
another paid transcription to force agreement.

**Recommendation:** the API is now a credible alternative for future drafts.
Resolve the shared form and chord policy questions before broad scaling; retain
human review for performance editions. One successful consumed score does not
establish collection-wide accuracy.

The [updated dashboard](http://127.0.0.1:8777/reproduction/astra62_api_rumichaca_20261118_v1/index.html)
shows four separate stages and links to Aviador and the Astra/Sol comparison.
[Portable metrics](eval/astra62_api_rumichaca_20261118_v1.json) retain exact values.
All 87 protected review/history files remain unchanged. Tests: **1,472 passed,
3 skipped**, with Ruff and Black clean.

## Fixed workflow and scope

The PDF bytes, generic author task, v2 review instructions, adaptive source tools,
renderer and `completion_gate_v1` match the saved subscription baseline. Each
reader sees only its own sealed author draft and the supplied PDF. Reference
editions, source labels, past results and human adjudications are unavailable to
both model stages. Both transports request `all_turns` reasoning context.

Author/reviewer deadlines remain 60/30 minutes. The conservative spending ceiling
is $60, allocated $36/$24, with no automatic retry. This is headroom above the
older full-score pair's approximately $15 estimate, not an expected charge.
The API retains the existing 25,000-token response allowance, 272,000-token input
tier and 128 source-tool-call stage limit. Record any effective allowance reduction
or interruption separately from transcription accuracy.

The API uses native function calls and encrypted reasoning continuation;
subscription uses Codex code-mode MCP and its client instruction envelope.
Differences in one trial do not establish a causal transport effect. No larger
batch or additional Aviador inference is part of this experiment.

## Evaluation boundaries

Both stages are sealed before reference grading. Report separately:

- Full-edition agreement for pitch/onset/duration, chords and exact bars. The
  reviewed edition includes human editorial decisions and is not a uniquely
  determined source transcription at every damaged location.
- Source-note fidelity under the original V3 labels and supplemental V4 mask.
  V4 audits 18 of 45 bars and 80 noteheads; it does not certify all source notes,
  chords, rests, form or reconstructions.
- Written-bar coverage, meter warnings, completion-gate dispositions and executable
  playback route. An accepted source exception does not prove the reading correct.
- Both stage times, input/output tokens, cached and reasoning subsets, and
  conservative API estimates. Actual billing and subscription dollars are unknown.

The historical graders and labels remain unchanged. The transport wrapper adds a
Rumichaca target and derives reference counts from the already-normalized frozen
edition. This is consumed development evidence, not a new heldout accuracy claim.

## Local evidence

The create-once root is
`out/eval/reproduction/astra62_api_rumichaca_20261118_v1/`. It holds prospective
plans, copied task templates, source-only input packets, audits, draft checkpoints,
validation feedback, private response receipts, sanitized traces, resource ledgers
and both-stage evaluation reports. The source tools never receive the credential.
Live review/history files are hashed before execution and verified at delivery.
Coordinator/setup usage is excluded from reported model-pair usage.
