# GPT-6.2 Astra and Sol subscription comparison

The November 17, 2026 experiment compares `gpt-6.2-astra` and `gpt-6.2-sol`
at explicit `xhigh` effort on Aviador system 8 and the complete Rumichaca score.
Each model gets one fresh author and one fresh independent reader of its own
draft. Both targets are preregistered for both models regardless of Aviador's
accuracy. No API generation, automatic paid fallback, or live review edits are
part of this experiment.

Open [the local comparison dashboard](http://127.0.0.1:8777/reproduction/astra_sol_62_subscription_20261117_v1/index.html)
when the local report server is running, or the experiment's `index.html` file.
[Portable metrics](eval/astra_sol_62_subscription_20261117_v1.json) retain both
stages and their report hashes without distributing source PDFs or ABC drafts.

## Fixed workflow

- Supplied PDF only; no recordings, reference editions, prior answers or human
  corrections reach either model.
- Existing adaptive source inspection and draft-validation tools, with
  `completion_gate_v1` and `assisted_review_v2`.
- Aviador author/reviewer limits: 30/20 minutes. Rumichaca: 60/30 minutes.
- One attempt per stage; no reference-guided retries or silent partial-author
  salvage. Initial and reviewed outputs remain separate and sealed before grading.
- Subscription authentication is enforced. Token usage is measured; subscription
  dollars and human correction time remain unknown.

The CLI is `0.169.0-alpha.11.1`. A versioned native tool-interface contract
preserves the historical runtime checks while accepting the inspected new wrapper
and deferred source-tool declarations. Local mock audits and a small synthetic
source-inspection/save canary run before either real task.

The installed catalog advertises different native context capacities: 272,000
tokens for Astra and 500,000 for Sol. Neither run overrides that capacity or its
default compaction threshold. Source tools, tasks, explicit effort and stage
deadlines match; a claim that every resource limit is identical would be false.
Cumulative input usage is not a measurement of peak context length.

## Evaluation boundaries

Aviador retains the historical seven-bar note/rest scorer. Ties, chord labels,
articulation and form are outside that passage's accuracy metric. Rumichaca uses
the existing full-edition comparison and both frozen source-note masks: v3
(nine bars) and supplemental v4 (18 bars, 80 noteheads). The v4 mask does not
certify source chords, rests, form or all 45 bars.

Edition agreement, source-note fidelity, meter validity, coverage and executable
form remain separate. A structurally complete score is not necessarily musically
correct. An alternative coherent form or reconstruction is not automatically a
proven reading error. Both author and reviewer metrics must be displayed so that
review regressions cannot be hidden.

Earlier Astra/Terra results remain historical context. They use older clients,
review instructions and completion policy, so differences from those results
cannot be attributed only to the model upgrade. This is a consumed development
comparison, not evidence of generalization to all 100 manuscripts.

Local evidence is create-once under
`out/eval/reproduction/astra_sol_62_subscription_20261117_v1/`, including task
copies, plans, runtime pins, source-only packets, audits, traces and outputs.
The initial preservation inventory covers 87 files across 19 saved reviews.

## Results

| Aviador system 8 | Astra 6.2 | Sol 6.2 |
| --- | ---: | ---: |
| Exact note/rest bars, author → reviewed | 7/7 → 7/7 | 7/7 → 7/7 |
| Strict note F1, author → reviewed | 1.0 → 1.0 | 1.0 → 1.0 |
| Author + reader elapsed time | 19m39s | 11m22s |
| Author + reader output tokens | 16,568 | 30,474 |
| Author + reader input tokens | 756,338 | 752,452 |
| Cached input, included above | 661,888 | 627,584 |

Both reproduce the seven-bar note/rest target, including the difficult accidental,
without a review regression. Sol is faster in this trial, while Astra uses fewer
output tokens. The metric does not certify the passage's ties, chords or form.

Sol's Rumichaca author and independent reader are complete. Their combined time
is 30m08s, with 71,645 output tokens. V4 audited-source note matches fall from
75/80 to 74/80 after review; edition note F1 falls from 85.79% to 83.76%, and
edition chord F1 from 40.54% to 29.73%. Source-note matching includes pitch,
onset, duration and tie endpoints, not just pitch.

Review improves Sol's executable form from 89 to 134 played bars, versus the
adopted edition's 132. A post-hoc alignment diagnostic shows the reviewed route
matches the edition after excluding two visits to Sol's unmatched reconstructed
opening pickup. The official form-exact metric remains false. Both drafts have
46 written bars and one declared source-meter exception; that exception is not
proof that the reconstructed pickup is correct. Four changed pitches account
for the note regression, and shifted chord onsets account for the chord decline.
Only one of those pitch changes lies within the audited V4 source subset; chord
placement is not source-certified by that audit. The sealed diagnostic is
`sol_rumichaca_postmortem.json` in the experiment directory.

| Rumichaca | Astra 6.2 | Sol 6.2 |
| --- | ---: | ---: |
| Audited source notes matched, author → reviewed | 80/80 → 80/80 | 75/80 → 74/80 |
| Edition note F1, author → reviewed | 98.45% → 98.97% | 85.79% → 83.76% |
| Edition chord F1, author → reviewed | 84.09% → 86.36% | 40.54% → 29.73% |
| Exact edition bars after review | 38/45 | 3/46 alignment columns |
| Written bars after review | 45 | 46, including reconstructed pickup |
| Played bars, author → reviewed (edition: 132) | 88 → 117 | 89 → 134 |
| Author + reader elapsed time | 76m34s | 30m08s |
| Author + reader output tokens | 57,178 | 71,645 |
| Author + reader input tokens | 3,559,539 | 3,342,011 |
| Cached input, included above | 3,396,224 | 3,078,400 |
| Unresolved completion warnings after review | 0 | 0, with one declared source exception |

Astra is the quality choice in this pilot: its independent review improves note
and chord agreement without losing any V4 source-note matches. Sol completes
Rumichaca about 2.54 times as quickly, but Astra uses 20.2% fewer output tokens.
Astra has slightly more total input but less uncached input. None of those
counts establishes exact subscription-quota consumption or monetary savings.
The two-case output totals are 73,746 for Astra and 102,119 for Sol; timing totals
are sums of model stages, not this coordinator's elapsed time.

Both models finish every execution stage normally. Astra's reviewed score passes
the completion gate without exceptions. Sol's gate accepts one documented pickup
exception; the unchanged historical grader still labels its score `partial`
because of that meter exception, despite alignment covering all 45 reference
bars. This is not a timeout or an unfinished model turn.

Neither reviewed playback route exactly matches the adopted edition. Keep form
review in the human workflow; an executable route is not proof of the intended
performance. Do not replace existing human scores with these experimental drafts.

Astra's reviewer fixes bar 23's C5 to C-sharp5 and changes the chord labels in
bars 6–7 from G7 to C7; bar 6's placement still differs from the edition. It also
makes the described D.C. executable. Its final route agrees with the edition for
116 visits, then takes first-ending bar 29 and stops where the edition continues
through bars 30–45. The reviewer records this endpoint as uncertain editorial
judgment, explicitly without a securely visible earlier Fine, and supplies the
second-ending continuation as an alternative. The two remaining note-event
disagreements are outside V4: C4 versus C5 at bar 30, and a two-beat versus
eighth-note C4 at bar 44. These are edition disagreements; this postmortem did
not perform new visual adjudication. `astra_rumichaca_postmortem.json` preserves
the diagnostic and input hashes.

For future drafts, prefer Astra 6.2 at `xhigh` for quality. Sol is a latency option,
not an established token-saving substitute. A possible Sol-author/Astra-reader
combination remains untested. These two consumed cases do not justify scaling
unreviewed transcription across the collection or a claim about all new models.

The artifact-derived compatibility audit passes **20/20** checks across the
two cases and stages, rechecking 2,098 evidence pins. It verifies shared task
inputs, prompts, tools, configured limits and graders; native context capacity
remains the disclosed model-specific difference above. Historical Astra results
use different generation conditions, so the stronger observed Rumichaca score
does not isolate the effect of the model version alone.

The full code suite passes: 1,407 tests, three skipped; Ruff and Black pass.
Both models passed local interface audits and subscription-only synthetic
inspection/save canaries. Setup used 242 Astra and 259 Sol output tokens,
recorded separately from the scored pairs. Two earlier local audit failures
made no inference requests. Independent code reviews found no blockers.
