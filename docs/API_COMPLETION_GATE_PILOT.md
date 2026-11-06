# Rumichaca API completion gate pilot — October 29, 2026

The fresh author and independent reviewer completed within the unchanged 60/30-minute
limits. **Keep API scaling paused.** This run improved note and chord agreement over
the earlier API run, but remains below the subscription note baseline and loses the
previous API run's match to the adopted performance route.

The PDF, author/review task text, review-v2 instructions, model (`gpt-6-astra`,
`xhigh`), output allowance, and numerical graders were held fixed against the
earlier API run. The experimental change was `completion_gate_v1`, including its
save-tool schema, feedback, instructions and completion enforcement. Neither
model received references, previous candidates or the human adjudications.

| After independent review | Subscription | Earlier API | Gated API |
| --- | ---: | ---: | ---: |
| Written bars | 45 | 45 | 45 |
| Edition note F1: pitch + onset + duration | 93.54% | 80.21% | **88.14%** |
| Audited-source note F1: nine bars only | 97.22% | 69.44% | **83.33%** |
| Chord agreement with adopted edition | 70.27% | 43.24% | **74.42%** |
| Exact edition bars, including chords | 23/45 | 10/45 | **21/45** |
| Meter-valid bars | 45/45 | 42/45 | **44/45** |
| Executed route matches adopted edition | No: 117 bars | Yes: 132 bars | **No: 116 bars** |

This is one fresh sample on a consumed development score. It does not isolate the
gate's causal effect or establish collection-wide accuracy. The nine source bars
are pre-existing model-audited labels, not whole-score human source truth. Edition
agreement includes editorial decisions; the subscription reviewer used v1, while
both API reviewers used v2. Historical reports and labels remain unchanged.

## What the gate achieved

Every final warning now has a structured disposition tied to the exact ABC, issue
and inspected crop. The author ended with three exceptions; the reviewer ended
with one. Both execution stages are `complete`, with gate status
`ready_with_source_exceptions`. The unchanged musical validator still reports
`partial`: the remaining meter mismatch has not been musically resolved.

The final exception is bar 8. The candidate writes `^G A6` with `L:1/8`: a G-sharp
eighth plus A dotted half, totaling **3.5 beats**. The frozen edition and earlier
API candidate have `z ^G A4`: an eighth rest, G-sharp eighth and A half, totaling
three beats. The arithmetic is correct; this is not a demonstrated ABC/parser
limitation. The inspected ink does not securely settle the opening curl/rest or
displaced dot. The exception records the model's uncertainty; it does not certify
that the manuscript intentionally has an irregular bar.

The author also wrote 29 D.C. return bars explicitly, producing 74 written bars:
45 source bars plus copies of bars 1–29. All 29 copies match their originals in
normalized notes, rests, chords and durations. This was deliberate form expansion,
not the earlier API author's bar-segmentation failure. Its written-note metric
penalizes the duplication and should be interpreted accordingly. The reviewer
restored 45 written bars but omitted one 16-bar pass through bars 30–45 compared
with the adopted edition's route.

## The three previously discussed bars

These are an offline comparison after both stages were sealed. They did not
provide feedback to either model, and they do not expand the frozen source labels.

- **Bar 2:** the spurious leading rest is gone. `^F A2 G e2` now matches the
  human-confirmed notes and rhythm. A7 still starts at beat 1.5 rather than beat
  zero in the edition.
- **Bar 16:** the new reading is `A,2 B,4` (A3 quarter, B-flat3 half), versus
  the edition's `D2 ^C4` (D4 quarter, C-sharp4 half). The rhythm is complete, but
  it remains a different uncertain pitch reconstruction. The human leading D is
  an externally supported editorial choice, not uniquely established PDF truth.
- **Bar 44:** `C2 CDGB & x2 C4` now fills the meter but retains a held-C overlay.
  It still differs from the human-confirmed single melodic line `D _D CEGB`.
  Passing the beat count did not settle the source reading.

The detailed diagnosis and 21 evidence hashes are saved as
`out/eval/reproduction/astra_api_rumichaca_gate_20261029_v1/known_bar_followup.json`.

## Resources and preservation

| API stage | Time | Input tokens | Cached input | Output tokens | Conservative estimate |
| --- | ---: | ---: | ---: | ---: | ---: |
| Author | 23m23s | 1,958,333 | 1,851,546 | 75,861 | $8.314271 |
| Independent review | 15m12s | 1,835,574 | 1,724,555 | 44,342 | $6.717130 |
| Total | **38m35s** | **3,793,907** | **3,576,101** | **120,203** | **$15.031401** |

Actual billing is unverified. The previous API pair's conservative estimate was
$11.989297, so this run cost about 25% more by that measure. Output tokens increased
from 100,535 to 120,203. About 94.3% of input tokens were cached; cache use does not
eliminate output cost or make the whole run cheaper automatically.

Saved locally: 99 generation requests with response receipts, 84 source crops,
11 draft checkpoints, structured exceptions, decisions, validation, timings and
usage. All 13 live human review files remain unchanged. Private response contents
are not embedded in the dashboard. There were no paid retries and no additional
Aviador or collection-wide runs.

An offline report adapter initially expected a flat preservation manifest rather
than its new wrapper. This failed before grading or creating the submission seal;
the adapter was corrected and covered with regression tests. No inference,
preregistration, candidate or numerical-grader files changed. The corrected
adapter and related report tests passed 48 tests, Ruff and Black. The dashboard
was opened and checked in the browser.

## Recommendation and evidence

Keep the successful subscription workflow unchanged. The new gate improves
accountability and exposes uncertainty, but cannot prove source correctness from
an exception record. The next work should address the remaining exception and
form interpretation offline before another paid trial. A fresh Aviador run is
not needed for this decision: its saved drafts pass the offline gate controls,
and its seven bars do not test Rumichaca's full-score form or these damaged readings.

- [Comparison dashboard](http://127.0.0.1:8777/reproduction/astra_api_rumichaca_gate_20261029_v1/index.html).
- [Portable report](eval/astra_api_rumichaca_gate_20261029_v1.json).
- [Completion-gate contract and offline controls](COMPLETION_GATE.md).
- [Earlier API pilot](API_RUMICHACA_PILOT.md).
- [Human adjudications — public copy](eval/rumichaca_human_adjudication_20261029_public.json).

The new adapter is `scripts/evaluate_completion_gate_pilot.py`. Its evaluation
and dashboard are create-once; do not rerun or overwrite the completed report.
