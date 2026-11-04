# Rumichaca API pilot — October 29, 2026

The fresh Astra API author and independent v2 reviewer both completed within the
original 60/30-minute limits. **Keep scaling paused:** review improved the draft
substantially and matched the adopted performance route, but musical agreement
remains below the saved subscription baseline.

| After independent review | Subscription, review v1 | API, review v2 |
| --- | ---: | ---: |
| Written bars | 45/45 | 45/45 |
| Edition note F1 (pitch, onset, duration) | 93.54% | 80.21% |
| Audited-source note F1 (nine bars only) | 97.22% | 69.44% |
| Chord agreement with edition | 70.27% | 43.24% |
| Exact edition bars, including chords | 23/45 | 10/45 |
| Meter-valid bars | 45/45 | 42/45 |
| Executed route matches adopted edition | No, 117 performed bars | Yes, 132 performed bars |

Source labels cover nine pre-existing model-audited bars, not the whole PDF.
The edition includes human editorial choices; chord or route differences are
not automatically source errors. This consumed development case is not an
unseen-score estimate. Review instructions and client orchestration differ, so
the result does not establish that API transport itself causes lower quality.

## What the traces show

The author interpreted long internal strokes as bar divisions, producing 67
divisions and 47 meter warnings. Its decisions explicitly retained these
boundaries. The independent reader reconsidered the beamed groups, regrouped
them into 45 bars, and raised edition note F1 from 51.28% to 80.21%.

Three meter warnings remain in the reviewed ABC:

- Bar 2: seven eighth-note units instead of six; the reader records uncertainty
  about a rest-like glyph and an unflagged note.
- Bar 16: four eighth-note units; the reader retained an undotted half note
  rather than infer a missing dot.
- Bar 44: eight eighth-note units, with an explicitly uncertain low-C voice
  reconstruction.

The reviewer also records accidental, duration, chord and note-versus-rest
uncertainties. It supplies a finite performance route that matches the edition.
These are useful diagnostic records, not musical approval.

Execution status is **complete**; evaluator status is **partial** because musical
validation still reports issues. This distinction is a concrete harness weakness:
the current finish contract accepts a parseable draft with unresolved meter
warnings. The warnings were already visible to the models; the missing part is
an explicit completion policy for resolving or classifying them. A future policy
should require a supported reading or a clearly labeled best-effort reconstruction,
and preserve genuine irregular bars as documented exceptions. It must not silently
pad rests or force the music to match the reference.

## Resources and trace preservation

| API stage | Time | Input tokens | Cached input | Output tokens | Conservative estimate |
| --- | ---: | ---: | ---: | ---: | ---: |
| Author | 26m12s | 2,122,472 | 2,022,454 (95.3%) | 67,427 | $7.894254 |
| Independent review | 12m48s | 1,016,443 | 957,143 (94.2%) | 33,108 | $4.095043 |
| Total | 39m00s | 3,138,915 | 2,979,597 (94.9%) | 100,535 | **$11.989297** |

Actual billing is unverified. These conservative estimates are not an invoice.
The saved subscription pair took 57m43s and used 76,382 output tokens; this API
pair used more output despite being faster. High cache use did not eliminate
generation cost. No per-run subscription dollar cost is known.

Saved locally: 87 generation requests and response receipts, 77 source crops,
seven draft checkpoints with validation, decision records, tool events, prompt
versions, source hashes, timing and usage. Private response receipts stay local.
Neither model received reference answers or previous candidates; the reviewer
received only its own new author's sealed draft. Both stages were sealed before
grading. The PDF, source tools and renderer match the subscription packet.

A local task-composition assertion failed before any API call; that zero-request
preparation failure is preserved. One paid author and one paid reader ran with
the fixed 25,000-token response allowance and no automatic retries. A dashboard
escaping defect was corrected offline; reports and scores were not rescored.
All 13 live human review files remain unchanged. 46 focused tests, Ruff and Black
passed, and the dashboard was checked in the browser.

## Evidence and next decision

- Dashboard: `http://127.0.0.1:8777/reproduction/astra_api_rumichaca_20261029_v1/index.html`.
- Local evidence: `out/eval/reproduction/astra_api_rumichaca_20261029_v1/`.
- [Portable metrics](eval/astra_api_rumichaca_20261029_v1.json).
- Offline grader: `scripts/evaluate_api_full_score_pilot.py`; presentation:
  `scripts/build_api_full_score_dashboard.py`. Numerical graders are unchanged.

Recommended next: inspect the saved bar-boundary, accidental and duration
disagreements offline, then propose one narrow, versioned completion-policy
improvement. Preserve this negative result and the reviewed score. Do not retry
or launch the six-score batch automatically. No human review or new paid run is
required to preserve and inspect these traces.
