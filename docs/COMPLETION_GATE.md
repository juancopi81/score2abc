# Completion gate v1 — offline validation

Subsequent live evidence: the [fresh Rumichaca API pilot](API_COMPLETION_GATE_PILOT.md)
completed with one documented source exception. Note agreement improved over the
earlier API run, but form agreement regressed and the subscription note baseline
remains stronger. The offline results below remain unchanged.

The opt-in `completion_gate_v1` prevents a parsed ABC draft from being labeled
complete while structural issues remain unresolved. It does not transcribe,
insert rests, judge musical correctness, or start another model call.

The ordinary subscription and API defaults retain their original source tools,
schema, instructions and completion rule. New experiments select the gate with
`--completion-policy completion_gate_v1` on `prepare` or `pair` in
`scripts/run_assisted_transcription.py`. The selection and implementation hash
are sealed before execution. Run and audit consume that sealed selection; they
cannot choose a different policy. Original prepared packets and results stay
immutable; new experiments require new packets.

## What happens when the model saves

1. The same local parser checks notation, bar durations and executable form.
   Every save preserves the ABC, decisions and parser results, including partial
   or invalid work.
2. The opt-in wrapper returns `completion_gate`: exact issue IDs, the ABC hash,
   unresolved issues and readiness. Parse errors/warnings, an empty score,
   unresolved playback route and incomplete decision records block readiness.
3. The model can inspect the supplied PDF and parallel passages, correct the
   ABC, and save again within its existing turn and time allowance. The gate
   itself makes no API call and never pads notes/rests automatically.
4. A genuinely irregular source bar or unpaired pickup can instead receive a
   structured source exception. It must cite the current issue and ABC hash, a
   prior inspected crop, location, adopted choice, reason and uncertainty.
   Missing, stale, duplicate, unmatched or malformed exceptions block completion.
   Unknown meter, parse errors and unresolved form cannot be waived this way.
5. Finalization recomputes the same policy. A complete claim with unresolved
   issues becomes a partial stage; useful output remains available. No automatic
   retry occurs. An interrupted exception sidecar fails closed while retaining
   the saved draft. The pair command stops after a partial author.

This is a save-triggered validation step inside our tool harness, not a new Codex
`hooks.json` hook. A structured exception proves that the decision is recorded
and linked to inspected image bytes. Its musical justification remains a model
assertion for independent/human review. Accepted exceptions stay visible as
`ready_with_source_exceptions` and are never labeled human-approved.

## Offline result, October 29, 2026

The final report is
`out/eval/diagnostics/completion_gate_20261029_v2/index.html`; its JSON includes
every checkpoint decision, input seal hashes and snapshots of implementation files.
The earlier v1 diagnostic snapshot was generated during development; v2 is the
final replay of the implemented policy.

All **35 saved checkpoints across nine stages** were checked. Each ABC was also
reparsed with the current local renderer: its completion issues matched the saved
parser output. All nine stage seals were verified before and after replay.
No references, old drafts or historical metrics were edited; no new musical
corrections or exceptions were supplied to the saved drafts.

| Saved final draft | Old stage status | New structural gate |
| --- | --- | --- |
| Subscription Aviador, author + reader | Complete | Both ready, no issues |
| Subscription Rumichaca, author + reader | Complete | Both ready, no issues |
| API Aviador v2, author + reader | Complete | Both ready, no issues |
| API Aviador with review instructions v2 | Complete | Ready, no issues |
| API Rumichaca, author | Complete | Blocked: 47 meter issues |
| API Rumichaca, reader | Complete | Blocked: bars 2, 16 and 44 |

There were **zero live model calls**. Separate controls exercise actual local
sandboxed tools and rendering: an overfull bar is blocked, its correction passes,
a complementary pickup passes, and simultaneous voices are not summed as
successive durations. Tests cover stale/invalid resolutions, evidence integrity,
interrupted writes, transport final status and policy tampering.

## What this does not establish

- It catches the known completion-policy problem. It does not prove that a model
  will make the right repairs, reduce its token use, or improve transcription F1.
- A wrong pitch or inserted rest can fit the meter. Older Aviador API drafts
  with the accidental disagreement still pass this structural gate.
- A resolved playback route can still differ from the intended musical form.
  PDF coverage is not established merely because parsed bars are valid.
- These scores have already informed development. This is a consumed regression
  replay, not a fresh held-out accuracy measurement.

Next, a bounded live experiment can test whether the model acts on the feedback.
Keep source-only inputs, time limits, model and musical grading fixed, recording
the gate as the changed harness version. Regular author/reader pairs require the
same policy version and implementation hash. A reviewer-only experiment using
an older author would need a separately explicit protocol; it is not silently
enabled by this change. No live experiment has been started by this work.

To reproduce the offline report into a new directory:

```sh
.venv/bin/python scripts/replay_completion_gate.py \
  --output out/eval/diagnostics/NEW_COMPLETION_REPLAY
```

Source artifacts and output directories are create-once. Parser drift or changed
stage seals stop the replay instead of rewriting historical evidence.
