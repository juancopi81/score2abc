# Complete-score grader and comparison dashboard

This slice makes existing and future complete-score outputs measurable without
calling a model. It does not yet execute isolated Astra or Terra agents. The six
reviewed editions are consumed development references, not held-out source truth.

## Delivered evidence

- Six verified PDFs and reviewed ABC editions remain in the immutable reference freeze.
- The separate V2 inventory records 30 source/editorial decisions and their evidence.
  Two Aviador descriptions were corrected from later human revision history; no
  reviewed music or frozen file changed. See [reference labels](EVAL_REFERENCE_LABELS.md).
- The ABC adapter reads written notes, simultaneous heads, rests, ties, chord
  symbols, meter and the performed bar route using the installed review renderer.
- Deterministic feedback flags parse, meter and route issues. Unconfirmed pickups
  and conflicting explicit voice barlines require inspection. It never pads rests.
- A callback-based repair hook allows zero to two revisions and preserves a copy
  of each attempt. It receives no answers or scores from the evaluator.
- Six controlled examples check equivalent notation, wrong pitch, duration, chord,
  missing bar and wrong return. Their dashboard says **synthetic**, uses fixture
  names, and cannot be ingested as Astra/Terra model evidence.

## Read the dashboard in five minutes

Open `out/eval/comparison/index.html` (or serve `out/eval` on localhost port 8777).
Read the current finding and run counts, then compare each piece's two model rows.
Open a piece only when you want mismatches, decisions and evidence files. The
separate grader demo linked near the title shows how measured differences appear.

All twelve real runs currently say **Not run**. A dash means unmeasured, never zero.
Model summaries, when present, average whole-piece metrics and show measured-piece
denominators. Cost and runtime totals also state their coverage. No overall winner
is supported while source fidelity and editorial judgments remain unmeasured.

| Metric | What it establishes now |
| --- | --- |
| Edition notes | Micro F1 within each piece for exact written onset, duration, MIDI pitch and tie flags after bar alignment. Duplicate simultaneous heads are retained. |
| Exact bars | Notes, visible rests, chord symbols and meter agree with the chosen edition. Extra bars lower the score. |
| Chords | Symbol and onset agreement with the edition, including its editorial harmony. |
| Form | Exact performed sequence of aligned edition bars, including endings and final stop. Different ABC encoding can match. |
| Bars aligned | Fraction of edition bars paired by the aligner. Same-length unrelated music can align; this is a count check, not accuracy. |
| Meter valid | Ordinary candidate bars whose total duration matches meter. Confirmed pickup pairs and unresolved partial bars are reported separately. |
| Source notes | Not scored yet. The inventory does not independently establish every event of a complete source bar. |
| Editorial acceptance | Not scored yet. Alternative coherent editions need adjudication, not string equality to one chosen reconstruction. |
| Cost / time / review | Supplied measured values require a pinned measurement record; unknowns remain null. Token pricing estimates are not billed cost. |

Voices' names are ignored in event matching; voice assignment itself, enharmonic
spelling, articulation, slurs and expressive timing are not graded. Written tie
heads are compared, not a separate acoustic attack/hold equivalence. Repeated
phrases can make bar alignment ambiguous. These are visible limitations, not
perfect scores for unsupported dimensions.

## Reproduce local artifacts

Use new destinations for create-once commands. Do not overwrite the reference
freeze, label V1/V2 sidecars or existing task packets.

```sh
uv run python scripts/build_eval_labels.py verify \
  out/eval/references/reviewed_six_20261028_v1 out/eval/labels/reviewed_six_v2

uv run python scripts/evaluate_complete_scores.py report \
  out/eval/references/reviewed_six_20261028_v1 out/eval/labels/reviewed_six_v2 \
  /tmp/new-comparison/report.json
uv run python scripts/build_model_comparison_dashboard.py \
  /tmp/new-comparison/report.json /tmp/new-comparison/index.html

uv run python scripts/evaluate_complete_scores.py self-test /tmp/new-grader-demo
uv run python scripts/build_model_comparison_dashboard.py \
  /tmp/new-grader-demo/report.json /tmp/new-grader-demo/index.html

uv run python scripts/evaluate_complete_scores.py prepare \
  out/eval/references/reviewed_six_20261028_v1 /tmp/new-pdf-only-tasks
uv run python scripts/evaluate_complete_scores.py validate draft.abc --decisions decisions.json
```

The adapter requires the existing optional local Node/abc2svg installation. It
fails explicitly when unavailable and never installs or fetches code. Python-only
grader/registry tests remain hermetic; renderer tests skip without that tool.

## Runner integration contract

`prepare` copies six PDFs, one generic instruction file and a twelve-task manifest.
It records the exact PDF/instruction hashes, model identifiers and identical
Extra High reasoning settings. It makes no model calls and says
`requires_enforced_runner`. A separate directory is not an access-control boundary.

Before real inference, run each task with a fresh context and enforced access to
only its PDF, generic instructions, trusted rendering/validation tools and its
own outputs. Do not expose this repository, labels, reviewed answers, histories,
per-work instructions or network musical sources. Both models must receive the
same tools and two-revision budget. Freeze the runner/tool versions and task
contract before making the twelve attempts. The current callback is the repair
integration point, not a provider adapter or a sandbox.

`report --registry RUNS.json` accepts schema version 1, `evidence_kind: "actual"`,
and a `runs` array. Each row records `slug`, `model_id`, exact `model`,
`reasoning_effort`, `status` (`complete`, `partial`, `failed`),
`authorship: "model"`, `input_pdf_sha256`, and hash-pinned output files:
`abc_path`/`abc_sha256`, `decision_log_path`/`decision_log_sha256`, and
`run_record_path`/`run_record_sha256`. Paths resolve relative to the registry.
Failed attempts remain visible and may omit outputs. No entries means not run.

The runner record contains schema version 1; matching model, reasoning, input
PDF, status and output hashes; the generic `instructions_sha256`; a `runner_name`;
`isolation: "input_only"`; `fresh_context: true`; `network_access: false`; and an
integer `revision_passes` from zero to two. This must be written by the runner,
not accepted as model self-report. Hash verification binds evidence but cannot
prove an untrusted runner told the truth. Byte-identical reference output is
flagged for a provenance audit, not automatically celebrated as perfection.

Optional `cost_usd`, `elapsed_seconds`, and `human_review_minutes` require a
`measurement_path` and `measurement_sha256`. That JSON maps each populated metric
to `{value, basis, evidence}`. Cost basis is `provider_reported`; elapsed basis is
`monotonic_clock` or `timestamp_log`; human effort basis is `active_review_timer`
or `manual_timing`. `evidence` identifies the retained receipt/log. Missing cost
is normal for a subscription agent. Preserve raw token usage separately in `usage`.

The report verifies both the frozen references and label sidecar before grading.
Invalid ABC stays failed; unresolved structural checks and missing edition bars
stay partial. A partial classification is a review signal: a missing editorial
bar may still be a defensible PDF-only choice. Never feed edition comparison
feedback back into a benchmark author.

## Next gate

Complete the source-supported event mask and adjudication rubric, implement and
verify enforced runner isolation, and run the same twelve task contracts. Keep
source fidelity, selected-edition agreement and acceptable alternative judgments
separate. Only then choose a harness/model and test genuinely fresh scores.
