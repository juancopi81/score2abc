# Complete-score evaluation

## Current status and local commands

The October 28 checkpoint and six-reference freeze are complete; see the
[checkpoint receipt](CHECKPOINT_20261028.md). The HTML reference-readiness page is
generated and browser-checked. Source/editorial annotation, musical grading,
agent-run isolation and the model comparison remain to be implemented.

Verify the existing snapshot; **do not recreate or overwrite it**:

```sh
uv run python scripts/freeze_reviewed_references.py verify \
  out/eval/references/reviewed_six_20261028_v1 --out-dir out
uv run python scripts/build_eval_dashboard.py \
  out/eval/references/reviewed_six_20261028_v1/manifest.json out/eval/index.html
```

Open `out/eval/index.html` directly, or serve its directory locally:

```sh
python3 -m http.server 8777 --bind 127.0.0.1 --directory out/eval
```

The archive recorded in the checkpoint restores `index.html` alongside
`references/reviewed_six_20261028_v1/`; its relative links then work offline.
Future snapshots must use a new name and remain outside live work/collection
trees. Creation uses `freeze OUT_DIR NEW_DESTINATION`; existing destinations are
rejected. Files under `inputs/` are PDFs only, but that directory separation is
not an inference sandbox by itself. Do not give a model access to the answer tree.

## Agreed scope

Compare Astra and Terra as complete-score transcribers using the same supplied
PDFs, instructions, tools and bounded self-correction. Each run delivers a playable
ABC edition and a decision log. Damaged passages receive a concrete best-effort
reading; uncertainty remains explicit. No recordings, web searches or other
editions may supply musical evidence.

Common forms, including variants of AA BB CC ABC, are a shared musical prior.
Readable source directions take precedence. Phrase connections and cadences can
support a reconstruction. Do not force a standard route, blindly repeat an ending,
or transfer a reviewed work's order to another piece. The route must identify the
physical bars and endings visited, including the final stop. Follow
[the collection notation convention](FORM_NOTATION.md).

Use the smallest supported intervention that completes the edition while retaining
readable material. An unmarked chord bar is not automatically damaged or incomplete.
An invented note, rest, chord or return must never become a source observation
merely because it sounds plausible or passes validation.

## Delivery sequence

1. Ship the correction-desk checkpoint through PR #13 after checks pass.
2. Freeze the exact six reviewed editions, PDFs and surviving histories. Keep a
   separately verified backup outside the working repository.
3. Build eval v1: align physical source bars and editorial bars, annotate source
   evidence versus reconstruction, normalize musical events, validate the grader,
   and produce a clear HTML dashboard.
4. Run one initial attempt per model per work: 12 complete attempts, with identical
   tool access and bounded repair. Record settings, usage, failures and outputs.
   These are an initial paired comparison, not a reliable estimate of run variance.
5. Freeze the selected harness before testing fresh scores. Do not reuse consumed
   development results as claims about unseen-score accuracy.

The six references are Amalia, Atardecer, Aviador, Carrizal, Coqueteos and
Rumichaca. They are consumed, musician-reviewed **performing editions**. Their
approval does not make every editorial addition uniquely inferable from the PDF.
Existing excerpt evaluators are useful components; they do not currently grade
this complete-score task.

## Reference and inference boundary

The initial immutable freeze keeps only PDFs under `inputs/`. Exact saved JSON,
ABC exports, histories, source-manifest records and referenced editorial evidence
belong on the answer side. A manifest records file hashes, surviving revisions and
any omitted context. A separate manifest checksum detects accidental manifest
changes; independently record that checksum outside the snapshot as well.

The freeze initially records annotation as pending. Preserve it unchanged and
create versioned annotation sidecars pinned to its checksum. Identify physical
source bars separately from inserted editorial bars; the two numbering systems
can differ. Notes and historical provenance retain their original wording, even
when older text says Draft and the current saved state is Reviewed.

Transcription runs require isolated input directories and contexts. Do not expose
answer files, reference manifests, human histories, score-specific prior answers,
or repository instructions containing benchmark results. A promise in a prompt
is not sufficient isolation when the same tools can open the answers. Generic
musical guidance and deterministic validator feedback are allowed equally for
both models. Freeze the harness, prompt, source hashes, model identifiers,
reasoning settings and run limits before the comparison.

## Checks that help the model

Deterministic tools validate parsing, rendering, pitch/event representation,
durations per voice, measure totals and explicit playback routes. Handle pickups,
ties, overlapping voices, meter changes and intentional exceptions explicitly.
Report an inconsistency with its location and evidence; do not automatically pad
every short bar with rests or discard notes to satisfy meter.

The model inspects the PDF and decides whether a flagged measure needs a different
duration, a missing note/rest, a segmentation correction or a documented exception.
It receives a bounded opportunity to revise. Unfinished, invalid and timed-out runs
remain visible outcomes rather than disappearing from the comparison.

Each editorial decision records its source location, observed reading, adopted
choice, reason, uncertainty and meaningful alternatives. Human acceptance remains
separate from model confidence. The reference annotation pass should extract
existing decisions first and ask Juan only about unresolved musical judgments.

## Evaluation dimensions

| Dimension | Initial scoring contract |
| --- | --- |
| Visible notation | Compare aligned source-supported pitches, onsets, durations, rests, chord symbols and relevant notation; show exact-bar matches alongside event metrics. |
| Reconstruction | Grade documented choices against evidence and accepted constraints/alternatives. Report unsupported or undisclosed changes separately. |
| Form | Compare coherent performed bar/ending routes, transitions and final stops; do not require the same ABC syntax. |
| Completeness | Count missing passages, unparseable outputs and unresolved structural failures. Meter validity alone is not accuracy. |
| Human burden | Record actual correction/adjudication effort when available, with the review desk's timing limitations stated. |
| Efficiency | Record runtime, model settings, input/output/cached tokens and attributable cost when available. Unknown cost stays unknown. |

Use semantic music comparisons, preserving polyphonic heads, voices, ties and
written versus performed events. Different ABC spellings/layouts can represent
equivalent music. Accepted alternatives must form coherent editions/routes;
do not award an impossible combination of individually allowed fragments.

Before model runs, test the grader with known pitch/rhythm/chord errors, missing
bars, wrong returns, editorial insertions and equivalent encodings. Keep metric
definitions and denominators fixed. Summarize paired results by score as well as
across the set; explain whether an aggregate weights notes or whole pieces.
Automated editorial judges may help triage but cannot be the sole ground truth.

## Five-minute dashboard

The dashboard is an offline HTML report with no external assets required. Its
first view should answer:

- Which system is ready to use, or why the evidence is insufficient to choose?
- Where did each system make mistakes, with a simple six-score comparison?
- Which uncertain passages need a musician, and what did each model choose?
- What quality, review effort and measured cost changed?

Keep source fidelity, editorial acceptability, form and completeness separate.
Use plain-language labels and examples before technical metrics. Unmeasured or
unsupported fields say so; they are neither zero nor perfect scores. Expandable
details should link to the source, proposed ABC, decision log and reproducibility
record. Show failures and coverage alongside successes.

The initial reference-readiness page is only the first layer. It must clearly say
that no comparison has run. A polished readiness page does not complete eval v1.
