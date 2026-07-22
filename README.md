# score2abc

Pipeline to convert handwritten Colombian scores into ABC notation plus metadata.

## Usage (uv)

Run the CLI through uv:

```bash
uv run python main.py ingest dataset dataset/metadata.csv out
uv run python main.py run out
uv run python main.py qa out
uv run python main.py export out
```

Notes:
- PDF rendering uses `pdf2image` and requires a local Poppler install.
- ABC previews render via `abc2svg` or `abcm2ps` if available; otherwise a placeholder SVG is written.
- MusicXML melody extraction defaults to committed fixtures under
  `dataset/musicxml/` so normal runs stay hermetic.
- ABC export now preserves canonical event timing, including implicit rests,
  simultaneous-note groups, and ties split across barlines/chord changes.
- Segmentation deskews each page once, then applies a gamma=3.5 curve to
  push faded pencil ink toward black while leaving the already-uniform
  paper white untouched. It then writes system crops plus annotation-band
  crops above and below each staff under each work's `systems/` directory.
  Broad horizontal proposals must contain five long, consistently spaced
  staff lines before they receive a system number. Rejected proposals are
  preserved as `rejected_candidate_page_*.png` with reasons in the per-page
  segment manifest; accepted systems are renumbered while retaining their
  original candidate index in metadata.
  Chord bands overlap the outer staff lines so chord symbols that sit
  against the staff aren't clipped. Per-page overlays and JSON bbox
  manifests (with the detected `page_rotation_degrees`) are written
  alongside for inspection.

### Melody VLM input crops

The first VLM melody spike milestone prepares local inputs only; it does not
call a model. Run the normal ingest/run pipeline first so `systems/system_*.png`
exists, then build measure-level crops and context files:

```bash
uv run python scripts/build_vlm_melody_inputs.py out \
  --slug jaime-llanos_12_aviador_pasillo_fulgencio-garcia \
  --system 1 \
  --overwrite
```

Outputs are written under `out/<slug>/vlm_melody_inputs/`:

- `measure_NNN_raw.png`: full-height measure crop from the detected system.
- `measure_NNN_staff.png`: tighter staff-region crop for clean model input.
- `measure_NNN_staff_overlay.png`: staff-line overlay for inspection/prompt experiments.
- `measure_NNN_context.json`: metadata, measure indices, detected barlines, and path references.
- `manifest.jsonl`: one record per measure for batch prompt experiments.

To index all local image variants for a selected measure without calling a
model, build the spike-only variant manifest:

```bash
uv run python scripts/build_vlm_melody_variants.py out \
  --slug jaime-llanos_12_aviador_pasillo_fulgencio-garcia \
  --system 1 --measure 3 \
  --variant all --overwrite
```

This writes `out/vlm_melody_variants_manifest.jsonl` plus
`out/<slug>/vlm_melody_inputs/variants_manifest.jsonl`. Derived pitch-ruler
filenames include both style and source crop so staff and staff-overlay variants
can be compared without overwriting each other. The `neighbor_context` variant
includes one adjacent measure on each side and marks the target with thin ticks
in white margins, allowing models to learn the writer's stem/barline style
without drawing helpers over the music.

To pre-render the prompt variants associated with those images, run:

```bash
uv run python scripts/build_vlm_melody_prompt_variants.py out \
  --slug jaime-llanos_12_aviador_pasillo_fulgencio-garcia \
  --system 1 --measure 3 \
  --variant all --prompt all --overwrite
```

This writes `out/vlm_melody_prompt_variants_manifest.jsonl` and prompt folders
under `out/vlm_melody_prompt_variants/`. Each manifest row ties one
`variant_id` to one compatible `prompt_id`, the exact image path, prompt files,
and optional JSON schema.

To plan or run a sweep from that manifest, use the batch runner. It defaults to
`--max-calls 0`, so the first command is a no-network dry run that records which
calls would happen:

```bash
uv run python scripts/run_vlm_melody_experiment_batch.py out \
  --slug jaime-llanos_12_aviador_pasillo_fulgencio-garcia \
  --system 1 --measure 3 \
  --variant-id staff \
  --prompt-id direct_pitch_v0 \
  --provider openai --model gpt-5.5 \
  --openai-reasoning-effort medium \
  --max-calls 0
```

Batch artifacts are written under `out/vlm_melody_batches/`. Increase
`--max-calls` only when you intentionally want live provider calls; add
`--journal` to snapshot each completed result with its exact image, prompts,
schema, fixture, eval report, and replay command.

The direct-call history, strict event benchmark, promoted review fixtures,
automatic notehead experiments, and anchored rhythm result are documented in
[`docs/VLM_MELODY_SPIKE.md`](docs/VLM_MELODY_SPIKE.md).

The cap-24 detector covers every annotated notehead in the four-measure
development slice, so the current spike exposes those proposals in a local,
GT-blind reviewer:

```bash
uv run python scripts/review_vlm_notehead_candidates.py out \
  --slug jaime-llanos_12_aviador_pasillo_fulgencio-garcia \
  --system 1 \
  --measure 1 --measure 2 --measure 3 --measure 4
```

Open the printed localhost URL. Confirm candidate circles, add any missing
notehead with the `+` control, and correct pitches before saving. Reviews and
overlays are written to
`out/<slug>/vlm_melody_reviews/system_NNN/measure_NNN/`. The browser never
receives ground truth or evaluation metrics; when independent coordinate GT is
available, hidden metrics are attached to `review.json` only after submission.
This is spike infrastructure and does not yet feed the production pipeline.

Promote completed reviews into deterministic, portable training fixtures before
using them in local experiments:

```bash
uv run python scripts/promote_vlm_notehead_reviews.py out \
  --slug jaime-llanos_12_aviador_pasillo_fulgencio-garcia \
  --system 1 \
  --measure 1 --measure 2 --measure 3 --measure 4
```

Build the leak-resistant event benchmark after the measure inputs exist:

```bash
uv run python scripts/build_vlm_melody_event_benchmark.py build out \
  --slug jaime-llanos_12_aviador_pasillo_fulgencio-garcia \
  --ground-truth dataset/ground_truth \
  --clef treble --time-signature 3/4 --key-hint "one flat: Bb"
```

The benchmark freezes request images, hashes, physical measure mappings, and
allowed context before reading truth. It separates development systems 1-2,
validation systems 7-8, and one-shot heldout system 3. System 3 has now been
consumed by the preregistered variable-threshold arm and must not be used for
further model selection. These local spike commands reproduce the strongest
development results without network calls:

```bash
uv run python scripts/experiments/spike_notehead_patch_templates.py out
uv run python scripts/experiments/spike_anchored_rhythm_parser.py --out-dir out
uv run python scripts/experiments/spike_meter_gap_resolver.py out
```

The original patch-selector report is supported by four-measure leave-one-out
evidence; the review-augmented arm expands model selection to 11 measures across
systems 1 and 7. The rhythm parser is an explicit HITL upper bound: it
uses promoted human centers and pitches as anchors, then predicts durations and
rests from pixels. The meter-gap arm adds the promoted system-7 reviews and
raises consumed system-8 strict note F1 to `0.426`, but still needs a fresh
independent score before pipeline integration. None of these paths is connected
to `score2abc run` yet.

The second-score gate froze Carrizal system 4 before truth. Its prediction
runner has no truth or MusicXML argument and refuses to overwrite the existing
freeze:

```bash
uv run python scripts/experiments/freeze_second_score_heldout.py out
```

Do not rerun it. The sealed requests, predictions, inference trace, hashes, and
overlays are under
`out/jaime-llanos_19_carrizal_pasillo_emilio-murillo/vlm_melody_fresh_heldout/system_004/`.
The independent transcription established that the seven automatic crops
contain eight physical measures: automatic crop 2 spans physical measures 2
and 3. The separate evaluator preserves that segmentation miss and verifies all
frozen hashes before reading MusicXML:

```bash
uv run python scripts/experiments/evaluate_second_score_heldout.py out \
  --musicxml out/jaime-llanos_19_carrizal_pasillo_emilio-murillo/\
vlm_melody_fresh_heldout/system_004/carrizal_system_4.musicxml
```

The sealed one-shot result was strict note F1 `0.325581`, ordered pitch
`0.269231`, rest F1 `0.25`, and `0/7` exact automatic crops. This is consumed
negative heldout evidence: the automatic recognizer remains spike-only and is
not ready for `score2abc run` integration.

### Current cross-score slice and heldout freezes

The Carrizal system-4 segmentation fix now gives `TP=9 FP=0 FN=0`; the Aviador
barline benchmark remains `TP=69 FP=7 FN=1`, aggregate F1 `0.945`. The consumed
Carrizal 8-crop v2 adjudication contains 20 noteheads (`18` high, `2` medium;
`14` candidate selections, `6` manual). It is agent-reviewed training material,
not human-reviewed or promotion-eligible ground truth.

The score-disjoint retraining selected configuration C: macro notehead F1
`0.706944`, conditional pitch accuracy on matched noteheads `0.813910`,
end-to-end correct-pitch recall `0.601531`, coordinate exactness `0.136364`, and coordinate-plus-pitch
exactness `0.090909`. The corrected replay model hash remains `6e2f17c...`.

La Chata system 7 completed the fresh third-score gate. Its v2 predictions were
truth-blind and sealed before transcription: 7 crops, 34 predicted heads,
prediction hash `89d5723...`, freeze hash `d140cf3...`, and seal hash
`56e5105...`. The one-shot result was note-count F1 `0.901408`, ordered-pitch
accuracy `0.435897`, and `1/7` exact crops. Rhythm, onset, rest, and meter remain
unscored because the frozen request lacked time/key context. This is now
consumed heldout evidence, not a model-selection set.

The seven-measure transcription is stored at
`out/jaime-llanos_64_la-chata_pasillo_luis-a-calvo/vlm_melody_third_score_heldout/v2/system_007/la_chata_system_007.musicxml`
and was evaluated with:

```bash
uv run python scripts/experiments/evaluate_frozen_third_score_heldout.py \
  out/jaime-llanos_64_la-chata_pasillo_luis-a-calvo/vlm_melody_third_score_heldout/v2/system_007/frozen/sealed_manifest.json \
  --musicxml out/jaime-llanos_64_la-chata_pasillo_luis-a-calvo/vlm_melody_third_score_heldout/v2/system_007/la_chata_system_007.musicxml
```

The transcription contains 37 noteheads in 25 onset groups. The simultaneous
heads share stems and are intentionally represented as chords in voice 1. A
create-once postmortem preserved the sealed inputs and separated two failure
modes:

- a visible one-sharp key change from measure 2 raises exact ordered-pitch
  matches from `17` to `20` without changing selected heads;
- the frozen x-only selector suppresses chord heads and also admits spurious
  onsets. Unfiltered recovery improves context-aware exact pitch groups from
  `12/25` to `17/25`, but overproduces `46` heads for 37 true heads.

A candidate-local stem filter narrows that result to 39 heads and `16/25`
exact pitch groups, but it is not adoptable: on 19 consumed Aviador/Carrizal
measures it changes candidate F1 from `0.791367` to `0.785714` by adding one
false positive and no true positives.

The visual key slice now handles the consumed initial one-flat signature plus
changed one-sharp, two-sharp, and two-flat signatures. Its six-case report is
`6/6`, including two repeat/double-bar controls, and a broader change-mode scan
fires only on the three actual changes across 89 Aviador, Carrizal, and La
Chata crops. Its work-scoped context now feeds the frozen La Chata pitch replay
directly: exact ordered pitches improve `17 -> 20`, exact pitch groups improve
`11 -> 12`, and candidate selection is unchanged. This is still a consumed,
bounded spike rather than independent general key recognition.

Automatic onset deletion remains rejected. The original work-disjoint group
filter is a no-op at `TP=63 FP=7 FN=6`, F1 `0.906475`; a candidate-patch veto
either remains a no-op or loses enough true Aviador groups to reduce aggregate
F1. A non-mutating meter-deficit validator is useful for review triage on the
consumed Aviador/Carrizal set: it catches `8/10` error measures with `0` false
alerts while flagging `8/19` measures. It does not generalize cleanly to La
Chata's count-only replay (`3/6` caught, one false alert), so it is not wired
into runtime.

Gato'e Fique system 3 is now the fresh fourth-score gate. Six layout-selected
crops and their automatic predictions were sealed before target MusicXML was
opened. The frozen context records an automatic initial one-flat prediction
and a provisional `Pasillo -> 3/4` metadata prior; neither is target truth.
Prediction hash: `2b5475a...`; freeze hash: `d74b230...`; seal hash:
`0d0e745...`. Accuracy is intentionally unknown pending the independent
transcription at
`out/jaime-llanos_49_gatoe-fique_pasillo_emilio-murillo/vlm_melody_fourth_score_heldout/v1/system_003/gatoe_fique_system_003.musicxml`.

Fresh segmentation now rejects La Chata's title/author band and renumbers only
musical systems. Existing historical `out/` crops remain unchanged so sealed
heldout hashes and prior evidence stay reproducible. The code and artifacts
remain spike-only.
Exact evidence and replay commands are in
[`docs/VLM_MELODY_SPIKE.md`](docs/VLM_MELODY_SPIKE.md).

For live transcription spikes, keep every tested image/prompt/config/result in a
journal folder after recording and evaluating a fixture. This makes later
prompt/model comparisons reproducible without relying on shell history:

```bash
uv run python scripts/record_vlm_melody_fixtures.py out \
  --slug jaime-llanos_12_aviador_pasillo_fulgencio-garcia \
  --system 1 --measure 3 \
  --input-kind staff \
  --provider openai --model gpt-5.5 \
  --transcription-mode pitch \
  --openai-reasoning-effort medium \
  --max-calls 1 --force

uv run python scripts/eval_vlm_melody_fixtures.py out \
  --slug jaime-llanos_12_aviador_pasillo_fulgencio-garcia \
  --system 1 --measure 3 \
  --input-kind staff \
  --provider openai --model gpt-5.5 \
  --transcription-mode pitch \
  --openai-reasoning-effort medium

uv run python scripts/journal_vlm_melody_experiment.py out \
  --slug jaime-llanos_12_aviador_pasillo_fulgencio-garcia \
  --system 1 --measure 3 \
  --input-kind staff \
  --provider openai --model gpt-5.5 \
  --transcription-mode pitch \
  --openai-reasoning-effort medium \
  --notes "Short hypothesis/result note"
```

Journals are written under `out/vlm_melody_experiments/` and include the exact
input image, prompt files, copied fixture, eval result, git snapshot, and replay
commands.

### Optional melody OMR backends

Optional melody-engine adapters are available behind explicit backend flags.
They shell out to locally installed OMR commands, copy or normalize generated
MusicXML into `out/<slug>/intermediate/musicxml.xml`, and validate it with the
same parser used for fixtures.

```bash
uv run python main.py run out --musicxml-backend homr
uv run python main.py run out --musicxml-backend audiveris
```

For a first smoke test, run only the labeled `Aviador` work:

```bash
uv run python main.py run out \
  --slug jaime-llanos_12_aviador_pasillo_fulgencio-garcia \
  --musicxml-backend homr \
  --homr-input page
```

To use a wrapper command or non-default executable:

```bash
uv run python main.py run out --musicxml-backend homr --homr-command "path/to/homr"
uv run python main.py run out --musicxml-backend audiveris --audiveris-command "path/to/audiveris"
```

Both optional OMR backends support the same image input modes:

- `page`: rendered PDF page.
- `deskewed-page`: deskewed/enhanced full page.
- `systems`: stitched detected system crops.

To compare homr input modes without overwriting outputs:

```bash
SLUG=jaime-llanos_12_aviador_pasillo_fulgencio-garcia
HOMR=/tmp/score2abc-homr/bin/homr

for MODE in page deskewed-page systems; do
  OUT="/tmp/score2abc-homr-${MODE}"
  rm -rf "$OUT"
  uv run python main.py ingest dataset dataset/metadata.csv "$OUT"
  uv run python main.py run "$OUT" \
    --slug "$SLUG" \
    --musicxml-backend homr \
    --homr-command "$HOMR" \
    --homr-input "$MODE"
  uv run python main.py eval "$OUT" --ground-truth dataset/ground_truth
done
```

Then compare the reports:

```bash
grep -n '"note_f1_avg"\|"time_signature_pred"\|"pred_notes"\|"truth_notes"' \
  /tmp/score2abc-homr-page/eval/report.json \
  /tmp/score2abc-homr-deskewed-page/eval/report.json \
  /tmp/score2abc-homr-systems/eval/report.json
```

`homr` is not a package dependency of score2abc. Install and license-review it
separately before using this backend. The current adapter supports rendered page,
deskewed page, and stitched-system-crop inputs.

To compare Audiveris input modes without overwriting outputs:

```bash
SLUG=jaime-llanos_12_aviador_pasillo_fulgencio-garcia
AUDIVERIS=audiveris

for MODE in page deskewed-page systems; do
  OUT="/tmp/score2abc-audiveris-${MODE}"
  rm -rf "$OUT"
  uv run python main.py ingest dataset dataset/metadata.csv "$OUT"
  uv run python main.py run "$OUT" \
    --slug "$SLUG" \
    --musicxml-backend audiveris \
    --audiveris-command "$AUDIVERIS" \
    --audiveris-input "$MODE"
  uv run python main.py eval "$OUT" --ground-truth dataset/ground_truth
done
```

Then compare the reports:

```bash
grep -n '"note_f1_avg"\|"time_signature_pred"\|"pred_notes"\|"truth_notes"' \
  /tmp/score2abc-audiveris-page/eval/report.json \
  /tmp/score2abc-audiveris-deskewed-page/eval/report.json \
  /tmp/score2abc-audiveris-systems/eval/report.json
```

`audiveris` is not a package dependency of score2abc. Install and
license-review it separately before using this backend. The current adapter
uses Audiveris batch export and supports rendered page, deskewed page, and
stitched-system-crop inputs.

## Dependency Management (uv)

Use uv for dependencies:

```bash
uv lock
uv sync
```

To add a new package:

```bash
uv add <package>
```

## Testing

Install test dependencies and run pytest:

```bash
uv sync --extra test
uv run pytest
```

Run the local CI-equivalent checks:

```bash
uv sync --extra dev
uv run ruff check .
uv run black --check .
uv run pytest
```

## Evaluation (M1)

Ground-truth events live under `dataset/ground_truth/` named by slug
(e.g., `dataset/ground_truth/<slug>.json`). Run evaluation against an `out/`
folder produced by the pipeline:

```bash
uv run python main.py eval out --ground-truth dataset/ground_truth
```

Evaluation now includes event-level precision/recall/F1 and enforces a minimum
coverage gate (`works_with_predictions / works_with_truth`).

If your ground truth starts in MuseScore, export an uncompressed `.musicxml`
file and convert it into the repo's canonical events JSON with:

```bash
uv run python -m score2abc.musicxml path/to/work.musicxml dataset/ground_truth/<slug>.json
```

## Pipeline execution status

- `ingest`, `run`, and `qa` now fail with non-zero exit codes if any work fails.
- Command-level status summaries are written to:
  - `out/ingest_status.json`
  - `out/run_status.json`
  - `out/qa_status.json`
- Per-work stage artifacts are written to `out/<slug>/stages/<stage>.json`.

## External Tools

### Poppler (required for PDF rendering)

Check whether Poppler is available:

```bash
pdftoppm -h
```

Install Poppler:

```bash
# macOS (Homebrew)
brew install poppler

# Ubuntu/Debian
sudo apt-get install poppler-utils
```

On Windows, install Poppler binaries (e.g., the `poppler-windows` builds by
oschwartz10612), then add the `bin` folder to your PATH.

### ABC preview renderers (optional)

Install one of these to render previews:

```bash
# abc2svg (npm)
npm i abc2svg

# abcm2ps (Homebrew)
brew install abcm2ps
```

If neither is installed, the pipeline writes a placeholder SVG preview.

## Dataset

`dataset/` is the current golden dataset of source PDFs, with a canonical metadata
table at `dataset/metadata.csv`. These files come from the
manuscript transcriptions of Colombian organist Jaime Llanos Gonzalez, later
photocopied and shared by triplista Jairo Rincon Gomez (June 2002) with the music
library of Universidad de Antioquia. The photocopies are low-legibility and in
fragile condition, so this dataset prioritizes preservation and accurate transcription.

## Filename convention (when possible)

`jaime-llanos_<num>_<titulo>_<genero>_<autor>.pdf`

Normalization rules:
- lowercase
- ASCII only (accents removed)
- words separated by hyphens
- punctuation removed (e.g., "Gato'e" -> `gatoe`)
- author last names separated by hyphens

Not all future sources will have clean or consistent names. If a file cannot be
normalized cleanly, keep the original filename and rely on the metadata table
as the source of truth for title/composer/genre.
