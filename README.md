# score2abc

Pipeline to convert handwritten Colombian scores into ABC notation plus metadata.

The collection inventory and current delivery plan are in
[`docs/COLLECTION_WORKFLOW.md`](docs/COLLECTION_WORKFLOW.md). The supplied book
contains 95 identifiable works; automatic whole-score transcription remains
experimental.

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
  staff lines before they receive a system number. A bounded staff-extent
  refinement preserves weak clef/meter preambles, while consecutive trailing
  ruled staves without enough non-staff musical ink are rejected. Rejected
  proposals are preserved as `rejected_candidate_page_*.png` with reasons in
  the per-page segment manifest; accepted systems are renumbered while
  retaining their original candidate index in metadata.
  Chord bands overlap the outer staff lines so chord symbols that sit
  against the staff aren't clipped. Per-page overlays and JSON bbox
  manifests (with the detected `page_rotation_degrees`) are written
  alongside for inspection.

### Local research sources

Keep supplied scores that are not cleared for redistribution under the
gitignored `dataset/local_restricted/` directory. Use the normal filename and
metadata conventions there, but write generated artifacts to a separate output
root so the tracked golden dataset remains reproducible:

```bash
uv run python main.py ingest \
  dataset/local_restricted dataset/local_restricted/metadata.csv out/local_restricted
uv run python main.py run out/local_restricted --slug <slug>
```

### Review and correct a transcription

```bash
uv run python main.py review out --slug jaime-llanos_12_aviador_pasillo_fulgencio-garcia --open-ui
```

The loopback-only editor shows manuscript pages/staves beside editable ABC and
rendered notation. Click a rendered note, rest, or barline to select its ABC;
use reference-tone playback to check a phrase. Save incomplete drafts, record
questions, then mark the complete score reviewed after comparing it with the
source. Unknown meter/key stay explicit until corrected. Supplied MusicXML is
identified as such, and stub melody output is never offered as a transcription.

Corrections are stored separately in `out/<slug>/overrides/review.json` with
revision checks and an original-ABC snapshot. Reopening preserves the exact ABC;
pipeline outputs and frozen research artifacts are unchanged. Download exports
the last saved, validated ABC, with `draft` or `reviewed` in its filename. An
unsaved edit disables download. Invalid notation can be saved as a draft but
cannot be approved or exported. Rendering validates syntax, not musical accuracy.

Preview, playback, approval and validated export require Node.js and the existing
optional local `abc2svg` installation; saving drafts still works without them.
The editor loads those assets locally and makes no recognition/API calls.
Use another port/output root to review restricted sources, for example
`uv run python main.py review out/local_restricted --port 8767`.
This first editor supports ABC text correction, not graphical note entry or
MusicXML export, and overrides do not yet feed back into canonical events/training.

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

Metadata remains the default key context. The independently supported visual
key slice is opt-in and accepts only strict initial/system-entry one-flat or
two-sharp signatures; unknown detections fail closed and inherit only a prior
accepted state:

```bash
uv run python scripts/build_vlm_melody_inputs.py out \
  --slug <slug> --system <n> --overwrite --key-context strict-visual
```

This option does not detect internal double-bar key changes.

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

The later edge-safe rule has now completed its separately frozen unseen-score
gate on No lo Creas system 8. Eleven automatic crops were sealed before
transcription; the paired rule preserves every baseline candidate ID and
coordinate and adds two companions, both in crop 1, without creating new
x-groups. A later audit found that the first physical measure's eleven-crop
mapping used a `9 + 3` note split where the frozen x-groups require `6 + 6`.
The immutable `evaluation_v1` is therefore superseded by the create-once mapping
erratum: note-count F1 still improves `0.584616 -> 0.626865`, but exact natural
diatonic positions remain `6 -> 6` and exact chord-size matches remain `4 -> 4`.
The preregistered promotion gate fails and the one-companion rule remains spike-only.
The evaluated transcription is at:

`out/local_restricted/jaime-llanos_73_no-lo-creas_pasillo_a-vasquez-pedrero/vlm_melody_independent_dyad_recovery_gate/v1/system_008/no_lo_creas_system_008.musicxml`

The evaluator verified the paired seal before opening that file. The corrected
mapping and authoritative interpretation are in `evaluation_v2_mapping_erratum/`.

A bounded multi-head follow-up improves the corrected No lo Creas evidence over
the dyad lane: note-count F1 `0.626865 -> 0.666667`, exact natural-diatonic
positions `6 -> 8`, exact chord sizes `4 -> 6`, and exact structure crops
`0 -> 1`. It preserves or improves consumed Alcira and La Chata results, while
Aviador/Carrizal candidate F1 remains `0.791367` with zero recovered false
positives. This is still model-selection evidence, not runtime integration. At
model-selection time, a genuinely unseen polyphonic score was still required for a
valid frozen gate because the remaining untouched corpus scores were monophonic. See
`docs/VLM_MELODY_SPIKE.md` for the full result and scope. Key, meter, rhythm, duration,
and rests are not scored by this gate.

That independent gate is now evaluated on local-restricted A medio palo system 7. A
full-staff barline fix yields seven physical crops and improves the Aviador A1 barline
benchmark from `F1=0.945` to `F1=0.952`. After the paired predictions were sealed, the
seven-measure transcription confirmed that all 13 additions were needed: note-count F1
improves `0.763636 -> 1.0`, exact natural-diatonic positions `13 -> 24`, exact chord-size
matches `0 -> 13`, and exact structure crops `0 -> 3`. The report is at:

`out/local_restricted/jaime-llanos_7_a-medio-palo_pasillo_m-garavito-w/vlm_melody_independent_multihead_recovery_gate/v1/system_007/evaluation_v1/report.json`

The recovery rule passes this independent gate and may enter an explicit opt-in spike
path, but not the default pipeline. Onset grouping remains wrong (`21` predicted groups
versus `17` truth groups), particularly in measures 3, 5, and 7. Do not rerun or overwrite
the frozen gate or `evaluation_v1`.

The opt-in is now available without changing canonical predictions:

```bash
uv run python scripts/experiments/run_third_score_heldout_inference.py \
  <prepared-manifest> \
  --model-dir <model-dir> \
  --inference-dirname <new-name> \
  --multihead-recovery \
  --no-freeze
```

It writes `edge_safe_stem_multihead_recovery_v1/` beside the baseline inference. The
sidecar is candidate-only, hash-pinned, and intentionally cannot enter the canonical
freezer.

A subsequent consumed visual audit corrected the interpretation of measures 3, 5, and 7:
the second selected point in measures 3 and 5 is an augmentation dot, and both selected
points in measure 7 are chord text below the staff. The selected v3 dotted-hollow repair
replaces those weak anchors only when it sees one shared-stem head pair plus matching weak
dots to the right and no additional leading-edge dyad. Exact staff-position matches improve
`24 -> 30`, onset groups `21 -> 18`, exact chord-size matches `13 -> 16`, and exact-structure
crops `3 -> 6`; Alcira, La Chata, and No lo Creas remain unchanged. The report is:

`out/vlm_melody_consumed_training/sparse_stem_dyad_repair_v3/report.json`

This is consumed model-selection evidence, not runtime promotion. The exact rule was then
frozen on previously unseen Desde Lejos system 7. Ten automatic crops were sealed before
truth; the rule accepts one replacement in automatic measure 2 and rejects the other nine.
The sealed manifest is:

`out/local_restricted/jaime-llanos_26_desde-lejos_pasillo_b-b/vlm_melody_independent_sparse_dyad_repair_gate/v1/system_007/frozen/sealed_manifest.json`

The finalized ten-measure transcription and raw-only coordinate review are stored at:

`out/local_restricted/jaime-llanos_26_desde-lejos_pasillo_b-b/vlm_melody_independent_sparse_dyad_repair_gate/v1/system_007/desde_lejos_system_007.musicxml`

The create-once evaluator verifies the complete seal before opening either input:

```bash
uv run python scripts/experiments/evaluate_independent_sparse_dyad_repair_gate.py \
  <sealed-manifest> \
  --musicxml <desde-lejos-system-7.musicxml> \
  --raw-review <raw-image-review.json>
```

`evaluation_v1` confirms the proposed two hollow heads and one frozen augmentation-dot
pair. In automatic measure 2, the comparison lane predicts staff positions `[-5,-5]` as
two singleton groups; the repair lane predicts the exact `[2,4]` (`G4,B4`) dyad in one
onset group. Across all ten crops, exact staff positions improve `9 -> 11`, exact chord-size
matches `8 -> 9`, and exact-structure crops `2 -> 3`; note-count F1 remains `0.941177`
because the replacement preserves two notes. The rule passes its independent bounded gate
and is now available as a chained opt-in spike sidecar, but it is not enabled in the default
pipeline:

```bash
uv run python scripts/experiments/run_third_score_heldout_inference.py \
  <prepared-manifest> \
  --model-dir <model-dir> \
  --inference-dirname <new-name> \
  --multihead-recovery \
  --sparse-dyad-repair \
  --no-freeze
```

This writes `sparse_stem_dyad_repair_v1/` beside the multi-head sidecar. It pins the exact
upstream lane, replays the v3 rule during verification, and records replacement diagnostics
and overlays. Canonical predictions remain byte-for-byte unchanged, and the canonical
freezer rejects the optional run. Omit `--sparse-dyad-repair` to retain the earlier
multi-head-only behavior. Do not rerun the gate or overwrite `evaluation_v1`.

To compose that exact repaired candidate lane into full spike events, use a separate
create-once sidecar:

```bash
uv run python scripts/experiments/materialize_repaired_full_event_sidecar.py \
  <inference-dir> --model-dir <model-dir>
```

This applies bounded pitch/key plus rhythm/rest/meter inference while preserving canonical
outputs byte-for-byte. It fails closed when the prepared request does not contain expected
meter context or when reaching the requested meter would require synthetic request-only rests.
Consumed comparisons require an explicit crop-to-physical-measure mapping and can be recorded with
`evaluate_consumed_repaired_full_event_sidecar.py`; they are postmortem evidence only. The
full-event composition still requires a newly frozen score before default pipeline integration.

The additive duration-aware variant keeps v1 replayable and overrides only an accepted,
single-onset sparse dotted-half dyad in a three-beat measure:

```bash
uv run python scripts/experiments/materialize_repaired_full_event_sidecar_v2.py \
  <inference-dir> --model-dir <model-dir>
```

It pins the sparse-repair diagnostics used by that decision, suppresses residual rests only for
the exact full-measure pattern, and otherwise reproduces v1 predictions. Publication remains
all-or-nothing: every measure must be materialized with valid meter.

The visual key slice now handles consumed initial one-flat and two-sharp
signatures plus changed one-sharp, two-sharp, and two-flat signatures. Its
seven-case report is `7/7`, including two repeat/double-bar controls and the
independently exposed Gato'e Fique miss. A broader change-mode scan still fires
only on the three actual changes across 89 Aviador, Carrizal, and La Chata
crops. Its work-scoped context feeds the frozen La Chata pitch replay directly:
exact ordered pitches improve `17 -> 20`, exact pitch groups improve `11 -> 12`,
and candidate selection is unchanged. This is still a consumed, bounded spike
rather than independent general key recognition. After the independent Alcira
gate, strict initial/system-entry state is now available to the spike inference
path through `--key-context strict-visual`. The default remains metadata-backed;
internal changes and unsupported signature counts still fail closed.

Automatic onset deletion remains rejected. The original work-disjoint group
filter is a no-op at `TP=63 FP=7 FN=6`, F1 `0.906475`; a candidate-patch veto
either remains a no-op or loses enough true Aviador groups to reduce aggregate
F1. A non-mutating meter-deficit validator is useful for review triage on the
consumed Aviador/Carrizal set: it catches `8/10` error measures with `0` false
alerts while flagging `8/19` measures. It does not generalize cleanly to La
Chata's count-only replay (`3/6` caught, one false alert), so it is not wired
into runtime.

Gato'e Fique system 3 completed the fresh fourth-score gate. Six
layout-selected crops and their automatic predictions were sealed before target
MusicXML was opened. The independent six-measure transcription produced
note-count F1 `0.941177` (`25` predicted heads versus `26` truth), ordered-pitch
accuracy `0.222222`, and `0/6` exact crops. The provisional `Pasillo -> 3/4`
prior matched the transcription, but rhythm/rest remain unscored because the
frozen output contains no usable onset or duration predictions.

The independent key result exposed the dominant pitch failure: the frozen
visual context predicted one flat, while the MusicXML contains two sharps. A
consumed replay with the correct key and identical selected candidates raises
exact ordered pitches from `6` to `18`, alignment accuracy from `0.222222` to
`0.666667`, and exact pitch groups from `5/24` to `16/24`. This is diagnostic
evidence, not a revised heldout score. Prediction hash: `2b5475a...`; freeze
hash: `d74b230...`; seal hash: `0d0e745...`. Evaluate with:

```bash
uv run python scripts/experiments/evaluate_frozen_fourth_score_heldout.py \
  out/jaime-llanos_49_gatoe-fique_pasillo_emilio-murillo/vlm_melody_fourth_score_heldout/v1/system_003/frozen/sealed_manifest.json \
  --musicxml out/jaime-llanos_49_gatoe-fique_pasillo_emilio-murillo/vlm_melody_fourth_score_heldout/v1/system_003/gatoe_fique_system_003.musicxml
```

The checked-out `out/` gate already contains the create-once `evaluation_v1`;
do not rerun it.

The consumed detector now recognizes the initial two-sharp signature while
retaining the existing six cases and 89-crop control sweep. An explicit review
action can also apply a human key correction without changing frozen notehead
selection or rhythm:

```bash
uv run python scripts/experiments/apply_vlm_melody_key_correction.py \
  out/jaime-llanos_49_gatoe-fique_pasillo_emilio-murillo/vlm_melody_fourth_score_heldout/v1/system_003/inference_v2/inference.jsonl \
  --key-event 1=2 \
  --output-dir out/jaime-llanos_49_gatoe-fique_pasillo_emilio-murillo/vlm_melody_fourth_score_heldout/v1/system_003/review_key_correction_v1
```

Repeat `--key-event START_MEASURE=FIFTHS` when a later system changes key.
The output is create-once and records invariant checks plus pitch overlays.

Coqueteos system 2 completed the fresh fifth-score full-event gate. Its six
frozen automatic crops map to seven physical measures because the pre-truth
segmentation missed the final internal barline. The one-shot result is note F1
`0.363636`, ordered pitch `0.375`, onset `0.28125`, duration `0.59375`, rest
F1 `0`, and `0/6` exact crops. The truth-blind review signal flags three real
error crops with no false alerts but misses three others. Postmortem barline
cleanup recovers all seven measure boundaries without changing any other
generated system or the sealed evaluation.

The consumed corrected replay now quantifies that cleanup separately. Reusing
the exact frozen model and context raises meter-valid crops from `5/6` to
`7/7`, but note F1 falls from `0.363636` to `0.280702`; segmentation alone is
therefore not a recognition improvement. The non-mutating meter sidecar flags
four of seven genuine error measures with precision `1.0` and recall
`0.571429`. A separate unreviewed proposal queue finds pitch-compatible
candidates for `27/31` expected notes, including all expected heads in measures
4, 5, and 7. Those assignments are not training data until visually reviewed.

```bash
uv run python scripts/experiments/build_consumed_cross_score_training_inputs.py out \
  --slug jaime-llanos_22_coqueteos_pasillo_fulgencio-garcia \
  --system 2 --namespace coqueteos_system_002_seg_v2 --expected-measures 7
uv run python scripts/experiments/spike_consumed_coqueteos_corrected_replay.py out
uv run python scripts/experiments/prepare_consumed_cross_score_proposals.py out \
  --mapping out/jaime-llanos_22_coqueteos_pasillo_fulgencio-garcia/vlm_melody_training_inputs/coqueteos_system_002_seg_v2/mapping.json \
  --consumption-mapping out/jaime-llanos_22_coqueteos_pasillo_fulgencio-garcia/vlm_melody_training_inputs/coqueteos_system_002_seg_v2/consumed_corrected_replay_v1/consumption_mapping.json
```

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
