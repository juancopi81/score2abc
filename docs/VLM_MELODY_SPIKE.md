# VLM Melody Spike

Status: bounded spike with one strong HITL result; automatic notehead recognition is still the
blocking stage. Nothing in this document is wired into the production pipeline.

## Current Decision

Do not promote direct VLM transcription, homr, Audiveris, or the current local notehead
detectors as the primary melody backend. Keep the spike infrastructure and review fixtures.

The most important result is a decomposition of the problem:

1. A high-recall proposal layer can expose every reviewed notehead.
2. A small local patch selector materially improves proposal precision on development data.
3. Once notehead centers and pitches are correct, a deterministic visual parser recovers the
   tested durations and rest exactly.
4. Direct image-to-score VLM calls and standalone stem detection do not generalize well enough.

The next automatic milestone is therefore not another broad transcription prompt. It is a
validated notehead selector and pitch mapper, followed by the already-working rhythm parser.

## Leak-Resistant Benchmark

`scripts/build_vlm_melody_event_benchmark.py` freezes inference requests, image hashes, physical
measure mappings, and allowed context before it reads canonical truth. Predictions are validated
as canonical note/rest events and must be persisted before evaluation.

The current split is:

| Split | Systems | Measures | Notes | Derived rests | Purpose |
| --- | --- | ---: | ---: | ---: | --- |
| Development | 1-2 | 17 | 56 | 10 | inspected data and fitting |
| Validation | 7-8 | 14 | 65 | 7 | model selection |
| One-shot heldout | 3 | 9 | 36 | 14 | consumed once by the threshold arm |

Systems 4-6 and 9-10 are quarantined because their generated/physical measure mapping is
ambiguous or was already used during segmentation review. Systems 7-8 have an explicit `-1`
stored-index correction. System 3 was opened once, only after the variable-threshold composed arm
passed its preregistered validation gate. It must not be used for further model selection.

The canonical JSON is sounding-event truth. It is not glyph truth: tied notes, dots, beams, and
printed rests can differ from the final sounding-event representation. Experiments must state
which semantic contract they predict.

Build the benchmark with:

```bash
uv run python scripts/build_vlm_melody_event_benchmark.py build out \
  --slug jaime-llanos_12_aviador_pasillo_fulgencio-garcia \
  --ground-truth dataset/ground_truth \
  --clef treble --time-signature 3/4 --key-hint "one flat: Bb"
```

Artifacts are written under `out/<slug>/vlm_melody_event_benchmark/`.

## Human Review Fixtures

The GT-blind reviewer was completed for system 1 measures 1-4:

- selected noteheads: `14/14`, with no false positives and no manual additions;
- automatic natural pitch: `11/14`;
- final natural pitch after review: `14/14`;
- pitch edits: five, including accidental edits;
- active times: 16.0, 30.0, 55.9, and 96.9 seconds;
- median active time: 43.0 seconds, so the original 10-second usability gate failed.

The review is accurate enough to be training data, but the current interaction is not yet fast
enough to be the product workflow. Reviews are promoted into portable, GT-free fixtures under
`tests/fixtures/vlm_melody/notehead_reviews/`:

```bash
uv run python scripts/promote_vlm_notehead_reviews.py out \
  --slug jaime-llanos_12_aviador_pasillo_fulgencio-garcia \
  --system 1 --measure 1 --measure 2 --measure 3 --measure 4
```

Promotion validates source hashes, strips absolute paths and hidden metrics, and refuses to
overwrite a fixture unless `--force` is explicit.

The same blind workflow is now complete for system 7 measures 1-7. It produced 35 final
noteheads, including three manual additions outside the original cap-24 proposals. All seven
fixtures pass promotion validation, and their ordered pitches match the canonical event audit
`35/35` after applying the written key signature. Together, systems 1 and 7 provide 11 reviewed
measures and 49 coordinate/pitch labels for selector development.

## Automatic Notehead Results

All coordinate metrics below use independent ellipses, not the reviewer selections themselves.

| Method | Precision | Recall | F1 | Decision |
| --- | ---: | ---: | ---: | --- |
| Grid detector, cap 4 | 0.500 | 0.571 | 0.533 | baseline |
| Grid detector, cap 24 | 0.146 | 1.000 | 0.255 | proposal layer only |
| First learned top-4 ranker, LOOCV | 0.563 | 0.643 | 0.600 | marginal |
| Staff-subtraction morphology | 0.333 | 0.571 | 0.421 | reject |
| Patch kNN top-4, LOOCV | 0.750 | 0.857 | 0.800 | material dev win |
| Patch kNN automatic count, LOOCV | 0.917 | 0.786 | 0.846 | material dev win |

The winning local experiment is
`scripts/experiments/spike_notehead_patch_templates.py`. Its automatic-count arm produced
`11 TP / 1 FP / 3 FN`, exact count on three of four folds, and exact selected sets on two of four
folds. This is only four-measure leave-one-out development evidence, not heldout proof.

Retraining with the system-7 reviews separated proposal coverage from classification quality:

- dense staff-grid proposals cover `49/49` reviewed noteheads;
- the dense patch-template selector reaches `0.842` S1+S7 measure-OOF candidate F1;
- the conservative cap-24 patch kNN reaches `0.936` OOF candidate F1, but cannot recover the
  three manually added heads and its absolute scores drift on system 8;
- neither selector alone beats the prior automatic event score on frozen system 8.

These results are preserved by
`spike_review_augmented_selector.py`, `spike_cap24_review_augmented_selector.py`, and
`spike_dense_patch_review_augmented_selector.py`. They show that high proposal recall is solved
on the reviewed slice, while score calibration and downstream rhythm interpretation remain
separate failure modes.

Two geometric hypotheses failed:

- affine staff rectification recovered one cap-8 development head but did not improve pitch-safe
  recall enough to pass its gate;
- local staff tracking left natural-pitch accuracy at `11/14`, while naive oval-center refinement
  reduced it to `8/14`. Cap-24 centers are pitch-grid proposals, not true notehead centroids.

The canonical-sequence pseudo-label bootstrap also stopped at its mandatory audit gate:
`11 TP / 3 FP / 3 FN`, precision/recall `0.786/0.786` versus the required `0.85/0.85`. The errors
were isolated to the pickup measure; no validation or heldout prediction was made by that arm.

A pickup-aware confidence filter recovered `1.00` precision and `0.778` recall on reviewed
non-pickup measures, but its downstream selector predicted `118` validation notes against `65`
truth notes. A later system-7 weak-supervision arm accepted `23/35` pseudo labels, then missed its
system-8 gate with ordered pitch `0.087` and count MAE `2.286`. Canonical sequence alignment is
therefore useful for proposing labels, but not reliable enough to replace independent coordinates.

Reproducible reports:

- `out/experiments/notehead_patch_templates/report.md`
- `out/experiments/notehead_candidate_classifier/report.md`
- `out/experiments/notehead_affine_rectification/report.md`
- `out/experiments/local_staff_tracking/report.json`
- `out/experiments/notehead_sequence_bootstrap/report.md`

## Rhythm And Rest Upper Bound

`scripts/experiments/spike_anchored_rhythm_parser.py` asks a narrower question: if a human or a
future selector supplies correct notehead centers and pitches, can local pixels recover rhythm?
It uses the promoted review fixtures as explicit oracle anchors, then inspects stem/flag, dot, and
residual-rest evidence. Predictions and overlays are written before benchmark truth is opened.

On system 1 measures 1-4:

| Arm | Duration accuracy | Rest F1 | Exact measures |
| --- | ---: | ---: | ---: |
| Layout/meter control | 0.500 | 0.000 | 1/4 |
| Visual only | 1.000 | 1.000 | 4/4 |
| Visual plus conservative 3/4 decoder | 1.000 | 1.000 | 4/4 |

This is the strongest result in the spike. It proves that the tested score pixels contain enough
information for duration/rest parsing once localization and pitch are solved. It does not prove
automatic transcription because the anchors are human-reviewed.

Evidence:

- `out/<slug>/vlm_melody_event_benchmark/anchored_rhythm_parser/report.md`
- `out/<slug>/vlm_melody_event_benchmark/anchored_rhythm_parser/measure_003_overlay.png`

## VLM Results

Direct and candidate-assisted calls were useful diagnostics but not a path to production:

- isolated and neighboring-measure prompts fixed one development measure but failed two blind
  system-2 measures;
- candidate-ID-only selection reached F1 `0.667` on one development measure and missed its gate;
- a full-system `gpt-5.6-sol`, medium-reasoning call on validation system 7 produced note F1
  `0.173`, ordered pitch accuracy `0.130`, duration accuracy `0.674`, rest F1 `0`, and `0/7`
  exact measures;
- that full-system arm stopped before system 3.

The exact full-system request, contact sheet, schema, raw response, usage, parsed prediction,
evaluation, and replay command are under:

`out/experiments/vlm_system_transcription/openai-gpt56sol-medium-system-contact-validation-s7/`

The earlier call ledger and all prompt/image experiments remain under
`out/vlm_melody_batches/`, `out/vlm_notehead_localization_batches/`,
`out/vlm_melody_experiments/`, and `out/experiments/vlm_candidate_id_selector/`.

## Composed Automatic Result

`scripts/experiments/spike_composed_melody_chain.py` composes cap-24 proposals, the local patch
selector, request-only pitch mapping, and the visual rhythm/rest parser. Candidate, anchor, rest,
and canonical event predictions are hashed before truth is loaded.

The original learned-count selector emitted exactly three notes for every validation measure. A
second arm used the development-fitted probability threshold instead, allowing visual evidence to
change the count.

| Arm / split | Note F1 | Ordered pitch | Duration | Rest F1 | Exact |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fixed learned count, validation | 0.056 | 0.149 | 0.328 | 0.000 | 0/14 |
| Variable threshold, validation | **0.264** | **0.446** | **0.662** | 0.000 | 0/14 |
| Variable threshold, one-shot system-3 heldout | 0.031 | 0.158 | 0.026 | 0.000 | 0/9 |

The validation arm beats the full-system VLM reference (`0.173` note F1 and `0.130` ordered
pitch), so the gate legitimately opened heldout once. The heldout score is non-zero but far too
small to justify integration. This is measurable progress and a useful failure boundary, not a
claim that automatic melody transcription is solved.

Evidence:

- `out/<slug>/vlm_melody_event_benchmark/composed_melody_chain/report.md`
- `out/<slug>/vlm_melody_event_benchmark/composed_melody_chain/threshold_selector/validation/`
- `out/<slug>/vlm_melody_event_benchmark/composed_melody_chain/threshold_selector/heldout/`

Other automatic hypotheses were stopped at validation:

- standalone stem endpoints retained high pitch-multiset recall (`0.846`) but overpredicted
  `174/65` notes;
- adding stem provenance to the patch selector did not beat the grid-only development method;
- repeated-measure visual retrieval reached validation note F1 `0` and found no exact measures.

These negative arms are preserved under the benchmark's `stem_endpoint_detector/`,
`hybrid_notehead_selector/`, and `measure_retrieval_transcriber/` directories.

### System-7 Review-Augmented Arm

`scripts/experiments/spike_meter_gap_resolver.py` trains the image selector only from promoted
S1+S7 reviews, freezes system-8 predictions, and then evaluates them. It also applies one narrow
GT-free musical repair: when visual note durations leave exactly half a beat unfilled in the
request's 3/4 meter and the first anchor follows a large leading gap, insert a leading eighth
rest instead of lengthening the final note.

| Arm / system-8 slice | Note F1 | Ordered pitch | Duration | Rest F1 | Exact |
| --- | ---: | ---: | ---: | ---: | ---: |
| Historical threshold selector | 0.321 | 0.333 | 0.700 | 0.000 | 0/7 |
| Dense patch selector alone | 0.262 | 0.469 | 0.719 | 0.000 | 0/7 |
| Dense patch + meter-gap resolver | **0.426** | **0.469** | **0.750** | **0.333** | **1/7** |

The repair made system 8 measure 2 fully exact: five pitches, five onsets, five durations, and
the leading rest. It raises strict note F1 by `0.162` absolute over the original `0.264` local
validation baseline and by `0.105` over the frozen same-target historical score. A candidate-gap
recovery sub-arm was selected as disabled (`score_margin=0`) because it did not improve S1+S7
OOF labels; no recovery override contributed to the reported gain.

This is a real model-selection win, not a new generalization claim. System 8 had already been
opened by earlier arms, and system 3 is consumed. The freeze manifest, inference records,
overlays, metrics, and training selection are under
`out/<slug>/vlm_melody_event_benchmark/meter_gap_resolver/`.

## Fresh Second-Score Freeze and Evaluation

Carrizal system 4 was selected using only page layout and crop segmentation. It is an interior
system for which the detector produced seven automatic crops, uses a different composer from
Aviador, and had no canonical events or MusicXML in the repository. The allowed context comes
only from its printed header: treble clef, `3/4`, and one flat (`Bb`).

`scripts/experiments/freeze_second_score_heldout.py` fit the unchanged S1+S7
review-trained selector/meter-gap configuration and wrote all Carrizal predictions before truth.
The runner has no truth, MusicXML, or evaluation parameter and refuses to overwrite an existing
freeze. The sealed state is:

- target: `jaime-llanos_19_carrizal_pasillo_emilio-murillo`, system 4, automatic crops 1-7;
- request SHA256: `bf061a1224eef6e473a5e0569d444d107b65724b0a7be19ef0a5d7575c428cf0`;
- prediction SHA256: `654f7671da37fe3d62e9b776a5b4ad5ab0cc0a44383ad5bddd864065c0cb430d`;
- freeze SHA256: `af7814c0d00888fbc7af3a579cfd746781558062cb9c5b498faadd65e9f28b96`;
- pre-truth status: `frozen_awaiting_canonical_musicxml`.

Artifacts are under
`out/jaime-llanos_19_carrizal_pasillo_emilio-murillo/vlm_melody_fresh_heldout/system_004/`.
The independent MuseScore transcription established eight physical measures. The detector missed
the barline between physical measures 2 and 3, so automatic crop 2 contains both measures. The
evaluation mapping was defined after transcription review and was never available to inference.

`scripts/experiments/evaluate_second_score_heldout.py` verifies the freeze and every frozen
artifact before parsing MusicXML. It then records the seven-crop/eight-measure mapping, imports
canonical sounding events, hashes the truth and evaluator, and writes an immutable evaluation.
The sealed result is:

| Metric | Carrizal system 4 |
| --- | ---: |
| Strict note F1 | `0.325581` |
| Ordered pitch | `0.269231` |
| Ordered onset | `0.423077` |
| Ordered duration | `0.423077` |
| Rest F1 | `0.250000` |
| Exact automatic crops | `0/7` |

The original freeze SHA256 remains
`af7814c0d00888fbc7af3a579cfd746781558062cb9c5b498faadd65e9f28b96`. The approved MusicXML
SHA256 is `d6450616b1a45f5d9fc5176f4c260f7b9f8f6ecefffb95f860c67762f58ed6d9`, and the mapped truth
SHA256 is `4e3ede2e3e3980bfaa606d504b2383eea737032209bc9b54a101124b17b84d95`.

This is consumed negative heldout evidence. Crop 3 reached note F1 `0.8` with all five onsets and
durations correct, but aggregate recognition produced no exact crop. The path remains spike-only.
The approved MusicXML, mapped truth, import manifest, and sealed evaluation are preserved under
`tests/fixtures/vlm_melody/fresh_heldout/`; they are deliberately outside `dataset/musicxml/` so
the partial-system transcription cannot enter normal fixture-backend runs.

## Next Gate

Before pipeline integration, the next composed automatic arm must:

1. keep the `0.426` system-8 result frozen as consumed model-selection evidence;
2. fit only on explicitly promoted training reviews;
3. persist complete predictions before reading truth;
4. use a new independent heldout score or an explicitly repaired/remapped quarantined system;
5. show meaningful heldout exact-event and rest accuracy without expected note counts or
   canonical sequences at inference.

Carrizal system 4 is now consumed and can be used only as explicit diagnostic/training material.
The next bounded arm should turn the missed barline into a segmentation regression, collect and
promote cross-writer notehead reviews on Carrizal, retrain without claiming heldout performance,
and freeze a third score before its transcription is opened. Further tuning on Aviador system 8
or reporting Carrizal as heldout again would be validation overfitting rather than stronger
evidence.

## Current Slice: Cross-Score Training and La Chata v2 Freeze

The Carrizal system-4 barline regression is fixed without changing the Aviador
benchmark: Carrizal system 4 is `TP=9 FP=0 FN=0`, while Aviador remains
`TP=69 FP=7 FN=1`, aggregate F1 `0.945`.

The versioned Carrizal 8-crop v2 adjudication contains 20 noteheads: 18 high
confidence and 2 medium, with 14 candidate selections and 6 manual placements.
These are consumed, agent-reviewed training labels. They are not human-reviewed
or promotion-eligible ground truth. Score-disjoint retraining selected
configuration C with:

| Metric | Selected C |
| --- | ---: |
| Macro notehead F1 | `0.706944` |
| Conditional pitch accuracy on matched noteheads | `0.813910` |
| End-to-end correct-pitch recall | `0.601531` |
| Coordinate exactness | `0.136364` |
| Coordinate-plus-pitch exactness | `0.090909` |

The corrected replay model hash remains `6e2f17c...`. The conditional pitch
metric is intentionally not an end-to-end accuracy claim; correct-pitch recall
is the metric that includes localization misses.

### La Chata System 7

La Chata system 7 completed the fresh third-score heldout gate. Its v2
predictions were truth-blind and sealed before the transcription was opened: 7
automatic crops, 34 predicted heads, prediction hash `89d5723...`, freeze hash
`d140cf3...`, and seal hash `56e5105...`. The v1 seal is historical and
superseded by the provenance audit; v2 is authoritative.

The independent seven-measure MusicXML transcription produced the one-shot
result below. Simultaneous heads share stems and are intentionally encoded as
chords in voice 1.

| Metric | La Chata v2 one-shot |
| --- | ---: |
| Predicted / truth noteheads | `34 / 37` |
| Note-count F1 | `0.901408` |
| Ordered-pitch alignment accuracy | `0.435897` |
| Exact automatic crops | `1 / 7` |

Rhythm, onset, rest, and meter metrics are
`not_scored_missing_frozen_context` because the frozen metadata did not contain
time-signature or key context. La Chata is now consumed evidence and cannot be
reused as a heldout gate.

The seven-measure transcription is stored at:

`out/jaime-llanos_64_la-chata_pasillo_luis-a-calvo/vlm_melody_third_score_heldout/v2/system_007/la_chata_system_007.musicxml`

The evaluator was run exactly once with:

```bash
uv run python scripts/experiments/evaluate_frozen_third_score_heldout.py \
  out/jaime-llanos_64_la-chata_pasillo_luis-a-calvo/vlm_melody_third_score_heldout/v2/system_007/frozen/sealed_manifest.json \
  --musicxml out/jaime-llanos_64_la-chata_pasillo_luis-a-calvo/vlm_melody_third_score_heldout/v2/system_007/la_chata_system_007.musicxml
```

The evaluator verified the v2 seal before reading MusicXML and preserved the
original inference/truth hashes.

### Consumed Polyphony and Key Postmortem

The revealed transcription contains 37 noteheads in 25 onset groups. Four
measures contain shared-stem chords. The system also has a visible one-sharp key
change after measure 1, while the frozen request had no key context. A separate
human-confirmed context file models that as a stateful event beginning at
measure 2; the automatic lane remains key-blind.

The create-once v3 postmortem reports every fixed configuration without
truth-based winner selection:

| Lane / configuration | Exact pitch groups | Group edit | Exact ordered pitches | Heads |
| --- | ---: | ---: | ---: | ---: |
| Automatic x-only baseline | `11/25` | `23` | `17/37` | `34` |
| Context x-only baseline | `12/25` | `22` | `20/37` | `34` |
| Context unfiltered recovery | `17/25` | `17` | `22/37` | `46` |
| Context stem-aware recovery (`0.8`/`0.9`) | `16/25` | `18` | `20/37` | `39` |

The stateful key hint helps pitch without changing localization. Chord recovery
helps group structure on La Chata, but the unfiltered arm overselects and the
stem-aware arm does not improve ordered-pitch matches. Neither is eligible for
runtime adoption.

The independent consumed-score regression audit covers 19 promoted Aviador and
Carrizal measures:

| Configuration | TP | FP | FN | Candidate F1 |
| --- | ---: | ---: | ---: | ---: |
| x-only baseline | `55` | `15` | `14` | `0.791367` |
| stem-aware `0.8`/`0.9` | `55` | `16` | `14` | `0.785714` |
| stem-aware `1.0` | `55` | `15` | `14` | `0.791367` |

The permissive stem thresholds add one false positive and no true positives;
the strict threshold is a no-op. This is consumed in-sample regression evidence,
not an accuracy claim. Reports are preserved at:

- `out/jaime-llanos_64_la-chata_pasillo_luis-a-calvo/vlm_melody_third_score_heldout/v2/system_007/postmortem_polyphonic_v3/`
- `out/vlm_melody_consumed_training/consumed_stem_aware_chord_recovery_regression_v1/`

### Automatic Visual Key State and Onset Gate

The next consumed-only slice separates two decisions.

First, `spike_consumed_key_state_detector.py` detects only an explicit sharp
signature immediately after a double bar. A single-bar sharp accidental plus a
notehead is a mandatory negative test. The detector fails closed before the
first confirmed event, propagates only confirmed exact fifths, and currently
does not support flats, multiple sharp glyphs, or initial system signatures.

On La Chata system 7 it finds the measure-2 change, emits `fifths=1`, and
propagates that state through measures 3-7. All 22 pure staff-crop controls from
Aviador systems 1/7 and Carrizal system 4 remain unknown. This is consumed
calibration evidence, not independent recognition accuracy.

`spike_consumed_visual_key_pitch_replay.py` verifies every source-image hash,
persists and hashes both prediction lanes before opening truth, and then scores
the same frozen x-only heads:

| Lane | Exact ordered pitches | Exact pitch groups | Ordered edit | Group edit |
| --- | ---: | ---: | ---: | ---: |
| No context | `17/37` | `11/25` | `22` | `23` |
| Automatic visual key | `20/37` | `12/25` | `19` | `22` |

The `+3` pitch gain exactly reproduces the earlier human-context result while
keeping candidate IDs, total heads, and onset groups identical. The consumed
pitch gate passes, but the claim remains limited to this already-open score and
this one signature pattern.

Second, `spike_consumed_onset_group_selector.py` fits a remove-only group
filter in work-disjoint Aviador/Carrizal folds. It does not earn adoption:

| Arm | TP | FP | FN | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Frozen onset groups | `63` | `7` | `6` | `0.900` | `0.913` | `0.906475` |
| Proposed group filter | `63` | `7` | `6` | `0.900` | `0.913` | `0.906475` |

Aviador contributes no false selected groups in this 11-measure view, so its
work-disjoint fold cannot teach the Carrizal false-onset boundary. La Chata
remains 34 predicted onset groups against 25 truth groups. This no-op result is
preserved rather than tuned in-sample.

Evidence:

- `out/vlm_melody_consumed_training/consumed_key_state_detector_v1/`
- `out/jaime-llanos_64_la-chata_pasillo_luis-a-calvo/vlm_melody_third_score_heldout/v2/system_007/postmortem_visual_key_pitch_v1/`
- `out/vlm_melody_consumed_training/consumed_onset_group_selector_v1/`

### Expanded Key Signatures and Meter-Deficit Review Triage

`spike_consumed_key_signature_detector.py` replaces the one-sharp-only visual
front end for consumed experimentation. It proposes accidental-sized windows
from vertical ink, skips cancellation naturals by matching the standard
left-to-right staff-position sequence, and keeps separate gates for an initial
treble-clef signature and a change after a double bar.

The materialized v3 seven-case report is exact:

| Case | Expected fifths | Predicted fifths |
| --- | ---: | ---: |
| Aviador initial one flat | `-1` | `-1` |
| Aviador change to two sharps | `+2` | `+2` |
| Aviador change to two flats | `-2` | `-2` |
| La Chata change to one sharp | `+1` | `+1` |
| Aviador repeat control 1 | unknown | unknown |
| Aviador repeat control 2 | unknown | unknown |
| Gato'e Fique initial two sharps | `+2` | `+2` |

A broader truth-blind change-mode scan covers 67 Aviador, 15 Carrizal, and 7
La Chata crops. It emits only the three actual changed signatures. These are
consumed controls, not an independent accuracy estimate. Evidence is at:

- `out/vlm_melody_consumed_training/consumed_key_signature_detector_v3/`

The expanded detector now emits work-scoped events and feeds the frozen La
Chata x-only pitch replay directly. Predictions are persisted before consumed
truth is opened, and evaluation labels remain comparison-only. The composed
result exactly matches the earlier one-sharp replay while proving the expanded
artifact is usable downstream:

| Lane | Exact ordered pitches | Exact pitch groups | Ordered edit | Group edit |
| --- | ---: | ---: | ---: | ---: |
| No key context | `17/37` | `11/25` | `22` | `23` |
| Expanded visual key | `20/37` | `12/25` | `19` | `22` |

Candidate selection is identical in both lanes. Evidence is at:

- `out/jaime-llanos_64_la-chata_pasillo_luis-a-calvo/vlm_melody_third_score_heldout/v2/system_007/postmortem_visual_key_signature_v2_pitch_v2/`

Candidate-level onset deletion was tested next with grayscale/binary raw and
staff-suppressed patches. A recall-first threshold keeps every false group; a
balanced threshold reduces Carrizal false groups but removes enough true
Aviador groups to lower aggregate F1. Automatic deletion remains rejected.

`spike_consumed_meter_deficit_validator.py` therefore changes the product
action: it never mutates onset groups. It applies the existing visual rhythm
parser to automatic anchors and flags a non-pickup measure when its visual
symbols underfill request meter context or the explicit `Pasillo -> 3 beats`
metadata prior.

| Evaluation | Errors caught | False alerts | Errors missed | Review load |
| --- | ---: | ---: | ---: | ---: |
| Aviador | `2` | `0` | `1` | `2/11` |
| Carrizal | `6` | `0` | `1` | `6/8` |
| Consumed aggregate | `8` | `0` | `2` | `8/19` |
| La Chata count-only | `3` | `1` | `3` | `4/7` |

The consumed aggregate passes the review-triage gate at precision `1.0`,
recall `0.8`, and F1 `0.888889`. La Chata does not pass a generalization gate,
so this remains a diagnostic review signal and is not connected to
`score2abc run`. Evidence is at:

- `out/vlm_melody_consumed_training/consumed_meter_deficit_validator_v1/`

The La Chata segmentation gap is now closed by a
truth-blind five-line system-eligibility gate. Rejected proposals remain
inspectable as crops with reasoned manifest records, while accepted systems are
renumbered and retain their original candidate index. Fresh regression accepts
all 10 Aviador systems, accepts Carrizal systems 1-11 while rejecting its
terminal incomplete four-line ruled tail, and rejects only La Chata's title
band before mapping its old musical systems 2-11 to new systems 1-10. All
accepted crops are pixel-identical to their prior musical counterparts, and no
frozen heldout artifact was rewritten.

### Fourth-Score One-Shot Evaluation

Gato'e Fique system 3 was selected from a three-score layout-only pool and
sealed before any target MusicXML was opened. The gate contains six automatic
crops. It pins configuration-C predictions, an automatic initial one-flat
result (`fifths=-1`), and an explicit provisional `Pasillo -> 3/4` metadata
prior. The prior is not presented as visual meter recognition or target truth.

The historical configuration-C model could not pass the provenance gate
because its implementation script changed after artifact creation. Replaying
the same Aviador+Carrizal training into a new create-once directory reproduced
the exact serialized model hash (`6e2f17c...`) and refreshed the implementation
provenance without using Gato'e Fique. The fourth-score prediction hash is
`2b5475a...`, freeze hash is `d74b230...`, and seal hash is `0d0e745...`.

The independent transcription also contains six physical measures, so the
evaluator used a deterministic one-to-one crop mapping. The one-shot result is:

| Metric | Result |
|---|---:|
| Predicted / truth noteheads | `25 / 26` |
| Note-count F1 | `0.941177` |
| Ordered-pitch alignment accuracy | `0.222222` |
| Exact ordered pitches | `6` |
| Exact automatic crops | `0 / 6` |

The provisional `3/4` context matched the MusicXML. The automatic key context
did not: it predicted one flat, while the transcription encodes two sharps.
The frozen prediction remains the authoritative heldout result.

A create-once consumed replay then changed only the supplied key context to two
sharps. Candidate IDs and note counts stayed identical. Exact ordered pitches
rose `6 -> 18`, ordered-pitch alignment accuracy rose
`0.222222 -> 0.666667`, edit distance fell `21 -> 9`, and exact pitch groups
rose `5/24 -> 16/24`. The context lane equals the diagnostic accidental oracle
on this slice, making initial key recognition the highest-leverage next fix.
This replay is postmortem evidence and cannot replace the sealed score.

After preserving that one-shot result, the initial signature was added as a
consumed detector case. The failure came from a fragmented clef that pushed the
search window into the first sharp, followed by greedy single-glyph matching.
The v3 detector accepts a bounded full-height broad sharp shape and scores
complete ordered accidental sequences. It now passes all seven consumed cases;
the 89-crop change-mode scan remains unchanged at three true hits.

The review-time fallback is
`apply_vlm_melody_key_correction.py`. It accepts repeatable
`START_MEASURE=FIFTHS` events, reuses frozen automatic anchors, recomputes only
pitch, and refuses to write if candidate IDs, coordinates, note counts, or
rhythm change. On Gato'e Fique, the create-once `+2` output has 25 unchanged
heads and 15 changed pitches and exactly matches the previously scored consumed
context lane. It does not rewrite `evaluation_v1`.

Evidence is at:

- `out/jaime-llanos_49_gatoe-fique_pasillo_emilio-murillo/vlm_melody_fourth_score_heldout/v1/system_003/`
- `out/jaime-llanos_49_gatoe-fique_pasillo_emilio-murillo/vlm_melody_fourth_score_heldout/v1/system_003/evaluation_v1/`
- `out/jaime-llanos_49_gatoe-fique_pasillo_emilio-murillo/vlm_melody_fourth_score_heldout/v1/system_003/postmortem_key_truth_context_v2/`
- `out/jaime-llanos_49_gatoe-fique_pasillo_emilio-murillo/vlm_melody_fourth_score_heldout/v1/system_003/review_key_correction_v1/`
- `out/vlm_melody_consumed_training/cross_score_notehead_v1_replay_20260722/`

Next: test event/rest accuracy and non-mutating onset/meter review triage on
fresh independent evidence. Keep visual-key recognition and explicit key
correction review-only until they generalize beyond the consumed suite.

### Fifth-Score One-Shot Evaluation

Coqueteos system 2 was frozen as the fifth fresh independent gate. The
pre-truth segmentation produced six automatic crops with width-spacing CV
`0.212794` under the unchanged layout threshold. This segmentation is part of
the sealed result and was not changed after transcription.

The gate uses the replayed cross-score selector trained only on Aviador and
Carrizal. Before any Coqueteos MusicXML was opened, it sealed the six source
crops, full event predictions, model/training provenance, treble clef, a
provisional metadata-derived `Pasillo -> 3/4` prior, and an unknown key because
the automatic initial-key detector was inconclusive. Important pins are:

- prepared manifest: `e6b92a4b75f627949acc1ea97091d26cf6b57d733ccd8db168a42060d35f2765`
- canonical predictions: `1950e076c46c8b61107ce70db985348828066924001ec0fc618bf7a2533d9157`
- model: `6e2f17c043e94a68cdd642a0dbd02a52bcefb011e493bb13b72a31d1f59cf2a6`
- freeze: `5aaf35921ed8951b38bdae0c9924f627cfd06e5b11cfe120b9f3508cc85abce3`
- sealed manifest: `ecaa6c990e7bfc4dfa5ac6f13ab222e156292378edef7aefd99e9e343eab7c2d`

The independent transcription contains seven valid `3/4` measures with one
flat. Automatic crops 1-5 map to physical measures 1-5; crop 6 maps to
physical measures 6-7. Crop 1 starts after a false leading boundary, so scoring
it against the complete first measure intentionally counts the lost beginning
as an automatic segmentation failure.

| Metric | Result |
|---|---:|
| Predicted / truth noteheads | `24 / 31` |
| Note F1 | `0.363636` |
| Ordered-pitch accuracy | `0.375` |
| Ordered-onset accuracy | `0.28125` |
| Ordered-duration accuracy | `0.59375` |
| Rest F1 | `0` |
| Exact automatic crops | `0 / 6` |
| Meter-valid automatic crops | `5 / 6` |

The provisional `3/4` context matched the MusicXML. Frozen key context was
unknown, while the transcription contains one flat. The truth-blind,
non-mutating meter sidecar had flagged automatic measures 3, 4, and 5. All
three are genuine event errors, but automatic measures 1, 2, and 6 are also
wrong, giving triage precision `1.0`, recall `0.5`, F1 `0.666667`, and review
load `3/6`. This supports review prioritization but not automatic deletion.

Transcription review also corrected the segmentation diagnosis. x=150 is an
upward note stem after musical ink, while x=2041 is the true barline before the
final dotted-half measure. The postmortem image-aware cleanup now yields
boundaries `[0, 541, 902, 1091, 1390, 1735, 2041, 2126]`. A replay over every
generated system changes only Coqueteos system 2, and the Aviador benchmark
remains `TP=69`, `FP=7`, `FN=1`, F1 `0.945`. The sealed six-crop evaluation
remains authoritative.

Evidence is at:

- `out/jaime-llanos_22_coqueteos_pasillo_fulgencio-garcia/vlm_melody_fifth_score_heldout/v1/system_002/evaluation_v1/`
- `out/jaime-llanos_22_coqueteos_pasillo_fulgencio-garcia/vlm_melody_fifth_score_heldout/v1/system_002/pretruth_meter_triage_v1/`

### Coqueteos Corrected-Segmentation Postmortem

After the one-shot gate was consumed, the corrected seven boundaries were
materialized in a separate `coqueteos_system_002_seg_v2` namespace. The replay
uses the exact frozen model hash `6e2f17c...` and the same truth-blind treble,
provisional `3/4`, and unknown-key context. It writes requests, predictions,
detailed inference, and meter-triage observations before opening the consumed
MusicXML. It never modifies the sealed six-crop gate.

| Metric | Sealed 6 crops | Corrected 7 crops | Delta |
|---|---:|---:|---:|
| Note F1 | `0.363636` | `0.280702` | `-0.082934` |
| Note precision | `0.416667` | `0.307692` | `-0.108975` |
| Note recall | `0.322581` | `0.258065` | `-0.064516` |
| Ordered pitch | `0.375` | `0.30303` | `-0.07197` |
| Ordered onset | `0.28125` | `0.333333` | `+0.052083` |
| Ordered duration | `0.59375` | `0.575758` | `-0.017992` |
| Meter-valid crops | `5/6` | `7/7` | structural gain |
| Exact crops | `0/6` | `0/7` | no gain |

This is a useful negative result: corrected segmentation restores physical
measure structure and meter coverage but does not fix selector, pitch, rhythm,
or rest recognition. The corrected review-only meter signal flags measures 3,
4, 5, and 7. All are genuine event errors, giving precision `1.0`, recall
`0.571429`, F1 `0.727273`, and review load `4/7`; all seven measures are wrong,
so the signal still misses three and remains unsuitable for automatic repair.

The corrected, GT-blind candidate builder plus consumed MusicXML alignment
produces an explicitly unreviewed proposal queue. It finds pitch-compatible
candidates for `27/31` expected notes:

| Measure | Expected | Proposed matches | Unresolved expected |
|---:|---:|---:|---:|
| 1 | 6 | 5 | 1 |
| 2 | 6 | 5 | 1 |
| 3 | 1 | 0 | 1 |
| 4 | 5 | 5 | 0 |
| 5 | 6 | 6 | 0 |
| 6 | 6 | 5 | 1 |
| 7 | 1 | 1 | 0 |

The frozen selector chose only two of six expected notes in measure 5 even
though all six have pitch-compatible candidates. This isolates selector recall
as the highest-value next target. The queue remains
`eligible_for_training=false` and `human_reviewed=false`; MusicXML events do
not provide notehead pixel coordinates, so deterministic assignments must be
visually adjudicated before retraining.

Reproduce the consumed slice:

```bash
uv run python scripts/experiments/build_consumed_cross_score_training_inputs.py out \
  --slug jaime-llanos_22_coqueteos_pasillo_fulgencio-garcia \
  --system 2 --namespace coqueteos_system_002_seg_v2 --expected-measures 7
uv run python scripts/experiments/spike_consumed_coqueteos_corrected_replay.py out
uv run python scripts/experiments/prepare_consumed_cross_score_proposals.py out \
  --mapping out/jaime-llanos_22_coqueteos_pasillo_fulgencio-garcia/vlm_melody_training_inputs/coqueteos_system_002_seg_v2/mapping.json \
  --consumption-mapping out/jaime-llanos_22_coqueteos_pasillo_fulgencio-garcia/vlm_melody_training_inputs/coqueteos_system_002_seg_v2/consumed_corrected_replay_v1/consumption_mapping.json
```

Evidence is at:

- `out/jaime-llanos_22_coqueteos_pasillo_fulgencio-garcia/vlm_melody_training_inputs/coqueteos_system_002_seg_v2/consumed_corrected_replay_v1/`
- `out/jaime-llanos_22_coqueteos_pasillo_fulgencio-garcia/vlm_melody_training_inputs/coqueteos_system_002_seg_v2/proposals/`
