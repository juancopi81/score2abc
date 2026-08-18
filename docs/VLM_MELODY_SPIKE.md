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

#### Human Candidate Adjudication

Consumed proposal measures can become explicit human-reviewed training
evidence only after every candidate is classified. The decision fixture pins
the raw image, candidate artifact, and automatic proposal by SHA256. The
materializer refuses stale sources, incomplete candidate partitions, and
overwrites:

```bash
uv run python scripts/experiments/materialize_consumed_human_candidate_review.py out \
  --decision-fixture \
  tests/fixtures/vlm_melody/human_candidate_reviews/coqueteos_system_002_measure_005.json
```

For Coqueteos system 2 measure 5, the human review confirmed all six noteheads
were present. The automatic proposal matched four, confused two sharp
fragments for noteheads, and therefore scored `TP=4`, `FP=2`, `FN=2`,
`F1=0.666667`. Rejected candidates are additionally classified as stems,
accidental fragments, barline fragments, slur fragments, or neighboring-system
leakage.

Measures 3 and 7 expose a different localization failure: each contains one
hollow dotted-half notehead, but the candidate generator places detections on
opposite rims. Their version-2 decisions preserve both rim candidates as
support and deterministically derive one notehead center using the mean of the
candidate centers. The resulting reviewed centers are:

- measure 3: `(62.0, 173.5)`, `F4`, supported by `c001` and `c002`;
- measure 7: `(52.0, 136.0)`, staff pitch `B4` and sounding pitch `Bb4`,
  supported by `c001` and `c003`.

This is consumed training evidence, not a heldout accuracy claim. It provides
a concrete target for a future hollow-head ellipse/ring proposal without
treating two rim detections as two notes.

#### Hollow-Notehead Center Proposal

The follow-up spike implements that target without expected pitches or note
counts. It considers pairs of strong local candidates on adjacent staff
positions, requires the manuscript's rising-right notehead diagonal, and then
accepts either an enclosed elongated white region aligned with the pair or a
high-coverage open contour with a light center. Staff-line intersections are
suppressed for the ring measurement. The source image and candidate artifact
are the only proposal inputs; reviewed coordinates enter only during scoring.
All 22 image/candidate pairs and their truth-source references are committed in
a SHA256-pinned fixture manifest, so this result does not depend on ignored
local `out/` state.

```bash
uv run python scripts/experiments/spike_consumed_hollow_notehead_proposals.py
```

On the currently reviewed consumed suite, the fixed rule produces:

| Review lane | Measures | Proposals | Recovered misses | False proposals |
|---|---:|---:|---:|---:|
| Aviador human-promoted | 11 | 0 | 0 | 0 |
| Carrizal agent-adjudicated | 8 | 3 | 3 | 0 |
| Coqueteos human-adjudicated | 3 | 2 | 2 | 0 |
| **Total** | **22** | **5** | **5** | **0** |

The Coqueteos proposals recover both dotted-half centers at `(62.0, 173.5)`
and `(52.0, 136.0)`. Review overlays and the create-once report are written to
`out/vlm_melody_consumed_training/hollow_notehead_proposals_v1/`.
The reproducible inputs live under
`tests/fixtures/vlm_melody/hollow_notehead_inputs/`.

This is a material consumed-data improvement, not a generalization result.
The thresholds were selected while inspecting these three score styles. Keep
the proposer out of runtime candidate generation until the rule is frozen and
tested on hollow notes from a newly reviewed score. The next engineering slice
should then compose accepted centers with the existing candidates and rerun
score-disjoint selector/regression evaluation without changing prior sealed
heldout artifacts.

#### Chispazo Unseen Morphology Gate

Chispazo system 4 was preregistered before candidate or hollow-proposal
generation. The committed selection pins the source PDF, raw system image,
eight visually checked measure boundaries, alignment implementation, and fixed
hollow-notehead rule. The create-once freeze then seals every raw crop,
candidate artifact, and proposal artifact without reading MusicXML, note
ground truth, or review files.

```bash
uv run python scripts/experiments/freeze_hollow_notehead_unseen_gate.py out

MANIFEST=out/jaime-llanos_25_chispazo_pasillo_pedro-morales-pino/vlm_melody_hollow_notehead_gate/v1/system_004/frozen/sealed_manifest.json
uv run python scripts/experiments/review_hollow_notehead_unseen_gate.py "$MANIFEST"
uv run python scripts/experiments/evaluate_hollow_notehead_unseen_gate.py "$MANIFEST"
```

The reviewer did not expose automatic candidates, proposal locations, or
proposal counts to the browser. All eight measures were finalized once. The
human review marked two hollow/open noteheads, in measures 4 and 6. Both are
matched by the frozen baseline staff-grid candidates, for baseline recall
`2/2`; therefore this system has zero baseline-missed hollow heads for the new
rule to recover. Nine candidate pairs passed its geometry prefilter across the
eight measures, but all failed the fixed pixel/shape gates, so the frozen rule
emitted zero proposals.

The create-once evaluator records `not_promoted` for three independent reasons:

- two reviewed hollow heads are below the evaluator's conservative five-head
  promotion minimum;
- baseline candidates already cover both reviewed heads, leaving no recovery
  opportunity;
- the frozen rule emitted no heldout proposal whose precision could be
  measured.

This result does not show a candidate regression: augmented recall remains
`2/2` and there are zero false proposals. It does show that the strong consumed
result (`5/5` recovered misses) did not receive a useful independent
generalization test on this target. Keep the rule out of runtime. A future
morphology gate is worthwhile only if truth-blind screening finds a new score
with several visually apparent hollow heads that the frozen baseline candidate
set genuinely misses; do not tune on Chispazo and then reuse it as heldout
evidence.

#### Independent Cross-Score Error Breakdown

The four independent score reports now have portable, SHA256-pinned copies
under `tests/fixtures/vlm_melody/cross_score_error_breakdown/`. The breakdown
reads only these already-frozen reports:

```bash
uv run python scripts/experiments/build_cross_score_error_breakdown.py
```

It excludes Aviador because that gate is within-score, Chispazo because it is
morphology-only, and all consumed postmortems because they are diagnostic
model-selection evidence. It also excludes Carrizal's merged crop 2 and
Coqueteos' merged final crop from downstream root-cause ranking. Their missing
boundaries remain visible as segmentation errors, but their pitch/onset errors
are not counted when selecting the next target.

Across the remaining 24 one-to-one crops from four independent scores:

- the count-capacity F1 upper bound is `0.878788` (`13` deficits and
  `11` surpluses); this checks per-crop count agreement only and is not
  note-level event F1;
- exact ordered pitch conditional on count capacity is `0.413793`, with
  `51/87` count-alignable notes wrong;
- onset accuracy is `0.483871` and duration accuracy is `0.677419` on the
  11 clean crops from Carrizal and Coqueteos;
- rest F1 is `0.222222`, but only five truth rests are available;
- candidate-pool coverage is `not_identifiable_from_frozen_reports`, not zero.

The report therefore selects `pitch_mapping_and_key_context`: it has the
largest supported clean-unit error burden across all four scores. This is an
engineering-priority result, not a causal claim that every pitch error comes
from the key signature. Staff geometry, note ordering, key state, and explicit
accidentals remain confounded inside the pitch metric.

The next controlled experiment must keep candidate IDs, coordinates, and note
counts fixed while comparing pitch-mapping variants. A runtime change requires
better exact ordered pitch on at least two new independent scores without
localization drift. The validated report and human-readable summary are under
`out/vlm_melody_cross_score_error_breakdown/v2/`.

#### Controlled Cross-Score Pitch Mapping

The follow-up experiment freezes the 24 clean one-to-one crops, 98 selected
noteheads, their candidate IDs, and their pixel coordinates in a portable
SHA256-pinned consumed fixture. It materializes and seals all five prediction
lanes before opening pitch truth:

```bash
uv run python scripts/experiments/spike_cross_score_pitch_mapping.py
```

| Lane | Exact pitches | Count capacity | Conditional accuracy |
|---|---:|---:|---:|
| Historical frozen baseline | `41` | `87` | `0.471264` |
| Global staff + frozen key | `40` | `87` | `0.459770` |
| Global staff + automatic key | `55` | `87` | `0.632184` |
| Locally tracked staff + frozen key | `40` | `87` | `0.459770` |
| Locally tracked staff + automatic key | `55` | `87` | `0.632184` |

The global replay differs from the historical mapper at two Coqueteos measure-5
pitch values and loses one truth match. Component decisions therefore use the
global-staff/frozen-key lane as the exact comparator. Against that comparator,
automatic key state adds `15` exact pitches: `+12` on Gato'e Fique and `+3` on
La Chata, while Carrizal and Coqueteos are unchanged. Candidate IDs,
coordinates, and note counts are identical across all 96 lane/measure
comparisons. Relative to the historical frozen baseline, the automatic-key
lane still adds `14` exact pitches.

Local common-shift staff tracking adds no pitch matches on any score. The
combined lane has the same result as automatic key state alone, so its apparent
pass is entirely attributable to key recognition. Advance only the key
component; reject this geometry tracker for now.

This remains consumed postmortem evidence. The detector recognizes La Chata's
one-sharp change and Gato'e Fique's initial two sharps, falls back to Carrizal's
frozen one-flat context when no stable staff is found, and remains unknown for
Coqueteos. No runtime or pipeline change is justified until the frozen key
component improves exact ordered pitch on two newly selected independent
scores with localization unchanged. The report, prediction seal, and summary
are under `out/vlm_melody_consumed_training/cross_score_pitch_mapping_v2/`;
the portable inputs are under
`tests/fixtures/vlm_melody/cross_score_pitch_mapping/`.

#### Independent Automatic-Key Gates

Two score-disjoint gates are now frozen before human transcription. Each gate
runs the configuration-C selector once and derives both pitch lanes from the
same candidate IDs and pixel coordinates. The baseline uses no key signature;
the automatic lane uses only the visual key state pinned at preparation time.
Meter, rhythm, onset, duration, and rest decoding are withheld so this gate
isolates pitch mapping.

| Gate | Key source | Automatic state | Crops | Selected heads |
|---|---|---:|---:|---:|
| Estrella del Caribe system 3 | initial signature in system 1 | `fifths=-1` | `6` | `24` |
| Sobre el Humo system 7 | double-bar change in system 7 | `fifths=-4` | `7` | `31` |

Selection invariance passes for all `55` selected heads. These counts and key
states are frozen predictions, not truth or accuracy results. An earlier
Estrella system-1 gate is excluded: pre-truth visual QA found that its first
automatic crop contained clef/key preamble rather than a measure.

The authoritative sealed manifests are:

- `out/jaime-llanos_41_estrella-del-caribe_danza_luis-a-calvo/vlm_melody_independent_key_gate/v1_estrella_initial_s3/system_003/frozen/sealed_manifest.json`
- `out/jaime-llanos_92_sobre-el-humo_bambuco_fulgencio-garcia/vlm_melody_independent_key_gate/v1_sobre_change/system_007/frozen/sealed_manifest.json`

Do not rerun them. The completed human transcriptions are stored at:

- `out/jaime-llanos_41_estrella-del-caribe_danza_luis-a-calvo/vlm_melody_independent_key_gate/v1_estrella_initial_s3/system_003/estrella_system_003.musicxml`
- `out/jaime-llanos_92_sobre-el-humo_bambuco_fulgencio-garcia/vlm_melody_independent_key_gate/v1_sobre_change/system_007/sobre_el_humo_system_007.musicxml`

The create-once evaluation verifies both frozen gates before opening either
MusicXML and records the two post-transcription false-split mappings. Estrella's
automatic one-flat state is correct and improves exact ordered-pitch matches
`14 -> 16`. Sobre el Humo's automatic four-flat state is wrong; the transcription
contains a two-sharp change, and exact matches regress `21 -> 4`. Aggregate
matches fall `35 -> 20`. Candidate IDs, coordinates, and counts remain identical
for all `55` selected heads, so this result isolates key state rather than
localization drift.

The preregistered promotion gate is therefore `not_promoted`. Keep automatic
key state out of runtime. Use Sobre el Humo only as consumed detector-debugging
evidence, require ambiguous change signatures to fail closed, retain Estrella
and prior signature controls as regressions, and use a new score-disjoint gate
for the next accuracy claim. The evaluation report is under
`out/vlm_melody_independent_key_gate_evaluation/v1/`.

#### Consumed Sobre Key-Detector Repair

Post-transcription review showed why the frozen Sobre prediction failed. The
position-sequence lane proposed four flat-like fragments carved from the two
physical sharp glyphs and nearby connecting ink. The detector now validates
changed signatures against a second, overlap-collapsed shape count. The old
four-flat proposal has only two independently supported flat regions and is
rejected. Two separate high-confidence sharp shapes then recover `fifths=+2`
despite their nonstandard handwritten vertical placement.

This repair preserves all seven prior consumed signature/control cases (`7/7`)
and leaves the 89-image change scan unchanged at three real hits. Replaying
`+2` through the already frozen Sobre candidate coordinates changes no IDs,
coordinates, or counts and improves exact ordered-pitch matches from the frozen
wrong-key result of `4` to `24`; the no-key lane had `21`. Combined with the
unchanged Estrella result, the repaired consumed projection is `40` exact pitch
matches versus `35` without key context and `20` in the original frozen
automatic lane.

The source gate and its original `fifths=-4` prediction remain immutable. This
is consumed model-selection evidence because the rule was designed after the
Sobre transcription was opened. It does not reverse the original `not_promoted`
decision. The exact regression image is committed under
`tests/fixtures/vlm_melody/key_signatures/`, and the local detector report is at
`out/vlm_melody_consumed_training/consumed_key_signature_detector_v4_sobre_repair/`.
The fixed-localization pitch replay is at
`out/vlm_melody_consumed_training/independent_key_detector_repair_v1/` and is
reproduced by `scripts/experiments/evaluate_consumed_key_detector_repair.py`.
The next accuracy claim still requires a newly frozen score-disjoint
change-signature gate.

#### Consumed Internal-Change Extraction

Sobre el Humo system 3 and Coqueteos system 7 exposed a separate input-contract
failure: normal measure crops split an internal double bar from the cancellation
and replacement signature immediately after it. The consumed full-system
scanner now uses existing barline detections only as x-coordinate hints, keeps
staff geometry from the complete system, and evaluates each hinted boundary in
place. Real-image regressions pin the user-confirmed change boundaries at x=1400
and x=566 respectively.

Human review labels Sobre's replacement signature as B-flat/E-flat
(`fifths=-2`). Coqueteos explicitly cancels the previous B-flat with a B-natural,
then writes F-sharp/C-sharp as the replacement signature (`fifths=+2`). The
detector records that natural as a cancellation prefix rather than counting it
as part of the new key.

The recovery remains restricted to full-system internal scans. Existing
left-edge hypotheses are suppressed in that mode unless they match a reviewed
internal pattern, and the internal double-bar spacing gate rejects tightly
paired note/bar strokes. Across 112 available system images, 10 non-staff or
unstable-staff crops fail closed and exactly the two reviewed systems emit a
key. These remain consumed regression results, not independent accuracy
evidence. The current report and overlays are under
`out/vlm_melody_consumed_training/internal_key_change_scan_v2/`.

Reproduce the scan with:

```bash
uv run python scripts/experiments/scan_consumed_internal_key_changes.py \
  out/jaime-llanos_92_sobre-el-humo_bambuco_fulgencio-garcia/systems/system_003.png \
  out/jaime-llanos_22_coqueteos_pasillo_fulgencio-garcia/systems/system_007.png \
  --out-dir <new-output-directory>
```

The next detector slice is a new score-disjoint gate frozen before its key or
transcription is opened. Until that passes, this internal scanner stays out of
the runtime pipeline.

#### Independent Chispazo Internal-Change Challenge

A truth-blind corpus audit selected Chispazo system 3 before any Chispazo
MusicXML was opened. The full system contains a credible internal double bar at
x=1284 followed by two flat-shaped glyphs. The precision-first detector still
returns `unknown`: its best positional candidate is `fifths=-2`, but it does not
pass the independent-shape acceptance gate. This is therefore an honest
score-disjoint false-negative challenge, not a selected detector success.

The frozen experiment preserves one score-disjoint selector pass over eight
automatic crops and 27 selected anchors. Candidate IDs, coordinates, and counts
are identical across lanes. The strict lane remains no-key because the detector
failed closed. A separate, explicitly non-promotable diagnostic lane applies
the top `-2` candidate only at system x >= 1284: 16 anchors remain before the
event and 11 receive the changed state. The diagnostic lane measures the value
of a correct acceptance decision; it cannot be used as automatic-key promotion
evidence.

The authoritative seal is:

- `out/jaime-llanos_25_chispazo_pasillo_pedro-morales-pino/vlm_melody_independent_key_gate/v1_chispazo_internal_change_s3/system_003/frozen/sealed_manifest.json`

Do not rerun or overwrite it. The complete visible-evidence system-3 MusicXML
transcription is at:

- `out/jaime-llanos_25_chispazo_pasillo_pedro-morales-pino/vlm_melody_independent_key_gate/v1_chispazo_internal_change_s3/system_003/chispazo_system_003.musicxml`

The create-once evaluator verified the seal before opening this transcription.
It maps nine physical measures to eight automatic crops; crop 5 spans physical
measures 5 and 6 because the automatic segmentation missed the key-change
barline. The MusicXML records `fifths=+2` at measure 1 and `fifths=-2` at
measure 6. The source does not repeat the inherited initial signature at this
system's left edge, and its final note is clipped or missing; these are recorded
as source limitations rather than annotation errors.

With the same 27 candidate IDs and coordinates in both lanes, the no-key lane
gets `10` exact ordered-pitch matches and `0.285714` alignment accuracy. The
stateful diagnostic lane gets `14` matches and `0.4` accuracy, a `+4` gain; its
sixth crop is exact (`5/5` pitches). This confirms that the visual two-flat
change is musically useful. It does **not** promote the detector: the strict
truth-blind detector returned `unknown`, while `-2` was retained only as an
inconclusive top-candidate diagnostic.

The immutable report is:

- `out/jaime-llanos_25_chispazo_pasillo_pedro-morales-pino/vlm_melody_independent_key_gate/v1_chispazo_internal_change_s3/system_003/evaluation_v1/report.json`

The strict detector repair and its consumed replay are documented below. A
different score-disjoint positive target still must be frozen before automatic
internal key state can be promoted.

The Chispazo create-once commands used before truth were:

```bash
uv run python scripts/experiments/freeze_independent_key_state_gates.py out \
  --case chispazo_internal_change_s3
uv run python scripts/experiments/run_independent_key_state_gate.py \
  out/jaime-llanos_25_chispazo_pasillo_pedro-morales-pino/\
vlm_melody_independent_key_gate/v1_chispazo_internal_change_s3/\
system_003/prepared_manifest.json
uv run python scripts/experiments/evaluate_chispazo_internal_key_gate.py out
```

#### Consumed Chispazo Strict-Detector Repair

Post-transcription analysis localized the miss to the second handwritten flat.
The first glyph retained strong independent flat morphology; the second was
positionally correct but merged with neighboring ink and fell below the generic
right-side-ink threshold. The repair does not simply lower that threshold. Its
fallback requires all of the following after a detected internal double bar:

- the top positional two-flat candidate beats the same-count two-sharp rival;
- both selected glyphs retain weak flat score, margin, left-stem, and right-ink
  support;
- at least one selected glyph retains the original strong flat morphology; and
- the pair follows the expected horizontal spacing and descending flat-anchor
  sequence.

The complete 112-system replay has the same 10 fail-closed non-staff crops, the
same 59 detected double bars, and exactly one changed decision: Chispazo system
3 at x=1284 moves from `unknown` to `fifths=-2`. Coqueteos system 7 and Sobre el
Humo system 3 keep their previous methods and predictions. The repaired hit set
is therefore exactly the three reviewed internal changes, with no unrelated new
hit. The report and overlays are under:

- `out/vlm_melody_consumed_training/internal_key_change_scan_v3_chispazo_repair/`

The consumed replay verifies the original Chispazo seal, rebuilds the stateful
`-2` lane from the repaired strict detector event, and proves that its candidate
IDs, coordinates, counts, and pitches match the previously frozen diagnostic
lane across all 27 anchors. Exact pitch remains `10 -> 14`, and ordered-pitch
alignment remains `0.285714 -> 0.4`. The report is:

- `out/vlm_melody_consumed_training/chispazo_key_detector_repair_v1/report.json`

This is model-selection evidence because the acceptance rule was designed after
the Chispazo transcription was opened. It does not convert the original gate
into a heldout success and does not make automatic key state pipeline-ready.

A visual audit of all 59 double-bar overlays found no unused positive internal
change in the current corpus. The only credible positives are the three systems
already consumed by detector development. Selecting another current system
after this sweep would manufacture a heldout claim. The next independent gate
therefore requires a new PDF/system outside these 112 images; it must be frozen
before its key or transcription is opened.

Reproduce the consumed repair after creating a fresh scan directory with:

```bash
uv run python scripts/experiments/evaluate_consumed_chispazo_key_detector_repair.py out \
  --detector-report <fresh-112-system-scan>/report.json \
  --output-dir <new-output-directory>
```

The earlier two-score gate commands were:

```bash
uv run python scripts/experiments/freeze_independent_key_state_gates.py out
uv run python scripts/experiments/run_independent_key_state_gate.py \
  out/jaime-llanos_41_estrella-del-caribe_danza_luis-a-calvo/vlm_melody_independent_key_gate/v1_estrella_initial_s3/system_003/prepared_manifest.json
uv run python scripts/experiments/run_independent_key_state_gate.py \
  out/jaime-llanos_92_sobre-el-humo_bambuco_fulgencio-garcia/vlm_melody_independent_key_gate/v1_sobre_change/system_007/prepared_manifest.json
uv run python scripts/experiments/evaluate_independent_key_state_gates.py out
```

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

### Frozen Alcira New-PDF Key-Change Gate

Alcira system 6 is the first new-PDF key-state target acquired after the
112-system consumed sweep. Before any MusicXML existed, system segmentation was
fixed and manually approved, then measure cleanup reduced ten false-heavy
slices to six coherent crops. The strict visual detector selected two sharp
glyphs and pinned `fifths=+2`. Because the changed signature occurs immediately
after the system's leading structural bar, the gate uses the left-edge
signature detector but replays the key state only after x=328, beyond both
selected glyphs.

One score-disjoint selector pass produced 18 anchors across six crops. The
no-key and automatic-key lanes have identical candidate IDs, coordinates, and
counts. The seal was written with `truth_accessed=false` and status
`frozen_awaiting_truth`:

- `out/local_restricted/jaime-llanos_5_alcira_bambuco_oriol-rangel/vlm_melody_independent_key_gate/v1_alcira_system_entry_change_s6/system_006/frozen/sealed_manifest.json`

Do not rerun or overwrite the prepared, inference, or frozen directories. The
six-measure transcription preserves the visible noteheads, including dyads,
rests, and durations, and encodes the two-sharp key signature in measure 1. It
is saved at:

- `out/local_restricted/jaime-llanos_5_alcira_bambuco_oriol-rangel/vlm_melody_independent_key_gate/v1_alcira_system_entry_change_s6/system_006/alcira_system_006.musicxml`

The create-once evaluator verified the seal before opening MusicXML. With the
same 18 selected candidates in both lanes, the strict automatic-key lane raises
exact ordered-pitch matches from `11` to `16` and ordered-pitch alignment
accuracy from `0.323529` to `0.470588`. The MusicXML confirms treble clef,
`3/4`, six physical measures, and one `fifths=+2` key event in measure 1.

Absolute note-count F1 is `0.666667` (`18` selected anchors versus `33` visible
MusicXML noteheads) because several source events are dyads while the frozen
selector often emits one melodic anchor. This limits the absolute transcription
score but does not confound the paired key-state delta: candidate IDs,
coordinates, and counts are identical. The gate therefore passes for strict
initial/system-entry key state only. Internal double-bar changes remain outside
this evidence and must continue to fail closed.

Create-once evidence is at:

- `out/local_restricted/jaime-llanos_5_alcira_bambuco_oriol-rangel/vlm_melody_independent_key_gate/v1_alcira_system_entry_change_s6/system_006/evaluation_v1/`

The bounded integration is now implemented behind the opt-in
`--key-context strict-visual` input-builder switch. It scans systems in score
order even when only a later system is selected, carries forward only a prior
accepted one-flat/two-sharp state, and applies a newly confirmed state only
after the detector's absolute system-x boundary. The benchmark request records
that crop origin, so candidate pitch mapping can resolve the boundary without
changing candidate IDs, coordinates, or counts.

The default metadata mode preserves its previous record shape. A portable
four-case acceptance fixture covers one flat, two sharps, an ordinary
accidental, and a title/non-staff rejection. A disposable real-Alcira smoke
produced six system-6 records with inherited `-1`, confirmed `+2`, and boundary
x=328 while leaving the sealed gate untouched. Internal double-bar changes and
unsupported signature counts remain out of this path.

The next main error is no longer this bounded key-state handoff. Alcira still
has 18 selected anchors for 33 visible MusicXML noteheads because several
events are dyads. The next spike should therefore isolate chord/dyad head
recovery with localization and key context held fixed, while leaving this key
mode opt-in until broader exact-event/rest evidence exists.

### Frozen No lo Creas Polyphonic Gate

The consumed Alcira/La Chata result selected one fixed recovery rule before a
new target was inspected: add at most one score-qualified, stem-supported
companion to an existing x-group, reject the crop's leading staff-space, and
never remove or reposition the baseline candidate. No lo Creas system 8 was
then selected as a new score-disjoint target because it contains repeated
stacked hollow-note chords and passes the existing automatic layout policy.

Before any target MusicXML or melody truth existed, the gate sealed eleven
automatic crops and replayed the provenance-refreshed score-disjoint selector.
The paired artifact keeps the generic canonical prediction for audit, but maps
both comparison lanes from the same frozen staff coordinates with natural
treble-clef diatonic positions. This isolates the recovery rule from key-state
and historical pitch-predictor differences. Meter, rhythm, duration, rests,
chromatic key, and accidentals are explicitly unsupported.

The fixed rule adds two companions, both in automatic crop 1. It preserves all
baseline candidate IDs and coordinates, reuses the two existing onset-group
identities, and creates no new x-group. Visual inspection of the frozen overlay
shows both additions on plausible stacked noteheads sharing the selected stems.

The finalized seven-measure MusicXML was initially evaluated with an explicit mapping
from eleven automatic crops. A post-evaluation audit found that the first physical
measure had been split `9 + 3` notes across automatic crops 1 and 2, while the frozen
x-groups show two onset groups in each crop. The correct split is `6 + 6`. The original
`evaluation_v1` remains immutable evidence of the mistake; it is superseded for
interpretation by create-once `evaluation_v2_mapping_erratum`.

With the corrected mapping, note-count F1 still improves `0.584616 -> 0.626865`,
recall `0.463415 -> 0.512195`, and precision `0.791667 -> 0.807692`. Exact natural
diatonic staff-position matches remain `6 -> 6`, chord-size alignment remains
`0.166667`, exact chord-size matches remain `4 -> 4`, and exact structure crops remain
`0`. The preregistered promotion condition therefore still fails, now without the
incorrect pitch-improvement claim. The one-companion rule remains spike-only.

Frozen artifacts are under:

- `out/local_restricted/jaime-llanos_73_no-lo-creas_pasillo_a-vasquez-pedrero/vlm_melody_independent_dyad_recovery_gate/v1/system_008/`
- seal: `frozen/sealed_manifest.json`
- paired overlay: `baseline_inference_v1/dyad_recovery_v1/overlays/measure_001.png`

Do not rerun or overwrite the prepared, inference, recovery, frozen,
`evaluation_v1`, or `evaluation_v2_mapping_erratum` directories. The evaluated
transcription is:

- `out/local_restricted/jaime-llanos_73_no-lo-creas_pasillo_a-vasquez-pedrero/vlm_melody_independent_dyad_recovery_gate/v1/system_008/no_lo_creas_system_008.musicxml`

The corrected authoritative post-freeze crop mapping is
`no_lo_creas_system_008_mapping.json`; it maps the seven physical measures across
the eleven automatic crops. The original create-once evaluator command was:

```bash
uv run python scripts/experiments/evaluate_independent_dyad_recovery_gate.py \
  out/local_restricted/jaime-llanos_73_no-lo-creas_pasillo_a-vasquez-pedrero/\
vlm_melody_independent_dyad_recovery_gate/v1/system_008/frozen/sealed_manifest.json \
  --musicxml out/local_restricted/jaime-llanos_73_no-lo-creas_pasillo_a-vasquez-pedrero/\
vlm_melody_independent_dyad_recovery_gate/v1/system_008/no_lo_creas_system_008.musicxml \
  --mapping out/local_restricted/jaime-llanos_73_no-lo-creas_pasillo_a-vasquez-pedrero/\
vlm_melody_independent_dyad_recovery_gate/v1/system_008/no_lo_creas_system_008_mapping.json
```

The original result is preserved under `evaluation_v1/report.json`. The mapping-only
correction is preserved under `evaluation_v2_mapping_erratum/report.json` and was
created by `supersede_independent_dyad_mapping_erratum.py`, which verifies every
frozen and source hash before rescoring. Do not rerun either result under the same or
a replacement version to search for a more favorable outcome.

The create-once erratum command shape is:

```bash
uv run python scripts/experiments/supersede_independent_dyad_mapping_erratum.py \
  <sealed-manifest> \
  --prior-evaluation <evaluation-v1-manifest> \
  --mapping <corrected-mapping.json>
```

### Consumed Multi-Head Chord Recovery Selection

The corrected No lo Creas result showed that adding one companion was too narrow to
recover the visible stacked chords. A bounded follow-up grid therefore tested a
multi-head variant on consumed Alcira, La Chata, corrected No lo Creas, and the
Aviador/Carrizal candidate reviews. The rule keeps the same onset groups, requires a
stem-supported vertical chain, fails closed near the crop's leading edge, and caps
recovery at two companions per group.

All eight preregistered parameter combinations produced the same scored behavior.
The selected conservative configuration is:

```text
minimum_y_gap_staff_spaces=1.0
maximum_y_gap_staff_spaces=3.0
minimum_score_ratio=0.5
minimum_stem_score=0.55
minimum_group_x_staff_spaces=1.0
maximum_recovered_heads_per_group=2
```

Against the corrected No lo Creas evidence, it improves over the previous dyad lane:

- note-count F1 `0.626865 -> 0.666667`;
- exact natural-diatonic staff-position matches `6 -> 8`;
- exact chord-size matches `4 -> 6`;
- chord-size alignment `0.166667 -> 0.25`;
- exact structure crops `0 -> 1`.

It also preserves or improves the consumed development sets: Alcira exact pitch
matches improve `16 -> 19` with note-count F1 `0.666667 -> 0.827586`; La Chata exact
pitch groups improve `12 -> 17`; and Aviador/Carrizal candidate F1 remains exactly
`0.791367` with zero recovered false positives.

The create-once result is:

- `out/vlm_melody_consumed_training/multihead_chord_recovery_v1/report.json`
- `out/vlm_melody_consumed_training/multihead_chord_recovery_v1/report.md`

A new output directory can reproduce the model-selection sweep:

```bash
uv run python scripts/experiments/spike_consumed_multihead_chord_recovery.py \
  out --output-dir <new-dir>
```

This does not make the rule runtime-eligible. The remaining untouched scores in the
current corpus are monophonic, so they cannot provide a positive score-disjoint chord
recovery gate. The next valid step is to freeze this exact selected configuration on
a genuinely unseen polyphonic score or system before its transcription is opened.
