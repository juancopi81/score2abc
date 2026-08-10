# Roadmap

This roadmap is milestone-based with checkboxes so progress can be tracked directly in this file. It prioritizes accuracy and a reproducible CLI pipeline before UI polish.

## M0 — Thin End-to-End Slice (baseline flow)

Goal: prove the full path PDF → ABC → preview → export works, even if accuracy is low.

- [x] Create CLI skeleton (`score2abc ingest/run/qa/export`) that wires the steps together.
- [x] Define a `WorkItem` manifest (metadata + pdf_path + slug) and iterate it in the runner.
- [x] PDF → image rendering at fixed DPI, store in `out/<slug>/pages/`.
- [x] Minimal staff/system crop (even if crude) saved to `out/<slug>/systems/`.
- [x] Stub `events.json` → ABC generator with a tiny sample event list.
- [x] Render ABC to SVG/PNG preview and store in `out/<slug>/final/`.
- [x] Export `out/index.md` catalog with metadata + ABC block.

Done when: a single PDF produces `melody.abc`, `melody_with_chords.abc`, preview, and is listed in `out/index.md`.

## M1 — Evaluation Harness + Dataset Plumbing

Goal: make accuracy measurable on the 10-PDF dataset.

- [x] Formalize dataset loader for `dataset/metadata.csv` (validate required fields).
- [x] Add a run log per work + a top-level summary report.
- [x] Add stage-level `stage.json` artifacts (inputs, params, hashes) for resume/caching.
- [x] Define accuracy metrics (note accuracy, chord accuracy, meter validity) and how to compute them.
- [x] Create a small labeled subset format (ground-truth ABC or events).
- [x] Add a “compare vs ground truth” script/report.
- [x] Add tests for manifest parsing/slugging, ABC formatting, and metrics/comparison code.

Done when: you can run an evaluation command that produces a numeric report for the dataset.

## M2 — Accuracy Lifts: Segmentation + Melody + Chords

Goal: improve recognition quality with prioritized de-risking.

- [x] Staff/system detection with robust line-spacing estimation.
- [x] Page-level deskew with row-peakiness estimator; chord bands overlap staff so glyphs nestled against outer staff lines aren't clipped.
- [ ] Additional preprocessing (contrast, denoise) if VLM/OMR readability demands it — deferred until recognition step drives the need.
- [x] Chord extraction via VLM on the above/below annotation bands → normalized symbols + measure alignment. _(chord-first pivot: validate VLM path on the easier target before investing in melody OMR; V0 uses a measure-only chord F1 metric and known-limited barline alignment)_
- [x] Harden chord measure alignment: debug/visualize detected barlines, avoid accidentals/stems, handle final right-edge barlines, and improve global measure offsets. _(staff-aware run-length detector with horizontal-drift tolerance; aggregate F1 0.903 on labeled aviador systems vs. 0.20 baseline; `measures_in_system` now subtracts leading and terminal barlines so global offsets stay correct.)_
- [x] Add first melody-backend MusicXML integration slice: staged MusicXML → `melody.json` + `events.json`.
- [x] Wire pipeline contract for `intermediate/musicxml.xml` via a `MusicXMLBackend` protocol and an `extract_musicxml` stage backed by a fixture backend (`dataset/musicxml/<slug>.musicxml`); manual drops still work, corrupt fixtures fail the work item.
- [x] Spike external MusicXML OMR backends behind the `MusicXMLBackend` protocol: optional `homr` and Audiveris CLI adapters, rendered/deskewed/system-collage input modes, MusicXML validation, and `Aviador` smoke/eval runs.
- [ ] Replace the fixture MusicXML backend with a real recognition path that produces `intermediate/musicxml.xml` from rendered pages or system crops. Current homr/Audiveris results are useful as a benchmark harness, but not accurate enough to become the primary melody path.
- [x] Add first local VLM melody-input builder: split detected system crops into measure-level raw/staff/overlay crops plus JSONL context, with no model calls or budget usage.
- [x] Prototype cost-capped VLM-assisted melody extraction on measure/system crops, including raw/staff/pitch-ruler inputs, candidate-assisted localization, and neighboring-measure context. The best context arm fixed one development measure but failed both blind measures; direct VLM transcription is not ready for pipeline integration. See `docs/VLM_MELODY_SPIKE.md`.
- [x] Test the notehead-selection gate without held-out leakage, then move to candidate-confirming HITL after affine rectification and a candidate-ID-only VLM both miss the fixed automatic-selection gate. The local reviewer preserves full proposal snapshots and corrections without exposing GT during review.
- [x] Run a timed GT-blind review of Aviador system 1 measures 1-4 and promote the result into portable training fixtures. Selection and final pitch reached `14/14`; automatic natural pitch reached `11/14`. The timing gate failed (43-second median), so the current reviewer is a labeling tool rather than the final product workflow.
- [x] Add a leak-resistant melody-event benchmark with frozen image requests and explicit development (systems 1-2), validation (systems 7-8), sealed heldout (system 3), and quarantined ambiguous systems.
- [x] Prove a visual rhythm/rest upper bound from reviewed notehead anchors. On system 1 measures 1-4, the pixel parser recovered `14/14` durations, the leading rest, and `4/4` exact measures. Automatic notehead selection remains the bottleneck.
- [x] Compose the strongest local selector with the rhythm parser and test it without truth-derived inference. The fixed-count arm failed validation (`0.056` note F1), while the variable-threshold arm improved validation to `0.264` note F1 and `0.446` ordered pitch. Its preregistered gate opened system 3 once; heldout note F1 was only `0.031`, so the path remains spike-only.
- [x] Promote the blind system-7 review slice (`35/35` ordered pitches across seven measures) and retrain the local selector on systems 1+7. Dense proposals cover all `49/49` reviewed heads; a narrow meter-gap resolver raises frozen system-8 strict note F1 from `0.264` to `0.426`, rest F1 from `0` to `0.333`, and produces the first exact automatic measure. System 8 is consumed model-selection evidence, not heldout proof.
- [x] Run the independent Carrizal second-score gate without changing frozen predictions. The transcription exposed eight physical measures in seven automatic crops (crop 2 merged measures 2+3). Segmentation-aware one-shot scoring reached note F1 `0.326`, ordered pitch `0.269`, rest F1 `0.25`, and `0/7` exact crops, so automatic pipeline integration remains blocked.
- [x] Repair and benchmark the Carrizal system-4 barline regression: system 4 now scores `TP=9 FP=0 FN=0`, while Aviador remains `TP=69 FP=7 FN=1`, F1 `0.945`.
- [x] Consume Carrizal as cross-score training material and select configuration C: macro notehead F1 `0.706944`, conditional pitch accuracy on matched noteheads `0.813910`, end-to-end correct-pitch recall `0.601531`, coordinate exactness `0.136364`, and coordinate-plus-pitch exactness `0.090909`. The 8-crop v2 adjudication has 20 heads (`18` high, `2` medium; `14` candidate, `6` manual) and is agent-reviewed, not human-promoted.
- [x] Freeze and evaluate La Chata system 7 as the third-score heldout gate. The authoritative v2 was sealed before transcription and scored note-count F1 `0.901408`, ordered-pitch accuracy `0.435897`, and `1/7` exact crops. Rhythm/rest remain unscored because frozen metadata lacked time/key context; this score is now consumed evidence.
- [x] Run a create-once La Chata polyphony/key postmortem and consumed-score regression audit. Stateful one-sharp context improves exact ordered pitches `17 -> 20`; unfiltered chord recovery improves context-aware exact pitch groups `12/25 -> 17/25` but overselects `46/37` heads. A candidate-local stem filter narrows this to `16/25` and `39/37`, but regresses consumed Aviador/Carrizal candidate F1 `0.791367 -> 0.785714`, so no recovery arm is runtime-eligible.
- [x] Add the first automatic visual key-change slice: a conservative double-bar-gated detector finds La Chata's explicit one-sharp change, emits exact `fifths=1`, and propagates it statefully. The consumed replay improves exact ordered pitches `17/37 -> 20/37` and exact pitch groups `11/25 -> 12/25` with identical head selection. This supports one sharp after a double bar only; it is not a general key-signature recognizer.
- [x] Expand the consumed visual key detector to initial state, flats, and multiple accidentals. The bounded report is `6/6` on initial one-flat, changed one-sharp/two-sharp/two-flat, and two double-bar controls; a change-mode scan fires only on the three actual changes across 89 consumed crops. Generalization beyond these handwritten patterns remains unproven.
- [x] Compose the expanded key detector with frozen pitch replay using work-scoped, truth-free events. On consumed La Chata the expanded report preserves identical head selection while improving exact ordered pitches `17/37 -> 20/37` and exact pitch groups `11/25 -> 12/25`, matching the earlier one-sharp-only replay without evaluation labels driving prediction.
- [x] Replace automatic false-onset deletion with a non-mutating meter-deficit review signal. Candidate-level deletion was also rejected: recall-first is a no-op and balanced thresholds regress aggregate onset F1. The review validator catches `8/10` consumed Aviador/Carrizal error measures with no false alerts, but its La Chata count-only replay catches only `3/6` with one false alert, so runtime integration remains blocked.
- [x] Reject non-musical system proposals before measure segmentation. A truth-blind five-line eligibility gate now preserves rejected crops/reasons and accepted-to-source numbering. Fresh regression keeps all 10 Aviador systems, keeps Carrizal systems 1-11 while rejecting its terminal incomplete four-line ruled tail, and maps La Chata's old musical systems 2-11 to new systems 1-10 while rejecting only the title/author band. Every accepted crop is pixel-identical to its prior musical counterpart; frozen heldout artifacts were not rewritten.
- [x] Prepare, freeze, and evaluate Gato'e Fique system 3 as a fourth-score truth-blind gate. Six automatic crops, configuration-C predictions, the automatic initial one-flat result, and a provisional `Pasillo -> 3/4` metadata prior were hash-sealed before MusicXML access. Independent scoring reached note-count F1 `0.941177`, ordered-pitch accuracy `0.222222`, and `0/6` exact crops. The transcription has two sharps; a consumed key-corrected replay preserves selected heads while raising ordered-pitch accuracy to `0.666667` and exact pitch groups from `5/24` to `16/24`.
- [x] Generalize initial key-signature recognition using the independently exposed Gato'e Fique miss, then add a review-time key correction/replay path. Ordered glyph-sequence matching now distinguishes its two sharps while retaining all six prior consumed signature/control cases (`7/7` total), and the 89-crop change scan still emits only the same three real changes. The create-once correction action replays pitch from explicit measure-indexed fifths while asserting candidate IDs, coordinates, counts, and rhythm are unchanged; the sealed heldout score remains untouched.
- [x] Complete the Coqueteos system-2 fifth-score independent gate. Six automatic crops and full event predictions were sealed before the seven-measure MusicXML was opened. Whole-measure scoring reached note F1 `0.363636`, ordered pitch `0.375`, onset `0.28125`, duration `0.59375`, rest F1 `0`, and `0/6` exact crops. The pre-truth meter triage flags three genuine error crops with no false alerts but misses the other three. Transcription review exposed a false leading stem and missed x=2041 barline; the postmortem correction recovers all seven boundaries while changing no other generated system and preserving Aviador F1 `0.945`.
- [x] Replay the exact frozen Coqueteos recognizer on corrected one-to-one segmentation without rewriting heldout evidence. Meter-valid crops improve `5/6 -> 7/7`, but note F1 regresses `0.363636 -> 0.280702`; segmentation is retained for structural correctness, not claimed as recognition gain. Corrected meter triage flags `4/7` genuine error measures at precision `1.0`, and the unreviewed proposal queue finds pitch-compatible candidates for `27/31` expected notes. The queue remains ineligible for training pending coordinate review.
- [x] Materialize complete human candidate reviews for the three highest-value Coqueteos measures and spike a GT-free hollow-notehead center proposer. Across 22 consumed reviewed measures from Aviador, Carrizal, and Coqueteos, the fixed rule adds five proposals, recovers five previously missed reviewed centers, and adds zero false proposals. This is consumed model-selection evidence; runtime integration remains blocked on an unseen-score gate.
- [x] Complete the preregistered Chispazo system-4 hollow-notehead gate. Eight raw-only reviews contain two hollow heads, both matched by the existing candidate set (`2/2` baseline recall). The frozen consumed rule emits no proposals, leaving zero recovery opportunities; the sample is also below the five-head promotion minimum. The gate is therefore `not_promoted`: this is clean heldout evidence to keep the rule out of runtime, not evidence that the rule regressed transcription.
- [x] Build a portable cross-score error breakdown from the four independent score gates. After excluding the two merged crops from root-cause ranking, the clean one-to-one count-capacity F1 upper bound is `0.878788`, but conditional exact pitch is only `0.413793` (`51/87` count-alignable notes wrong). Candidate coverage is not identifiable from these reports, onset is `0.483871` on the two-score full-event subset, and unsupported rhythm fields remain explicit. The next engineering target is pitch mapping and key context with note identities and coordinates frozen.
- [x] Isolate automatic key state from staff geometry on the 24 clean consumed crops. With candidate IDs, coordinates, and note counts fixed, automatic key state raises exact ordered-pitch matches `40/87 -> 55/87` (`+12` Gato'e Fique, `+3` La Chata, no component-level regressions). Local common-shift staff tracking adds `0` matches. The key component passes the consumed development gate but remains out of runtime pending two new independent scores; the geometry component is rejected.
- [x] Prepare and seal two independent automatic-key-state gates before truth. Estrella del Caribe system 3 pins an initial one-flat prediction from system 1 (`6` crops, `24` selected heads); Sobre el Humo system 7 pins a four-flat double-bar change (`7` crops, `31` selected heads). Both gates use one selector pass and prove candidate IDs, coordinates, and counts identical across no-key and automatic-key lanes. Estrella system 1 was sealed but rejected during pre-truth visual QA because its first crop was clef/key preamble; it is not evaluation evidence. The authoritative two gates now await human MusicXML.
- [ ] Generalize the onset validator on independent evidence before connecting it to the pipeline. It must preserve transcription outputs and improve review prioritization without La Chata-style false alerts.
- [ ] Implement musical validation/repair (meter enforcement, quantization).

Next main M2 focus: transcribe the two sealed independent systems, then evaluate
the already-frozen no-key and automatic-key lanes. Require better exact ordered
pitch on both scores without localization drift or a score-level regression;
otherwise keep automatic key state out of runtime. Do not carry the local
common-shift staff tracker into that gate: it added no pitch matches on the
consumed comparison. The Chispazo hollow-head rule and meter-based note
deletion remain out of runtime. Pipeline integration remains gated on
independent exact-event/rest evidence. See `docs/VLM_MELODY_SPIKE.md`.

Done when: evaluation shows clear improvement over M1 baseline and meter validity is 100% after repair.

## M3 — Human-in-the-Loop (HITL) Editing Loop

Goal: easy correction for remaining errors with patches fed back into the dataset.

- [ ] Streamlit (or equivalent) UI showing: PDF/crop, current ABC, rendered score, metadata.
- [ ] Optional MIDI playback of the rendered ABC.
- [ ] Editing panel for ABC + chord symbols with measure focus.
- [ ] Save edits as patches/overrides and revalidate.
- [ ] Define `review/` bundle outputs and `overrides/patches.json` ingestion path.

Done when: a user can correct a flagged work end-to-end in minutes and the corrections persist.

## M4 — QA Scoring + Review Prioritization

Goal: focus human time where it matters most.

- [ ] Render & compare scoring (visual similarity + heuristic flags).
- [ ] Confidence thresholds to mark “needs review”.
- [ ] Ranked review queue in `out/index.md` or a separate report.

Done when: the system produces a prioritized list of measures/works to review with useful flags.

## M5 — Export & Dataset Polish

Goal: stable outputs suitable for downstream usage.

- [ ] Finalize output formats and naming conventions.
- [ ] Ensure ABC exports are consistent and reproducible.
- [ ] Document dataset packaging and versioning.

Done when: a full dataset run produces clean, versionable outputs without manual cleanup.

---

## Parallel Work (Optional)

If you want to split efforts, these can happen alongside M2–M4:

- [ ] CLI plumbing + logging (M0/M1) can progress while segmentation/OMR spikes are prototyped.
- [ ] HITL UI skeleton can be built early using placeholder ABC/preview assets.
- [ ] Evaluation harness can be built before perfect recognition, so improvements are measurable.
- [ ] Add per-work parallelism once stage I/O contracts are stable.
