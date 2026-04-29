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
- [ ] Harden chord measure alignment: debug/visualize detected barlines, avoid accidentals/stems, handle final right-edge barlines, and improve global measure offsets.
- [x] Add first melody-backend MusicXML integration slice: staged MusicXML → `melody.json` + `events.json`.Add first melody-backend MusicXML integration slice: staged MusicXML → `melody.json` + `events.json`.
- [ ] Wire Melody Engine A OMR output to produce `intermediate/musicxml.xml` automatically.
- [ ] Implement musical validation/repair (meter enforcement, quantization).

Next main M2 focus: automatic Melody Engine A OMR output wiring, then musical validation/repair.

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
