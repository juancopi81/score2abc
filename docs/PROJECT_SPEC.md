# Handwritten Colombian Scores → ABC Transcription Pipeline (Project Spec)

## 1) Overview

This project converts a folder of PDF scores into a clean, searchable dataset where each work includes:

- **ABC notation** for the **single melodic line**
- **Chord symbols** (traditional “cifrado” above the staff)
- **Metadata** (title, composer, genre/rhythm, source file, page/system positions)
- **QA artifacts** (rendered notation previews + confidence/flags)

The target material is **handwritten / low-quality scans** of Colombian folk/Andean repertoire: one melody staff + chord symbols.

---

## 2) Goals

### Product goals

- Batch-process **N PDFs** (one PDF = one work).
- Produce **high-accuracy ABC** for the melody and **clean chord symbols** aligned to measures/beats.
- Create outputs that are:
  - easy to read
  - easy to version-control
  - easy to load into other tools (MIDI rendering, MusicXML conversion, web player)

### Engineering goals

- **Robustness > cleverness**: handle noisy scans using preprocessing + validation + human review for only the hard parts.
- Use Python as the primary implementation language.
- Allow optional use of **commercial VLMs** (e.g., GPT-4o / Gemini) for **chord OCR** and **ambiguity resolution**, but avoid relying on them for full free-form transcription.

---

## 3) Non-goals (for v1)

- Multi-voice polyphony or piano-style grand staff.
- Lyrics extraction.
- Perfect engraving reproduction (we only need correct pitch/duration/chords).
- Fully “hands-off” transcription with zero review (we aim for minimal review).

---

## 4) Inputs

### Input manifest (canonical structured input)

The pipeline consumes a **manifest** of works. Each entry is a `WorkItem`
containing metadata + source file path + output slug. This makes the pipeline
deterministic and allows batching, caching, and parallel execution.

Example (JSON lines or list):

```json
{
  "slug": "jaime-llanos_01_acuata_pasillo_garcia",
  "pdf_path": "dataset/acuatA.pdf",
  "metadata": {
    "title": "Acuata",
    "composer": "Fulgencio García",
    "rhythm": "Pasillo",
    "time_signature": "3/4",
    "key_hint": "Em"
  }
}
```

### Required input folder

- `input_dir/`
  - `*.pdf` (each PDF is one work; may contain 1+ pages)

### Current dataset (v0)

- `dataset/` contains **10 PDFs** with clean metadata at `dataset/metadata.csv`.
- **Current assumption:** each PDF has **one page** and contains **one song**.

### Golden dataset (current source PDFs)

The current `dataset/` folder is a golden dataset derived from the manuscript
transcriptions of Colombian organist Jaime Llanos Gonzalez, later photocopied
and shared by triplista Jairo Rincon Gomez (June 2002) with the music library of
Universidad de Antioquia. The photocopies are low-legibility and in fragile
condition, so preservation and accurate transcription are a priority.

#### Filename convention (when possible)

`jaime-llanos_<num>_<titulo>_<genero>_<autor>.pdf`

Normalization rules:

- lowercase
- ASCII only (accents removed)
- words separated by hyphens
- punctuation removed (e.g., "Gato'e" -> `gatoe`)
- author last names separated by hyphens

If a future source cannot be normalized cleanly, keep its original filename and
use the metadata table as the source of truth for title/composer/genre.

### Metadata input (provided by you)

A cleaned table (CSV/JSON) keyed by PDF filename or title:

- `title`
- `composer`
- `rhythm/genre` (pasillo, bambuco, danza, joropo, etc.)
- Optional: `time_signature`, `key_hint`, `tempo_hint`

Example `metadata.csv`:

```csv
pdf_file,title,composer,rhythm,time_signature,key_hint
acuatA.pdf,Acuata,Fulgencio García,Pasillo,3/4,Em
...
```

For the current golden dataset, the canonical table lives at `dataset/metadata.csv`.

### Labeled subset (ground truth)

For evaluation, place labeled events under `dataset/ground_truth/` using the
work slug as the filename: `dataset/ground_truth/<slug>.json`.

---

## 5) Outputs (Recommended Format)

### Output folder structure

For each work: `out/<slug>/`

- `source.pdf` (copied)
- `metadata.json`
- `stages/` (per-stage inputs/params/hashes for resume/caching)
- `pages/` (rendered page images)
- `systems/` (cropped staff systems + chord region crops)
- `intermediate/`
  - `musicxml.xml` (if produced by an OMR engine)
  - `events.json` (canonical note/chord event representation)
  - `chords.json` (chord symbols + positions/confidence)

- `final/`
  - `melody.abc`
  - `melody_with_chords.abc`
  - `preview.svg` (rendered from ABC)
  - `preview.png` (optional)

In `melody_with_chords.abc`, chords are embedded as ABC chord annotations (e.g., `"Em"c2 "B7"d2 |`).

- `qa/`
  - `report.json` (scores, warnings, flags)
  - `overlay.png` (optional: visual diff/overlay)
- `review/` (bundle of crops + previews + ABC for human review)
- `overrides/`
  - `patches.json` (human edits applied to the canonical events)

### Top-level index

- `out/index.md` — a human-readable catalog containing for each work:
  - Title / Composer / Rhythm
  - Links to outputs
  - Embedded ABC block

---

## 6) Canonical Data Model

All recognition outputs are normalized into `events.json`:

### Notes

```json
{
  "measure": 1,
  "onset_beats": 0.0,
  "duration_beats": 0.5,
  "pitch_midi": 64,
  "accidental": null,
  "tie": false
}
```

### Chords

```json
{
  "measure": 1,
  "onset_beats": 0.0,
  "symbol": "Em"
}
```

This model is the source of truth. ABC is generated from it.

---

## 7) System Architecture

### Pipeline I/O contracts (per work)

Each stage is deterministic and writes outputs under `out/<slug>/...`. A stage
may be skipped if its inputs and parameters have not changed (resume/caching).
Each stage writes `stages/<stage>.json` with inputs, params, and hashes.

1. **Load** → copy `source.pdf`, write `metadata.json`, render `pages/`
2. **Preprocess** → write `pages/*_enhanced.png` (or variants)
3. **Segment** → write `systems/*/system_image.png` + `systems/*/chord_region.png`
4. **Recognize (melody)** → write `intermediate/musicxml.xml` (if applicable)
5. **Normalize** → write canonical `intermediate/events.json`
6. **Chord OCR** → write `intermediate/chords.json` (symbols + positions/confidence)
7. **Validate/Repair** → write updated `events.json` + `qa/report.json`
8. **ABC Generation** → write `final/melody.abc` + `final/melody_with_chords.abc`
9. **QA Render/Score** → write `final/preview.svg` + `qa/flags.json`
10. **Review Bundle** → write `review/` package for human review
11. **Apply Overrides** → read `overrides/patches.json`, re-run validate/export
12. **Export Catalog** → update `out/index.md`

**Stage I/O quick table (per work)**

| Stage            | Inputs                  | Outputs                                                    |
| ---------------- | ----------------------- | ---------------------------------------------------------- |
| Load             | manifest entry + PDF    | `source.pdf`, `metadata.json`, `pages/*`                   |
| Preprocess       | `pages/*`               | `pages/*_enhanced.png` (variants)                          |
| Segment          | preprocessed pages      | `systems/*/system_image.png`, `systems/*/chord_region.png` |
| Melody OMR       | system crops            | `intermediate/musicxml.xml` (if applicable)                |
| Normalize        | MusicXML                | `intermediate/events.json`                                 |
| Chord OCR        | chord crops             | `intermediate/chords.json`                                 |
| Validate/Repair  | events + chords         | updated `events.json`, `qa/report.json`                    |
| ABC + QA         | events                  | `final/*.abc`, `final/preview.svg`, `qa/flags.json`        |
| Review/Overrides | review bundle + patches | patched outputs + regenerated ABC/QA                       |

### High-level pipeline

1. **PDF → images** (high DPI)
2. **Image preprocessing** (deskew, contrast normalize, denoise)
3. **Staff/system detection** (segment page into staff systems)
4. **Melody recognition** (OMR engine(s) + normalization)
5. **Chord symbol extraction** (OCR/VLM on chord region)
6. **Musical validation & repair** (meter checks, quantization, key consistency)
7. **ABC generation**
8. **QA render & scoring** (render ABC, compare to source, flag uncertain measures)
9. **Human review loop** (only for flagged measures)
10. **Finalize + export catalog**

---

## 8) Core Engineering Decisions

### 8.1 Preprocessing (critical for accuracy)

Use a deterministic OpenCV/scikit-image preprocessing stack:

- deskew / dewarp (start with deskew; add dewarp if needed)
- adaptive contrast (e.g., CLAHE)
- denoise (median/bilateral)
- binarized variant (adaptive threshold)
- keep multiple representations: `gray`, `enhanced`, `binary`

Store preprocessing parameters in `metadata.json` to reproduce runs.

### 8.2 Staff/system segmentation (avoid brittle heuristics)

Implement a staff-line detection module to estimate:

- staff line spacing
- staff angle
- staff region bounding boxes

Output: per-page list of staff system crops, each with:

- `system_image.png`
- `chord_region.png` (strip above staff for chord OCR)

### 8.3 Multi-engine melody recognition

Use **at least two** recognition strategies:

- **Engine A**: a deep OMR model (e.g., `oemer`/`homr` producing MusicXML)
- **Engine B**: a fallback engine OR a second pass model / different preprocessing view

Do **not** trust raw output directly:

- always convert to canonical `events.json`
- always run validation & repair

### 8.4 Chord extraction (VLM-friendly and high ROI)

Chord symbols are often easier than notes for a VLM:

- crop only the chord strip above the staff
- prompt strictly: “extract chord symbols with approximate x-position ordering”
- normalize symbols:
  - `Em`, `Emin`, `E-` → `Em`
  - `B7`, `B 7` → `B7`
  - handle slash chords if present (`D/F#`)

Align chords to measures by x-position mapping:

- detect barlines (vertical lines) in system crop
- map each chord’s x coordinate → measure index
- optional: beat alignment if chord placement is clear; otherwise store as “measure-level”

### 8.5 Musical validation & repair (the accuracy backbone)

Implement a “music constraints” module using `music21`-style logic:

- **Meter enforcement**: each measure must sum to `time_signature`
- **Quantization**: snap durations to allowed set (e.g., {1/8, 1/4, 3/8, 1/2} in beats) with penalties
- **Key consistency**: infer likely key from pitch histogram + key_hint; reduce accidental noise
- **Tie handling**: allow ties crossing barlines if needed
- **Outlier detection**: flag improbable leaps / too many accidentals for the style

Any repair must be logged in `report.json` with before/after.

### 8.6 QA scoring and “render & compare”

For each system (or measure):

- render ABC → SVG/PNG (`abc2svg`, `abcm2ps`, or equivalent)
- compute a similarity score against the source crop:
  - edge-based similarity
  - structural match heuristics (notehead density vs positions)

- emit flags:
  - “bar count mismatch”
  - “meter fix applied”
  - “low visual similarity”
  - “chord OCR low confidence”

### 8.7 Human review loop (minimal but decisive)

Provide a lightweight review UI (Streamlit):

- shows: source PDF/crop, rendered preview, current ABC, metadata, and flags
- optional: MIDI playback of the rendered ABC
- allows:
  - selecting among 2–3 candidate hypotheses (from multi-engine outputs)
  - editing ABC for only flagged measures
  - editing chord symbols list

Edits are saved as patches/overrides in `overrides/patches.json` and re-validated
so the corrected ABC becomes part of the dataset.

---

## 9) CLI / Developer Experience

### CLI commands

- `score2abc ingest <input_dir> <metadata.csv> <out_dir>`
- `score2abc run <out_dir> [--workers N] [--use-vlm]`
- `score2abc qa <out_dir> [--open-ui]`
- `score2abc eval <out_dir> --ground-truth <dir>`
- `score2abc export <out_dir> --format index.md`

### Configuration

- `config.yaml` for:
  - DPI, preprocessing params
  - time signatures per rhythm default
  - model selection (OMR engine A/B)
  - VLM provider + prompt templates
  - thresholds for QA flags

### Logging

- structured JSON logs per work + a global run log
- store timing per stage for profiling

### Batch runner + parallelism

The CLI should iterate a manifest of `WorkItem`s and run the per-work pipeline.
Start with sequential execution for debuggability. When stable, add per-work
parallelism (thread pool for I/O-bound steps, process pool for CPU-heavy steps).
Avoid intra-work parallelism until the pipeline is reliable.

---

## 10) Dependencies (Python-first)

Core:

- `pdf2image` (+ poppler), `opencv-python`, `numpy`, `scikit-image`
- `pydantic` for schemas
- `music21` (or equivalent) for musical validation and transformations
- ABC rendering tool (external binary or python wrapper)

Optional:

- OMR engine(s) integration (as subprocess or python API)
- Commercial VLM SDK (OpenAI / Gemini) for chord OCR + disambiguation
- Streamlit for review UI

---

## 11) Quality Metrics / Acceptance Criteria

Per work:

- **Measure completeness**: 100% measures have valid meter after repair
- **Chord extraction**: ≥ 95% chord symbols correctly recognized on a small labeled set
- **Note accuracy**: target ≥ 90–95% on a labeled subset (you can label 5–10 works)
- **Review rate**: < 20% of measures require human intervention after pipeline maturity

---

## 12) Milestones (with checkpoints)

### Milestone 0 — Thin End-to-End Slice (baseline flow)

- CLI skeleton (`score2abc ingest/run/qa/export`) wires the steps together
- `WorkItem` manifest iteration + slugging
- PDF → image rendering at fixed DPI
- Minimal staff/system crop (even if crude)
- Stub `events.json` → ABC generator
- Render ABC previews (SVG/PNG)
- Export `out/index.md` catalog with metadata + ABC block
  **Done when:** a single PDF produces `melody.abc`, `melody_with_chords.abc`, previews, and appears in `out/index.md`.

### Milestone 1 — Evaluation Harness + Dataset Plumbing

- formalize dataset loader for `dataset/metadata.csv` (validate required fields)
- run log per work + top-level summary report
- stage-level `stage.json` artifacts (inputs, params, hashes) for resume/caching
- define accuracy metrics (note accuracy, chord accuracy, meter validity)
- small labeled subset format (ground-truth ABC or events)
- compare vs ground truth report/script
- tests for manifest parsing/slugging, ABC formatting, and metrics/comparison code
  **Done when:** an evaluation command produces a numeric report for the dataset.

### Milestone 2 — Accuracy Lifts: Segmentation + Melody + Chords

- staff/system detection with robust line-spacing estimation
- improve preprocessing (deskew, contrast, denoise) and save variants
- integrate Melody Engine A → MusicXML → `events.json`
- implement musical validation/repair (meter enforcement, quantization)
- chord OCR extraction + normalization + alignment to measures
  **Done when:** evaluation improves over M1 baseline and meter validity is 100% after repair.

### Milestone 3 — Human-in-the-Loop (HITL) Editing Loop

- Streamlit (or equivalent) UI showing: PDF/crop, current ABC, rendered score, metadata
- optional MIDI playback of rendered ABC
- editing panel for ABC + chord symbols with measure focus
- save edits as patches/overrides and revalidate
- define `review/` bundle outputs and `overrides/patches.json` ingestion path
  **Done when:** a user can correct a flagged work end-to-end in minutes and the corrections persist.

### Milestone 4 — QA Scoring + Review Prioritization

- render & compare scoring (visual similarity + heuristic flags)
- confidence thresholds to mark “needs review”
- ranked review queue in `out/index.md` or a separate report
  **Done when:** the system produces a prioritized list of measures/works to review with useful flags.

### Milestone 5 — Export & Dataset Polish

- finalize output formats and naming conventions
- ensure ABC exports are consistent and reproducible
- document dataset packaging and versioning
  **Done when:** a full dataset run produces clean, versionable outputs without manual cleanup.

---

## 13) Risks & Mitigations

- **Handwriting variability** → multi-engine + validation + review UI
- **Low scan quality** → strong preprocessing + optional SR
- **Chord OCR mistakes** → constrained prompts + normalization + alignment sanity checks
- **Weird notation quirks** (grace notes, unclear ties) → flagging + targeted manual correction

---

## 14) Deliverables (v1)

- `score2abc` CLI tool
- per-work outputs: `melody.abc`, `melody_with_chords.abc`, previews, QA report
- `out/index.md` catalog
- Streamlit review UI for flagged measures
