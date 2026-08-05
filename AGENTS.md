# Repository Guidelines

## Project Structure & Module Organization

- `main.py` is the entry point for the CLI.
- `score2abc/` contains the Python package code (schemas, CLI, pipeline, and helpers).
- `score2abc/chord_ocr/` holds the chord-OCR backends (Gemini + fixture + cache wrappers), the prompt, normalization, and barline/measure alignment logic.
- `score2abc/chords.py` orchestrates the `extract_chords` pipeline stage and picks a `ChordOCR` backend from `--use-vlm`.
- `score2abc/render.py` deskews/enhances pages, proposes horizontal bands, rejects candidates without five long consistently spaced staff lines, and persists accepted/source numbering plus rejected-candidate diagnostics.
- `score2abc/melody/` holds the MusicXML melody pipeline: `backend.py` defines the `MusicXMLBackend` protocol plus fixture, optional homr, and optional Audiveris backends that power the `extract_musicxml` stage, while `musicxml.py` parses MusicXML into `melody.json` / `events.json`.
- `score2abc/utils/` holds shared logging and timing utilities.
- `scripts/` holds one-off developer utilities (e.g., `record_vlm_fixtures.py`).
- `docs/PROJECT_SPEC.md` contains the end-to-end product specification and planned architecture.
- `docs/ROADMAP.md` tracks milestone progress with checkboxes — update it when milestone items ship.
- `CLAUDE.md` is a thin pointer for Claude Code back to this file.
- `pyproject.toml` defines the package metadata and Python version requirement.
- `dataset/` holds the golden PDF sources; filenames follow the documented normalization rules when possible.
- `dataset/ground_truth/` holds labeled events for evaluation (`<slug>.json`).
- `dataset/musicxml/` holds committed MusicXML fixtures (`<slug>.musicxml`) consumed by the `extract_musicxml` stage until a real OMR engine is wired in. If a slug has no fixture, the stage is skipped and the pipeline falls back to stub melody notes; a fixture that fails to parse fails the work item rather than silently falling back.
- `tests/` holds pytest-based unit tests.
- `tests/fixtures/vlm/` holds committed chord-OCR responses so hermetic runs (`use_vlm=False`) and CI never call Gemini.
- `.cache/vlm/` (gitignored) holds live Gemini responses captured by `--use-vlm` runs.

## Build, Test, and Development Commands

- `uv run python main.py ingest dataset dataset/metadata.csv out` runs ingest.
- `uv run python main.py run out` runs the pipeline stubs.
- `uv run python main.py qa out` checks for previews.
- `uv run python main.py export out` writes `out/index.md`.
- `uv run python main.py eval out --ground-truth dataset/ground_truth` runs the evaluation report.
- `uv run python main.py run out --use-vlm` enables the live Gemini chord-OCR path (requires `GEMINI_API_KEY`; see VLM notes below).
- `uv run python main.py run out --slug <slug>` runs only selected manifest entries; repeat `--slug` for multiple works.
- `uv run python main.py run out --musicxml-backend homr --homr-input page|deskewed-page|systems` enables the optional external homr melody-OMR path; homr must be installed separately and is not a project dependency.
- `uv run python main.py run out --musicxml-backend audiveris --audiveris-input page|deskewed-page|systems` enables the optional external Audiveris melody-OMR path; Audiveris must be installed separately and is not a project dependency.
- `uv run python scripts/record_vlm_fixtures.py out --slug <slug>` captures chord-OCR fixtures from a rendered work; add `--band below` when chords are visually placed below the staff.
- `uv run python scripts/build_vlm_melody_prompt_variants.py out --slug <slug> --system <n> --measure <n> --variant all --prompt all` pre-renders spike prompt files and associates each compatible `prompt_id` with an image `variant_id`.
- `uv run python scripts/run_vlm_melody_experiment_batch.py out --slug <slug> --system <n> --measure <n> --variant-id <variant> --prompt-id <prompt> --provider openai --max-calls 0` plans a no-network melody VLM sweep; increase `--max-calls` intentionally for live calls and add `--journal` to snapshot each result.
- `uv run python scripts/journal_vlm_melody_experiment.py out --slug <slug> --system <n> --measure <n> --input-kind <kind> --provider <provider>` snapshots a melody VLM spike after record/eval, including image, prompt, config, fixture, eval result, and replay commands.
- `uv run python scripts/review_vlm_notehead_candidates.py out --slug <slug> --system <n> --measure <n>` starts the spike-only, GT-blind localhost reviewer for confirming cap-24 notehead candidates, adding missing heads, correcting pitches, and recording active review time.
- `uv run python scripts/promote_vlm_notehead_reviews.py out --slug <slug> --system <n> --measure <n>` validates saved reviews and promotes deterministic, GT-free training fixtures under `tests/fixtures/vlm_melody/notehead_reviews/`.
- `uv run python scripts/build_vlm_melody_event_benchmark.py build out --slug <slug> --ground-truth dataset/ground_truth --clef treble --time-signature 3/4 --key-hint "one flat: Bb"` freezes the spike's inference requests before materializing canonical event truth.
- `uv run python scripts/experiments/spike_notehead_patch_templates.py out` reproduces the four-measure leave-one-out local notehead-selector benchmark.
- `uv run python scripts/experiments/spike_anchored_rhythm_parser.py --out-dir out` reproduces the explicit HITL rhythm/rest upper bound from promoted notehead anchors.
- `uv run python scripts/experiments/spike_meter_gap_resolver.py out` retrains the dense selector on promoted systems 1+7 reviews, freezes system-8 predictions, and reproduces the spike-only meter-gap repair gate.
- `uv run python scripts/experiments/freeze_second_score_heldout.py out` creates the one-shot, truth-blind Carrizal system-4 freeze from the S1+S7 model. The checked-out `out/` artifact is already frozen; the command intentionally refuses to overwrite it.
- `uv run python scripts/experiments/evaluate_second_score_heldout.py out --musicxml <path>` verifies the immutable Carrizal freeze, maps its seven automatic crops to eight physical MusicXML measures, and writes or reuses the sealed one-shot evaluation. It refuses changed freeze or truth hashes.
- `uv run python scripts/experiments/evaluate_frozen_third_score_heldout.py <v2-sealed-manifest> --musicxml <path>` verifies the authoritative, truth-blind La Chata system-7 v2 freeze and scores pitch/count metrics only; rhythm/rest remain `not_scored_missing_frozen_context` when time/key metadata is absent. The expected MusicXML path is `out/jaime-llanos_64_la-chata_pasillo_luis-a-calvo/vlm_melody_third_score_heldout/v2/system_007/la_chata_system_007.musicxml`; use `docs/templates/vlm_heldout_crop_mapping.example.json` if the physical-measure count is not seven. Do not run this evaluator before the transcription is finalized.
- `uv run python scripts/experiments/freeze_fourth_score_heldout.py prepare out` prepares the create-once Gato'e Fique system-3 fourth-score gate, including truth-blind visual-key and metadata-rhythm context. The checked-out `out/` artifact is already prepared and frozen; do not rerun it.
- `uv run python scripts/experiments/run_fourth_score_heldout_inference.py <prepared-manifest> --model-dir <model-dir>` replays and freezes a prepared fourth-score gate with a model that must exclude the target score. The sealed Gato'e Fique artifact uses the provenance-refreshed `cross_score_notehead_v1_replay_20260722` model, whose serialized model hash is identical to the historical configuration-C model.
- `uv run python scripts/experiments/evaluate_frozen_fourth_score_heldout.py <sealed-manifest> --musicxml <path>` verifies and scores the six-crop Gato'e Fique gate. The checked-out `out/` artifact is already evaluated as `evaluation_v1`; do not rerun or overwrite it.
- `uv run python scripts/experiments/freeze_fifth_score_heldout.py prepare out` prepares the create-once Coqueteos system-2 fifth-score gate with six automatic crops, provisional `Pasillo -> 3/4` context, and an unknown key. The checked-out `out/` artifact is already prepared, frozen, and evaluated; do not rerun it.
- `uv run python scripts/experiments/triage_frozen_heldout_meter_deficits.py <sealed-manifest>` writes a create-once, truth-blind review sidecar without modifying the frozen predictions. Coqueteos measures 3-5 are flagged for review; this is prioritization evidence, not a transcription result.
- `uv run python scripts/experiments/evaluate_frozen_fifth_score_heldout.py <sealed-manifest> --musicxml <path> --mapping <path>` verifies and scores full note, pitch, onset, duration, rest, measure, and meter metrics for the Coqueteos gate. The checked-out `out/` artifact already contains `evaluation_v1` using the seven-measure whole-measure mapping; do not rerun or overwrite it.
- `uv run python scripts/experiments/build_consumed_cross_score_training_inputs.py out --slug jaime-llanos_22_coqueteos_pasillo_fulgencio-garcia --system 2 --namespace coqueteos_system_002_seg_v2 --expected-measures 7` creates corrected, GT-blind Coqueteos crops and candidate artifacts. `uv run python scripts/experiments/spike_consumed_coqueteos_corrected_replay.py out` then reuses the exact frozen model/context, writes prediction and meter-triage sidecars before opening consumed MusicXML, and compares the seven-crop replay with the sealed six-crop baseline. Both outputs are consumed postmortems, not heldout claims.
- `uv run python scripts/experiments/prepare_consumed_cross_score_proposals.py out --mapping <corrected-mapping.json> --consumption-mapping <consumption-mapping.json>` creates a review-only candidate assignment queue for any contiguous one-to-one corrected consumed system. The checked-out Coqueteos queue remains `eligible_for_training=false` until its pixel coordinates are visually adjudicated.
- `uv run python scripts/experiments/materialize_consumed_human_candidate_review.py out --decision-fixture <path>` validates a complete human candidate partition, including deterministic centers derived from hollow-head rim candidates, freezes it as consumed spike-training evidence, and renders a human-versus-automatic comparison overlay.
- `uv run python scripts/experiments/spike_consumed_hollow_notehead_proposals.py` reproduces the GT-free hollow-notehead center proposer over 22 consumed reviewed measures from its committed SHA256-pinned fixture bundle. It pairs diagonal rim candidates, checks closed or strongly supported open contours, writes review overlays, and opens review truth only after proposals are fixed. This is consumed evidence and must pass an unseen-score gate before candidate-pipeline integration.
- `uv run python scripts/experiments/spike_consumed_polyphonic_pitch_repair.py <inference.jsonl> <truth.jsonl> <evaluation-report.json> <model.json> --context-hints <hints.json> --output-dir <new-dir>` reproduces a create-once consumed postmortem for frozen/external key context, chord-aware scoring, and diagnostic recovery variants.
- `uv run python scripts/experiments/spike_consumed_chord_recovery_regression.py --out-dir out --output-dir <new-dir>` audits the same fixed recovery variants against consumed Aviador and Carrizal candidate labels. It is in-sample regression evidence, not an accuracy claim.
- `uv run python scripts/experiments/spike_consumed_key_state_detector.py --input-dir <measure-crops> --out-dir <new-dir>` detects conservative explicit one-sharp changes after double bars, writes overlays plus stateful `context_hints.json`, and otherwise fails closed as inherited/unknown.
- `uv run python scripts/experiments/spike_consumed_key_signature_detector.py --event 'PATH|initial|1|NAME' --expected-fifths NAME=-1 --change-scan-dir <measure-dir> --out-dir <new-dir>` reproduces the consumed initial/change key-signature detector for sharps, flats, and multiple accidentals. Repeat `--event`, `--expected-fifths`, and `--change-scan-dir`; use `NAME=none` for labeled negative controls.
- `uv run python scripts/experiments/spike_consumed_visual_key_pitch_replay.py --detector-report <expanded-report.json> --context-hints <evaluation-labels.json> --inference <inference.jsonl> --model <model.json> --truth <truth.jsonl> --output-dir <new-dir>` scopes expanded key-signature events to the inference work, persists baseline and automatic visual-key predictions before opening consumed truth, then measures the pitch-only gain with identical candidate selection. Use `--slug <slug>` only when inference rows do not carry one unambiguous work slug.
- `uv run python scripts/experiments/apply_vlm_melody_key_correction.py <inference.jsonl> --key-event 1=2 --output-dir <new-dir>` applies an explicit human-reviewed key signature without rerunning notehead selection. Repeat `--key-event START_MEASURE=FIFTHS` for later key changes; output is create-once and hard-fails if candidate IDs, coordinates, counts, or note rhythm change.
- `uv run python scripts/experiments/spike_consumed_onset_group_selector.py --out-dir out --output-dir <new-dir> --overlays` reproduces the rejected work-disjoint onset-group filter and its La Chata count-only audit.
- `uv run python scripts/experiments/spike_consumed_meter_deficit_validator.py --out-dir out --output-dir <new-dir> --overlays` reproduces the non-mutating review-triage signal that flags visual meter deficits. Its consumed gate passes, but La Chata generalization does not, so it must not delete notes or enter runtime yet.
- `uv run python scripts/debug_barlines.py out --slug <slug>` renders detected barlines over system crops for chord alignment debugging.
- `uv lock` updates the lockfile when dependencies change.
- `uv sync` installs dependencies from the lockfile.
- `uv sync --extra test` installs the test dependencies.
- `uv sync --extra vlm` installs the live Gemini backend dependency (`google-genai`).
- `uv run pytest` runs the test suite.
- `uv add <package>` adds a dependency and refreshes the lockfile.

## Coding Style & Naming Conventions

- Language: Python 3.11+ (per `pyproject.toml`).
- Indentation: 4 spaces, PEP 8 conventions.
- Naming: `snake_case` for functions/variables, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants.
- Formatting: `black` (line length 100).
- Linting: `ruff` (rules: E, F, I, B; line length 100).

## Testing Guidelines

- Tests live in `tests/` with files named `test_*.py`.
- Use pytest conventions (fixtures, asserts).
- Local quality checks:
  - `uv sync --extra dev`
  - `uv run ruff check .`
  - `uv run black --check .`
  - `uv run pytest`

## Commit & Pull Request Guidelines

- Use Conventional Commits: `<type>: <description>` (scope optional).
- Allowed types: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`, `build`, `ci`.
- Examples: `feat: add CLI skeleton`, `docs: add roadmap`, `chore: initial commit`.
- For PRs, include: summary, related issue (if any), and notes on how to validate changes.

## VLM Chord OCR (Gemini)

- Default pipeline runs remain hermetic: `score2abc run` uses `FixtureChordOCR(tests/fixtures/vlm/)` to replay committed fixtures and never calls the network. Missing fixtures are logged and treated as empty detections.
- To exercise the live path, install the optional extra and export your Gemini key (free tier is enough for small fixture-recording runs, but it is rate-limited):
  ```
  uv sync --extra vlm
  export GEMINI_API_KEY=...   # from https://aistudio.google.com/apikey
  uv run python main.py run out --use-vlm
  ```
- Live responses land in `.cache/vlm/` (gitignored). Promote an entry to a committed fixture by copying it into `tests/fixtures/vlm/` — the filename is the SHA256-based fixture key and is stable across machines.
- Re-record fixtures after a chord-crop, prompt, or model change:
  ```
  uv run python scripts/record_vlm_fixtures.py out --slug <slug>
  uv run python scripts/record_vlm_fixtures.py out --slug <slug> --band below
  ```
- `--band below` is useful when chords are visually placed below the staff and avoids spending Gemini calls on above-staff crops. Pass `--band above` to record only above-staff crops; the default `--band both` preserves the previous behavior. Pass `--force` to overwrite existing fixtures, `--model` to override the Gemini model, and repeat `--slug` to target multiple works.
- To inspect current measure-alignment limits, render detected barlines over system crops:
  ```
  uv run python scripts/debug_barlines.py out --slug <slug>
  ```

## Agent-Specific Notes

- Keep this guide updated as the CLI, pipeline stages, and test suite are introduced.
- When you add new directories (e.g., `score2abc/`, `scripts/`, `configs/`), document their roles here.
