# Repository Guidelines

## Project Structure & Module Organization

- `main.py` is the entry point for the CLI.
- `score2abc/` contains the Python package code (schemas, CLI, pipeline, and helpers).
- `score2abc/chord_ocr/` holds the chord-OCR backends (Gemini + fixture + cache wrappers), the prompt, normalization, and barline/measure alignment logic.
- `score2abc/chords.py` orchestrates the `extract_chords` pipeline stage and picks a `ChordOCR` backend from `--use-vlm`.
- `score2abc/melody/` holds the MusicXML melody pipeline: `backend.py` defines the `MusicXMLBackend` protocol plus fixture and optional homr backends that power the `extract_musicxml` stage, while `musicxml.py` parses MusicXML into `melody.json` / `events.json`.
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
- `uv run python main.py run out --musicxml-backend homr` enables the optional external homr melody-OMR path; homr must be installed separately and is not a project dependency.
- `uv run python scripts/record_vlm_fixtures.py out --slug <slug>` captures chord-OCR fixtures from a rendered work; add `--band below` when chords are visually placed below the staff.
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
