# Repository Guidelines

## Project Structure & Module Organization
- `main.py` is the entry point for the CLI.
- `score2abc/` contains the Python package code (schemas, CLI, pipeline, and helpers).
- `score2abc/chord_ocr/` holds the chord-OCR backends (Gemini + fixture + cache wrappers), the prompt, normalization, and barline/measure alignment logic.
- `score2abc/chords.py` orchestrates the `extract_chords` pipeline stage and picks a `ChordOCR` backend from `--use-vlm`.
- `score2abc/utils/` holds shared logging and timing utilities.
- `scripts/` holds one-off developer utilities (e.g., `record_vlm_fixtures.py`).
- `docs/PROJECT_SPEC.md` contains the end-to-end product specification and planned architecture.
- `docs/ROADMAP.md` tracks milestone progress with checkboxes — update it when milestone items ship.
- `CLAUDE.md` is a thin pointer for Claude Code back to this file.
- `pyproject.toml` defines the package metadata and Python version requirement.
- `dataset/` holds the golden PDF sources; filenames follow the documented normalization rules when possible.
- `dataset/ground_truth/` holds labeled events for evaluation (`<slug>.json`).
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
- `uv run python scripts/record_vlm_fixtures.py out --slug <slug>` captures a set of chord-OCR fixtures from a rendered work.
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
- Default runs are hermetic: `score2abc run` uses `FixtureChordOCR(tests/fixtures/vlm/)` and never calls the network. Missing fixtures are logged and treated as empty detections.
- To exercise the live path, install the optional extra and export your Gemini key (free tier is sufficient):
  ```
  uv sync --extra vlm
  export GEMINI_API_KEY=...   # from https://aistudio.google.com/apikey
  uv run python main.py run out --use-vlm
  ```
- Live responses land in `.cache/vlm/` (gitignored). Promote an entry to a committed fixture by copying it into `tests/fixtures/vlm/` — the filename is the SHA256-based fixture key and is stable across machines.
- Re-record fixtures after a chord-crop, prompt, or model change:
  ```
  uv run python scripts/record_vlm_fixtures.py out --slug aviador
  ```
  Pass `--force` to overwrite existing fixtures, `--model` to override the Gemini model, and repeat `--slug` to target multiple works.

## Agent-Specific Notes
- Keep this guide updated as the CLI, pipeline stages, and test suite are introduced.
- When you add new directories (e.g., `score2abc/`, `scripts/`, `configs/`), document their roles here.
