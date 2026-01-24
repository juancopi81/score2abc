# Repository Guidelines

## Project Structure & Module Organization
- `main.py` is the entry point for the CLI.
- `score2abc/` contains the Python package code (schemas, CLI, pipeline, and helpers).
- `score2abc/utils/` holds shared logging and timing utilities.
- `docs/PROJECT_SPEC.md` contains the end-to-end product specification and planned architecture.
- `pyproject.toml` defines the package metadata and Python version requirement.
- `dataset/` holds the golden PDF sources; filenames follow the documented normalization rules when possible.
- There are no test or asset directories yet; add them under `tests/` and `assets/` when needed.

## Build, Test, and Development Commands
- `uv run python main.py ingest dataset dataset/metadata.csv out` runs ingest.
- `uv run python main.py run out` runs the pipeline stubs.
- `uv run python main.py qa out` checks for previews.
- `uv run python main.py export out` writes `out/index.md`.
- `uv lock` updates the lockfile when dependencies change.
- `uv sync` installs dependencies from the lockfile.
- `uv add <package>` adds a dependency and refreshes the lockfile.

## Coding Style & Naming Conventions
- Language: Python 3.11+ (per `pyproject.toml`).
- Indentation: 4 spaces, PEP 8 conventions.
- Naming: `snake_case` for functions/variables, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants.
- No formatter or linter is configured yet; if you add one (e.g., Ruff/Black), document it here.

## Testing Guidelines
- No tests are present yet. When adding tests, use `tests/` with files named `test_*.py`.
- Prefer `pytest` conventions (fixtures, asserts) if you introduce a test framework.

## Commit & Pull Request Guidelines
- Use Conventional Commits: `<type>: <description>` (scope optional).
- Allowed types: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`, `build`, `ci`.
- Examples: `feat: add CLI skeleton`, `docs: add roadmap`, `chore: initial commit`.
- For PRs, include: summary, related issue (if any), and notes on how to validate changes.

## Agent-Specific Notes
- Keep this guide updated as the CLI, pipeline stages, and test suite are introduced.
- When you add new directories (e.g., `score2abc/`, `scripts/`, `configs/`), document their roles here.
