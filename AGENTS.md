# Repository Guidelines

## Project Structure & Module Organization
- `main.py` is the current entry point and only executable module.
- `docs/PROJECT_SPEC.md` contains the end-to-end product specification and planned architecture.
- `pyproject.toml` defines the package metadata and Python version requirement.
- `dataset/` holds the golden PDF sources; filenames follow the documented normalization rules when possible.
- There are no test or asset directories yet; add them under `tests/` and `assets/` when needed.

## Build, Test, and Development Commands
- `python main.py` runs the current entry point (prints a placeholder message).
- `python -m venv .venv` then `source .venv/bin/activate` sets up a local virtual environment.
- `python -m pip install -e .` installs the project in editable mode (once dependencies are added).

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
