# score2abc

Pipeline to convert handwritten Colombian scores into ABC notation plus metadata.

## Usage (uv)

Run the CLI through uv:

```bash
uv run python main.py ingest dataset dataset/metadata.csv out
uv run python main.py run out
uv run python main.py qa out
uv run python main.py export out
```

Notes:
- PDF rendering uses `pdf2image` and requires a local Poppler install.
- ABC previews render via `abc2svg` or `abcm2ps` if available; otherwise a placeholder SVG is written.
- ABC export now preserves canonical event timing, including implicit rests,
  simultaneous-note groups, and ties split across barlines/chord changes.
- Segmentation now writes both raw and locally deskewed system/annotation crops
  under each work's `systems/` directory, plus per-page overlays and JSON bbox
  manifests for inspection.

## Dependency Management (uv)

Use uv for dependencies:

```bash
uv lock
uv sync
```

To add a new package:

```bash
uv add <package>
```

## Testing

Install test dependencies and run pytest:

```bash
uv sync --extra test
uv run pytest
```

Run the local CI-equivalent checks:

```bash
uv sync --extra dev
uv run ruff check .
uv run black --check .
uv run pytest
```

## Evaluation (M1)

Ground-truth events live under `dataset/ground_truth/` named by slug
(e.g., `dataset/ground_truth/<slug>.json`). Run evaluation against an `out/`
folder produced by the pipeline:

```bash
uv run python main.py eval out --ground-truth dataset/ground_truth
```

Evaluation now includes event-level precision/recall/F1 and enforces a minimum
coverage gate (`works_with_predictions / works_with_truth`).

If your ground truth starts in MuseScore, export an uncompressed `.musicxml`
file and convert it into the repo's canonical events JSON with:

```bash
uv run python -m score2abc.musicxml path/to/work.musicxml dataset/ground_truth/<slug>.json
```

## Pipeline execution status

- `ingest`, `run`, and `qa` now fail with non-zero exit codes if any work fails.
- Command-level status summaries are written to:
  - `out/ingest_status.json`
  - `out/run_status.json`
  - `out/qa_status.json`
- Per-work stage artifacts are written to `out/<slug>/stages/<stage>.json`.

## External Tools

### Poppler (required for PDF rendering)

Check whether Poppler is available:

```bash
pdftoppm -h
```

Install Poppler:

```bash
# macOS (Homebrew)
brew install poppler

# Ubuntu/Debian
sudo apt-get install poppler-utils
```

On Windows, install Poppler binaries (e.g., the `poppler-windows` builds by
oschwartz10612), then add the `bin` folder to your PATH.

### ABC preview renderers (optional)

Install one of these to render previews:

```bash
# abc2svg (npm)
npm i abc2svg

# abcm2ps (Homebrew)
brew install abcm2ps
```

If neither is installed, the pipeline writes a placeholder SVG preview.

## Dataset

`dataset/` is the current golden dataset of source PDFs, with a canonical metadata
table at `dataset/metadata.csv`. These files come from the
manuscript transcriptions of Colombian organist Jaime Llanos Gonzalez, later
photocopied and shared by triplista Jairo Rincon Gomez (June 2002) with the music
library of Universidad de Antioquia. The photocopies are low-legibility and in
fragile condition, so this dataset prioritizes preservation and accurate transcription.

## Filename convention (when possible)

`jaime-llanos_<num>_<titulo>_<genero>_<autor>.pdf`

Normalization rules:
- lowercase
- ASCII only (accents removed)
- words separated by hyphens
- punctuation removed (e.g., "Gato'e" -> `gatoe`)
- author last names separated by hyphens

Not all future sources will have clean or consistent names. If a file cannot be
normalized cleanly, keep the original filename and rely on the metadata table
as the source of truth for title/composer/genre.
