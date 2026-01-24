from __future__ import annotations

import argparse
import sys
from pathlib import Path

from score2abc.utils import Timer, get_logger


def _validate_path(path: Path, label: str, logger) -> bool:
    if not path.exists():
        logger.error("%s not found: %s", label, path)
        return False
    return True


def _cmd_ingest(args: argparse.Namespace) -> int:
    logger = get_logger("score2abc.ingest")
    input_dir = Path(args.input_dir)
    metadata_csv = Path(args.metadata_csv)
    out_dir = Path(args.out_dir)

    logger.info("Ingest started")
    logger.info("Input dir: %s", input_dir)
    logger.info("Metadata: %s", metadata_csv)
    logger.info("Output dir: %s", out_dir)

    with Timer("ingest", logger=logger):
        ok = True
        ok &= _validate_path(input_dir, "Input directory", logger)
        ok &= _validate_path(metadata_csv, "Metadata CSV", logger)
        if not ok:
            return 1
        out_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Ingest stub complete (no pipeline steps yet)")

    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    logger = get_logger("score2abc.run")
    out_dir = Path(args.out_dir)
    logger.info("Run started")
    logger.info("Output dir: %s", out_dir)
    logger.info("Workers: %s", args.workers)
    logger.info("Use VLM: %s", args.use_vlm)

    with Timer("run", logger=logger):
        if not _validate_path(out_dir, "Output directory", logger):
            return 1
        logger.info("Run stub complete (no pipeline steps yet)")

    return 0


def _cmd_qa(args: argparse.Namespace) -> int:
    logger = get_logger("score2abc.qa")
    out_dir = Path(args.out_dir)
    logger.info("QA started")
    logger.info("Output dir: %s", out_dir)
    logger.info("Open UI: %s", args.open_ui)

    with Timer("qa", logger=logger):
        if not _validate_path(out_dir, "Output directory", logger):
            return 1
        logger.info("QA stub complete (no pipeline steps yet)")

    return 0


def _cmd_export(args: argparse.Namespace) -> int:
    logger = get_logger("score2abc.export")
    out_dir = Path(args.out_dir)
    logger.info("Export started")
    logger.info("Output dir: %s", out_dir)
    logger.info("Format: %s", args.format)

    with Timer("export", logger=logger):
        if not _validate_path(out_dir, "Output directory", logger):
            return 1
        logger.info("Export stub complete (no pipeline steps yet)")

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="score2abc",
        description="Handwritten Colombian scores to ABC pipeline.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("ingest", help="Ingest PDFs and metadata")
    ingest.add_argument("input_dir", help="Directory containing PDF files")
    ingest.add_argument("metadata_csv", help="Metadata CSV file")
    ingest.add_argument("out_dir", help="Output directory")
    ingest.set_defaults(func=_cmd_ingest)

    run = subparsers.add_parser("run", help="Run the pipeline")
    run.add_argument("out_dir", help="Output directory")
    run.add_argument("--workers", type=int, default=1, help="Number of workers")
    run.add_argument("--use-vlm", action="store_true", help="Enable VLM-assisted steps")
    run.set_defaults(func=_cmd_run)

    qa = subparsers.add_parser("qa", help="Run QA and review tooling")
    qa.add_argument("out_dir", help="Output directory")
    qa.add_argument("--open-ui", action="store_true", help="Open review UI")
    qa.set_defaults(func=_cmd_qa)

    export = subparsers.add_parser("export", help="Export catalog and artifacts")
    export.add_argument("out_dir", help="Output directory")
    export.add_argument("--format", default="index.md", help="Export format")
    export.set_defaults(func=_cmd_export)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "func", None)
    if handler is None:
        parser.print_help()
        return 1
    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
