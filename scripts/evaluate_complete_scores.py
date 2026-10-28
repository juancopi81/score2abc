"""Prepare, validate and compare complete-score outputs without making model calls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from score2abc.score_eval import build_report, validate_candidate
from score2abc.score_eval_config import load_config
from score2abc.score_eval_harness import build_grader_demo, prepare_inputs


def write_report(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    config = commands.add_parser("config", help="write an editable initial system configuration")
    config.add_argument("output", type=Path)
    prepare = commands.add_parser(
        "prepare", help="create PDF-only task packet; does not launch agents"
    )
    prepare.add_argument("freeze", type=Path)
    prepare.add_argument("destination", type=Path)
    prepare.add_argument(
        "--config", type=Path, help="system configuration; defaults to Astra/Terra"
    )
    prepare.add_argument(
        "--trial", type=int, default=1, help="one shared trial number for all tasks"
    )
    evaluate = commands.add_parser(
        "report", help="evaluate registered outputs, or show all runs as unrun"
    )
    evaluate.add_argument("freeze", type=Path)
    evaluate.add_argument("labels", type=Path)
    evaluate.add_argument("output", type=Path)
    evaluate.add_argument("--registry", type=Path)
    evaluate.add_argument("--config", type=Path, help="require a matching configured cohort")
    validate = commands.add_parser("validate", help="give truth-free feedback on one draft")
    validate.add_argument("abc", type=Path)
    validate.add_argument("--decisions", type=Path)
    demo = commands.add_parser(
        "self-test", help="six controlled synthetic cases; not model results"
    )
    demo.add_argument("destination", type=Path)
    args = parser.parse_args()
    if args.command == "config":
        write_report(args.output, load_config())
        print(args.output)
    elif args.command == "prepare":
        data = prepare_inputs(
            args.freeze, args.destination, load_config(args.config), trial=args.trial
        )
        print(
            json.dumps({"tasks": len(data["tasks"]), "execution_status": data["execution_status"]})
        )
    elif args.command == "report":
        configuration = args.config or (load_config() if args.registry is None else None)
        write_report(
            args.output, build_report(args.freeze, args.labels, args.registry, configuration)
        )
        print(args.output)
    elif args.command == "validate":
        decisions = json.loads(args.decisions.read_text()) if args.decisions else None
        print(
            json.dumps(
                validate_candidate(args.abc.read_text(), decisions), ensure_ascii=False, indent=2
            )
        )
    else:
        write_report(args.destination / "report.json", build_grader_demo(args.destination))
        print(args.destination / "report.json")


if __name__ == "__main__":
    main()
