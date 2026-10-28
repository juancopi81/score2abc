#!/usr/bin/env python3
"""Bootstrap, canary, run and collect bounded subscription evaluation tasks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from score2abc.score_eval_config import digest  # noqa: E402
from score2abc.score_eval_runner import (  # noqa: E402
    RUNTIME_ROOT,
    prepare_runtime,
    run_task,
    subscription_call,
    verify_runtime,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    bootstrap = sub.add_parser("bootstrap")
    bootstrap.add_argument("destination", type=Path, nargs="?", default=RUNTIME_ROOT)
    bootstrap.add_argument("--codex", type=Path)
    canary = sub.add_parser("canary")
    canary.add_argument("destination", type=Path)
    canary.add_argument("--model", required=True)
    canary.add_argument("--effort", default="xhigh")
    canary.add_argument("--runtime", type=Path, default=RUNTIME_ROOT)
    run = sub.add_parser("run")
    run.add_argument("packet", type=Path)
    run.add_argument("destination", type=Path)
    run.add_argument("--task-id", required=True)
    run.add_argument("--runtime", type=Path, default=RUNTIME_ROOT)
    collect = sub.add_parser("collect")
    collect.add_argument("destination", type=Path)
    collect.add_argument("registries", type=Path, nargs="+")
    args = parser.parse_args()
    if args.action == "bootstrap":
        result = prepare_runtime(args.destination, args.codex)
        print(json.dumps(result, indent=2))
    elif args.action == "run":
        print(
            json.dumps(
                run_task(args.packet, args.task_id, args.destination, args.runtime), indent=2
            )
        )
    elif args.action == "canary":
        from PIL import Image, ImageDraw

        runtime = verify_runtime(args.runtime)
        args.destination.mkdir(parents=True, exist_ok=False)
        image = Image.new("RGB", (420, 160), "white")
        ImageDraw.Draw(image).text((20, 30), "C D E | F G A |", fill="black", font_size=28)
        path = args.destination / "input.png"
        image.save(path)
        prompt = (
            "Read the six note letters in this synthetic image. Return JSON with abc and "
            "decisions. ABC: treble, Cmajor, 3/4, quarter notes, two bars. Include one synthetic "
            "decision with id canary, category schema_check, location bar1, adopted_choice C, "
            "reason schema exercise, uncertainty none, and one alternative "
            "D_not_adopted_synthetic_test. "
            "Use only the image; no tools or external information."
        )
        calls = []
        for index in range(2):
            call = subscription_call(
                runtime,
                args.runtime.resolve(),
                args.model,
                args.effort,
                [path],
                prompt,
                args.destination / f"call_{index}",
            )
            calls.append(call)
            if call["status"] != "complete" or not call["response"]["decisions"]:
                break
            prompt = (
                "Recheck this synthetic two-bar draft against the same image. Keep all six "
                "quarter notes, with 3/4 meter. Return complete JSON abc and decisions. "
                + json.dumps(call["response"])
            )
        result = {
            "schema_version": 1,
            "kind": "synthetic_subscription_canary",
            "model": args.model,
            "reasoning_effort": args.effort,
            "runtime_sha256": digest(runtime),
            "calls": calls,
            "status": (
                "passed"
                if len(calls) == 2 and all(x["status"] == "complete" for x in calls)
                else "failed"
            ),
        }
        (args.destination / "canary.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({k: v for k, v in result.items() if k != "calls"}, indent=2))
        if result["status"] != "passed":
            raise SystemExit(1)
    else:
        if args.destination.exists():
            raise ValueError("Combined registry is create-once")
        records, identities, seen = [], [], set()
        for path in args.registries:
            path = path.resolve()
            registry = json.loads(path.read_text())
            if registry.get("schema_version") != 2 or registry.get("evidence_kind") != "actual":
                raise ValueError("Only actual schema2 registries may be collected")
            identities.append(
                digest({"config": registry["config"], "generation": registry["generation"]})
            )
            for record in registry["runs"]:
                key = record["task_id"]
                if key in seen:
                    raise ValueError("Duplicate task attempt")
                seen.add(key)
                for field in (
                    "abc_path",
                    "decision_log_path",
                    "measurement_path",
                    "run_record_path",
                ):
                    if record.get(field):
                        record[field] = str((path.parent / record[field]).resolve())
                records.append(record)
        if len(set(identities)) != 1:
            raise ValueError("Cannot collect incompatible configurations or generations")
        output = {**registry, "runs": records}
        args.destination.parent.mkdir(parents=True, exist_ok=True)
        args.destination.write_text(json.dumps(output, indent=2) + "\n")
        print(json.dumps({"runs": len(records), "registry": str(args.destination)}, indent=2))


if __name__ == "__main__":
    main()
