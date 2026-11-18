"""Run API transcription and independent review for a list of manuscript PDFs."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROMPTS = ROOT / "docs/prompts"
MODEL = "gpt-6.2-astra"


def stamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def budget_cents(value: str) -> int:
    if re.fullmatch(r"[0-9]+(?:\.[0-9]{1,2})?", value):
        dollars, _, cents = value.partition(".")
        total = int(dollars) * 100 + int(cents.ljust(2, "0"))
        if total > 0:
            return total
    raise argparse.ArgumentTypeError("Enter a positive dollar budget with at most two decimals")


def dollars(cents: int) -> str:
    return f"{cents // 100}.{cents % 100:02d}"


def source_hash(pdf: Path) -> str:
    if not pdf.is_file() or pdf.stat().st_size > 50_000_000:
        raise ValueError(f"Expected a source PDF of at most 50 MB: {pdf.name}")
    data = pdf.read_bytes()
    if not data.startswith(b"%PDF"):
        raise ValueError(f"Expected a source PDF: {pdf.name}")
    return hashlib.sha256(data).hexdigest()


def make_plan(pdfs: list[Path], ceiling: int) -> dict:
    share, remainder = divmod(ceiling, len(pdfs))
    if share < 1:
        raise ValueError("The total budget must allocate at least one cent to each PDF")
    works, hashes = [], set()
    for number, path in enumerate(pdfs, 1):
        pdf = path.expanduser().resolve()
        digest = source_hash(pdf)
        if digest in hashes:
            raise ValueError(f"Duplicate source PDF: {path.name}")
        hashes.add(digest)
        works.append(
            {
                "key": f"score_{number:03d}",
                "title": path.stem,
                "source_pdf": {"path": str(pdf), "sha256": digest},
                "budget_usd": dollars(share + (number <= remainder)),
            }
        )
    return {
        "schema_version": 1,
        "kind": "api_transcription_batch",
        "auth_mode": "openai_api",
        "model": MODEL,
        "reasoning_effort": "xhigh",
        "completion_policy": "completion_gate_v1",
        "deadline_seconds": {"initial": 3600, "reviewed": 1800},
        "budget": {"batch_ceiling_usd": dollars(ceiling), "automatic_retries": 0},
        "works": works,
    }


def write_status(path: Path, status: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def run_batch(plan: dict, output: Path) -> int:
    handoff = (PROMPTS / "collection_handoff_v1.txt").read_text()
    author = (PROMPTS / "collection_author_v1.txt").read_text() + "\n" + handoff
    reader = (
        "Audit the complete supplied manuscript, including every musical row and its form.\n\n"
        + (PROMPTS / "assisted_review_v2.txt").read_text()
        + "\n"
        + handoff
    )
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    (output / "works").mkdir()
    (output / "author.txt").write_text(author)
    (output / "reviewer.txt").write_text(reader)
    plan_path = output / "plan.json"
    plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n")
    status = {
        "state": "running",
        "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        "started_at": stamp(),
        "pairs": [],
    }
    status_path = output / "batch_status.json"
    write_status(status_path, status)
    row = None
    exit_code = 1
    try:
        for work in plan["works"]:
            key = work["key"]
            row = {"key": key, "state": "running", "started_at": stamp()}
            status["pairs"].append(row)
            write_status(status_path, status)
            pdf = Path(work["source_pdf"]["path"])
            if source_hash(pdf) != work["source_pdf"]["sha256"]:
                raise ValueError(f"Source changed after planning: {pdf.name}")

            target = output / "works" / key
            log = output / f"{key}.log"
            print(f"Starting {work['title']} (up to ${work['budget_usd']})", flush=True)
            with log.open("x") as stream:
                result = subprocess.run(
                    [
                        sys.executable,
                        str(ROOT / "scripts/run_assisted_transcription.py"),
                        "pair",
                        "--backend",
                        "api",
                        "--pdf",
                        str(pdf),
                        "--prompt",
                        str(output / "author.txt"),
                        "--review-prompt",
                        str(output / "reviewer.txt"),
                        "--output",
                        str(target),
                        "--model",
                        plan["model"],
                        "--author-seconds",
                        str(plan["deadline_seconds"]["initial"]),
                        "--review-seconds",
                        str(plan["deadline_seconds"]["reviewed"]),
                        "--completion-policy",
                        plan["completion_policy"],
                        "--budget-usd",
                        work["budget_usd"],
                    ],
                    cwd=ROOT,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            pair_file = target / "pair_status.json"
            pair = json.loads(pair_file.read_text()) if pair_file.exists() else {}
            complete = result.returncode == 0 and pair.get("complete") is True
            row.update(state="complete" if complete else "incomplete", ended_at=stamp())
            write_status(status_path, status)
            if not complete:
                status["state"] = "stopped_without_retry"
                print(f"Stopped after {work['title']}. See {log}.", file=sys.stderr)
                break
            print(f"Completed {work['title']}", flush=True)
        else:
            status["state"] = "all_pairs_complete_awaiting_delivery_checks"
            exit_code = 0
    except KeyboardInterrupt:
        status["state"] = "interrupted"
        if row and row["state"] == "running":
            row.update(state="interrupted", ended_at=stamp())
        exit_code = 130
    except (OSError, ValueError) as error:
        status["state"] = "stopped_without_retry"
        if row and row["state"] == "running":
            row.update(state="incomplete", ended_at=stamp())
        print(f"Batch stopped: {error}", file=sys.stderr)
    finally:
        status["ended_at"] = stamp()
        write_status(status_path, status)
    print(f"Saved batch results in {output}", flush=True)
    return exit_code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, action="append", required=True, help="Repeat per score")
    parser.add_argument("--budget-usd", type=budget_cents, required=True, help="Total API ceiling")
    parser.add_argument("--output", type=Path, required=True, help="New directory for the batch")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan without API calls")
    args = parser.parse_args()
    try:
        output = args.output.expanduser()
        if output.exists() or output.is_symlink():
            raise ValueError("Choose a new output directory")
        plan = make_plan(args.pdf, args.budget_usd)
        if args.dry_run:
            print(json.dumps(plan, ensure_ascii=False, indent=2))
            return 0
        return run_batch(plan, output.resolve())
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    sys.exit(main())
