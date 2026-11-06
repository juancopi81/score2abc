"""Explicit API/subscription entry point for the source-only author/reader workflow.

Preparation and audits are offline. `run` and `pair` can invoke a model. API credentials
stay in the host process; they are never copied into the model's input bundle.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import PIL  # noqa: E402
from PIL import Image  # noqa: E402

from scripts.experiments import run_astra_baseline_reproduction as shared  # noqa: E402


def load_api_key(env_file: Path | None = None) -> str:
    """Load only the selected variable, without executing shell configuration."""
    if os.environ.get("OPENAI_API_KEY"):
        return os.environ["OPENAI_API_KEY"]
    if env_file is not None and env_file.is_file() and not env_file.is_symlink():
        for line in env_file.read_text().splitlines():
            match = re.match(r"^\s*(?:export\s+)?OPENAI_API_KEY\s*=\s*(.*)$", line)
            if match:
                try:
                    values = shlex.split(match.group(1), comments=True)
                except ValueError:
                    raise ValueError("API credential entry cannot be parsed") from None
                if len(values) == 1 and values[0]:
                    return values[0]
    raise ValueError("OPENAI_API_KEY is unavailable; no request was sent")


def subscription_runtime(explicit: Path | None) -> Path:
    """Reuse a verified installed runtime; never silently refresh an old pin."""
    if explicit is not None:
        shared.legacy.verify_runtime(explicit)
        return explicit
    candidates = sorted(
        (ROOT / "out/eval/runtime").glob("codex_subscription*"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for candidate in candidates:
        if not candidate.is_dir():
            continue
        try:
            shared.legacy.verify_runtime(candidate)
            return candidate
        except (OSError, ValueError, KeyError):
            continue
    raise ValueError("No current verified subscription runtime; prepare a new runtime first")


def verify_incomplete_checkpoint_author(initial_run: Path, source_hash: str, model: str) -> dict:
    """Admit only the explicit diagnostic review of a sealed output-limit interruption."""
    shared.verify_seal(initial_run, "astra_reproduction_stage")
    provenance = json.loads((initial_run / "stage_provenance.json").read_text())
    if any(
        key in provenance
        for key in ("completion_policy", "completion_policy_sha256", "completion_gate")
    ):
        raise ValueError("Legacy diagnostic review cannot change an author's completion policy")
    expected = {
        "workflow_id": shared.WORKFLOW_ID,
        "stage": "initial",
        "model": model,
        "reasoning_effort": shared.EFFORT,
        "source_pdf_sha256": source_hash,
        "auth_mode": "openai_api",
        "reference_access": False,
        "status": "partial",
        "complete": False,
        "completed_turn": False,
        "quarantined": False,
        "timed_out": False,
        "returncode": 1,
        "error": "Response did not complete; no continuation retry",
    }
    if any(provenance.get(key) != value for key, value in expected.items()):
        raise ValueError("Diagnostic review requires a matching source-only interrupted API author")
    selected = provenance.get("selected_checkpoint")
    if not isinstance(selected, str) or not re.fullmatch(r"artifacts/drafts/draft_\d{3}", selected):
        raise ValueError("Diagnostic review requires the author's saved checkpoint")
    checkpoint_dir = initial_run / selected
    checkpoints = sorted(initial_run.glob("artifacts/drafts/draft_*/checkpoint.json"))
    required = [checkpoint_dir / name for name in ("checkpoint.json", "normalized.json")]
    required += [
        root / name
        for root in (checkpoint_dir, initial_run)
        for name in ("score.abc", "decisions.json")
    ]
    if (
        not checkpoints
        or checkpoints[-1].parent != checkpoint_dir
        or any(not path.is_file() for path in required)
    ):
        raise ValueError("Diagnostic review requires the latest complete saved checkpoint files")
    checkpoint = json.loads((checkpoint_dir / "checkpoint.json").read_text())
    normalized = json.loads((checkpoint_dir / "normalized.json").read_text())
    if normalized.get("valid") is not True or checkpoint.get("complete_claimed") is not False:
        raise ValueError("Diagnostic review requires a valid, explicitly unfinished checkpoint")
    for name, key in (("score.abc", "abc_sha256"), ("decisions.json", "decisions_sha256")):
        digest = shared.sha((checkpoint_dir / name).read_bytes())
        if checkpoint.get(key) != digest or shared.sha((initial_run / name).read_bytes()) != digest:
            raise ValueError("Diagnostic review checkpoint and exported candidate differ")
    responses = sorted((initial_run / "api/private").glob("response_*.json"))
    if not responses:
        raise ValueError("Diagnostic review requires the sealed final API response metadata")
    response = json.loads(responses[-1].read_text())
    details = response.get("incomplete_details")
    if (
        response.get("status") != "incomplete"
        or not isinstance(details, dict)
        or details.get("reason") != "max_output_tokens"
    ):
        raise ValueError("Diagnostic review requires a confirmed max_output_tokens interruption")
    return {
        "diagnostic_review_of_incomplete_author": True,
        "original_author_status": provenance["status"],
        "original_author_error": provenance["error"],
        "original_author_seal_sha256": shared.sha((initial_run / "seal.json").read_bytes()),
        "original_author_checkpoint": selected,
        "original_author_last_response_sha256": shared.sha(responses[-1].read_bytes()),
        "original_author_incomplete_reason": "max_output_tokens",
    }


def prepare_api(
    pdf: Path,
    prompt_file: Path,
    output: Path,
    stage: str,
    limit_seconds: int,
    budget_usd: str | None,
    initial_run: Path | None = None,
    model: str = shared.MODEL,
    *,
    review_incomplete_checkpoint: bool = False,
    no_dollar_cap: bool = False,
    completion_policy: str | None = None,
) -> dict:
    from scripts.experiments import assisted_api

    selected_tools = shared.selected_source_tools(completion_policy)
    if model != shared.MODEL:
        raise ValueError("API pricing is currently pinned for gpt-6-astra only")
    if stage not in {"initial", "reviewed"} or not 60 <= limit_seconds <= 3600:
        raise ValueError("Unsupported stage or allowance")
    if (stage == "reviewed") != (initial_run is not None):
        raise ValueError("Reviewed stage requires its own sealed initial run")
    if review_incomplete_checkpoint and stage != "reviewed":
        raise ValueError("Incomplete-checkpoint review applies only to an API reviewed stage")
    if no_dollar_cap == (budget_usd is not None):
        raise ValueError("Select exactly one explicit dollar budget or no-dollar-cap policy")
    budget = None if no_dollar_cap else Decimal(budget_usd)
    if budget is not None and (not budget.is_finite() or budget <= 0):
        raise ValueError("Positive finite API budget required")
    data = pdf.read_bytes()
    if not data.startswith(b"%PDF") or len(data) > 50_000_000:
        raise ValueError("Expected a bounded source PDF")
    prompt = prompt_file.read_text()
    initial_hash = None
    diagnostic = {}
    if initial_run is not None:
        shared.verify_seal(initial_run, "astra_reproduction_stage")
        provenance = json.loads((initial_run / "stage_provenance.json").read_text())
        if review_incomplete_checkpoint:
            if completion_policy is not None:
                raise ValueError("Legacy incomplete-author diagnostics cannot use a new policy")
            diagnostic = verify_incomplete_checkpoint_author(initial_run, shared.sha(data), model)
        else:
            shared.verify_initial_author(
                provenance,
                shared.sha(data),
                model,
                **({"completion_policy": completion_policy} if completion_policy else {}),
            )
        if provenance.get("auth_mode") != "openai_api":
            raise ValueError("API reader requires its own API author's draft")
        candidate = (initial_run / "score.abc").read_text()
        decisions = json.loads((initial_run / "decisions.json").read_text())
        initial_hash = shared.sha(candidate.encode())
        prompt += (
            "\n\n<untrusted_candidate_draft>\n"
            + json.dumps({"abc": candidate, "decisions": decisions}, ensure_ascii=False)
            + "\n</untrusted_candidate_draft>\n"
        )
    binaries = {}
    for name in ("pdfinfo", "pdftoppm", "node", "abc2svg"):
        found = shutil.which(name)
        if not found:
            raise ValueError(f"Installed {name} required")
        binaries[name] = Path(found).resolve()
    info = subprocess.run(
        [str(binaries["pdfinfo"]), str(pdf)],
        capture_output=True,
        text=True,
        check=True,
        timeout=15,
    )
    pages = next(
        int(line.split(":", 1)[1]) for line in info.stdout.splitlines() if line.startswith("Pages:")
    )
    if not 1 <= pages <= 8:
        raise ValueError("Source page count exceeds bounded tool support")
    output.mkdir(parents=True, exist_ok=False)
    bundle = output / "bundle"
    bundle.mkdir()
    (bundle / "source.pdf").write_bytes(data)
    (bundle / "prompt.txt").write_text(prompt)
    (bundle / "base.txt").write_text(shared.BASE)
    for name, path in shared.source_tool_files(completion_policy).items():
        shutil.copyfile(path, bundle / name)
    for name in ("score_eval_parse.js", "review_playback.js"):
        shutil.copyfile(ROOT / "score2abc" / name, bundle / name)
    (bundle / "renderer").mkdir()
    for name in ("abc2svg-1.js", "toaudio-1.js"):
        shutil.copyfile(binaries["abc2svg"].parent / name, bundle / "renderer" / name)
    shutil.copytree(
        Path(PIL.__file__).parent, bundle / "PIL", ignore=shutil.ignore_patterns("__pycache__")
    )
    shared.write(
        bundle / "tool_config.json",
        {
            "source_pdf_sha256": shared.sha(data),
            "pages": pages,
            "node": str(binaries["node"]),
            "pdftoppm": str(binaries["pdftoppm"]),
            **({"completion_policy": completion_policy} if completion_policy else {}),
        },
    )
    subprocess.run(
        [
            str(binaries["pdftoppm"]),
            "-f",
            "1",
            "-l",
            "1",
            "-r",
            "160",
            "-singlefile",
            "-png",
            str(bundle / "source.pdf"),
            str(bundle / "overview"),
        ],
        capture_output=True,
        check=True,
        timeout=30,
    )
    with Image.open(bundle / "overview.png") as overview:
        overview.thumbnail((2000, 2000), Image.Resampling.LANCZOS)
        overview.save(bundle / "overview.png")
    python = Path(sys.executable).resolve()
    implementation = [Path(__file__), Path(assisted_api.__file__), Path(shared.__file__)]
    metadata = {
        "schema_version": 1,
        "workflow_id": shared.WORKFLOW_ID,
        "stage": stage,
        "model": model,
        "reasoning_effort": shared.EFFORT,
        "backend": "api",
        "auth_mode": "openai_api",
        "limit_seconds": limit_seconds,
        "budget_usd": str(budget) if budget is not None else None,
        "spending_policy": "fixed_output_no_dollar_cap" if no_dollar_cap else "dollar_cap",
        "max_output_tokens_per_response": assisted_api.MAX_OUTPUT_TOKENS,
        "source_pdf_sha256": shared.sha(data),
        "prompt_sha256": shared.sha(prompt.encode()),
        "prompt_template_sha256": shared.sha(prompt_file.read_bytes()),
        "reference_access": False,
        "independent_reader": stage == "reviewed",
        "initial_abc_sha256": initial_hash,
        "python": str(python),
        "python_sha256": shared.sha(python.read_bytes()),
        "helper_binary_files": {
            str(binaries[n]): shared.sha(binaries[n].read_bytes()) for n in ("node", "pdftoppm")
        },
        "implementation": {
            str(p.relative_to(ROOT)): shared.sha(p.read_bytes()) for p in implementation
        },
        "tools_sha256": shared.sha(shared.encoded(selected_tools.TOOLS)),
        "server_instructions_sha256": shared.sha(selected_tools.SERVER_INSTRUCTIONS.encode()),
        "native_tool_mode": "responses_function_calling",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "transport_differences": [
            "Responses native functions replace Codex code-mode MCP wrapper",
            "BASE and source-server instructions are supplied directly",
            (
                "Fixed response output allowance; no dollar-driven truncation"
                if no_dollar_cap
                else "Per-stage dollar cap and per-response output-token cap added"
            ),
        ],
        **diagnostic,
    }
    if completion_policy is not None:
        metadata["completion_policy"] = completion_policy
        metadata["completion_policy_sha256"] = shared.sha(
            shared.source_tool_files(completion_policy)["completion_gate.py"].read_bytes()
        )
    shared.write(output / "preparation.json", metadata)
    shared.seal(output, "astra_reproduction_input")
    return metadata


def audit_api(packet: Path, output: Path) -> dict:
    shared.verify_seal(packet, "astra_reproduction_input")
    metadata = json.loads((packet / "preparation.json").read_text())
    if metadata.get("backend") != "api" or metadata.get("reference_access") is not False:
        raise ValueError("Expected source-only API packet")
    bundle = packet / "bundle"
    selected_tools = shared.verify_source_bundle(bundle, metadata)
    expected = {
        "base.txt": shared.sha(shared.BASE.encode()),
        "source.pdf": metadata["source_pdf_sha256"],
        "prompt.txt": metadata["prompt_sha256"],
    }
    for name, value in expected.items():
        if shared.sha((bundle / name).read_bytes()) != value:
            raise ValueError("API input contract mismatch")
    for name, value in metadata["implementation"].items():
        if shared.sha((ROOT / name).read_bytes()) != value:
            raise ValueError("Prepared implementation changed")
    output.mkdir(parents=True, exist_ok=False)
    record = {
        "schema_version": 1,
        "status": "passed",
        "backend": "api",
        "scope": "offline integrity and source-only tool contract; no model invocation",
        "input_seal_sha256": shared.sha((packet / "seal.json").read_bytes()),
        "tools": [tool["name"] for tool in selected_tools.TOOLS],
        "model": metadata["model"],
        "reasoning_effort": metadata["reasoning_effort"],
        "reference_access": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if metadata.get("completion_policy") is not None:
        record["completion_policy"] = metadata["completion_policy"]
        record["completion_policy_sha256"] = metadata["completion_policy_sha256"]
    shared.write(output / "audit.json", record)
    shared.seal(output, "astra_reproduction_audit")
    return record


def execute_pair(args) -> dict:
    """Run both fresh sessions with one explicit backend; never fall back or retry."""
    api = args.backend == "api"
    no_cap = getattr(args, "no_dollar_cap", False)
    policy = getattr(args, "completion_policy", None)
    shared.selected_source_tools(policy)
    policy_options = {"completion_policy": policy} if policy is not None else {}
    if api:
        from scripts.experiments import assisted_api

        if no_cap:
            if args.budget_usd is not None:
                raise ValueError("Choose either --budget-usd or --no-dollar-cap")
            stage_budgets = {"initial": None, "reviewed": None}
        else:
            total = Decimal(args.budget_usd or "0")
            if not total.is_finite() or total <= 0:
                raise ValueError("API pair requires --budget-usd or explicit --no-dollar-cap")
            stage_budgets = {
                "initial": str(total * Decimal("0.6")),
                "reviewed": str(total * Decimal("0.4")),
            }
        key = load_api_key(args.env_file)
        runtime = None
    else:
        if args.budget_usd or no_cap:
            raise ValueError("Subscription runs cannot promise a dollar cap")
        runtime = subscription_runtime(args.runtime_dir)
        key = None
    args.output.mkdir(parents=True, exist_ok=False)
    results = {}
    for stage, prompt, seconds in (
        ("initial", args.prompt, args.author_seconds),
        ("reviewed", args.review_prompt, args.review_seconds),
    ):
        packet, audit, result = (
            args.output / (stage + "_packet"),
            args.output / (stage + "_audit"),
            args.output / stage,
        )
        initial = args.output / "initial" if stage == "reviewed" else None
        if api:
            prepare_api(
                args.pdf,
                prompt,
                packet,
                stage,
                seconds,
                stage_budgets[stage],
                initial,
                args.model,
                **({"no_dollar_cap": True} if no_cap else {}),
                **policy_options,
            )
            audit_api(packet, audit)
            provenance = assisted_api.run(
                packet, result, api_key=key, budget_usd=stage_budgets[stage]
            )
        else:
            shared.prepare(
                args.pdf,
                prompt,
                packet,
                stage,
                seconds,
                initial_run=initial,
                model=args.model,
                runtime_dir=runtime,
                **policy_options,
            )
            shared.audit_packet(packet, audit)
            provenance = shared.run(packet, result, audit)
        results[stage] = provenance
        if (
            provenance.get("status") != "complete"
            or provenance.get("returncode") != 0
            or provenance.get("complete") is not True
        ):
            break
    summary = {
        "backend": args.backend,
        "stages": results,
        "complete": len(results) == 2
        and all(
            p.get("status") == "complete" and p.get("complete") is True for p in results.values()
        ),
        "budget_usd": args.budget_usd if api else None,
        "no_dollar_cap": no_cap,
        "automatic_retries": 0,
        "musical_accuracy": "not_graded",
        **policy_options,
    }
    shared.write(args.output / "pair_status.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--pdf", type=Path, required=True)
    prep.add_argument("--prompt", type=Path, required=True)
    prep.add_argument("--stage", choices=("initial", "reviewed"), required=True)
    prep.add_argument("--limit-seconds", type=int, required=True)
    prep.add_argument("--model", default=shared.MODEL)
    prep.add_argument("--initial-run", type=Path)
    prep.add_argument("--runtime-dir", type=Path)
    prep.add_argument("--budget-usd")
    prep.add_argument("--no-dollar-cap", action="store_true")
    prep.add_argument("--completion-policy", choices=shared.COMPLETION_POLICIES)
    prep.add_argument(
        "--review-incomplete-checkpoint",
        action="store_true",
        help="Diagnostic API review of a valid checkpoint interrupted by max_output_tokens",
    )
    audit = commands.add_parser("audit")
    audit.add_argument("--packet", type=Path, required=True)
    run = commands.add_parser("run")
    run.add_argument("--packet", type=Path, required=True)
    run.add_argument("--audit-dir", type=Path, required=True)
    run.add_argument("--env-file", type=Path, default=ROOT / ".env")
    pair = commands.add_parser("pair", help="Prepare, audit and run one fresh author/reader pair")
    pair.add_argument("--pdf", type=Path, required=True)
    pair.add_argument("--prompt", type=Path, required=True)
    pair.add_argument("--review-prompt", type=Path, required=True)
    pair.add_argument("--author-seconds", type=int, default=1800)
    pair.add_argument("--review-seconds", type=int, default=1200)
    pair.add_argument("--model", default=shared.MODEL)
    pair.add_argument("--budget-usd")
    pair.add_argument("--no-dollar-cap", action="store_true")
    pair.add_argument("--completion-policy", choices=shared.COMPLETION_POLICIES)
    pair.add_argument("--runtime-dir", type=Path)
    pair.add_argument("--env-file", type=Path, default=ROOT / ".env")
    for command in (prep, audit, run, pair):
        command.add_argument("--backend", choices=("api", "subscription"), required=True)
        command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "pair":
        result = execute_pair(args)
    elif args.command == "prepare":
        if args.review_incomplete_checkpoint and (
            args.backend != "api" or args.stage != "reviewed"
        ):
            parser.error("--review-incomplete-checkpoint requires --backend api --stage reviewed")
        common = (args.pdf, args.prompt, args.output, args.stage, args.limit_seconds)
        if args.backend == "api":
            if bool(args.budget_usd) == args.no_dollar_cap or args.runtime_dir:
                parser.error("API prepare requires one spending policy and no CLI runtime")
            result = prepare_api(
                *common,
                args.budget_usd,
                args.initial_run,
                args.model,
                review_incomplete_checkpoint=args.review_incomplete_checkpoint,
                no_dollar_cap=args.no_dollar_cap,
                **({"completion_policy": args.completion_policy} if args.completion_policy else {}),
            )
        else:
            if args.budget_usd or args.no_dollar_cap:
                parser.error("Subscription runs cannot promise a dollar cap")
            result = shared.prepare(
                *common,
                initial_run=args.initial_run,
                model=args.model,
                runtime_dir=subscription_runtime(args.runtime_dir),
                **({"completion_policy": args.completion_policy} if args.completion_policy else {}),
            )
    elif args.command == "audit":
        result = (audit_api if args.backend == "api" else shared.audit_packet)(
            args.packet, args.output
        )
    elif args.backend == "subscription":
        result = shared.run(args.packet, args.output, args.audit_dir)
    else:
        from scripts.experiments import assisted_api

        shared.verify_seal(args.audit_dir, "astra_reproduction_audit")
        audit_record = json.loads((args.audit_dir / "audit.json").read_text())
        if (
            audit_record.get("status") != "passed"
            or audit_record.get("backend") != "api"
            or audit_record.get("input_seal_sha256")
            != shared.sha((args.packet / "seal.json").read_bytes())
        ):
            raise ValueError("Matching API audit required")
        metadata = json.loads((args.packet / "preparation.json").read_text())
        result = assisted_api.run(
            args.packet,
            args.output,
            api_key=load_api_key(args.env_file),
            budget_usd=metadata["budget_usd"],
            pricing=assisted_api.ASTRA_PRICING,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
