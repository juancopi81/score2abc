"""Create-once, evidence-derived compatibility audit for the assisted comparison.

No model calls, metric recomputation, reference interpretation, or artifact edits.
Only finalized reports from cleanly completed CLI turns may be audited; score
completeness remains a separate outcome. Reviewer candidate content is checked
against its own sealed author and then excluded from cross-model fingerprints.
"""

from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from score2abc import score_eval_runner as legacy  # noqa: E402

CASES = ("aviador_s8", "rumichaca_full")
STAGES = ("initial", "reviewed")
DIMENSIONS = ("data", "prompts", "tools", "limits", "grader")
MODELS = {"astra": "gpt-6-astra", "terra": "gpt-6.1-terra"}
RUNNER = "scripts/experiments/run_astra_baseline_reproduction.py"
ADAPTER = "scripts/experiments/evaluate_astra_reproduction.py"
WORKFLOW = "adaptive_source_inspection_author_then_independent_reader_v1"
NUMERICAL_FUNCTIONS = (
    "abc_to_historical",
    "ledger_consistency",
    "_advance_gate",
    "_aviador",
    "_rumichaca",
)
EXECUTION_FUNCTIONS = (
    "run",
    "mcp_broker",
    "mirror",
    "verify_events",
    "stage_status",
    "server_policy",
)
GRADER_FILES = (
    "historical_scorer",
    "historical_adapter",
    "abc_normalizer",
    "abc_parser",
    "complete_score_grader",
    "source_grader",
)
GRADE_CONTRACT_KEYS = {
    "aviador_s8": (
        "slug",
        "system_index",
        "grading_only_expected_measure_indices",
        "historical_reference_pins",
        "historical_protocol",
        "advance_gate",
        "unsupported_scope",
    ),
    "rumichaca_full": ("slug", "frozen_reference", "labels"),
}
PREFIX = "\n\n<untrusted_candidate_draft>\n"
SUFFIX = "\n</untrusted_candidate_draft>\n"


def byte_sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def encoded(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()


def digest(value: object) -> str:
    return byte_sha(encoded(value))


class Evidence:
    def __init__(self):
        self.pins: dict[Path, str] = {}

    def read(self, path: Path) -> bytes:
        if path.is_symlink():
            raise ValueError(f"Symlink evidence is forbidden: {path}")
        path = path.resolve()
        value = path.read_bytes()
        fingerprint = byte_sha(value)
        if path in self.pins and self.pins[path] != fingerprint:
            raise ValueError(f"Evidence changed during audit: {path}")
        self.pins[path] = fingerprint
        return value

    def json(self, path: Path) -> dict:
        return json.loads(self.read(path))

    def hash(self, path: Path) -> str:
        return byte_sha(self.read(path))

    def verify(self, path: Path, expected: str) -> None:
        if self.hash(path) != expected:
            raise ValueError(f"Evidence hash mismatch: {path}")

    def recheck(self) -> None:
        for path, expected in list(self.pins.items()):
            self.verify(path, expected)


def verify_directory(evidence: Evidence, root: Path, kind: str) -> dict:
    seal = evidence.json(root / "seal.json")
    actual = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Symlink in sealed directory: {path}")
        if path.is_file() and path.name != "seal.json":
            actual[str(path.relative_to(root))] = evidence.hash(path)
    if seal.get("kind") != kind or seal.get("files") != actual:
        raise ValueError(f"Sealed directory differs: {root}")
    return seal


def resolve_pin(record: dict, base: Path) -> Path:
    path = Path(record["path"])
    return path if path.is_absolute() else base / path


def normalize_pins(value: object) -> object:
    if isinstance(value, dict):
        return {
            key: normalize_pins(item)
            for key, item in value.items()
            if not (key == "path" and "sha256" in value)
        }
    if isinstance(value, list):
        return [normalize_pins(item) for item in value]
    return value


def ast_fingerprints(source: str, names: tuple[str, ...], *, include_imports: bool = False) -> dict:
    tree = ast.parse(source)
    found = {
        node.name: digest(ast.dump(node, include_attributes=False))
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names
    }
    if set(found) != set(names):
        raise ValueError("Pinned implementation lacks an audited function")
    if include_imports:
        found["imports"] = digest(
            [
                ast.dump(node, include_attributes=False)
                for node in tree.body
                if isinstance(node, (ast.Import, ast.ImportFrom))
            ]
        )
        found["shared_constants"] = digest(
            [
                ast.dump(node, include_attributes=False)
                for node in tree.body
                if isinstance(node, ast.Assign)
                and any(
                    isinstance(target, ast.Name) and target.id in {"STAGES", "WORKFLOW"}
                    for target in node.targets
                )
            ]
        )
    return found


def git_source(ref: str, relative: str) -> str:
    # ref is passed as an argv value, never shell text; git show is read-only.
    return subprocess.check_output(["git", "show", f"{ref}:{relative}"], cwd=ROOT, text=True)


def effective_cli(source: str, metadata: dict) -> dict:
    """Evaluate only the pinned pure CLI builders, never the runner entry point."""
    tree = ast.parse(source)
    selected = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"controls", "command", "verify_model_selection"}
    ]
    namespace = {"Path": Path, "json": json, "legacy": legacy}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or target.id not in {
            "MODEL",
            "EFFORT",
            "SUPPORTED_MODELS",
        }:
            continue
        value = node.value
        if isinstance(value, ast.Tuple):
            namespace[target.id] = tuple(
                namespace[item.id] if isinstance(item, ast.Name) else ast.literal_eval(item)
                for item in value.elts
            )
        else:
            namespace[target.id] = ast.literal_eval(value)
    exec(
        compile(ast.Module(body=selected, type_ignores=[]), "<pinned-cli-builders>", "exec"),
        namespace,
    )
    workspace = Path("/private/tmp/COMPARISON_WORKSPACE")
    argv = namespace["command"](workspace, metadata)
    if argv[argv.index("--model") + 1] != metadata["model"]:
        raise ValueError("CLI requested model differs from prepared provenance")
    argv[argv.index("--model") + 1] = "<selected_model>"
    return {"argv": argv, "controls": namespace["controls"](workspace, metadata)}


def prompt_evidence(
    prompt: str, stage: str, own_abc: str | None = None, own_decisions: object = None
) -> tuple[dict, dict]:
    if stage == "initial":
        return {"generic_prompt_sha256": byte_sha(prompt.encode()), "candidate_template": None}, {}
    generic, found, candidate = prompt.partition(PREFIX)
    if not found or not candidate.endswith(SUFFIX):
        raise ValueError("Reader prompt lacks the exact sealed-candidate injection template")
    value = json.loads(candidate[: -len(SUFFIX)])
    expected = {"abc": own_abc, "decisions": own_decisions}
    expected_prompt = generic + PREFIX + json.dumps(expected, ensure_ascii=False) + SUFFIX
    if value != expected or prompt != expected_prompt:
        raise ValueError("Reader candidate differs from its own sealed initial draft")
    return {
        "generic_prompt_sha256": byte_sha(generic.encode()),
        "candidate_template": {
            "prefix": PREFIX,
            "suffix": SUFFIX,
            "encoding": "json.dumps(ensure_ascii=False)",
            "keys": ["abc", "decisions"],
        },
        "own_initial_draft_verified": True,
    }, {
        "own_abc_sha256": byte_sha(own_abc.encode()),
        "injected_candidate_sha256": byte_sha(candidate[: -len(SUFFIX)].encode()),
    }


def request_controls(request: dict, base: str, prompt: str, overview_sha: str) -> dict:
    """Check transported context, excluding only per-session identifiers and image path."""
    rows = request.get("input", [])
    if (
        len(rows) != 4
        or rows[0].get("type") != "additional_tools"
        or rows[0].get("role") != "developer"
    ):
        raise ValueError("Transport contains unexpected initial context")
    for row, expected in zip(rows[1:3], (base, legacy.PERMISSIONS_INSTRUCTIONS), strict=True):
        if (
            row.get("type") != "message"
            or row.get("role") != "developer"
            or row.get("content") != [{"type": "input_text", "text": expected}]
        ):
            raise ValueError("Transport developer instructions differ from supplied context")
    user = rows[3]
    content = user.get("content", [])
    if user.get("type") != "message" or user.get("role") != "user" or len(content) != 4:
        raise ValueError("Transport user context differs")
    wrapper = content[0].get("text", "")
    if (
        content[0].get("type") != "input_text"
        or not wrapper.startswith('<image name=[Image #1] path="/')
        or not wrapper.endswith('/overview.png">')
    ):
        raise ValueError("Unexpected source overview wrapper")
    if content[2:] != [
        {"type": "input_text", "text": "</image>"},
        {"type": "input_text", "text": prompt},
    ]:
        raise ValueError("Transport prompt differs from prepared prompt")
    image = content[1]
    if (
        set(image) != {"type", "image_url"}
        or image["type"] != "input_image"
        or not image["image_url"].startswith("data:image/png;base64,")
    ):
        raise ValueError("Unexpected transported image modality")
    if (
        byte_sha(base64.b64decode(image["image_url"].split(",", 1)[1], validate=True))
        != overview_sha
    ):
        raise ValueError("Transported overview differs from supplied pixels")
    excluded = {"input", "client_metadata", "prompt_cache_key", "model"}
    return {key: value for key, value in request.items() if key not in excluded}


def observe_runtime_binaries(evidence: Evidence, recorded: dict) -> dict:
    observations = {}
    for name, expected in recorded.items():
        path = Path(name)
        observed = evidence.hash(path) if path.is_file() and not path.is_symlink() else None
        observations[name] = {
            "recorded_sha256": expected,
            "current_sha256": observed,
            "current_status": (
                "matches_sealed_preparation"
                if observed == expected
                else "changed_since_preparation" if observed is not None else "unavailable"
            ),
        }
    return observations


def clean_runtime_turn(provenance: dict) -> bool:
    """A finished model session is distinct from a complete transcription."""
    return (
        type(provenance.get("complete")) is bool
        and provenance.get("status") == ("complete" if provenance["complete"] else "partial")
        and type(provenance.get("returncode")) is int
        and provenance["returncode"] == 0
        and provenance.get("completed_turn") is True
        and provenance.get("quarantined") is False
        and provenance.get("timed_out") is False
        and "error" in provenance
        and provenance["error"] is None
    )


def author_review_policy(metadata: dict, stage: str) -> dict | None:
    allowed = metadata.get("allow_partial_author", False)
    if type(allowed) is not bool or (stage != "reviewed" and allowed):
        raise ValueError("Partial-author review policy must be an explicit reviewer boolean")
    return {"allow_partial_author": allowed} if stage == "reviewed" else None


def load_report(
    evidence: Evidence,
    cohort: Path,
    case_id: str,
    system: str,
    plan: dict,
    plan_path: Path,
    *,
    case_root: Path | None = None,
) -> tuple[dict, dict]:
    case = case_root if case_root is not None else cohort / case_id
    report = evidence.json(case / "evaluation/report.json")
    if (
        report.get("schema_version") != 1
        or report.get("kind") != "astra_assisted_workflow_reproduction_report"
        or report.get("target_id") != case_id
        or report.get("model") != MODELS[system]
        or report.get("reasoning_effort") != "xhigh"
        or report.get("workflow_id") != WORKFLOW
        or report.get("both_stages_verified_before_reference_content") is not True
        or set(report.get("stages", {})) != set(STAGES)
        or any(
            not clean_runtime_turn(lane.get("runtime", {}))
            or lane.get("completion_claim", lane["runtime"]["complete"])
            is not lane["runtime"]["complete"]
            for lane in report.get("stages", {}).values()
        )
    ):
        raise ValueError(f"A finalized matching report is required: {system}/{case_id}")
    pinned = {}
    for record in report["pins"]:
        path = resolve_pin(record, ROOT).resolve()
        if path in pinned and pinned[path] != record["sha256"]:
            raise ValueError("Conflicting report evidence pins")
        pinned[path] = record["sha256"]

    def report_pin(path: Path) -> None:
        path = path.resolve()
        if path not in pinned:
            raise ValueError(f"Artifact absent from final report pins: {path}")
        evidence.verify(path, pinned[path])

    report_pin(plan_path)
    report_pin(case / "submission_seal.json")
    submission = evidence.json(case / "submission_seal.json")
    if (
        submission.get("kind") != "astra_assisted_workflow_submission_seal"
        or submission.get("status") != "both_stages_frozen_before_reference_scoring"
        or submission.get("target_id") != case_id
        or submission.get("source_pdf_sha256") != plan["targets"][case_id]["source_pdf"]["sha256"]
        or submission.get("plan_sha256") != evidence.hash(plan_path)
        or submission.get("stages") != report.get("stage_artifacts")
        or submission.get("reference_feedback_given") is not False
    ):
        raise ValueError("Submission seal/report/plan binding differs")
    if plan.get("model") != MODELS[system] or plan.get("source_policy") != "supplied_pdf_only":
        raise ValueError("Plan model or source policy differs")
    stages = {}
    for stage in STAGES:
        packet, audit_dir, run_dir = case / f"{stage}_packet", case / f"{stage}_audit", case / stage
        for directory, kind in (
            (packet, "astra_reproduction_input"),
            (audit_dir, "astra_reproduction_audit"),
            (run_dir, "astra_reproduction_stage"),
        ):
            report_pin(directory / "seal.json")
            verify_directory(evidence, directory, kind)
        metadata = evidence.json(packet / "preparation.json")
        author_review_policy(metadata, stage)
        audit = evidence.json(audit_dir / "audit.json")
        provenance = evidence.json(run_dir / "stage_provenance.json")
        if report["stages"][stage]["runtime"] != provenance:
            raise ValueError("Report runtime differs from sealed stage provenance")
        for key, expected in {
            "model": MODELS[system],
            "reasoning_effort": "xhigh",
            "workflow_id": WORKFLOW,
            "stage": stage,
            "reference_access": False,
        }.items():
            if metadata.get(key) != expected or provenance.get(key) != expected:
                raise ValueError(f"Stage identity differs at {key}")
        if (
            audit.get("status") != "passed"
            or audit.get("input_seal_sha256") != evidence.hash(packet / "seal.json")
            or provenance.get("input_seal_sha256") != audit["input_seal_sha256"]
            or provenance.get("audit_sha256") != evidence.hash(audit_dir / "seal.json")
        ):
            raise ValueError("Prepared input/audit/run linkage differs")
        for key, filename in (
            ("abc", "score.abc"),
            ("decisions", "decisions.json"),
            ("provenance", "stage_provenance.json"),
        ):
            record = submission["stages"][stage][key]
            path = resolve_pin(record, case)
            if path.resolve() != (run_dir / filename).resolve():
                raise ValueError("Submission artifact does not bind the audited stage file")
            report_pin(path)
            evidence.verify(path, record["sha256"])
        stages[stage] = SimpleNamespace(
            packet=packet,
            audit_dir=audit_dir,
            run_dir=run_dir,
            metadata=metadata,
            audit=audit,
            provenance=provenance,
        )
    if (
        stages["initial"].provenance["complete"] is False
        and stages["reviewed"].metadata.get("allow_partial_author") is not True
    ):
        raise ValueError("Review of a partial author requires an explicit sealed policy exception")
    return report, stages


def stage_dimensions(
    evidence: Evidence,
    stage: str,
    lane: SimpleNamespace,
    initial: SimpleNamespace,
    report: dict,
    plan: dict,
    case_id: str,
    runner_source: str,
    adapter_source: str,
) -> tuple[dict, dict]:
    metadata, audit, provenance = lane.metadata, lane.audit, lane.provenance
    bundle = lane.packet / "bundle"
    if byte_sha(runner_source.encode()) != metadata["implementation"][RUNNER]:
        raise ValueError("Runner source does not match the actual prepared implementation")
    for relative, expected in metadata["implementation"].items():
        if relative != RUNNER:
            evidence.verify(ROOT / relative, expected)
    source_sha, overview_sha = evidence.hash(bundle / "source.pdf"), evidence.hash(
        bundle / "overview.png"
    )
    if (
        source_sha != metadata["source_pdf_sha256"]
        or source_sha != provenance["source_pdf_sha256"]
        or source_sha != plan["targets"][case_id]["source_pdf"]["sha256"]
        or audit["input_image_sha256"] != overview_sha
    ):
        raise ValueError("PDF/overview identity differs between preparation and audit")
    tool_config = evidence.json(bundle / "tool_config.json")
    prompt = evidence.read(bundle / "prompt.txt").decode()
    if (
        byte_sha(prompt.encode()) != metadata["prompt_sha256"]
        or audit["prompt_sha256"] != metadata["prompt_sha256"]
    ):
        raise ValueError("Prompt identity differs between preparation and audit")
    own_abc = evidence.read(initial.run_dir / "score.abc").decode() if stage == "reviewed" else None
    own_decisions = (
        evidence.json(initial.run_dir / "decisions.json") if stage == "reviewed" else None
    )
    prompts, injection_proof = prompt_evidence(prompt, stage, own_abc, own_decisions)
    prompts["base_sha256"] = evidence.hash(bundle / "base.txt")
    if stage == "reviewed" and (
        metadata["initial_abc_sha256"] != byte_sha(own_abc.encode())
        or provenance["initial_abc_sha256"] != metadata["initial_abc_sha256"]
        or provenance["independent_reader"] is not True
    ):
        raise ValueError("Independent reader is not bound to its own initial draft")
    requests = evidence.json(lane.audit_dir / "requests.json")
    request = evidence.json(lane.audit_dir / "request.json")
    if requests[0] != request or digest(request) != audit["request_sha256"]:
        raise ValueError("Initial request differs from audited transport")
    inventory_output = [
        row
        for row in requests[1]["input"]
        if row.get("type") == "custom_tool_call_output" and row.get("call_id") == "audit-inventory"
    ]
    inventory = json.loads(inventory_output[0]["output"][1]["text"])
    wrapper = request["input"][0]["tools"]
    if (
        inventory != audit["native_inventory"]
        or digest(inventory) != metadata["registry_sha256"]
        or audit["registry_sha256"] != metadata["registry_sha256"]
        or digest(wrapper) != metadata["wrapper_sha256"]
        or audit["wrapper_sha256"] != metadata["wrapper_sha256"]
    ):
        raise ValueError("Native capabilities differ from pinned wrapper/inventory")
    if (
        request["model"] != metadata["model"]
        or request["reasoning"]["effort"] != metadata["reasoning_effort"]
    ):
        raise ValueError("Request selected a different model/effort")
    catalog = evidence.json(bundle / "model_catalog.json")
    if (
        len(catalog["models"]) != 1
        or catalog["models"][0]["slug"] != metadata["model"]
        or any(catalog["models"][0].get(k) != v for k, v in metadata["catalog_overrides"].items())
    ):
        raise ValueError("Prepared model catalog differs from selected capabilities")
    tool_bundle = {
        str(path.relative_to(bundle)): evidence.hash(path)
        for path in sorted(bundle.rglob("*"))
        if path.is_file()
        and path.name
        not in {"source.pdf", "overview.png", "prompt.txt", "base.txt", "model_catalog.json"}
    }
    tools = {
        "bundle_files": tool_bundle,
        "native_wrapper": wrapper,
        "native_inventory": inventory,
        "catalog_overrides": metadata["catalog_overrides"],
        "native_tool_mode": metadata["native_tool_mode"],
        "cli": effective_cli(runner_source, metadata),
        "transport_controls": request_controls(
            request, evidence.read(bundle / "base.txt").decode(), prompt, overview_sha
        ),
        "execution_functions": ast_fingerprints(runner_source, EXECUTION_FUNCTIONS),
        "binary_files": {
            **metadata["base_runtime"]["binary_files"],
            **metadata["host_binary_files"],
            **metadata["helper_binary_files"],
            metadata["python"]: metadata["python_sha256"],
        },
    }
    # The installed app may have been upgraded after a sealed run. Its current
    # bytes cannot establish which executable ran historically. Compare the
    # preparation's sealed hashes and disclose availability separately below.
    injection_proof["runtime_binary_observations"] = observe_runtime_binaries(
        evidence, tools["binary_files"]
    )
    deadline = metadata["limit_seconds"]
    if (
        deadline != provenance["deadline_seconds"]
        or deadline != plan["stage_protocol"]["deadline_seconds"][case_id][stage]
    ):
        raise ValueError("Runtime deadline differs from the preregistered allowance")
    helper_ast = ast.parse(evidence.read(bundle / "astra_source_tools.py").decode())
    constants = {
        target.id: ast.literal_eval(node.value)
        for node in helper_ast.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
        and target.id
        in {"MAX_TOOL_CALLS", "MAX_DRAFTS", "MAX_DRAFT_BYTES", "ALLOWED_DPI", "MAX_IMAGE_SIDE"}
    }
    implementations = report["implementations"]
    if byte_sha(adapter_source.encode()) != implementations["adapter"]["sha256"]:
        raise ValueError("Evaluator wrapper source does not match finalized report")
    for name in GRADER_FILES:
        record = implementations[name]
        evidence.verify(resolve_pin(record, ROOT), record["sha256"])
    grader = {
        "core_files": {name: implementations[name]["sha256"] for name in GRADER_FILES},
        "renderer": implementations["renderer"],
        "numerical_functions": ast_fingerprints(
            adapter_source, NUMERICAL_FUNCTIONS, include_imports=True
        ),
        "contract": normalize_pins(
            {key: plan["targets"][case_id][key] for key in GRADE_CONTRACT_KEYS[case_id]}
        ),
    }
    return {
        "data": {
            "source_pdf_sha256": source_sha,
            "overview_sha256": overview_sha,
            "pages": tool_config["pages"],
            "source_policy": plan["source_policy"],
            "reference_access": provenance["reference_access"],
        },
        "prompts": prompts,
        "tools": tools,
        "limits": {
            "deadline_seconds": deadline,
            "reasoning_effort": metadata["reasoning_effort"],
            "tool_bounds": constants,
            "author_review_policy": author_review_policy(metadata, stage),
        },
        "grader": grader,
    }, injection_proof


def differing_paths(left: object, right: object, prefix: str = "") -> list[str]:
    if left == right:
        return []
    if isinstance(left, dict) and isinstance(right, dict):
        result = []
        for key in sorted(set(left) | set(right)):
            if key not in left or key not in right:
                result.append(f"{prefix}.{key}".strip("."))
                continue
            result.extend(
                differing_paths(left.get(key), right.get(key), f"{prefix}.{key}".strip("."))
            )
        return result
    return [prefix or "value"]


def compare_dimensions(case_id: str, stage: str, astra: dict, terra: dict) -> list[dict]:
    rows = []
    for dimension in DIMENSIONS:
        left, right = astra[dimension], terra[dimension]
        differences = differing_paths(left, right)
        rows.append(
            {
                "case_id": case_id,
                "stage": stage,
                "dimension": dimension,
                "status": "mismatched" if differences else "matched",
                "fingerprints": {"astra": digest(left), "terra": digest(right)},
                "basis": (
                    "Verified normalized evidence matches."
                    if not differences
                    else "Differences: " + ", ".join(differences[:8])
                ),
                "differences": differences,
            }
        )
    return rows


def select_runner_source(sources: list[str], expected: str) -> str:
    for source in sources:
        if byte_sha(source.encode()) == expected:
            return source
    raise ValueError("No supplied runner source matches the sealed preparation implementation")


def audit_comparison(
    astra_root: Path,
    terra_root: Path,
    *,
    baseline_ref: str = "892256d",
    terra_rumichaca_root: Path | None = None,
    runner_sources: tuple[Path, ...] = (),
) -> dict:
    case_roots = {
        "astra": {case: astra_root / case for case in CASES},
        "terra": {case: terra_root / case for case in CASES},
    }
    if terra_rumichaca_root is not None:
        case_roots["terra"]["rumichaca_full"] = terra_rumichaca_root
    # Refuse to publish an audit for a partially generated cohort.
    for roots in case_roots.values():
        for root in roots.values():
            if not (root / "evaluation/report.json").is_file():
                raise ValueError(
                    "All four finalized reports are required before compatibility audit"
                )
    evidence = Evidence()
    plan_paths = {
        "astra": astra_root / "evaluation_plan.json",
        "terra": terra_root.parent / "evaluation_plan.json",
    }
    plans = {name: evidence.json(path) for name, path in plan_paths.items()}
    sources = [
        git_source(baseline_ref, RUNNER),
        evidence.read(ROOT / RUNNER).decode(),
        *(evidence.read(path).decode() for path in runner_sources),
    ]
    adapter_sources = {
        "astra": git_source(baseline_ref, ADAPTER),
        "terra": evidence.read(ROOT / ADAPTER).decode(),
    }
    fingerprints, normalized, injection_proofs, runtime_observations = {}, {}, {}, {}
    execution_outcomes = {}
    for system, root in (("astra", astra_root), ("terra", terra_root)):
        fingerprints[system], normalized[system], injection_proofs[system] = {}, {}, {}
        runtime_observations[system] = {}
        execution_outcomes[system] = {}
        for case_id in CASES:
            report, stages = load_report(
                evidence,
                root,
                case_id,
                system,
                plans[system],
                plan_paths[system],
                case_root=case_roots[system][case_id],
            )
            fingerprints[system][case_id] = evidence.hash(
                case_roots[system][case_id] / "evaluation/report.json"
            )
            normalized[system][case_id], injection_proofs[system][case_id] = {}, {}
            runtime_observations[system][case_id] = {}
            execution_outcomes[system][case_id] = {}
            for stage in STAGES:
                provenance = stages[stage].provenance
                execution_outcomes[system][case_id][stage] = {
                    "runtime_turn_completed_cleanly": clean_runtime_turn(provenance),
                    "status": provenance["status"],
                    "score_complete": provenance["complete"],
                    "author_review_policy": author_review_policy(stages[stage].metadata, stage),
                }
                dimensions, proof = stage_dimensions(
                    evidence,
                    stage,
                    stages[stage],
                    stages["initial"],
                    report,
                    plans[system],
                    case_id,
                    select_runner_source(sources, stages[stage].metadata["implementation"][RUNNER]),
                    adapter_sources[system],
                )
                normalized[system][case_id][stage] = dimensions
                runtime_observations[system][case_id][stage] = proof.pop(
                    "runtime_binary_observations"
                )
                injection_proofs[system][case_id][stage] = proof
    checks = [
        row
        for case_id in CASES
        for stage in STAGES
        for row in compare_dimensions(
            case_id, stage, normalized["astra"][case_id][stage], normalized["terra"][case_id][stage]
        )
    ]
    evidence.recheck()
    return {
        "schema_version": 1,
        "kind": "assisted_model_comparison_compatibility_audit",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "report_sha256": fingerprints,
        "case_roots": {
            system: {case: str(path.resolve()) for case, path in roots.items()}
            for system, roots in case_roots.items()
        },
        "checks": checks,
        "normalized_evidence": normalized,
        "reviewer_injection_proofs": injection_proofs,
        "runtime_binary_observations": runtime_observations,
        "execution_outcomes": execution_outcomes,
        "basis": {
            "baseline_git_ref": baseline_ref,
            "selected_model_excluded_from_cli_fingerprint": True,
            "intrinsic_model_catalog_fields_excluded": True,
            "model_own_candidate_excluded_after_self_binding": True,
            "numerical_adapter_functions": list(NUMERICAL_FUNCTIONS),
            "all_referenced_evidence_rechecked": True,
            "runtime_binary_basis": "Hashes recorded in sealed preparation; no historical rerun.",
            "current_runtime_availability_is_separate_from_comparison": True,
            "score_completeness_is_separate_from_runtime_completion": True,
        },
        "evidence_pins": [
            {"path": str(path), "sha256": expected}
            for path, expected in sorted(evidence.pins.items())
        ],
    }


def write_audit(output: Path, report: dict) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        handle.write(encoded(report).decode())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("astra_root", type=Path)
    parser.add_argument("terra_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--baseline-ref", default="892256d")
    parser.add_argument("--terra-rumichaca-root", type=Path)
    parser.add_argument(
        "--runner-source",
        type=Path,
        action="append",
        default=[],
        help="Archived pinned runner source",
    )
    args = parser.parse_args()
    report = audit_comparison(
        args.astra_root,
        args.terra_root,
        baseline_ref=args.baseline_ref,
        terra_rumichaca_root=args.terra_rumichaca_root,
        runner_sources=tuple(args.runner_source),
    )
    write_audit(args.output, report)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "checks": dict(Counter(row["status"] for row in report["checks"])),
                "differences": [
                    row["basis"] for row in report["checks"] if row["status"] != "matched"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
