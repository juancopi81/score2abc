"""Validated system configurations and content-addressed evaluation identities.

Generation identities contain only source inputs and authoring machinery. Answer
and label identities are added by the grader, never to the model's task packet.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from copy import deepcopy
from pathlib import Path

DEFAULT_CONFIG = {
    "schema_version": 1,
    "id": "reviewed-six-astra-terra-v1",
    "dataset_version": "reviewed-six-pdf-v1",
    "labels_version": "reviewed_six_v3",
    "grader_version": "complete-score-v2",
    "harness": {"id": "pdf-only-tool-less-v1", "max_revision_passes": 2},
    "systems": [
        {
            "id": "astra",
            "label": "GPT-6 Astra",
            "provider": "openai",
            "adapter": "codex_subscription",
            "model": "gpt-6-astra",
            "reasoning_effort": "xhigh",
            "settings": {},
        },
        {
            "id": "terra",
            "label": "GPT-6.1 Terra",
            "provider": "openai",
            "adapter": "codex_subscription",
            "model": "gpt-6.1-terra",
            "reasoning_effort": "xhigh",
            "settings": {},
        },
    ],
}
_ID = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,95}$")
_HASH = re.compile(r"^[0-9a-f]{64}$")
_PACKAGE = Path(__file__).resolve().parent


def digest(value: object) -> str:
    """Canonical content hash; NaN and unserializable settings are rejected."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def _name(value: object, description: str) -> None:
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise ValueError(f"Invalid {description}")


def load_config(value: Path | dict | None = None) -> dict:
    """Validate configuration, without claiming its provider adapter is installed."""
    config = deepcopy(DEFAULT_CONFIG if value is None else value)
    if isinstance(value, (str, Path)):
        config = json.loads(Path(value).read_text())
    keys = {
        "schema_version",
        "id",
        "dataset_version",
        "labels_version",
        "grader_version",
        "harness",
        "systems",
    }
    if (
        not isinstance(config, dict)
        or set(config) != keys
        or type(config["schema_version"]) is not int
        or config["schema_version"] != 1
    ):
        raise ValueError("Unsupported evaluation configuration schema or fields")
    for key in ("id", "dataset_version", "labels_version", "grader_version"):
        _name(config[key], key)
    harness = config["harness"]
    if not isinstance(harness, dict) or set(harness) != {"id", "max_revision_passes"}:
        raise ValueError("Harness needs an id and max_revision_passes")
    _name(harness["id"], "harness id")
    if (
        type(harness["max_revision_passes"]) is not int
        or not 0 <= harness["max_revision_passes"] <= 2
    ):
        raise ValueError("Harness permits zero to two revision passes")
    systems = config["systems"]
    if not isinstance(systems, list) or not 1 <= len(systems) <= 20:
        raise ValueError("Configure one to twenty systems")
    system_keys = {"id", "label", "provider", "adapter", "model", "reasoning_effort", "settings"}
    seen = set()
    for system in systems:
        if not isinstance(system, dict) or set(system) != system_keys:
            raise ValueError("Unsupported system fields")
        for key in ("id", "provider", "adapter"):
            _name(system[key], f"system {key}")
        deterministic = system["provider"] == "local" and system["adapter"] == "deterministic"
        if deterministic:
            if system["model"] is not None or system["reasoning_effort"] is not None:
                raise ValueError("Deterministic systems have no model or reasoning effort")
        else:
            if not isinstance(system["model"], str) or not re.fullmatch(
                r"[a-zA-Z0-9][a-zA-Z0-9_.:/-]{0,159}", system["model"]
            ):
                raise ValueError("Invalid system model")
        if system["id"] in seen:
            raise ValueError("Duplicate system id")
        seen.add(system["id"])
        if not isinstance(system["label"], str) or not 1 <= len(system["label"]) <= 120:
            raise ValueError("System label must be nonempty text")
        if system["reasoning_effort"] is not None:
            _name(system["reasoning_effort"], "reasoning setting")
        if not isinstance(system["settings"], dict):
            raise ValueError("System settings must be an object")
    digest(config)
    return config


def identity(identifier: str, manifest: dict) -> dict:
    return {"id": identifier, "sha256": digest(manifest), "manifest": manifest}


def verify_identity(value: dict) -> None:
    if (
        not isinstance(value, dict)
        or not {"id", "sha256", "manifest"}.issubset(value)
        or set(value) - {"id", "sha256", "manifest", "provenance"}
        or not isinstance(value.get("manifest"), dict)
        or not isinstance(value.get("sha256"), str)
        or not _HASH.fullmatch(value["sha256"])
        or value["sha256"] != digest(value["manifest"])
    ):
        raise ValueError("Evaluation identity hash is invalid")
    _name(value["id"], "version id")


def code_provenance() -> dict:
    """Commit is descriptive; exact implementation hashes identify dirty work too."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_PACKAGE,
            capture_output=True,
            text=True,
            check=True,
            timeout=3,
        ).stdout.strip()
        dirty = (
            subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                cwd=_PACKAGE,
                capture_output=True,
                text=True,
                check=True,
                timeout=3,
            ).stdout
            != ""
        )
        return {"commit": commit, "tracked_worktree_dirty": dirty}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "tracked_worktree_dirty": None}


def _files(names: tuple[str, ...]) -> dict:
    result = {}
    for name in names:
        path = _PACKAGE / name
        if not path.is_file():
            raise ValueError(f"Versioned evaluation implementation is missing: {name}")
        result[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def renderer_identity() -> dict:
    """Pin renderer code, not only the wrapper or a mutable executable name."""
    renderer, node = shutil.which("abc2svg"), shutil.which("node")
    if not renderer or not node:
        return {"status": "unavailable", "files": {}, "node_version": None}
    directory = Path(renderer).resolve().parent
    names = ("abc2svg-1.js", "toaudio-1.js")
    if not all((directory / name).is_file() for name in names):
        return {"status": "unavailable", "files": {}, "node_version": None}
    try:
        version = subprocess.run(
            [node, "--version"],
            capture_output=True,
            text=True,
            check=True,
            timeout=3,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        version = None
    return {
        "status": "available",
        "node_version": version,
        "files": {
            name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in names
        },
    }


def dataset_identity(frozen: dict, config: dict) -> dict:
    manifest = {
        "split": "consumed_development",
        "task": "complete_performing_edition_from_supplied_pdf",
        "source_policy": "pdf_only",
        "works": [
            {"slug": work["slug"], "pdf_sha256": frozen["files"][work["source_pdf_path"]]["sha256"]}
            for work in frozen["works"]
        ],
    }
    return identity(config["dataset_version"], manifest)


def runner_identity() -> dict:
    """Import lazily so the runner can use configuration helpers without a cycle."""
    from score2abc.score_eval_runner import runtime_identity

    return runtime_identity()


def generation_identity(frozen: dict, config: dict, instructions: str) -> dict:
    """Freeze model-visible machinery; this never includes answers or labels."""
    harness = identity(
        config["harness"]["id"],
        {
            "policy": deepcopy(config["harness"]),
            "source_policy": "pdf_only",
            "instructions_sha256": hashlib.sha256(instructions.encode()).hexdigest(),
            "implementation": _files(
                (
                    "score_eval_config.py",
                    "score_eval_harness.py",
                    "score_eval_runner.py",
                    "score_eval.py",
                    "score_eval_abc.py",
                    "score_eval_parse.js",
                    "review_playback.js",
                )
            ),
            "renderer": renderer_identity(),
            "runner_runtime": runner_identity(),
        },
    )
    harness["provenance"] = code_provenance()
    generation = {"dataset": dataset_identity(frozen, config), "harness": harness}
    generation["sha256"] = digest({key: generation[key]["sha256"] for key in generation})
    return generation


def verify_generation(generation: dict, config: dict, frozen: dict | None = None) -> None:
    if not isinstance(generation, dict) or set(generation) != {"dataset", "harness", "sha256"}:
        raise ValueError("Generation identity is missing or invalid")
    for key in ("dataset", "harness"):
        verify_identity(generation[key])
    if generation["sha256"] != digest({k: generation[k]["sha256"] for k in ("dataset", "harness")}):
        raise ValueError("Generation identity hash differs")
    if generation["dataset"]["id"] != config["dataset_version"]:
        raise ValueError("Dataset version differs from configuration")
    if frozen is not None and generation["dataset"] != dataset_identity(frozen, config):
        raise ValueError("Run dataset differs from frozen sources")
    harness = generation["harness"]
    if (
        harness["id"] != config["harness"]["id"]
        or harness["manifest"].get("policy") != config["harness"]
    ):
        raise ValueError("Run harness differs from configuration")
    if (
        harness["manifest"].get("source_policy") != "pdf_only"
        or not _HASH.fullmatch(str(harness["manifest"].get("instructions_sha256", "")))
        or not harness["manifest"].get("implementation")
    ):
        raise ValueError("Run harness is not pinned to PDF-only instructions and implementation")


def configured_systems(config: dict, generation: dict) -> list[dict]:
    return [
        {
            **deepcopy(system),
            "harness_version": generation["harness"]["id"],
            "system_sha256": digest(
                {"system": system, "harness_sha256": generation["harness"]["sha256"]}
            ),
        }
        for system in config["systems"]
    ]


def task_id(task: dict) -> str:
    """One piece and trial under one source and system identity, independent of paths."""
    return digest(
        {
            key: task[key]
            for key in (
                "slug",
                "model_id",
                "system_sha256",
                "generation_sha256",
                "input_pdf_sha256",
                "trial",
            )
        }
    )


def verify_prepared_task(task: dict, contract: dict) -> dict:
    """Bind execution fields to the frozen system, PDF and repair policy before a call."""
    configuration = load_config(contract.get("config"))
    if "config" not in contract or contract.get("schema_version") != 2:
        raise ValueError("Prepared task requires a versioned configuration")
    generation = contract.get("generation")
    verify_generation(generation, configuration)
    if (
        contract.get("source_policy") != "pdf_only"
        or contract.get("execution_status") != "not_run"
        or contract.get("isolation_status") != "requires_enforced_runner"
    ):
        raise ValueError("Prepared task source/execution policy changed")
    systems = configured_systems(configuration, generation)
    if contract.get("systems") != systems:
        raise ValueError("Prepared system metadata differs from configuration")
    model = next((item for item in systems if item["id"] == task.get("model_id")), None)
    if model is None:
        raise ValueError("Prepared task references an unknown system")
    for key in ("model", "reasoning_effort", "provider", "adapter", "settings", "system_sha256"):
        if task.get(key) != model[key]:
            raise ValueError(f"Prepared task differs from configured system: {key}")
    policy = generation["harness"]["manifest"]
    if (
        task.get("generation_sha256") != generation["sha256"]
        or task.get("instructions_sha256") != policy["instructions_sha256"]
        or type(task.get("max_revision_passes")) is not int
        or task["max_revision_passes"] != policy["policy"]["max_revision_passes"]
        or task.get("status") != "not_run"
    ):
        raise ValueError("Prepared task instructions or repair policy differs from generation")
    sources = [
        work
        for work in generation["dataset"]["manifest"]["works"]
        if work["slug"] == task.get("slug")
    ]
    if len(sources) != 1 or task.get("input_pdf_sha256") != sources[0]["pdf_sha256"]:
        raise ValueError("Prepared PDF differs from versioned dataset")
    if task.get("input_pdf_path") != f"inputs/{task['slug']}/source.pdf":
        raise ValueError("Prepared PDF path differs from source-only packet layout")
    if (
        type(contract.get("trial")) is not int
        or contract["trial"] < 1
        or type(task.get("trial")) is not int
        or task["trial"] != contract["trial"]
    ):
        raise ValueError("Prepared task trial differs from cohort")
    if task.get("task_id") != task_id(task):
        raise ValueError("Prepared task identity differs from its content")
    return deepcopy(task)


def grading_versions(config: dict, generation: dict, frozen_root: Path, labels_root: Path) -> dict:
    grader_files = [
        "score_eval_config.py",
        "score_eval.py",
        "score_eval_abc.py",
        "score_eval_parse.js",
        "review_playback.js",
        "eval_labels.py",
    ]
    if (_PACKAGE / "score_eval_source.py").is_file():
        grader_files.append("score_eval_source.py")
    versions = {
        "dataset": generation["dataset"],
        "harness": generation["harness"],
        "labels": identity(
            config["labels_version"],
            {
                "labels_manifest_sha256": hashlib.sha256(
                    (labels_root / "manifest.json").read_bytes()
                ).hexdigest(),
                "reference_manifest_sha256": hashlib.sha256(
                    (frozen_root / "manifest.json").read_bytes()
                ).hexdigest(),
            },
        ),
        "grader": identity(
            config["grader_version"],
            {
                "implementation": _files(tuple(grader_files)),
                "renderer": renderer_identity(),
            },
        ),
        "config": identity(config["id"], config),
    }
    versions["grader"]["provenance"] = code_provenance()
    return versions
