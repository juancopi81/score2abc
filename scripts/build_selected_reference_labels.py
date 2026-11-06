"""Build/verify a standalone, two-reader label sidecar without changing benchmark defaults."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from score2abc.reference_freeze import verify_freeze
from score2abc.score_eval_abc import normalize_abc
from score2abc.selected_reference_labels import compile_source_labels, digest


def build_payload(frozen: Path, evidence: dict[str, bytes], label_set: str) -> dict:
    if set(evidence) != {"packet.json", "author.json", "review.json", "decisions.json"}:
        raise ValueError("Unexpected evidence file inventory")
    if not isinstance(label_set, str) or not label_set.strip():
        raise ValueError("Label set needs a name")
    manifest = verify_freeze(frozen)
    if len(manifest["works"]) != 1:
        raise ValueError("Selected labels require one separately frozen edition")
    work = manifest["works"][0]
    packet = json.loads(evidence["packet.json"])
    if (
        packet["freeze_manifest_sha256"] != digest((frozen / "manifest.json").read_bytes())
        or packet["reference_abc_sha256"] != work["abc_sha256"]
        or packet["reference_pdf_sha256"] != manifest["files"][work["source_pdf_path"]]["sha256"]
    ):
        raise ValueError("Audit packet does not match the frozen edition")
    normalized = normalize_abc((frozen / work["answer_abc_path"]).read_text())
    labels = compile_source_labels(
        evidence["packet.json"],
        normalized,
        [json.loads(evidence[name]) for name in ("author.json", "review.json")],
    )
    decisions = json.loads(evidence["decisions.json"])
    identifiers = set()
    for decision in decisions["decisions"]:
        if not decision.get("id") or decision["id"] in identifiers or not decision.get("evidence"):
            raise ValueError("Editorial decisions need unique IDs and pinned evidence")
        identifiers.add(decision["id"])
        for item in decision["evidence"]:
            relative = item["path"]
            if relative not in manifest["files"]:
                raise ValueError("Decision evidence is not in the frozen reference")
            raw = (frozen / relative).read_bytes()
            if digest(raw) != item["sha256"]:
                raise ValueError("Decision evidence hash differs")
            field = item.get("field")
            content = json.loads(raw)[field] if field else raw.decode("utf-8")
            if (
                not isinstance(item.get("quote"), str)
                or not item["quote"]
                or item["quote"] not in content
            ):
                raise ValueError("Decision quote is absent from its frozen evidence")
    return {
        "schema_version": 1,
        "label_set": label_set,
        "reference_manifest_sha256": packet["freeze_manifest_sha256"],
        "reference_abc_sha256": work["abc_sha256"],
        "reference_revision": work["revision"],
        "benchmark_registration": "separate_sidecar; not added to reviewed_six",
        "human_approved_source_audit": False,
        "evidence_sha256": {name: digest(raw) for name, raw in evidence.items()},
        "works": [{"slug": work["slug"], **labels, "decisions": decisions["decisions"]}],
    }


def verify(root: Path, frozen: Path) -> dict:
    root = root.resolve()
    raw = (root / "manifest.json").read_bytes()
    if (root / "manifest.sha256").read_text().strip() != digest(raw):
        raise ValueError("Label manifest seal changed")
    manifest = json.loads(raw)
    expected = {"labels.json", "packet.json", "author.json", "review.json", "decisions.json"}
    if set(manifest["files"]) != expected:
        raise ValueError("Unexpected sidecar file inventory")
    actual = {
        p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() or p.is_symlink()
    }
    if actual != expected | {"manifest.json", "manifest.sha256"}:
        raise ValueError("Label sidecar file set changed")
    payloads = {}
    for name, pin in manifest["files"].items():
        p = root / name
        if p.is_symlink():
            raise ValueError("Symlink in label sidecar")
        payloads[name] = p.read_bytes()
        if digest(payloads[name]) != pin["sha256"] or len(payloads[name]) != pin["size"]:
            raise ValueError("Label sidecar content changed")
    stored = json.loads(payloads.pop("labels.json"))
    if stored != build_payload(frozen, payloads, stored["label_set"]):
        raise ValueError("Label sidecar no longer reproduces from the pinned audits")
    return stored


def publish(frozen: Path, evidence: dict[str, bytes], destination: Path, label_set: str) -> dict:
    frozen, destination = frozen.resolve(), destination.resolve()
    if destination.is_relative_to(frozen):
        raise ValueError("Labels must remain outside the immutable reference")
    if destination.exists():
        raise ValueError("Label destination already exists")
    payload = build_payload(frozen, evidence, label_set)
    files = {
        **evidence,
        "labels.json": (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode(),
    }
    manifest = {
        "schema_version": 1,
        "files": {n: {"sha256": digest(b), "size": len(b)} for n, b in files.items()},
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir()
    try:
        for name, raw in files.items():
            (destination / name).write_bytes(raw)
        raw = (json.dumps(manifest, indent=2) + "\n").encode()
        (destination / "manifest.json").write_bytes(raw)
        (destination / "manifest.sha256").write_text(digest(raw) + "\n")
        return verify(destination, frozen)
    except Exception:
        shutil.rmtree(destination)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("build")
    create.add_argument("frozen", type=Path)
    create.add_argument("destination", type=Path)
    create.add_argument("--label-set", required=True)
    for name in ("packet", "author", "review", "decisions"):
        create.add_argument(f"--{name}", type=Path, required=True)
    check = commands.add_parser("verify")
    check.add_argument("frozen", type=Path)
    check.add_argument("destination", type=Path)
    args = parser.parse_args()
    if args.command == "build":
        evidence = {
            name + ".json": getattr(args, name).read_bytes()
            for name in ("packet", "author", "review", "decisions")
        }
        result = publish(args.frozen, evidence, args.destination, args.label_set)
    else:
        result = verify(args.destination, args.frozen)
    print(
        json.dumps(
            {"label_set": result["label_set"], "coverage": result["works"][0]["coverage"]}, indent=2
        )
    )


if __name__ == "__main__":
    main()
