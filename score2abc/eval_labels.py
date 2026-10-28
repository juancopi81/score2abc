"""Conservative labels for the frozen, reviewed six-score editorial collection."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

V1_INVENTORY_PATH = Path(__file__).with_name("eval_labels_reviewed_six_v1.json")
INVENTORY_PATH = Path(__file__).with_name("eval_labels_reviewed_six_v2.json")
KNOWN_INVENTORIES = {
    "reviewed_six_v1": V1_INVENTORY_PATH,
    "reviewed_six_v2": INVENTORY_PATH,
}
WORK_LIMITS = {
    "amalia": (7, 48, 48),
    "atardecer": (10, 57, 57),
    "aviador": (None, 62, 64),
    "carrizal": (8, 48, 48),
    "coqueteos": (9, 52, 52),
    "rumichaca": (7, 45, 45),
}
VALID_ORIGINS = {"pdf", "human_musical_judgment", "external_recording", "unclear"}
VALID_STATUSES = {
    "supported_by_review_notes",
    "supported_by_current_abc_and_history",
    "needs_adjudication",
}
VALID_DIMENSIONS = {"notes", "rests", "chords", "form", "tempo", "duration"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _check_frozen(frozen_root: Path, expected_sha256: str) -> dict:
    manifest_path = frozen_root / "manifest.json"
    if manifest_path.is_symlink() or (frozen_root / "manifest.sha256").is_symlink():
        raise ValueError("frozen manifest path is a symlink")
    if _sha256(manifest_path) != expected_sha256:
        raise ValueError("frozen reference manifest hash mismatch")
    if (frozen_root / "manifest.sha256").read_text().strip() != expected_sha256:
        raise ValueError("frozen reference checksum file mismatch")
    return _read_json(manifest_path)


def _check_file(frozen_root: Path, manifest: dict, relative_path: str) -> str:
    if relative_path not in manifest["files"]:
        raise ValueError(f"unlisted frozen evidence: {relative_path}")
    supplied_path = frozen_root / relative_path
    if supplied_path.is_symlink():
        raise ValueError(f"frozen evidence path is a symlink: {relative_path}")
    path = supplied_path.resolve()
    if not path.is_relative_to(frozen_root.resolve()):
        raise ValueError(f"evidence escapes reference root: {relative_path}")
    actual = _sha256(path)
    if actual != manifest["files"][relative_path]["sha256"]:
        raise ValueError(f"frozen evidence hash mismatch: {relative_path}")
    return actual


def _location(location: dict | None, *, max_system: int | None, max_bar: int | None) -> None:
    if location is None:
        return
    if not isinstance(location, dict) or not location:
        raise ValueError("invalid location")
    if "bar" in location:
        if set(location) != {"bar"} or not 1 <= location["bar"] <= max_bar:
            raise ValueError(f"invalid bar location: {location}")
    elif "bar_range" in location:
        span = location["bar_range"]
        if (
            set(location) != {"bar_range"}
            or len(span) != 2
            or not 1 <= span[0] <= span[1] <= max_bar
        ):
            raise ValueError(f"invalid bar range: {location}")
    elif set(location) == {"system", "measure"}:
        if not 1 <= location["system"] <= max_system or not 1 <= location["measure"] <= 12:
            raise ValueError(f"invalid system-measure location: {location}")
    else:
        raise ValueError(f"invalid location fields: {location}")


def build_labels(frozen_root: Path, inventory_path: Path = INVENTORY_PATH) -> dict:
    """Validate the hand-authored inventory against frozen bytes and build labels."""
    frozen_root = Path(frozen_root)
    inventory = _read_json(Path(inventory_path))
    if inventory["schema_version"] != 1 or set(inventory["works"]) != set(WORK_LIMITS):
        raise ValueError("unsupported or incomplete label inventory")
    reference = _check_frozen(frozen_root, inventory["reference_manifest_sha256"])
    if len(reference["works"]) != 6:
        raise ValueError("expected six frozen works")
    output = {
        "schema_version": 1,
        "label_set": inventory.get("label_set", "reviewed_six_v1"),
        "reference_name": inventory["reference_name"],
        "reference_manifest_sha256": inventory["reference_manifest_sha256"],
        "inventory_sha256": _sha256(Path(inventory_path)),
        "source_metric_policy": "pending_exact_independent_source_alignment",
        "confidence_semantics": "not_assessed; evidence status is not a human approval grade",
        "works": [],
    }
    for frozen_work in reference["works"]:
        slug = frozen_work["slug"]
        key = next((name for name in WORK_LIMITS if f"_{name}_" in slug), None)
        if key is None:
            raise ValueError(f"unexpected work: {slug}")
        max_system, max_source_bar, max_edition_bar = WORK_LIMITS[key]
        review_path = frozen_work["answer_review_path"]
        abc_path = frozen_work["answer_abc_path"]
        review_sha = _check_file(frozen_root, reference, review_path)
        abc_sha = _check_file(frozen_root, reference, abc_path)
        if review_sha != frozen_work["review_sha256"] or abc_sha != frozen_work["abc_sha256"]:
            raise ValueError(f"frozen work hash mismatch: {slug}")
        review = _read_json(frozen_root / review_path)
        abc = (frozen_root / abc_path).read_text()
        if review["abc"] != abc:
            raise ValueError(f"review ABC export differs: {slug}")
        if (
            review["revision"] != frozen_work["revision"]
            or review["review_state"] != frozen_work["review_state"]
        ):
            raise ValueError(f"frozen review metadata differs: {slug}")
        work_inventory = inventory["works"][key]
        decisions = []
        for item in work_inventory["decisions"]:
            decision = dict(item)
            if decision["status"] not in VALID_STATUSES:
                raise ValueError(f"invalid decision status: {decision['id']}")
            if decision["evidence_origin"] not in VALID_ORIGINS:
                raise ValueError(f"invalid evidence origin: {decision['id']}")
            _location(decision["source_location"], max_system=max_system, max_bar=max_source_bar)
            _location(decision["edition_location"], max_system=max_system, max_bar=max_edition_bar)
            evidence = []
            for source in decision["evidence"]:
                if source["kind"] == "review":
                    path = review_path
                    content = review["review_notes"]
                elif source["kind"] == "abc":
                    path = abc_path
                    content = abc
                elif source["kind"] == "history":
                    revision = source["revision"]
                    prefix = f"answers/{slug}/overrides/history/{revision}-"
                    paths = [name for name in reference["files"] if name.startswith(prefix)]
                    if len(paths) != 1:
                        raise ValueError(f"missing or ambiguous history revision: {revision}")
                    path = paths[0]
                    content = _read_json(frozen_root / path)["abc"]
                elif source["kind"] == "packet":
                    path = (
                        f"answers/{slug}/context/out/collection/{source['packet']}/decisions.json"
                    )
                    content = (frozen_root / path).read_text()
                else:
                    raise ValueError(f"unknown evidence type: {source['kind']}")
                evidence_sha = _check_file(frozen_root, reference, path)
                if source["quote"] not in content:
                    raise ValueError(f"evidence quote absent: {decision['id']}")
                evidence.append({"path": path, "sha256": evidence_sha, "quote": source["quote"]})
            decision["evidence"] = evidence
            decision["confidence"] = "not_assessed"
            decisions.append(decision)
        decision_by_id = {item["id"]: item for item in decisions}
        if len(decision_by_id) != len(decisions):
            raise ValueError(f"duplicate decision ID: {slug}")
        exclusions = []
        for item in work_inventory["exclusions"]:
            exclusion = dict(item)
            for decision_id in exclusion.get("decision_ids", []):
                if decision_id not in decision_by_id:
                    raise ValueError(f"unknown exclusion decision: {decision_id}")
            for bar in exclusion.get("bars", []):
                _location({"bar": bar}, max_system=None, max_bar=max_edition_bar)
            if "bar_range" in exclusion:
                _location(
                    {"bar_range": exclusion["bar_range"]}, max_system=None, max_bar=max_edition_bar
                )
            if not exclusion["dimensions"] or not set(exclusion["dimensions"]) <= VALID_DIMENSIONS:
                raise ValueError("invalid exclusion dimensions")
            exclusions.append(exclusion)
        mapping = None
        if key == "aviador":
            path = (
                f"answers/{slug}/context/out/collection/"
                "aviador_B_cadence_20261028_v1/decisions.json"
            )
            mapping_sha = _check_file(frozen_root, reference, path)
            packet = _read_json(frozen_root / path)
            mapping = {int(src): dst for src, dst in packet["source_to_edition_bar"].items()}
            if set(mapping) != set(range(1, 63)) or set(mapping.values()) != set(range(1, 65)) - {
                29,
                45,
            }:
                raise ValueError("invalid Aviador source-to-edition mapping")
            mapping = {
                "source_to_edition_bar": {str(src): dst for src, dst in mapping.items()},
                "evidence_path": path,
                "evidence_sha256": mapping_sha,
            }
        output["works"].append(
            {
                "slug": slug,
                "title": frozen_work["title"],
                "review_revision": frozen_work["revision"],
                "review_state": frozen_work["review_state"],
                "abc_sha256": abc_sha,
                "review_sha256": review_sha,
                "coverage": "decision_level_only; unmentioned bars pending source audit",
                "source_metric_eligible_measures": [],
                "source_to_edition_mapping": mapping,
                "decisions": decisions,
                "source_metric_exclusions": exclusions,
            }
        )
    return output


def create_labels(root: Path, frozen_root: Path, inventory_path: Path = INVENTORY_PATH) -> dict:
    """Write a new sidecar exactly once, after complete in-memory validation."""
    root = Path(root)
    if root.exists():
        raise FileExistsError(f"label sidecar already exists: {root}")
    labels = build_labels(frozen_root, inventory_path)
    label_set = labels["label_set"]
    if label_set not in KNOWN_INVENTORIES or _sha256(Path(inventory_path)) != _sha256(
        KNOWN_INVENTORIES[label_set]
    ):
        raise ValueError("unregistered label inventory")
    root.mkdir(parents=True, exist_ok=False)
    content = json.dumps(labels, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    (root / "manifest.json").write_text(content, encoding="utf-8")
    if labels["label_set"] != "reviewed_six_v1":
        (root / "inventory.json").write_bytes(Path(inventory_path).read_bytes())
    (root / "manifest.sha256").write_text(_sha256(root / "manifest.json") + "\n")
    return labels


def verify_labels(root: Path, frozen_root: Path, inventory_path: Path | None = None) -> dict:
    """Check the sidecar and all pinned frozen evidence without writing files."""
    root = Path(root)
    if root.is_symlink() or any(path.is_symlink() for path in root.iterdir()):
        raise ValueError("label sidecar contains a symlink")
    actual = (root / "manifest.json").read_text(encoding="utf-8")
    if (root / "manifest.sha256").read_text().strip() != _sha256(root / "manifest.json"):
        raise ValueError("label sidecar checksum mismatch")
    saved = json.loads(actual)
    label_set = saved.get("label_set")
    if label_set not in KNOWN_INVENTORIES:
        raise ValueError(f"unknown label set: {label_set}")
    canonical_inventory = Path(inventory_path or KNOWN_INVENTORIES[label_set])
    if _sha256(canonical_inventory) != saved["inventory_sha256"]:
        raise ValueError("versioned inventory checksum mismatch")
    inventory_copy = root / "inventory.json"
    expected_files = {"manifest.json", "manifest.sha256"}
    if label_set == "reviewed_six_v2":
        expected_files.add("inventory.json")
        if _sha256(inventory_copy) != saved["inventory_sha256"]:
            raise ValueError("sidecar inventory checksum mismatch")
    if {path.name for path in root.iterdir()} != expected_files:
        raise ValueError("unexpected sidecar files")
    expected = build_labels(
        frozen_root, inventory_copy if label_set == "reviewed_six_v2" else canonical_inventory
    )
    if saved != expected:
        raise ValueError("label sidecar differs from pinned inventory or references")
    return expected
