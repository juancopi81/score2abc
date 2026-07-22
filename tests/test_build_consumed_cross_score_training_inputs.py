import hashlib
import json
import shutil
from pathlib import Path

import pytest

from score2abc.manifest import write_manifest_jsonl
from score2abc.schemas import WorkItem, WorkMetadata
from scripts.experiments.build_consumed_cross_score_training_inputs import (
    DEFAULT_NAMESPACE,
    DEFAULT_SLUG,
    build_consumed_cross_score_training_inputs,
)


def test_builds_isolated_eight_measure_namespace_with_blind_candidates(
    tmp_path: Path,
) -> None:
    out_dir = _make_carrizal_out(tmp_path)
    work_dir = out_dir / DEFAULT_SLUG
    protected = {
        namespace: _sentinel(work_dir / namespace)
        for namespace in (
            "vlm_melody_inputs",
            "vlm_melody_reviews",
            "vlm_melody_fresh_heldout",
            "notehead_reviews",
        )
    }

    destination = build_consumed_cross_score_training_inputs(out_dir)

    assert destination == (work_dir / "vlm_melody_training_inputs" / DEFAULT_NAMESPACE).resolve()
    assert {name: path.read_text(encoding="utf-8") for name, path in protected.items()} == {
        name: "do-not-touch" for name in protected
    }

    namespace_manifest = _json(destination / "manifest.json")
    assert namespace_manifest["split_status"] == "consumed_training"
    assert namespace_manifest["review_status"] == "unreviewed"
    assert namespace_manifest["eligible_for_training"] is False
    assert namespace_manifest["eligible_for_promotion"] is False
    assert namespace_manifest["measure_count"] == 8
    assert namespace_manifest["provenance"]["ground_truth_files_read"] == []

    segmentation = _json(destination / "segmentation.json")
    assert segmentation["measure_count"] == 8
    assert len(segmentation["measure_boundaries_x_fraction"]) == 9
    assert len(segmentation["crops"]) == 8
    assert _recorded_hash_matches(segmentation["source_system"])
    for crop in segmentation["crops"]:
        for key in ("measure_raw", "measure_staff", "measure_staff_overlay", "context"):
            assert _recorded_hash_matches(crop[key])

    input_rows = _jsonl(destination / "inputs_manifest.jsonl")
    candidate_rows = _jsonl(destination / "candidates_manifest.jsonl")
    assert [row["system_measure_index"] for row in input_rows] == list(range(1, 9))
    assert [row["identity"]["system_measure_index"] for row in candidate_rows] == list(range(1, 9))
    for row in candidate_rows:
        assert row["review_status"] == "unreviewed"
        assert row["candidate_generation"]["ground_truth_files_read"] == []
        assert row["candidate_generation"]["strategy"] == "staff-grid-density"
        assert _recorded_hash_matches(row["artifacts"]["candidates"])
        assert _recorded_hash_matches(row["artifacts"]["candidate_overlay"])
        candidate_payload = _json(_recorded_path(row["artifacts"]["candidates"]))
        assert candidate_payload["provenance"]["ground_truth_files_read"] == []

    mapping = _json(destination / "mapping.json")
    mapping_keys = _recursive_keys(mapping)
    assert "musicxml" not in mapping_keys
    assert "proposal" not in mapping_keys
    assert "expected" not in mapping_keys
    assert [row["system_measure_index"] for row in mapping["crops"]] == list(range(1, 9))
    assert [row["physical_measure_numbers"] for row in mapping["crops"]] == [
        [index] for index in range(1, 9)
    ]


def test_generation_is_deterministic_for_same_source_and_namespace(tmp_path: Path) -> None:
    out_dir = _make_carrizal_out(tmp_path)
    first = build_consumed_cross_score_training_inputs(out_dir)
    first_hashes = _tree_hashes(first)

    shutil.rmtree(first)
    second = build_consumed_cross_score_training_inputs(out_dir)

    assert _tree_hashes(second) == first_hashes


def test_refuses_existing_and_protected_namespaces(tmp_path: Path) -> None:
    out_dir = _make_carrizal_out(tmp_path)
    destination = build_consumed_cross_score_training_inputs(out_dir)

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        build_consumed_cross_score_training_inputs(out_dir)
    assert destination.is_dir()

    for namespace in (
        "vlm_melody_inputs",
        "vlm_melody_reviews",
        "fresh_heldout",
        "notehead_reviews",
    ):
        with pytest.raises(ValueError, match="protected"):
            build_consumed_cross_score_training_inputs(out_dir, namespace=namespace)

    with pytest.raises(ValueError, match="Unsafe"):
        build_consumed_cross_score_training_inputs(out_dir, namespace="../escape")


def _make_carrizal_out(tmp_path: Path) -> Path:
    out_dir = tmp_path / "out"
    work_dir = out_dir / DEFAULT_SLUG
    systems_dir = work_dir / "systems"
    systems_dir.mkdir(parents=True)
    source_pdf = work_dir / "source.pdf"
    source_pdf.write_bytes(b"%PDF-1.4\n")
    fixture_dir = Path(__file__).parent / "fixtures" / "barlines" / DEFAULT_SLUG
    shutil.copyfile(fixture_dir / "system_004.png", systems_dir / "system_004.png")
    write_manifest_jsonl(
        [
            WorkItem(
                slug=DEFAULT_SLUG,
                pdf_path=source_pdf,
                metadata=WorkMetadata(
                    title="Carrizal",
                    composer="Emilio Murillo",
                    rhythm="pasillo",
                    time_signature="3/4",
                    key_hint="one flat: Bb",
                ),
            )
        ],
        out_dir / "manifest.jsonl",
    )
    return out_dir


def _sentinel(directory: Path) -> Path:
    directory.mkdir(parents=True)
    path = directory / "sentinel.txt"
    path.write_text("do-not-touch", encoding="utf-8")
    return path


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _recorded_path(record: dict[str, str]) -> Path:
    path = Path(record["path"])
    return path if path.is_absolute() else Path(__file__).resolve().parents[1] / path


def _recorded_hash_matches(record: dict[str, str]) -> bool:
    path = _recorded_path(record)
    return path.is_file() and _sha256(path) == record["sha256"]


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _recursive_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return {str(key).lower() for key in value} | {
            nested_key for nested in value.values() for nested_key in _recursive_keys(nested)
        }
    if isinstance(value, list):
        return {nested_key for nested in value for nested_key in _recursive_keys(nested)}
    return set()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
