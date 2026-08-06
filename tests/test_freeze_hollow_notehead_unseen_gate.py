import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from score2abc.manifest import write_manifest_jsonl
from score2abc.schemas import WorkItem, WorkMetadata
from scripts.build_vlm_melody_inputs import StaffEstimate
from scripts.build_vlm_notehead_candidates import GridCandidate
from scripts.experiments import freeze_hollow_notehead_unseen_gate as gate
from scripts.experiments.spike_consumed_hollow_notehead_proposals import HollowProposal


def test_freezes_eight_ordered_truth_blind_measures_with_individual_proposals(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out_dir = _make_pipeline_out(tmp_path)
    selection_manifest = _make_selection_manifest(tmp_path, out_dir)
    proposal_calls = _patch_deterministic_generation(monkeypatch)
    forbidden_paths = _make_forbidden_truth_traps(tmp_path, out_dir)
    opened_forbidden: list[Path] = []
    original_path_open = Path.open

    def guarded_open(path: Path, *args: object, **kwargs: object):
        normalized_parts = {part.casefold() for part in path.resolve().parts}
        is_forbidden_namespace = bool(
            normalized_parts.intersection({"ground_truth", "musicxml", "reviews"})
        )
        if is_forbidden_namespace or any(
            path.resolve() == forbidden.resolve() for forbidden in forbidden_paths
        ):
            opened_forbidden.append(path)
            raise AssertionError(f"Forbidden truth/review path opened: {path}")
        return original_path_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    manifest_path = gate.freeze_hollow_notehead_unseen_gate(
        out_dir,
        selection_manifest=selection_manifest,
    )

    assert opened_forbidden == []
    assert len(proposal_calls) == 8
    assert manifest_path == (
        out_dir
        / gate.DEFAULT_SLUG
        / gate.OUTPUT_SUBDIR
        / "v1"
        / "system_004"
        / "frozen"
        / "sealed_manifest.json"
    )
    payload = gate.verify_sealed_manifest(manifest_path)
    assert payload["kind"] == gate.SEALED_KIND
    assert payload["status"] == "frozen_awaiting_human_review"
    assert payload["split"] == "fresh_heldout_morphology"
    assert payload["truth_accessed"] is False
    assert payload["gate"]["end_to_end_transcription_claim"] is False
    assert payload["gate"]["eligible_for_candidate_pipeline_integration"] is False
    assert payload["measure_count"] == 8
    assert _record_hash_matches(payload["selection"], manifest_path.parent)
    assert [row["identity"]["system_measure_index"] for row in payload["measures"]] == list(
        range(1, 9)
    )

    frozen_root = manifest_path.parent
    for row in payload["measures"]:
        assert set(row) == {
            "identity",
            "raw_image",
            "candidate_artifact",
            "proposal_artifact",
        }
        for key in ("raw_image", "candidate_artifact", "proposal_artifact"):
            assert set(row[key]) == {"path", "sha256"}
            assert _record_hash_matches(row[key], frozen_root)
        candidate = _json(_record_path(row["candidate_artifact"], frozen_root))
        assert candidate["source_image_size_px"] == {"width": 200, "height": 200}
        assert candidate["staff_lines_y_px"] == [60, 80, 100, 120, 140]
        assert candidate["candidate_count"] == 2
        assert candidate["provenance"]["ground_truth_files_read"] == []
        proposal = _json(_record_path(row["proposal_artifact"], frozen_root))
        assert proposal["proposal_count"] == 1
        assert proposal["proposals"][0]["support_candidate_ids"] == ["c001", "c002"]
        assert proposal["provenance"]["ground_truth_files_read"] == []

    serialized_values = json.dumps(payload, sort_keys=True).lower()
    assert "dataset/ground_truth" not in serialized_values
    assert ".musicxml" not in serialized_values
    assert "/reviews/" not in serialized_values

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        gate.freeze_hollow_notehead_unseen_gate(
            out_dir,
            selection_manifest=selection_manifest,
        )


def test_hash_integrity_fails_after_candidate_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out_dir = _make_pipeline_out(tmp_path)
    selection_manifest = _make_selection_manifest(tmp_path, out_dir)
    _patch_deterministic_generation(monkeypatch)
    manifest_path = gate.freeze_hollow_notehead_unseen_gate(
        out_dir,
        selection_manifest=selection_manifest,
    )
    payload = _json(manifest_path)
    candidate_path = _record_path(
        payload["measures"][0]["candidate_artifact"],
        manifest_path.parent,
    )
    candidate_path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="hash drift"):
        gate.verify_sealed_manifest(manifest_path)


def test_wrong_measure_count_fails_without_publishing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out_dir = _make_pipeline_out(tmp_path)
    selection_manifest = _make_selection_manifest(tmp_path, out_dir)
    monkeypatch.setattr(gate, "detect_barlines", lambda _: [index / 7 for index in range(8)])
    monkeypatch.setattr(
        gate,
        "measure_boundaries_for_system",
        lambda _path, values: list(values),
    )

    with pytest.raises(ValueError, match="Expected 8 measures"):
        gate.freeze_hollow_notehead_unseen_gate(
            out_dir,
            selection_manifest=selection_manifest,
        )

    assert not _destination(out_dir).exists()


def test_detected_boundaries_must_match_preregistration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out_dir = _make_pipeline_out(tmp_path)
    selection_manifest = _make_selection_manifest(tmp_path, out_dir)
    changed = [index / 8 for index in range(9)]
    changed[4] += 0.01
    monkeypatch.setattr(gate, "detect_barlines", lambda _: list(changed))
    monkeypatch.setattr(
        gate,
        "measure_boundaries_for_system",
        lambda _path, values: list(values),
    )

    with pytest.raises(ValueError, match="changed from preregistration"):
        gate.freeze_hollow_notehead_unseen_gate(
            out_dir,
            selection_manifest=selection_manifest,
        )

    assert not _destination(out_dir).exists()


def test_selection_rule_hash_drift_fails_before_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out_dir = _make_pipeline_out(tmp_path)
    selection_manifest = _make_selection_manifest(tmp_path, out_dir)
    selection = _json(selection_manifest)
    selection["frozen_rule"]["sha256"] = "0" * 64
    selection_manifest.write_text(json.dumps(selection) + "\n", encoding="utf-8")
    proposal_calls = _patch_deterministic_generation(monkeypatch)

    with pytest.raises(ValueError, match="hash drift"):
        gate.freeze_hollow_notehead_unseen_gate(
            out_dir,
            selection_manifest=selection_manifest,
        )

    assert proposal_calls == []
    assert not _destination(out_dir).exists()


def test_stale_temp_directory_fails_closed(
    tmp_path: Path,
) -> None:
    out_dir = _make_pipeline_out(tmp_path)
    selection_manifest = _make_selection_manifest(tmp_path, out_dir)
    destination = _destination(out_dir)
    temp_dir = destination.with_name(".frozen.tmp")
    temp_dir.mkdir(parents=True)
    (temp_dir / "partial.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="stale"):
        gate.freeze_hollow_notehead_unseen_gate(
            out_dir,
            selection_manifest=selection_manifest,
        )

    assert not destination.exists()
    assert (temp_dir / "partial.json").is_file()


def test_malformed_generated_candidate_fails_without_publishing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out_dir = _make_pipeline_out(tmp_path)
    selection_manifest = _make_selection_manifest(tmp_path, out_dir)
    _patch_deterministic_generation(monkeypatch)
    original_builder = gate._build_candidate_payload

    def malformed_builder(*args: object, **kwargs: object) -> dict:
        payload = original_builder(*args, **kwargs)
        payload.pop("staff_lines_y_px")
        return payload

    monkeypatch.setattr(gate, "_build_candidate_payload", malformed_builder)

    with pytest.raises(ValueError, match="five staff lines"):
        gate.freeze_hollow_notehead_unseen_gate(
            out_dir,
            selection_manifest=selection_manifest,
        )

    destination = _destination(out_dir)
    assert not destination.exists()
    assert not destination.with_name(".frozen.tmp").exists()


def _patch_deterministic_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[int, int]]:
    boundaries = [index / 8 for index in range(9)]
    monkeypatch.setattr(gate, "detect_barlines", lambda _: list(boundaries))
    monkeypatch.setattr(
        gate,
        "measure_boundaries_for_system",
        lambda _path, values: list(values),
    )
    monkeypatch.setattr(
        gate,
        "_estimate_staff",
        lambda _image: StaffEstimate(
            y0=40,
            y1=160,
            line_ys=(60, 80, 100, 120, 140),
        ),
    )
    candidates = [
        GridCandidate(
            bbox=(40, 85, 50, 95),
            score=0.91,
            features={"ink_density": 0.5},
        ),
        GridCandidate(
            bbox=(50, 65, 60, 75),
            score=0.89,
            features={"ink_density": 0.48},
        ),
    ]
    monkeypatch.setattr(
        gate,
        "detect_staff_grid_density_candidates",
        lambda *_args, **_kwargs: list(candidates),
    )
    calls: list[tuple[int, int]] = []

    def propose(image: Image.Image, payload: dict):
        calls.append(image.size)
        assert payload["source_image_size_px"] == {
            "width": image.width,
            "height": image.height,
        }
        return (
            [
                HollowProposal(
                    support_candidate_ids=("c001", "c002"),
                    center=(50.0, 80.0),
                    bbox=(40, 65, 61, 96),
                    contour_kind="closed",
                    score=0.94,
                    features={"ring_density": 0.4},
                )
            ],
            [
                {
                    "support_candidate_ids": ["c001", "c002"],
                    "accepted": True,
                }
            ],
        )

    monkeypatch.setattr(gate.hollow, "propose_hollow_notehead_centers", propose)
    return calls


def _make_pipeline_out(tmp_path: Path) -> Path:
    out_dir = tmp_path / "out"
    work_dir = out_dir / gate.DEFAULT_SLUG
    systems_dir = work_dir / "systems"
    systems_dir.mkdir(parents=True)
    source_pdf = work_dir / "source.pdf"
    source_pdf.write_bytes(b"%PDF-1.4\n")
    image = Image.new("RGB", (1600, 200), "white")
    draw = ImageDraw.Draw(image)
    for y in (60, 80, 100, 120, 140):
        draw.line((0, y, image.width - 1, y), fill="black", width=2)
    image.save(systems_dir / "system_004.png")
    write_manifest_jsonl(
        [
            WorkItem(
                slug=gate.DEFAULT_SLUG,
                pdf_path=source_pdf,
                metadata=WorkMetadata(
                    title="Chispazo",
                    composer="Pedro Morales Pino",
                    rhythm="pasillo",
                    time_signature="3/4",
                    key_hint=None,
                ),
            )
        ],
        out_dir / "manifest.jsonl",
    )
    return out_dir


def _make_selection_manifest(tmp_path: Path, out_dir: Path) -> Path:
    work_dir = out_dir / gate.DEFAULT_SLUG
    source_pdf = work_dir / "source.pdf"
    system_path = work_dir / "systems/system_004.png"
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": gate.SELECTION_KIND,
                "split_status": "unseen_morphology_gate",
                "eligible_for_end_to_end_transcription_claim": False,
                "identity": {
                    "slug": gate.DEFAULT_SLUG,
                    "system_index": gate.DEFAULT_SYSTEM_INDEX,
                },
                "source": {
                    "dataset_pdf": {
                        "path": str(source_pdf),
                        "sha256": _sha256(source_pdf),
                    },
                    "system_image": {
                        "path": str(system_path),
                        "sha256": _sha256(system_path),
                        "width_px": 1600,
                        "height_px": 200,
                    },
                },
                "segmentation": {
                    "expected_measure_count": 8,
                    "measure_boundaries_x_fraction": [index / 8 for index in range(9)],
                    "alignment_source": {
                        "path": str(gate.ALIGNMENT_SOURCE),
                        "sha256": _sha256(gate.ALIGNMENT_SOURCE),
                    },
                },
                "frozen_rule": {
                    "path": str(gate.RULE_SOURCE),
                    "sha256": _sha256(gate.RULE_SOURCE),
                },
                "selection_policy": {
                    "automatic_candidate_outputs_inspected": False,
                    "automatic_hollow_proposals_inspected": False,
                    "review_truth_available_at_selection": False,
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return selection_path


def _make_forbidden_truth_traps(tmp_path: Path, out_dir: Path) -> list[Path]:
    paths = [
        tmp_path / "dataset/ground_truth" / f"{gate.DEFAULT_SLUG}.json",
        tmp_path / "dataset/musicxml" / f"{gate.DEFAULT_SLUG}.musicxml",
        out_dir / gate.DEFAULT_SLUG / "reviews/hollow_truth.json",
    ]
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"trap": true}\n', encoding="utf-8")
    return paths


def _destination(out_dir: Path) -> Path:
    return out_dir / gate.DEFAULT_SLUG / gate.OUTPUT_SUBDIR / "v1" / "system_004" / "frozen"


def _record_path(record: dict[str, str], root: Path) -> Path:
    path = Path(record["path"])
    return path if path.is_absolute() else root / path


def _record_hash_matches(record: dict[str, str], root: Path) -> bool:
    path = _record_path(record, root)
    return path.is_file() and _sha256(path) == record["sha256"]


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
