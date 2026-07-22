import json
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from scripts.experiments import freeze_third_score_heldout as spike


def test_prepare_selects_first_layout_candidate_and_records_provenance(tmp_path: Path) -> None:
    primary = spike.Candidate("primary", 7, "primary")
    fallback = spike.Candidate("fallback", 4, "fallback")
    _system_image(tmp_path, fallback, width=800, barlines=(20, 200, 400, 600, 780))

    result = spike.prepare_third_score(
        tmp_path,
        namespace="test-v1",
        candidate_pool=(primary, fallback),
        policy=_test_policy(),
    )

    prepared_path = Path(result["prepared_manifest"])
    prepared = json.loads(prepared_path.read_text(encoding="utf-8"))
    selection = json.loads((prepared_path.parent / "selection.json").read_text(encoding="utf-8"))

    assert prepared["status"] == "prepared_awaiting_model_predictions"
    assert prepared["split"] == "fresh_heldout"
    assert prepared["truth_accessed"] is False
    assert prepared["target"] == {"slug": "fallback", "system_index": 4}
    assert selection["candidate_pool"][0]["exists"] is False
    assert selection["selected"]["source_system_sha256"]
    assert selection["selected"]["detected_barlines_sha256"]
    assert selection["selected"]["cleaned_boundaries_sha256"]
    assert selection["selected"]["crop_manifest_sha256"]
    assert prepared["artifacts"]["requests"]["row_sha256"]
    assert prepared["artifacts"]["evaluator"]["version"] == spike.EVALUATOR_VERSION
    assert all("fallback" in path for path in prepared["forbidden_truth_paths"])


def test_prepare_hashes_are_deterministic_and_truth_independent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate = spike.Candidate("same-score", 2, "only")
    first = tmp_path / "first"
    second = tmp_path / "second"
    for root in (first, second):
        _system_image(root, candidate, width=800, barlines=(20, 200, 400, 600, 780))
        truth = root / "dataset/ground_truth/same-score.json"
        musicxml = root / "dataset/musicxml/same-score.musicxml"
        truth.parent.mkdir(parents=True)
        musicxml.parent.mkdir(parents=True)
        truth.write_text("not valid json and must remain unread", encoding="utf-8")
        musicxml.write_text("not valid musicxml and must remain unread", encoding="utf-8")

    original_open = Path.open

    def guarded_open(path: Path, *args, **kwargs):
        normalized = path.as_posix()
        if "/dataset/ground_truth/" in normalized or "/dataset/musicxml/" in normalized:
            raise AssertionError(f"truth path was opened during layout preparation: {path}")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)

    result_a = spike.prepare_third_score(
        first,
        namespace="deterministic-v1",
        candidate_pool=(candidate,),
        policy=_test_policy(),
    )
    result_b = spike.prepare_third_score(
        second,
        namespace="deterministic-v1",
        candidate_pool=(candidate,),
        policy=_test_policy(),
    )

    assert result_a["selection_sha256"] == result_b["selection_sha256"]
    assert result_a["requests_sha256"] == result_b["requests_sha256"]
    assert result_a["evaluator_sha256"] == result_b["evaluator_sha256"]
    monkeypatch.undo()
    assert (first / "dataset/ground_truth/same-score.json").read_text() == (
        "not valid json and must remain unread"
    )


def test_prepare_refuses_to_overwrite_versioned_namespace(tmp_path: Path) -> None:
    candidate = spike.Candidate("fresh", 3, "only")
    _system_image(tmp_path, candidate, width=800, barlines=(20, 200, 400, 600, 780))
    kwargs = {
        "namespace": "immutable-v1",
        "candidate_pool": (candidate,),
        "policy": _test_policy(),
    }
    spike.prepare_third_score(tmp_path, **kwargs)

    with pytest.raises(ValueError, match="already exists"):
        spike.prepare_third_score(tmp_path, **kwargs)


def test_freeze_hash_pins_explicit_model_training_and_predictions(tmp_path: Path) -> None:
    candidate = spike.Candidate("fresh", 3, "only")
    _system_image(tmp_path, candidate, width=800, barlines=(20, 200, 400, 600, 780))
    prepared = spike.prepare_third_score(
        tmp_path,
        namespace="freeze-v1",
        candidate_pool=(candidate,),
        policy=_test_policy(),
    )
    artifacts = tmp_path / "external"
    artifacts.mkdir()
    predictions = artifacts / "predictions.jsonl"
    model = artifacts / "model.json"
    training = artifacts / "training.json"
    predictions.write_text('{"notes":[]}\n', encoding="utf-8")
    model.write_text('{"threshold":0.5}\n', encoding="utf-8")
    training.write_text('{"reviews":["a","b"]}\n', encoding="utf-8")

    result = spike.freeze_prepared_third_score(
        Path(prepared["prepared_manifest"]),
        predictions_path=predictions,
        model_artifact_paths=(model,),
        training_artifact_paths=(training,),
    )
    freeze = json.loads(Path(result["freeze"]).read_text(encoding="utf-8"))

    assert freeze["status"] == "frozen_awaiting_truth"
    assert freeze["truth_accessed"] is False
    assert freeze["model_artifacts"][0]["source_sha256"] == spike._sha256(model)
    assert freeze["training_artifacts"][0]["source_sha256"] == spike._sha256(training)
    assert freeze["predictions"]["source_sha256"] == spike._sha256(predictions)
    for pin in [freeze["predictions"], *freeze["model_artifacts"], *freeze["training_artifacts"]]:
        snapshot = (
            Path(prepared["prepared_manifest"]).parent / pin["snapshot_path_relative_to_namespace"]
        )
        assert spike._sha256(snapshot) == pin["snapshot_sha256"]

    with pytest.raises(ValueError, match="cannot be overwritten"):
        spike.freeze_prepared_third_score(
            Path(prepared["prepared_manifest"]),
            predictions_path=predictions,
            model_artifact_paths=(model,),
            training_artifact_paths=(training,),
        )


def test_freeze_hash_is_deterministic_for_identical_pinned_inputs(tmp_path: Path) -> None:
    candidate = spike.Candidate("fresh", 3, "only")
    roots = (tmp_path / "first", tmp_path / "second")
    prepared = []
    for root in roots:
        _system_image(root, candidate, width=800, barlines=(20, 200, 400, 600, 780))
        prepared.append(
            spike.prepare_third_score(
                root,
                namespace="freeze-deterministic-v1",
                candidate_pool=(candidate,),
                policy=_test_policy(),
            )
        )

    artifacts = tmp_path / "shared"
    artifacts.mkdir()
    predictions = artifacts / "predictions.jsonl"
    model = artifacts / "model.json"
    training = artifacts / "training.json"
    predictions.write_text('{"notes":[]}\n', encoding="utf-8")
    model.write_text('{"threshold":0.5}\n', encoding="utf-8")
    training.write_text('{"reviews":["a","b"]}\n', encoding="utf-8")

    frozen = [
        spike.freeze_prepared_third_score(
            Path(report["prepared_manifest"]),
            predictions_path=predictions,
            model_artifact_paths=(model,),
            training_artifact_paths=(training,),
        )
        for report in prepared
    ]

    assert frozen[0]["freeze_sha256"] == frozen[1]["freeze_sha256"]
    assert frozen[0]["sealed_manifest_sha256"] == frozen[1]["sealed_manifest_sha256"]


def test_freeze_rejects_truth_artifact_path(tmp_path: Path) -> None:
    candidate = spike.Candidate("fresh", 3, "only")
    _system_image(tmp_path, candidate, width=800, barlines=(20, 200, 400, 600, 780))
    prepared = spike.prepare_third_score(
        tmp_path,
        namespace="truth-path-v1",
        candidate_pool=(candidate,),
        policy=_test_policy(),
    )
    truth = tmp_path / "dataset/ground_truth/fresh.json"
    truth.parent.mkdir(parents=True)
    truth.write_text("{}\n", encoding="utf-8")
    model = tmp_path / "model.json"
    training = tmp_path / "training.json"
    model.write_text("{}\n", encoding="utf-8")
    training.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="forbidden before freeze"):
        spike.freeze_prepared_third_score(
            Path(prepared["prepared_manifest"]),
            predictions_path=truth,
            model_artifact_paths=(model,),
            training_artifact_paths=(training,),
        )


@pytest.mark.parametrize(
    ("role", "relative_path"),
    (
        ("predictions", "out/fresh/intermediate/canonical_truth.json"),
        ("model", "out/fresh/truth_cache/model.json"),
        ("training", "out/fresh/intermediate/musicxml.xml"),
    ),
)
def test_freeze_rejects_target_output_truth_paths_for_every_artifact_role(
    tmp_path: Path,
    role: str,
    relative_path: str,
) -> None:
    prepared_path = _prepared_manifest(tmp_path, namespace=f"target-output-{role}-v1")
    safe = _artifact(tmp_path / "safe.json")
    forbidden = _artifact(tmp_path / relative_path)
    kwargs = {
        "predictions_path": safe,
        "model_artifact_paths": (safe,),
        "training_artifact_paths": (safe,),
    }
    if role == "predictions":
        kwargs["predictions_path"] = forbidden
    elif role == "model":
        kwargs["model_artifact_paths"] = (forbidden,)
    else:
        kwargs["training_artifact_paths"] = (forbidden,)

    with pytest.raises(ValueError, match="forbidden before freeze"):
        spike.freeze_prepared_third_score(prepared_path, **kwargs)

    assert not (prepared_path.parent / "frozen").exists()


@pytest.mark.parametrize(
    "relative_path",
    (
        "dataset/ground_truth/fresh.json",
        "dataset/musicxml/fresh.musicxml",
        "dataset/musicxml/fresh.xml",
        "dataset/musicxml/fresh.mxl",
    ),
)
def test_freeze_rejects_each_declared_target_dataset_truth_path(
    tmp_path: Path,
    relative_path: str,
) -> None:
    prepared_path = _prepared_manifest(tmp_path, namespace="target-dataset-v1")
    forbidden = _artifact(tmp_path / relative_path)
    safe = _artifact(tmp_path / "safe.json")

    with pytest.raises(ValueError, match="forbidden before freeze"):
        spike.freeze_prepared_third_score(
            prepared_path,
            predictions_path=forbidden,
            model_artifact_paths=(safe,),
            training_artifact_paths=(safe,),
        )

    assert not (prepared_path.parent / "frozen").exists()


@pytest.mark.parametrize(
    "relative_path",
    (
        "out/unrelated-score/intermediate/canonical_truth.json",
        "out/unrelated-score/intermediate/musicxml.xml",
        "dataset/ground_truth/unrelated-score.json",
        "dataset/musicxml/unrelated-score.musicxml",
    ),
)
def test_external_artifact_validation_allows_unrelated_score_truth_paths(
    tmp_path: Path,
    relative_path: str,
) -> None:
    artifact = _artifact(tmp_path / relative_path)

    spike._validate_external_artifact(artifact, target_slug="fresh")


def test_external_artifact_validation_rejects_forbidden_symlink_path(tmp_path: Path) -> None:
    safe = _artifact(tmp_path / "safe.json")
    forbidden = tmp_path / "out/fresh/intermediate/truth_snapshot.json"
    forbidden.parent.mkdir(parents=True)
    forbidden.symlink_to(safe)

    with pytest.raises(ValueError, match="forbidden before freeze"):
        spike._validate_external_artifact(forbidden, target_slug="fresh")


def _prepared_manifest(tmp_path: Path, *, namespace: str) -> Path:
    candidate = spike.Candidate("fresh", 3, "only")
    _system_image(tmp_path, candidate, width=800, barlines=(20, 200, 400, 600, 780))
    prepared = spike.prepare_third_score(
        tmp_path,
        namespace=namespace,
        candidate_pool=(candidate,),
        policy=_test_policy(),
    )
    return Path(prepared["prepared_manifest"])


def _artifact(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}\n", encoding="utf-8")
    return path


def _test_policy() -> spike.LayoutPolicy:
    return spike.LayoutPolicy(
        min_width_px=500,
        min_height_px=50,
        min_measure_count=2,
        max_measure_count=8,
        min_crop_width_px=50,
        max_spacing_cv=0.6,
    )


def _system_image(
    out_dir: Path,
    candidate: spike.Candidate,
    *,
    width: int,
    barlines: tuple[int, ...],
) -> Path:
    path = out_dir / candidate.slug / "systems" / f"system_{candidate.system_index:03d}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("L", (width, 120), "white")
    draw = ImageDraw.Draw(image)
    for y in (35, 45, 55, 65, 75):
        draw.line((0, y, width - 1, y), fill="black", width=1)
    for x in barlines:
        draw.line((x, 31, x, 79), fill="black", width=3)
    image.save(path)
    return path
