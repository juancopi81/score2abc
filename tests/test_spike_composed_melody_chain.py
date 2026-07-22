from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from scripts.experiments import spike_composed_melody_chain as spike
from scripts.experiments import spike_notehead_patch_templates as selector


class DetectorScoreScorer:
    def score(self, candidate: selector.CandidatePatch) -> float:
        return candidate.detector_score


def test_composes_selected_candidates_into_canonical_events(tmp_path: Path) -> None:
    request, measure = _request_and_measure(tmp_path)
    model = _model(learned_count=2)

    composed = spike.compose_measure(request, measure, model, out_dir=tmp_path)

    assert [row["candidate_id"] for row in composed.candidate_predictions[:2]] == [
        "c002",
        "c003",
    ]
    assert [anchor["source"]["candidate_id"] for anchor in composed.anchors] == [
        "c003",
        "c002",
    ]
    assert [anchor["pitch"] for anchor in composed.anchors] == ["F5", "Bb4"]
    assert len(composed.prediction["notes"]) == 2
    assert composed.prediction["identity"] == request["identity"]
    assert composed.prediction["inference_provenance"]["review_anchors_used"] is False


def test_rhythm_inference_receives_only_predicted_candidate_anchors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request, measure = _request_and_measure(tmp_path)
    captured: list[dict] = []
    original = spike.rhythm.extract_anchor_features

    def capture(image: Image.Image, anchors: list[dict], staff_lines: list[int]) -> list[dict]:
        captured.extend(anchors)
        return original(image, anchors, staff_lines)

    monkeypatch.setattr(spike.rhythm, "extract_anchor_features", capture)

    spike.compose_measure(request, measure, _model(learned_count=2), out_dir=tmp_path)

    assert captured
    assert {anchor["source"]["kind"] for anchor in captured} == {"automatic_candidate"}
    assert {(anchor["center"]["x"], anchor["center"]["y"]) for anchor in captured} == {
        (55.0, 20.0),
        (95.0, 60.0),
    }
    assert all("review" not in anchor["source"] for anchor in captured)


def test_custom_pitch_predictor_and_selector_id_are_recorded(tmp_path: Path) -> None:
    request, measure = _request_and_measure(tmp_path)

    composed = spike.compose_measure(
        request,
        measure,
        _model(learned_count=2),
        out_dir=tmp_path,
        selector_method_id="fixture_selector",
        pitch_predictor=lambda candidate, request, image: "C4",
    )

    assert [anchor["pitch"] for anchor in composed.anchors] == ["C4", "C4"]
    assert {anchor["source"]["selector_method"] for anchor in composed.anchors} == {
        "fixture_selector"
    }
    assert composed.prediction["inference_provenance"]["notehead_selector"] == "fixture_selector"


def test_threshold_selector_uses_fitted_threshold_instead_of_learned_count(
    tmp_path: Path,
) -> None:
    request, measure = _request_and_measure(tmp_path)
    model = _model(learned_count=1, learned_threshold=0.5)

    composed = spike.compose_measure(
        request,
        measure,
        model,
        out_dir=tmp_path,
        selection_mode=spike.THRESHOLD_SELECTOR,
    )

    assert len(composed.anchors) == 2
    assert [row["candidate_id"] for row in composed.candidate_predictions if row["selected"]] == [
        "c002",
        "c003",
    ]
    provenance = composed.prediction["inference_provenance"]
    assert provenance["selection_mode"] == spike.THRESHOLD_SELECTOR
    assert provenance["learned_score_threshold"] == 0.5
    assert provenance["threshold_fit_from_training_reviews_only"] is True


def test_threshold_fit_uses_inner_measure_isolation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    training = [_training_measure(tmp_path, measure) for measure in (1, 2, 3)]
    fitted_measure_sets: list[tuple[int, ...]] = []

    def fake_fit(rows: list[selector.LabeledCandidate], **_: object) -> DetectorScoreScorer:
        fitted_measure_sets.append(tuple(sorted({row.candidate.measure for row in rows})))
        return DetectorScoreScorer()

    def fake_select(
        measures: list[selector.MeasureData], scores: dict[tuple[int, str], float]
    ) -> tuple[float, dict]:
        assert {measure.measure for measure in measures} == {1, 2, 3}
        assert {measure for measure, _ in scores} == {1, 2, 3}
        return 0.5, {"f1": 0.75, "source": "inner_loocv"}

    monkeypatch.setattr(spike.selector, "_fit_patch_scorer", fake_fit)
    monkeypatch.setattr(spike.selector, "_select_training_threshold", fake_select)

    model = spike.fit_selector(training)

    assert fitted_measure_sets == [(2, 3), (1, 3), (1, 2), (1, 2, 3)]
    assert model.learned_threshold == 0.5
    assert model.threshold_training_metrics == {"f1": 0.75, "source": "inner_loocv"}


def test_truth_loader_runs_only_after_freeze_is_verified(tmp_path: Path) -> None:
    prediction_path = tmp_path / "experiment/validation/predictions.jsonl"
    inference_path = tmp_path / "experiment/validation/inference.jsonl"
    freeze_path = tmp_path / "experiment/validation/freeze.json"
    prediction_path.parent.mkdir(parents=True)
    prediction_path.write_text("", encoding="utf-8")
    inference_path.write_text("", encoding="utf-8")
    freeze_path.write_text(
        json.dumps(
            {
                "status": "not_frozen",
                "split": "validation",
                "target_count": 0,
                "artifacts": [],
            }
        ),
        encoding="utf-8",
    )
    artifacts = spike.SplitArtifacts(
        split="validation",
        prediction_path=prediction_path,
        inference_path=inference_path,
        freeze_path=freeze_path,
        overlay_paths=(),
        freeze_sha256=spike._sha256(freeze_path),
    )
    called = False

    def truth_loader(path: Path) -> list[dict]:
        nonlocal called
        called = True
        return []

    with pytest.raises(ValueError, match="Invalid prediction freeze"):
        spike.evaluate_frozen_split(
            tmp_path / "benchmark",
            artifacts=artifacts,
            truth_loader=truth_loader,
        )

    assert called is False

    freeze_path.write_text(
        json.dumps(
            {
                "status": "frozen_before_truth",
                "split": "validation",
                "target_count": 0,
                "artifacts": [
                    {"path": str(prediction_path), "sha256": spike._sha256(prediction_path)},
                    {"path": str(inference_path), "sha256": spike._sha256(inference_path)},
                ],
            }
        ),
        encoding="utf-8",
    )
    frozen_artifacts = spike.SplitArtifacts(
        split="validation",
        prediction_path=prediction_path,
        inference_path=inference_path,
        freeze_path=freeze_path,
        overlay_paths=(),
        freeze_sha256=spike._sha256(freeze_path),
    )

    report = spike.evaluate_frozen_split(
        tmp_path / "benchmark",
        artifacts=frozen_artifacts,
        truth_loader=truth_loader,
    )

    assert called is True
    assert report["summary"]["targets"] == 0


def test_split_artifacts_are_deterministic(tmp_path: Path) -> None:
    request, measure = _request_and_measure(tmp_path)
    composed = spike.compose_measure(request, measure, _model(learned_count=2), out_dir=tmp_path)
    requests_path = tmp_path / "requests.jsonl"
    spike._write_jsonl(requests_path, [request])

    first = spike.freeze_split_predictions(
        split="validation",
        composed=[composed],
        output_dir=tmp_path / "experiment",
        requests_path=requests_path,
        training=[],
        selector_mode="test",
    )
    first_hashes = _artifact_hashes(first)
    second = spike.freeze_split_predictions(
        split="validation",
        composed=[composed],
        output_dir=tmp_path / "experiment",
        requests_path=requests_path,
        training=[],
        selector_mode="test",
    )

    assert _artifact_hashes(second) == first_hashes


def test_one_shot_heldout_evaluation_reuses_only_matching_freeze(tmp_path: Path) -> None:
    output_dir = tmp_path / "experiment"
    output_dir.mkdir()
    metrics = {"summary": {"note_f1": 0.25}}
    (output_dir / "report.json").write_text(
        json.dumps(
            {
                "threshold_selector": {
                    "heldout": {
                        "metrics": metrics,
                        "artifacts": {"freeze_sha256": "frozen-predictions"},
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    artifacts = spike.SplitArtifacts(
        split="heldout",
        prediction_path=tmp_path / "predictions.jsonl",
        inference_path=tmp_path / "inference.jsonl",
        freeze_path=tmp_path / "heldout/freeze.json",
        overlay_paths=(),
        freeze_sha256="frozen-predictions",
    )

    reused = spike._load_previous_heldout_evaluation(output_dir, artifacts=artifacts)

    assert reused == metrics

    changed = spike.SplitArtifacts(
        split="heldout",
        prediction_path=artifacts.prediction_path,
        inference_path=artifacts.inference_path,
        freeze_path=artifacts.freeze_path,
        overlay_paths=(),
        freeze_sha256="changed-predictions",
    )
    with pytest.raises(ValueError, match="refusing to reopen heldout truth"):
        spike._load_previous_heldout_evaluation(output_dir, artifacts=changed)


def _request_and_measure(tmp_path: Path) -> tuple[dict, selector.UnlabeledMeasure]:
    slug = "fixture"
    relative = Path(slug) / "measure.png"
    image_path = tmp_path / relative
    image_path.parent.mkdir(parents=True)
    image = Image.new("RGB", (140, 125), "white")
    image.save(image_path)
    image_hash = spike._sha256(image_path)
    request = {
        "schema_version": 1,
        "split": "validation",
        "identity": {
            "slug": slug,
            "system_index": 7,
            "system_measure_index": 1,
            "global_measure_index": 46,
        },
        "images": {"raw": {"path_relative_to_out": relative.as_posix(), "sha256": image_hash}},
        "staff_geometry": {"raw_staff_lines_y_px": [20, 40, 60, 80, 100]},
        "allowed_context": {
            "expected_measure_beats": "3",
            "allow_pickup": False,
            "key_hint": "one flat: Bb",
        },
    }
    candidates = (
        _candidate("c001", rank=1, x=25.0, y=40.0, score=0.1),
        _candidate("c002", rank=2, x=95.0, y=60.0, score=0.9),
        _candidate("c003", rank=3, x=55.0, y=20.0, score=0.8),
    )
    measure = selector.UnlabeledMeasure(
        measure=1,
        source_image=image_path,
        source_sha256=image_hash,
        staff_lines=(20, 40, 60, 80, 100),
        staff_spacing=20.0,
        candidates=candidates,
    )
    return request, measure


def _candidate(
    candidate_id: str, *, rank: int, x: float, y: float, score: float
) -> selector.CandidatePatch:
    return selector.CandidatePatch(
        measure=1,
        id=candidate_id,
        rank=rank,
        center_x=x,
        center_y=y,
        bbox=(round(x - 4), round(y - 3), round(x + 4), round(y + 3)),
        detector_score=score,
        patches={"binary_raw": (0.0,)},
    )


def _model(*, learned_count: int, learned_threshold: float | None = None) -> spike.SelectorModel:
    return spike.SelectorModel(
        scorer=DetectorScoreScorer(),  # type: ignore[arg-type]
        learned_count=learned_count,
        probability_center=0.5,
        probability_scale=0.2,
        training_measures=(1, 2, 3, 4),
        training_positive_count=14,
        learned_threshold=learned_threshold,
    )


def _training_measure(tmp_path: Path, measure: int) -> selector.MeasureData:
    positive = _candidate(
        f"m{measure}-positive",
        rank=1,
        x=20.0,
        y=20.0,
        score=0.8,
    )
    negative = _candidate(
        f"m{measure}-negative",
        rank=2,
        x=40.0,
        y=40.0,
        score=0.2,
    )
    positive = selector.CandidatePatch(**{**positive.__dict__, "measure": measure})
    negative = selector.CandidatePatch(**{**negative.__dict__, "measure": measure})
    return selector.MeasureData(
        measure=measure,
        source_image=tmp_path / f"measure-{measure}.png",
        source_sha256=f"image-{measure}",
        review_path=tmp_path / f"review-{measure}.json",
        review_sha256=f"review-{measure}",
        staff_lines=(20, 40, 60, 80, 100),
        staff_spacing=20.0,
        rows=(
            selector.LabeledCandidate(candidate=positive, label=1),
            selector.LabeledCandidate(candidate=negative, label=0),
        ),
    )


def _artifact_hashes(artifacts: spike.SplitArtifacts) -> dict[str, str]:
    return {
        str(path): spike._sha256(path)
        for path in (
            artifacts.prediction_path,
            artifacts.inference_path,
            artifacts.freeze_path,
            *artifacts.overlay_paths,
        )
    }
