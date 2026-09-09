import json
from pathlib import Path

from PIL import Image

from scripts.experiments import spike_vlm_candidate_id_selector as selector


def test_run_experiment_journals_and_evaluates_gt_blind_selection(
    tmp_path: Path, monkeypatch
) -> None:
    out_dir = tmp_path / "out"
    input_dir = tmp_path / "inputs"
    input_dir.mkdir(parents=True)
    detail = input_dir / "detail.png"
    gallery = input_dir / "gallery.png"
    Image.new("RGB", (80, 70), "white").save(detail)
    Image.new("RGB", (160, 140), "white").save(gallery)
    context = input_dir / "context.json"
    context.write_text("{}\n", encoding="utf-8")
    canonical = input_dir / "canonical.json"
    canonical.write_text("{}\n", encoding="utf-8")
    candidate_path = input_dir / "candidates.json"
    candidates = [
        {
            "id": f"c{index:03d}",
            "center": {"x": 10.0 + index, "y": 20.0},
            "score": 1.0 / index,
            "features": {},
        }
        for index in range(1, 25)
    ]
    candidate_path.write_text(
        json.dumps(
            {
                "candidates": candidates,
                "staff_lines_y_px": [10, 20, 30, 40, 50],
            }
        ),
        encoding="utf-8",
    )
    record = {
        "system_measure_index": 1,
        "images": [
            {"role": "context", "path": str(detail)},
            {"role": "detail", "path": str(detail)},
            {"role": "candidate_gallery", "path": str(gallery)},
        ],
        "candidate_artifact_path": str(candidate_path),
        "context_path": str(context),
        "canonical_ground_truth_path": str(canonical),
    }
    monkeypatch.setattr(
        selector,
        "build_vlm_notehead_localization_inputs",
        lambda *args, **kwargs: [record],
    )
    gt_dir = tmp_path / "gt"
    gt_dir.mkdir()
    monkeypatch.setattr(selector, "GT_DIR", gt_dir)
    monkeypatch.setattr(selector, "CACHE_DIR", tmp_path / "cache")
    gt_path = selector.evaluator._ground_truth_path(
        gt_dir,
        slug="example",
        system_index=1,
        measure=1,
    )
    gt_path.write_text(
        json.dumps(
            {
                "noteheads": [
                    {
                        "id": "n001",
                        "pitch": "F5",
                        "center": {"x": 11.0, "y": 20.0},
                        "annotation_geometry": {
                            "bbox_px": {"left": 6, "top": 15, "right": 16, "bottom": 25},
                            "radius_x_px": 5,
                            "radius_y_px": 5,
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    def fake_client(payload):
        schema = payload["text"]["format"]["schema"]
        assert "uniqueItems" not in schema["properties"]["selected_candidate_ids"]
        assert "ground_truth" not in json.dumps(payload).lower()
        return {
            "id": "resp_test",
            "output_text": '{"selected_candidate_ids":["c001"]}',
            "usage": {"input_tokens": 10, "output_tokens": 2},
        }

    report = selector.run_experiment(
        out_dir,
        slug="example",
        system_index=1,
        measures=[1],
        model="gpt-test",
        reasoning_effort="low",
        max_output_tokens=128,
        timeout_seconds=5,
        max_calls=1,
        force=True,
        run_id="test-run",
        provider_client=fake_client,
    )

    assert report["live_calls"] == 1
    assert report["aggregate"] == {
        "evaluated_measure_count": 1,
        "tp": 1,
        "fp": 0,
        "fn": 0,
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
        "exact_measure_count": 1,
    }
    assert report["gate"]["passed"] is True
    request = json.loads(
        (
            out_dir / "experiments/vlm_candidate_id_selector/test-run/measure_001/request.json"
        ).read_text(encoding="utf-8")
    )
    assert request["candidate_ids"] == [f"c{index:03d}" for index in range(1, 25)]


def test_duplicate_candidate_ids_are_rejected_after_provider_response(
    tmp_path: Path, monkeypatch
) -> None:
    artifact = {
        "candidates": [{"id": "c001", "center": {"x": 10, "y": 10}}],
        "staff_lines_y_px": [10, 20, 30, 40, 50],
    }
    item = selector.PreparedRequest(
        measure=1,
        request=None,  # type: ignore[arg-type]
        candidate_artifact=artifact,
        cache_path=tmp_path / "cache.json",
        journal_dir=tmp_path,
    )
    monkeypatch.setattr(selector, "GT_DIR", tmp_path)

    result = selector._evaluate_response(
        {
            "status": "called",
            "raw_response": '{"selected_candidate_ids":["c001","c001"]}',
        },
        item=item,
        slug="example",
        system_index=1,
    )

    assert result["gt_status"] == "not_evaluated"
    assert "duplicates" in result["error"]
