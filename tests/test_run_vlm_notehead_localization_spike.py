import base64
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from scripts.run_vlm_notehead_localization_spike import (
    ANNOTATION_ELLIPSE_MARGIN,
    GeminiLocalizationTransport,
    LocalizationRequest,
    LocalizationResponse,
    OpenAILocalizationTransport,
    build_request,
    default_model_for_provider,
    evaluate_candidate_coverage,
    evaluate_coordinate_localization,
    evaluate_event_pitch_localization,
    localization_schema,
    parse_and_validate_payload,
    run_batch,
    treble_pitch_for_y,
)


class _FakeTransport:
    def __init__(self, responses: dict[str, dict | str | Exception]) -> None:
        self.responses = responses
        self.requests: list[LocalizationRequest] = []

    def localize(self, request: LocalizationRequest) -> LocalizationResponse:
        self.requests.append(request)
        value = self.responses[request.experiment_id]
        if isinstance(value, Exception):
            raise value
        return LocalizationResponse(
            raw_response=value if isinstance(value, str) else json.dumps(value),
            response_id=f"response-{request.experiment_id}",
            usage={"input_tokens": 10, "output_tokens": 5},
            provider_response={"raw": request.experiment_id},
        )


class _FakeGeminiModels:
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls: list[dict] = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            text=self.text,
            response_id="gemini-response",
            usage_metadata=SimpleNamespace(
                to_json_dict=lambda: {"prompt_token_count": 9, "candidates_token_count": 4}
            ),
        )


def test_provider_payloads_send_all_role_images_in_order_with_strict_schemas(
    tmp_path: Path,
) -> None:
    setup = _make_manifest_setup(tmp_path, task_kinds=("candidate-assisted-localization",))
    record = setup["records"][0]
    request = _request(setup, record)
    openai_payloads = []

    def fake_openai(payload):
        openai_payloads.append(payload)
        return {"id": "openai-response", "output_text": json.dumps(_candidate_payload())}

    openai = OpenAILocalizationTransport(
        model="gpt-test",
        reasoning_effort="xhigh",
        max_output_tokens=1234,
        client=fake_openai,
    )
    response = openai.localize(request)

    assert response.response_id == "openai-response"
    payload = openai_payloads[0]
    assert payload["store"] is False
    assert payload["reasoning"] == {"effort": "xhigh"}
    assert payload["max_output_tokens"] == 1234
    assert payload["text"]["format"]["strict"] is True
    schema = payload["text"]["format"]["schema"]
    selected = schema["properties"]["selected_candidates"]["items"]
    assert selected["properties"]["candidate_id"]["enum"] == ["c001"]
    assert selected["additionalProperties"] is False
    content = payload["input"][0]["content"]
    image_parts = [part for part in content if part["type"] == "input_image"]
    assert [part["detail"] for part in image_parts] == ["original", "original"]
    decoded = [base64.b64decode(part["image_url"].split(",", 1)[1]) for part in image_parts]
    assert decoded == [path.read_bytes() for path in setup["images"]]

    models = _FakeGeminiModels(json.dumps(_candidate_payload()))
    gemini = GeminiLocalizationTransport(
        model="gemini-test",
        max_output_tokens=222,
        client=SimpleNamespace(models=models),
    )
    gemini.localize(request)
    call = models.calls[0]
    inline_parts = [part for part in call["contents"][0]["parts"] if "inline_data" in part]
    assert [base64.b64decode(part["inline_data"]["data"]) for part in inline_parts] == [
        path.read_bytes() for path in setup["images"]
    ]
    assert call["config"]["response_mime_type"] == "application/json"
    assert call["config"]["max_output_tokens"] == 222
    assert "additionalProperties" not in json.dumps(call["config"]["response_schema"])


def test_openai_transport_preserves_incomplete_response_metadata(tmp_path: Path) -> None:
    setup = _make_manifest_setup(tmp_path, task_kinds=("direct-localization",))
    request = _request(setup, setup["records"][0])
    incomplete = {
        "id": "response-incomplete",
        "status": "incomplete",
        "incomplete_details": {"reason": "max_output_tokens"},
        "usage": {"output_tokens": 4096},
        "output": [],
    }
    transport = OpenAILocalizationTransport(model="gpt-test", client=lambda payload: incomplete)

    response = transport.localize(request)

    assert response.raw_response == ""
    assert response.response_id == "response-incomplete"
    assert response.usage == {"output_tokens": 4096}
    assert response.provider_response["incomplete_details"] == {"reason": "max_output_tokens"}


def test_candidate_schema_requires_refined_points_and_application_validates_ids(
    tmp_path: Path,
) -> None:
    setup = _make_manifest_setup(tmp_path, task_kinds=("candidate-assisted-localization",))
    request = _request(setup, setup["records"][0])
    schema = localization_schema("candidate-assisted-localization", candidate_ids=("c001", "c002"))

    assert "c001=(0.250000, 0.250000)" in request.user_prompt
    assert "starting points only, not presumed correct" in request.user_prompt
    assert "First identify every visible rest" in request.user_prompt
    assert schema["properties"]["rest_symbols"]["type"] == "array"
    for rejected_symbol in ("stems", "flags", "accidentals", "barlines", "rests"):
        assert rejected_symbol in request.system_prompt
    assert "selected_candidate_ids" not in schema["properties"]
    assert schema["properties"]["selected_candidates"]["items"]["required"] == [
        "candidate_id",
        "x_fraction",
        "y_fraction",
        "confidence",
        "evidence",
    ]
    invalid = _candidate_payload(candidate_id="not-a-candidate")
    with pytest.raises(ValueError, match="Unknown selected candidate IDs"):
        parse_and_validate_payload(json.dumps(invalid), request)

    missing_refinement = {
        "selected_candidates": [{"candidate_id": "c001"}],
        "missing_noteheads": [],
        "rest_symbols": [],
        "overall_confidence": 0.8,
        "comments": "",
    }
    with pytest.raises(ValueError, match="keys must be exactly"):
        parse_and_validate_payload(json.dumps(missing_refinement), request)


def test_fixture_key_is_stable_and_hashes_roles_bytes_context_and_config(
    tmp_path: Path,
) -> None:
    setup = _make_manifest_setup(tmp_path, task_kinds=("direct-localization",))
    record = setup["records"][0]
    first = _request(setup, record)
    second = _request(setup, json.loads(json.dumps(record)))
    assert first.fixture_key == second.fixture_key

    role_changed = json.loads(json.dumps(record))
    role_changed["images"][0]["role"] = "different-role"
    assert _request(setup, role_changed).fixture_key != first.fixture_key

    config_changed = build_request(
        out_dir=setup["out_dir"],
        manifest_path=setup["manifest"],
        record=record,
        provider="openai",
        model="gpt-test",
        reasoning_effort="medium",
        image_detail="original",
        max_output_tokens=4097,
    )
    assert config_changed.fixture_key != first.fixture_key

    setup["images"][0].write_bytes(setup["images"][0].read_bytes() + b"changed")
    assert _request(setup, record).fixture_key != first.fixture_key


def test_coordinate_evaluation_uses_ellipse_primary_and_reports_pitch_separately() -> None:
    ground_truth = _coordinate_gt()
    predicted = [
        {
            "x_fraction": 0.29,
            "y_fraction": 0.20,
            "confidence": 0.9,
            "evidence": "inside ellipse but outside strict x tolerance",
        }
    ]

    result = evaluate_coordinate_localization(
        predicted,
        ground_truth,
        raw_image_size=(100.0, 100.0),
        staff_lines_y_px=[20, 30, 40, 50, 60],
    )

    assert result["localization"]["annotation_ellipse_margin"] == ANNOTATION_ELLIPSE_MARGIN
    assert result["localization"]["tp"] == 1
    assert result["localization"]["precision"] == 1.0
    assert result["strict_center_diagnostic"]["tp"] == 0
    assert result["pitch"]["predicted_natural_pitches"] == ["F5"]
    assert result["pitch"]["ground_truth_natural_pitches"] == ["F5"]
    assert result["pitch"]["accuracy"] == 1.0

    wrong_pitch = evaluate_coordinate_localization(
        [{**predicted[0], "x_fraction": 0.25, "y_fraction": 0.25}],
        ground_truth,
        raw_image_size=(100.0, 100.0),
        staff_lines_y_px=[20, 30, 40, 50, 60],
    )
    assert wrong_pitch["localization"]["tp"] == 1
    assert wrong_pitch["pitch"]["predicted_natural_pitches"] == ["E5"]
    assert wrong_pitch["pitch"]["accuracy"] == 0.0


def test_raw_candidate_region_coverage_and_pitch_are_independent() -> None:
    coverage = evaluate_candidate_coverage(
        {"candidates": [{"id": "c001", "center": {"x": 25, "y": 25}}]},
        _coordinate_gt(),
        staff_lines_y_px=[20, 30, 40, 50, 60],
    )

    assert coverage is not None
    assert coverage["coverage"] == 1.0
    assert coverage["matched_pitch_accuracy"] == 0.0
    assert coverage["pitch_assignments"][0]["predicted_natural_pitch"] == "E5"


def test_event_pitch_evaluation_uses_global_measure_and_ignores_accidentals() -> None:
    result = evaluate_event_pitch_localization(
        [
            {
                "x_fraction": 0.2,
                "y_fraction": 0.2,
                "confidence": 0.9,
                "evidence": "oval",
            },
            {
                "x_fraction": 0.6,
                "y_fraction": 0.3,
                "confidence": 0.9,
                "evidence": "oval",
            },
        ],
        {
            "notes": [
                {"measure": 0, "pitch_midi": 60},
                {"measure": 4, "pitch_midi": 78, "accidental": 1},
                {"measure": 4, "pitch_midi": 75, "accidental": 1},
            ]
        },
        global_measure_index=4,
        raw_image_size=(100, 100),
        staff_lines_y_px=[20, 30, 40, 50, 60],
    )

    assert result["predicted_natural_pitches"] == ["F5", "D5"]
    assert result["ground_truth_natural_pitches"] == ["F5", "D5"]
    assert result["exact_count"] is True
    assert result["ordered_pitch_accuracy"] == 1.0


@pytest.mark.parametrize(
    ("y_px", "pitch"),
    [(10, "A5"), (20, "F5"), (25, "E5"), (30, "D5"), (40, "B4"), (60, "E4")],
)
def test_treble_staff_pitch_mapping(y_px: float, pitch: str) -> None:
    assert treble_pitch_for_y(y_px, [20, 30, 40, 50, 60]) == pitch


def test_batch_isolates_failures_and_writes_complete_unique_journals(tmp_path: Path) -> None:
    setup = _make_manifest_setup(
        tmp_path,
        task_kinds=("direct-localization", "candidate-assisted-localization"),
    )
    transport = _FakeTransport(
        {
            "direct-1": RuntimeError("provider unavailable"),
            "candidate-assisted-2": _candidate_payload(),
        }
    )
    summary = run_batch(
        out_dir=setup["out_dir"],
        manifest_path=setup["manifest"],
        provider="openai",
        model="gpt-test",
        openai_reasoning_effort="high",
        max_calls=2,
        max_output_tokens=321,
        timeout_seconds=17,
        run_id="isolated",
        fixtures_dir=tmp_path / "fixtures",
        transport=transport,
    )

    assert len(transport.requests) == 2
    assert summary["overall"]["statuses"] == {"evaluated": 1, "failed": 1}
    assert summary["overall"]["gt_statuses"] == {"coordinate": 1, "not_evaluated": 1}
    assert (
        summary["by_task"]["candidate-assisted-localization"]["candidate_coverage"]["coverage"]
        == 1.0
    )
    assert (
        summary["by_task"]["candidate-assisted-localization"]["candidate_coverage"][
            "matched_pitch_accuracy"
        ]
        == 0.0
    )
    batch_dir = Path(summary["paths"]["batch_dir"])
    results = _read_jsonl(batch_dir / "batch_manifest.jsonl")
    journals = [Path(result["journal"]) for result in results]
    assert len(journals) == len(set(journals)) == 2
    for journal in journals:
        assert (journal / "config.json").exists()
        assert (journal / "raw_response.txt").exists()
        assert (journal / "parsed_payload.json").exists()
        assert (journal / "response_metadata.json").exists()
        assert (journal / "evaluation.json").exists()
        assert (journal / "result.json").exists()
        assert (journal / "inputs" / "manifest_record.json").exists()

    failed_journal = journals[0]
    assert (failed_journal / "raw_response.txt").read_text(encoding="utf-8") == ""
    successful_journal = journals[1]
    image_copies = sorted((successful_journal / "images").iterdir())
    assert [path.read_bytes() for path in image_copies] == [
        path.read_bytes() for path in setup["images"]
    ]
    assert (successful_journal / "prompts" / "system.txt").exists()
    assert (successful_journal / "prompts" / "user.txt").exists()
    assert (successful_journal / "prompts" / "schema.json").exists()
    experiment = json.loads((successful_journal / "experiment.json").read_text(encoding="utf-8"))
    assert "--experiment-id candidate-assisted-2" in experiment["replay_command"]
    assert "--max-calls 1" in experiment["replay_command"]
    assert "--force" in experiment["replay_command"]


def test_batch_isolates_parse_failure_from_later_record(tmp_path: Path) -> None:
    setup = _make_manifest_setup(
        tmp_path,
        task_kinds=("direct-localization", "candidate-assisted-localization"),
    )
    transport = _FakeTransport(
        {
            "direct-1": "not json",
            "candidate-assisted-2": _candidate_payload(),
        }
    )
    summary = run_batch(
        out_dir=setup["out_dir"],
        manifest_path=setup["manifest"],
        provider="openai",
        model="gpt-test",
        max_calls=2,
        run_id="parse-isolation",
        fixtures_dir=tmp_path / "fixtures",
        transport=transport,
    )

    results = _read_jsonl(Path(summary["paths"]["batch_manifest"]))
    assert [result["status"] for result in results] == ["failed", "evaluated"]
    assert results[0]["failure_stage"] == "parse"
    assert Path(results[0]["journal"]).exists()
    assert Path(results[1]["journal"]).exists()


def test_batch_distinguishes_no_ground_truth_from_failure(tmp_path: Path) -> None:
    setup = _make_manifest_setup(tmp_path, task_kinds=("direct-localization",))
    record = setup["records"][0]
    record["coordinate_ground_truth_path"] = None
    record["canonical_ground_truth_path"] = str(tmp_path / "missing-canonical.json")
    setup["manifest"].write_text(json.dumps(record) + "\n", encoding="utf-8")
    summary = run_batch(
        out_dir=setup["out_dir"],
        manifest_path=setup["manifest"],
        provider="openai",
        model="gpt-test",
        max_calls=1,
        run_id="no-gt",
        fixtures_dir=tmp_path / "fixtures",
        transport=_FakeTransport({"direct-1": _direct_payload()}),
    )

    assert summary["overall"]["statuses"] == {"no_gt": 1}
    assert summary["overall"]["gt_statuses"] == {"missing_canonical_gt": 1}
    result = _read_jsonl(Path(summary["paths"]["batch_manifest"]))[0]
    assert result["status"] == "no_gt"
    assert Path(result["journal"]).exists()


def test_dry_run_needs_no_api_key_and_filters_without_creating_journals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    setup = _make_manifest_setup(
        tmp_path,
        task_kinds=("direct-localization", "candidate-assisted-localization"),
    )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    summary = run_batch(
        out_dir=setup["out_dir"],
        manifest_path=setup["manifest"],
        provider="openai",
        model=default_model_for_provider("openai"),
        max_calls=0,
        run_id="dry",
        fixtures_dir=tmp_path / "fixtures",
        selected_task_kinds={"candidate-assisted-localization"},
        selected_slugs={"demo"},
        selected_systems={1},
        selected_measures={2},
    )

    assert default_model_for_provider("openai") == "gpt-5.6-sol"
    assert default_model_for_provider("gemini") == "gemini-3.1-pro-preview"
    assert summary["dry_run"] is True
    assert summary["overall"]["statuses"] == {"dry_run": 1}
    batch_dir = Path(summary["paths"]["batch_dir"])
    assert len(_read_jsonl(batch_dir / "selected_records.jsonl")) == 1
    assert not list((batch_dir / "journals").iterdir())
    assert not (tmp_path / "fixtures").exists()


def _request(setup: dict, record: dict) -> LocalizationRequest:
    return build_request(
        out_dir=setup["out_dir"],
        manifest_path=setup["manifest"],
        record=record,
        provider="openai",
        model="gpt-test",
        reasoning_effort="medium",
        image_detail="original",
        max_output_tokens=4096,
    )


def _candidate_payload(candidate_id: str = "c001") -> dict:
    return {
        "selected_candidates": [
            {
                "candidate_id": candidate_id,
                "x_fraction": 0.25,
                "y_fraction": 0.20,
                "confidence": 0.95,
                "evidence": "refined oval center",
            }
        ],
        "missing_noteheads": [],
        "rest_symbols": [],
        "overall_confidence": 0.9,
        "comments": "best effort",
    }


def _direct_payload() -> dict:
    return {
        "noteheads": [
            {
                "x_fraction": 0.25,
                "y_fraction": 0.20,
                "confidence": 0.95,
                "evidence": "filled oval",
            }
        ],
        "rest_symbols": [],
        "overall_confidence": 0.9,
        "comments": "best effort",
    }


def _coordinate_gt() -> dict:
    return {
        "kind": "vlm_melody_notehead_human_ground_truth",
        "notehead_count": 1,
        "noteheads": [
            {
                "id": "n001",
                "order": 1,
                "pitch": "F#5",
                "center": {"x": 20, "y": 20},
                "annotation_geometry": {
                    "bbox_px": {"left": 20, "top": 14, "right": 30, "bottom": 26},
                    "radius_x_px": 5,
                    "radius_y_px": 6,
                },
            }
        ],
    }


def _make_manifest_setup(tmp_path: Path, *, task_kinds: tuple[str, ...]) -> dict:
    out_dir = tmp_path / "out"
    inputs = out_dir / "demo" / "inputs"
    inputs.mkdir(parents=True)
    raw = inputs / "measure_raw.png"
    overlay = inputs / "candidate_overlay.png"
    Image.new("RGB", (100, 100), "white").save(raw)
    Image.new("RGB", (100, 100), "gray").save(overlay)
    context = inputs / "context.json"
    context.write_text(json.dumps({"clef_hint": "treble", "private": "input context"}))
    candidates = inputs / "candidates.json"
    candidates.write_text(
        json.dumps({"candidates": [{"id": "c001", "center": {"x": 25, "y": 25}}]})
    )
    coordinate_gt = inputs / "coordinate_gt.json"
    coordinate_gt.write_text(json.dumps(_coordinate_gt()))
    canonical_gt = inputs / "canonical.json"
    canonical_gt.write_text(
        json.dumps(
            {
                "notes": [
                    {"measure": 1, "pitch_midi": 66, "accidental": 1},
                ]
            }
        )
    )
    source_context = {
        "staff_lines_y_px": [20, 30, 40, 50, 60],
        "raw_image_size": {"width": 100, "height": 100},
    }
    records = []
    for index, task_kind in enumerate(task_kinds, start=1):
        experiment_id = (
            f"direct-{index}"
            if task_kind == "direct-localization"
            else f"candidate-assisted-{index}"
        )
        records.append(
            {
                "schema_version": 1,
                "experiment_id": experiment_id,
                "task_kind": task_kind,
                "slug": "demo",
                "system_index": 1,
                "system_measure_index": 2,
                "global_measure_index": 1,
                "context_path": str(context),
                "images": [
                    {"role": "raw_measure", "path": str(raw)},
                    {"role": "candidate_overlay", "path": str(overlay)},
                ],
                "candidate_artifact_path": (
                    str(candidates) if task_kind == "candidate-assisted-localization" else None
                ),
                "coordinate_ground_truth_path": str(coordinate_gt),
                "canonical_ground_truth_path": str(canonical_gt),
                "source_context": source_context,
            }
        )
    manifest = out_dir / "vlm_notehead_localization_manifest.jsonl"
    manifest.write_text("\n".join(json.dumps(record) for record in records) + "\n")
    return {
        "out_dir": out_dir,
        "manifest": manifest,
        "records": records,
        "images": [raw, overlay],
    }


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
