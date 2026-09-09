import json
from pathlib import Path

from PIL import Image

from scripts.run_vlm_melody_experiment_batch import (
    PromptVariantVLMRequest,
    PromptVariantVLMResponse,
    main,
    run_batch,
)


class _FakePromptVariantTranscriber:
    def __init__(self) -> None:
        self.requests: list[PromptVariantVLMRequest] = []

    def transcribe(self, request: PromptVariantVLMRequest) -> PromptVariantVLMResponse:
        self.requests.append(request)
        if request.output_mode == "free_response":
            return PromptVariantVLMResponse(
                raw_response="There appear to be two noteheads, but duration is unclear."
            )
        return PromptVariantVLMResponse(
            raw_response=json.dumps(
                {
                    "items": [
                        {
                            "kind": "note",
                            "pitch": "A4",
                            "duration": "1/4",
                            "accidental": "none",
                            "confidence": 0.91,
                            "evidence": "single notehead near A4",
                        }
                    ],
                    "comments": "ok",
                    "overall_confidence": 0.88,
                    "uncertainties": [],
                }
            )
        )


def test_run_vlm_melody_experiment_batch_dry_run_writes_plan_without_fixture(
    tmp_path: Path,
) -> None:
    out_dir, manifest_path, ground_truth_dir = _make_prompt_variant_out(tmp_path)
    fixtures_dir = tmp_path / "fixtures"

    summary = run_batch(
        out_dir=out_dir,
        prompt_variant_manifest=manifest_path,
        provider="openai",
        model="gpt-5.5",
        openai_reasoning_effort="medium",
        fixtures_dir=fixtures_dir,
        ground_truth_dir=ground_truth_dir,
        max_calls=0,
        run_id="dry-run",
    )

    batch_dir = Path(summary["paths"]["batch_dir"])
    assert summary["dry_run"] is True
    assert summary["counts"]["live_calls"] == 0
    assert summary["counts"]["statuses"] == {"dry_run": 3}
    assert (batch_dir / "summary.json").exists()
    assert (batch_dir / "batch_manifest.jsonl").exists()
    assert not list(fixtures_dir.glob("*.json"))


def test_run_vlm_melody_experiment_batch_cli_filters_selected_records(
    tmp_path: Path,
) -> None:
    out_dir, manifest_path, _ground_truth_dir = _make_prompt_variant_out(tmp_path)

    assert (
        main(
            [
                str(out_dir),
                "--prompt-variant-manifest",
                str(manifest_path),
                "--prompt-id",
                "direct_pitch_v0",
                "--variant-id",
                "staff",
                "--slug",
                "demo",
                "--system",
                "1",
                "--measure",
                "3",
                "--run-id",
                "filtered",
            ]
        )
        == 0
    )

    selected = _read_jsonl(out_dir / "vlm_melody_batches" / "filtered" / "selected_records.jsonl")
    assert [record["prompt_variant_id"] for record in selected] == ["staff__direct_pitch_v0"]


def test_run_vlm_melody_experiment_batch_writes_fixture_eval_and_journal(
    tmp_path: Path,
) -> None:
    out_dir, manifest_path, ground_truth_dir = _make_prompt_variant_out(tmp_path)
    fixtures_dir = tmp_path / "fixtures"
    transcriber = _FakePromptVariantTranscriber()

    summary = run_batch(
        out_dir=out_dir,
        prompt_variant_manifest=manifest_path,
        provider="openai",
        model="gpt-5.5",
        openai_reasoning_effort="medium",
        fixtures_dir=fixtures_dir,
        ground_truth_dir=ground_truth_dir,
        max_calls=10,
        journal=True,
        run_id="structured",
        selected_prompt_variant_ids={"staff__direct_pitch_v0"},
        transcriber=transcriber,
    )

    assert len(transcriber.requests) == 1
    assert transcriber.requests[0].system_prompt == "System structured prompt."
    assert transcriber.requests[0].user_prompt == "User structured prompt."
    assert summary["counts"]["statuses"] == {"evaluated": 1}
    assert summary["metrics"]["note_count_match_rate"] == 1.0
    assert summary["metrics"]["pitch_order_accuracy_avg"] == 1.0
    assert summary["metrics"]["duration_order_accuracy_avg"] == 1.0

    result = _read_jsonl(Path(summary["paths"]["batch_manifest"]))[0]
    fixture_path = Path(result["fixture"])
    journal_path = Path(result["journal"])
    report_path = Path(result["report"])
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    assert fixture["prompt_variant_id"] == "staff__direct_pitch_v0"
    assert fixture["prompt_id"] == "direct_pitch_v0"
    assert fixture["variant_id"] == "staff"
    assert fixture["model_id"] == "openai:gpt-5.5:reasoning-medium"
    assert fixture["transcription"]["items"][0]["pitch"] == "A4"
    assert report_path.exists()
    assert (journal_path / "artifacts" / "fixture.json").exists()
    assert (journal_path / "prompts" / "system.txt").read_text(encoding="utf-8") == (
        "System structured prompt."
    )
    assert (journal_path / "eval_result.json").exists()


def test_run_vlm_melody_experiment_batch_free_response_skips_structured_eval(
    tmp_path: Path,
) -> None:
    out_dir, manifest_path, ground_truth_dir = _make_prompt_variant_out(tmp_path)
    fixtures_dir = tmp_path / "fixtures"
    transcriber = _FakePromptVariantTranscriber()

    summary = run_batch(
        out_dir=out_dir,
        prompt_variant_manifest=manifest_path,
        provider="openai",
        model="gpt-5.5",
        fixtures_dir=fixtures_dir,
        ground_truth_dir=ground_truth_dir,
        max_calls=1,
        journal=True,
        run_id="free-response",
        selected_prompt_variant_ids={"staff__free_response_describe_v1"},
        transcriber=transcriber,
    )

    assert summary["counts"]["statuses"] == {"not_structured": 1}
    result = _read_jsonl(Path(summary["paths"]["batch_manifest"]))[0]
    fixture = json.loads(Path(result["fixture"]).read_text(encoding="utf-8"))
    assert fixture["output_mode"] == "free_response"
    assert fixture["transcription"]["items"] == []
    assert "two noteheads" in fixture["transcription"]["raw_response"]
    assert Path(result["journal"]).exists()
    assert json.loads(Path(result["report"]).read_text(encoding="utf-8"))["status"] == (
        "not_structured"
    )


def _make_prompt_variant_out(tmp_path: Path) -> tuple[Path, Path, Path]:
    out_dir = tmp_path / "out"
    prompt_root = out_dir / "vlm_melody_prompt_variants" / "demo" / "system_001"
    measure_dir = out_dir / "demo" / "vlm_melody_inputs" / "system_001"
    prompt_root.mkdir(parents=True)
    measure_dir.mkdir(parents=True)

    image_path = measure_dir / "measure_003_staff.png"
    context_path = measure_dir / "measure_003_context.json"
    Image.new("RGB", (16, 16), color="white").save(image_path)
    context = {
        "slug": "demo",
        "title": "Demo",
        "rhythm": "pasillo",
        "clef_hint": "treble",
        "time_signature_hint": "3/4",
        "key_hint": None,
        "system_index": 1,
        "system_measure_index": 3,
        "global_measure_index": 2,
        "display_measure_number": 3,
        "allow_pickup": False,
        "expected_measure_beats": "3",
    }
    context_path.write_text(json.dumps(context), encoding="utf-8")

    structured_dir = prompt_root / "measure_003" / "staff__direct_pitch_v0"
    free_dir = prompt_root / "measure_003" / "staff__free_response_describe_v1"
    other_dir = prompt_root / "measure_004" / "staff__direct_pitch_v0"
    structured_record = _write_prompt_variant_record(
        structured_dir,
        image_path=image_path,
        context_path=context_path,
        prompt_variant_id="staff__direct_pitch_v0",
        prompt_id="direct_pitch_v0",
        variant_id="staff",
        output_mode="json_schema",
        system_prompt="System structured prompt.",
        user_prompt="User structured prompt.",
        with_schema=True,
        system_measure_index=3,
        global_measure_index=2,
    )
    free_record = _write_prompt_variant_record(
        free_dir,
        image_path=image_path,
        context_path=context_path,
        prompt_variant_id="staff__free_response_describe_v1",
        prompt_id="free_response_describe_v1",
        variant_id="staff",
        output_mode="free_response",
        system_prompt="System free prompt.",
        user_prompt="User free prompt.",
        with_schema=False,
        system_measure_index=3,
        global_measure_index=2,
    )
    other_record = _write_prompt_variant_record(
        other_dir,
        image_path=image_path,
        context_path=context_path,
        prompt_variant_id="staff__direct_pitch_v0_measure4",
        prompt_id="direct_pitch_v0",
        variant_id="staff",
        output_mode="json_schema",
        system_prompt="Other system prompt.",
        user_prompt="Other user prompt.",
        with_schema=True,
        system_measure_index=4,
        global_measure_index=3,
    )
    manifest_path = out_dir / "vlm_melody_prompt_variants_manifest.jsonl"
    _write_jsonl(manifest_path, [structured_record, free_record, other_record])

    ground_truth_dir = tmp_path / "ground_truth"
    ground_truth_dir.mkdir()
    (ground_truth_dir / "demo.json").write_text(
        json.dumps(
            {
                "time_signature": "3/4",
                "notes": [
                    {
                        "measure": 2,
                        "onset_beats": 0.0,
                        "duration_beats": 1.0,
                        "pitch_midi": 69,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return out_dir, manifest_path, ground_truth_dir


def _write_prompt_variant_record(
    prompt_dir: Path,
    *,
    image_path: Path,
    context_path: Path,
    prompt_variant_id: str,
    prompt_id: str,
    variant_id: str,
    output_mode: str,
    system_prompt: str,
    user_prompt: str,
    with_schema: bool,
    system_measure_index: int,
    global_measure_index: int,
) -> dict:
    prompt_dir.mkdir(parents=True)
    system_prompt_path = prompt_dir / "system.txt"
    user_prompt_path = prompt_dir / "user.txt"
    schema_path = prompt_dir / "schema.json"
    system_prompt_path.write_text(system_prompt, encoding="utf-8")
    user_prompt_path.write_text(user_prompt, encoding="utf-8")
    if with_schema:
        schema_path.write_text(json.dumps(_schema()), encoding="utf-8")

    return {
        "prompt_variant_id": prompt_variant_id,
        "prompt_id": prompt_id,
        "variant_id": variant_id,
        "input_kind": "staff",
        "transcription_mode": "pitch",
        "output_mode": output_mode,
        "slug": "demo",
        "system_index": 1,
        "system_measure_index": system_measure_index,
        "global_measure_index": global_measure_index,
        "display_measure_number": system_measure_index,
        "image_path": str(image_path),
        "context_path": str(context_path),
        "system_prompt_path": str(system_prompt_path),
        "user_prompt_path": str(user_prompt_path),
        "schema_path": str(schema_path) if with_schema else None,
    }


def _schema() -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "kind": {"type": "string"},
                        "pitch": {"type": "string"},
                        "duration": {"type": "string"},
                        "accidental": {"type": "string"},
                        "confidence": {"type": "number"},
                        "evidence": {"type": "string"},
                    },
                    "required": [
                        "kind",
                        "pitch",
                        "duration",
                        "accidental",
                        "confidence",
                        "evidence",
                    ],
                },
            },
            "comments": {"type": "string"},
            "overall_confidence": {"type": "number"},
            "uncertainties": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["items", "comments", "overall_confidence", "uncertainties"],
    }


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
