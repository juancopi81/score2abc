from __future__ import annotations

import importlib.util
import json
import logging
from pathlib import Path

from PIL import Image

from score2abc.melody.vlm import (
    MelodyVLMItem,
    MelodyVLMRequest,
    MelodyVLMTranscription,
    OpenAIMelodyVLM,
    _gemini_response_schema,
    fixture_model_id,
    melody_fixture_key,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


record_vlm_melody_fixtures = _load_script(
    "record_vlm_melody_fixtures",
    REPO_ROOT / "scripts" / "record_vlm_melody_fixtures.py",
)
eval_vlm_melody_fixtures = _load_script(
    "eval_vlm_melody_fixtures",
    REPO_ROOT / "scripts" / "eval_vlm_melody_fixtures.py",
)
journal_vlm_melody_experiment = _load_script(
    "journal_vlm_melody_experiment",
    REPO_ROOT / "scripts" / "journal_vlm_melody_experiment.py",
)


class _FakeMelodyVLM:
    model_id = "fake-model"
    prompt_version = "melody-vlm-v0"

    def __init__(self) -> None:
        self.requests: list[MelodyVLMRequest] = []

    def transcribe(self, request: MelodyVLMRequest) -> MelodyVLMTranscription:
        self.requests.append(request)
        return MelodyVLMTranscription(
            items=(
                MelodyVLMItem(
                    kind="note",
                    pitch="A3",
                    duration="1/8",
                    accidental=None,
                    confidence=0.9,
                ),
            ),
            comments="ok",
            raw_response='{"items":[],"comments":"ok"}',
        )


def test_record_fixtures_respects_input_kind_and_call_cap(tmp_path: Path) -> None:
    out_dir = _make_vlm_out(tmp_path)
    records = record_vlm_melody_fixtures._load_manifest(out_dir)
    fixtures_dir = tmp_path / "fixtures"
    transcriber = _FakeMelodyVLM()

    written, skipped, would_call = record_vlm_melody_fixtures.record_fixtures(
        records,
        out_dir=out_dir,
        fixtures_dir=fixtures_dir,
        input_kinds=("staff",),
        model_id=transcriber.model_id,
        transcription_mode="pitch",
        transcriber=transcriber,
        max_calls=1,
        force=False,
        logger=logging.getLogger("test"),
        context_overrides={
            "time_signature_hint": "3/4",
            "expected_measure_beats": "1.5",
        },
    )

    assert (written, skipped, would_call) == (1, 0, 1)
    assert [request.input_kind for request in transcriber.requests] == ["staff"]
    assert transcriber.requests[0].context["time_signature_hint"] == "3/4"
    assert transcriber.requests[0].context["expected_measure_beats"] == "1.5"
    assert len(list(fixtures_dir.glob("*.json"))) == 1


def test_eval_vlm_melody_fixtures_compares_pitch_and_duration(tmp_path: Path) -> None:
    out_dir = _make_vlm_out(tmp_path)
    records = record_vlm_melody_fixtures._load_manifest(out_dir)
    fixtures_dir = tmp_path / "fixtures"
    transcriber = _FakeMelodyVLM()
    record_vlm_melody_fixtures.record_fixtures(
        records[:1],
        out_dir=out_dir,
        fixtures_dir=fixtures_dir,
        input_kinds=("staff",),
        model_id=transcriber.model_id,
        transcription_mode="pitch",
        transcriber=transcriber,
        max_calls=1,
        force=False,
        logger=logging.getLogger("test"),
        context_overrides=None,
    )
    ground_truth_dir = tmp_path / "ground_truth"
    ground_truth_dir.mkdir()
    (ground_truth_dir / "demo.json").write_text(
        json.dumps(
            {
                "time_signature": "3/4",
                "notes": [
                    {
                        "measure": 0,
                        "onset_beats": 0.0,
                        "duration_beats": 0.5,
                        "pitch_midi": 57,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report = eval_vlm_melody_fixtures.evaluate_records(
        records[:1],
        out_dir=out_dir,
        ground_truth_dir=ground_truth_dir,
        fixtures_dir=fixtures_dir,
        input_kinds=("staff",),
        model_id=transcriber.model_id,
    )

    assert report["summary"]["evaluated"] == 1
    assert report["summary"]["note_count_match_rate"] == 1.0
    assert report["summary"]["pitch_order_accuracy_avg"] == 1.0
    assert report["summary"]["duration_order_accuracy_avg"] == 1.0


def test_journal_vlm_melody_experiment_snapshots_artifacts_and_prompts(
    tmp_path: Path,
) -> None:
    out_dir = _make_vlm_out(tmp_path)
    records = record_vlm_melody_fixtures._load_manifest(out_dir)
    fixtures_dir = tmp_path / "fixtures"
    transcriber = _FakeMelodyVLM()
    record_vlm_melody_fixtures.record_fixtures(
        records[:1],
        out_dir=out_dir,
        fixtures_dir=fixtures_dir,
        input_kinds=("pitch_ruler_panel",),
        model_id=transcriber.model_id,
        transcription_mode="pitch",
        transcriber=transcriber,
        max_calls=1,
        force=False,
        logger=logging.getLogger("test"),
        context_overrides=None,
    )
    ground_truth_dir = tmp_path / "ground_truth"
    ground_truth_dir.mkdir()
    (ground_truth_dir / "demo.json").write_text(
        json.dumps(
            {
                "time_signature": "3/4",
                "notes": [
                    {
                        "measure": 0,
                        "onset_beats": 0.0,
                        "duration_beats": 0.5,
                        "pitch_midi": 57,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    journal_path = journal_vlm_melody_experiment.journal_experiment(
        out_dir=out_dir,
        slug="demo",
        system_index=1,
        system_measure_index=1,
        input_kind="pitch_ruler_panel",
        provider="gemini",
        model="fake-model",
        transcription_mode="pitch",
        fixtures_dir=fixtures_dir,
        ground_truth_dir=ground_truth_dir,
        journal_root=tmp_path / "journal",
        run_id="demo-run",
    )

    assert (journal_path / "artifacts" / "input.png").exists()
    assert (journal_path / "artifacts" / "context.json").exists()
    assert (journal_path / "artifacts" / "fixture.json").exists()
    assert "left pitch reference gutter" in (journal_path / "prompts" / "user.txt").read_text(
        encoding="utf-8"
    )
    assert "two visual regions" in (journal_path / "prompts" / "system.txt").read_text(
        encoding="utf-8"
    )

    experiment = json.loads((journal_path / "experiment.json").read_text(encoding="utf-8"))
    eval_result = json.loads((journal_path / "eval_result.json").read_text(encoding="utf-8"))
    assert experiment["config"]["input_kind"] == "pitch_ruler_panel"
    assert experiment["eval_summary"]["evaluated"] == 1
    assert "record_vlm_melody_fixtures.py" in experiment["replay_commands"]["record_fixture"]
    assert eval_result["results"][0]["pred_pitches"] == ["A3"]


def test_eval_vlm_melody_fixtures_compares_staff_position(tmp_path: Path) -> None:
    out_dir = _make_vlm_out(tmp_path)
    records = record_vlm_melody_fixtures._load_manifest(out_dir)
    fixtures_dir = tmp_path / "fixtures"
    context_path = Path(records[0]["paths"]["context"])
    context = json.loads(context_path.read_text(encoding="utf-8"))
    image_path = Path(records[0]["paths"]["measure_staff"])
    key = melody_fixture_key(
        image_path,
        prompt_version="melody-vlm-staff-position-v0",
        model_id="fake-model",
        input_kind="staff",
        context=context,
    )
    fixtures_dir.mkdir()
    (fixtures_dir / f"{key}.json").write_text(
        json.dumps(
            {
                "prompt_version": "melody-vlm-staff-position-v0",
                "model_id": "fake-model",
                "input_kind": "staff",
                "image_path": str(image_path),
                "context_path": str(context_path),
                "context": context,
                "transcription": {
                    "items": [
                        {
                            "kind": "note",
                            "pitch": "",
                            "duration": "1/8",
                            "accidental": "none",
                            "confidence": 0.9,
                            "notehead_x_fraction": 0.5,
                            "staff_position": -4,
                        }
                    ],
                    "comments": "ok",
                    "raw_response": "{}",
                },
            }
        ),
        encoding="utf-8",
    )
    ground_truth_dir = tmp_path / "ground_truth"
    ground_truth_dir.mkdir()
    (ground_truth_dir / "demo.json").write_text(
        json.dumps(
            {
                "time_signature": "3/4",
                "notes": [
                    {
                        "measure": 0,
                        "onset_beats": 0.0,
                        "duration_beats": 0.5,
                        "pitch_midi": 57,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report = eval_vlm_melody_fixtures.evaluate_records(
        records[:1],
        out_dir=out_dir,
        ground_truth_dir=ground_truth_dir,
        fixtures_dir=fixtures_dir,
        input_kinds=("staff",),
        model_id="fake-model",
        transcription_mode="staff_position",
    )

    result = report["results"][0]
    assert report["summary"]["staff_position_order_accuracy_avg"] == 1.0
    assert result["pred_staff_positions"] == [-4]
    assert result["truth_staff_positions"] == [-4]


def test_eval_vlm_melody_fixtures_converts_notehead_y_to_staff_position(
    tmp_path: Path,
) -> None:
    out_dir = _make_vlm_out(tmp_path)
    records = record_vlm_melody_fixtures._load_manifest(out_dir)
    fixtures_dir = tmp_path / "fixtures"
    context_path = Path(records[0]["paths"]["context"])
    context = json.loads(context_path.read_text(encoding="utf-8"))
    image_path = Path(records[0]["paths"]["measure_staff"])
    key = melody_fixture_key(
        image_path,
        prompt_version="melody-vlm-notehead-y-v0",
        model_id="fake-model",
        input_kind="staff",
        context=context,
    )
    fixtures_dir.mkdir()
    (fixtures_dir / f"{key}.json").write_text(
        json.dumps(
            {
                "prompt_version": "melody-vlm-notehead-y-v0",
                "model_id": "fake-model",
                "input_kind": "staff",
                "image_path": str(image_path),
                "context_path": str(context_path),
                "context": context,
                "transcription": {
                    "items": [
                        {
                            "kind": "note",
                            "pitch": "",
                            "duration": "1/8",
                            "accidental": "none",
                            "confidence": 0.9,
                            "notehead_x_fraction": 0.5,
                            "notehead_y_fraction": 7 / 9,
                        }
                    ],
                    "comments": "ok",
                    "raw_response": "{}",
                },
            }
        ),
        encoding="utf-8",
    )
    ground_truth_dir = tmp_path / "ground_truth"
    ground_truth_dir.mkdir()
    (ground_truth_dir / "demo.json").write_text(
        json.dumps(
            {
                "time_signature": "3/4",
                "notes": [
                    {
                        "measure": 0,
                        "onset_beats": 0.0,
                        "duration_beats": 0.5,
                        "pitch_midi": 57,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report = eval_vlm_melody_fixtures.evaluate_records(
        records[:1],
        out_dir=out_dir,
        ground_truth_dir=ground_truth_dir,
        fixtures_dir=fixtures_dir,
        input_kinds=("staff",),
        model_id="fake-model",
        transcription_mode="notehead_y",
    )

    result = report["results"][0]
    assert report["summary"]["staff_position_order_accuracy_avg"] == 1.0
    assert result["pred_staff_positions"] == [-4]
    assert len(result["pred_notehead_y_fractions"]) == 1


def test_melody_fixture_key_includes_input_kind(tmp_path: Path) -> None:
    image = tmp_path / "measure.png"
    image.write_bytes(b"same image")
    context = {"clef_hint": "treble", "time_signature_hint": None}

    raw_key = melody_fixture_key(
        image,
        prompt_version="p",
        model_id="m",
        input_kind="raw",
        context=context,
    )
    staff_key = melody_fixture_key(
        image,
        prompt_version="p",
        model_id="m",
        input_kind="staff",
        context=context,
    )

    assert raw_key != staff_key


def test_melody_fixture_key_includes_prompt_context(tmp_path: Path) -> None:
    image = tmp_path / "measure.png"
    image.write_bytes(b"same image")

    baseline_key = melody_fixture_key(
        image,
        prompt_version="p",
        model_id="m",
        input_kind="staff",
        context={"clef_hint": "treble", "time_signature_hint": None},
    )
    hinted_key = melody_fixture_key(
        image,
        prompt_version="p",
        model_id="m",
        input_kind="staff",
        context={"clef_hint": "treble", "time_signature_hint": "3/4"},
    )

    assert baseline_key != hinted_key


def test_load_manifest_merges_pitch_ruler_sidecar(tmp_path: Path) -> None:
    out_dir = _make_vlm_out(tmp_path)

    records = record_vlm_melody_fixtures._load_manifest(out_dir)

    assert "measure_pitch_ruler" in records[0]["paths"]
    assert records[0]["pitch_ruler"]["clef"] == "treble"
    assert "measure_pitch_ruler_panel" in records[0]["paths"]


def test_openai_melody_vlm_uses_responses_payload_with_original_detail(
    tmp_path: Path,
) -> None:
    image = tmp_path / "measure.png"
    Image.new("RGB", (10, 10), color="white").save(image)
    requests = []

    def fake_client(payload):
        requests.append(payload)
        return {
            "output": [
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": json.dumps(
                                {
                                    "items": [
                                        {
                                            "kind": "note",
                                            "pitch": "G4",
                                            "duration": "1/4",
                                            "accidental": "none",
                                            "confidence": 0.8,
                                        }
                                    ],
                                    "comments": "ok",
                                }
                            ),
                        }
                    ],
                }
            ]
        }

    transcriber = OpenAIMelodyVLM(client=fake_client, model="gpt-5.5")
    transcription = transcriber.transcribe(
        MelodyVLMRequest(
            image_path=image,
            context={"clef_hint": "treble", "time_signature_hint": "3/4"},
            input_kind="staff",
            transcription_mode="pitch",
        )
    )

    payload = requests[0]
    image_part = payload["input"][0]["content"][1]
    assert payload["model"] == "gpt-5.5"
    assert "temperature" not in payload
    assert payload["reasoning"] == {"effort": "none"}
    assert payload["max_output_tokens"] == 4096
    assert payload["text"]["format"]["type"] == "json_schema"
    assert payload["text"]["format"]["strict"] is True
    assert image_part["detail"] == "original"
    assert image_part["image_url"].startswith("data:image/png;base64,")
    assert transcription.items[0].pitch == "G4"


def test_openai_melody_vlm_can_enable_reasoning_effort(tmp_path: Path) -> None:
    image = tmp_path / "measure.png"
    Image.new("RGB", (10, 10), color="white").save(image)
    requests = []

    def fake_client(payload):
        requests.append(payload)
        return {
            "output": [
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": json.dumps({"items": [], "comments": "ok"}),
                        }
                    ],
                }
            ]
        }

    transcriber = OpenAIMelodyVLM(
        client=fake_client,
        model="gpt-5.5",
        reasoning_effort="medium",
    )
    transcriber.transcribe(
        MelodyVLMRequest(
            image_path=image,
            context={"clef_hint": "treble"},
            input_kind="staff",
            transcription_mode="pitch",
        )
    )

    assert requests[0]["reasoning"] == {"effort": "medium"}


def test_openai_melody_vlm_uses_pitch_ruler_prompt(tmp_path: Path) -> None:
    image = tmp_path / "measure_pitch_ruler.png"
    Image.new("RGB", (10, 10), color="white").save(image)
    requests = []

    def fake_client(payload):
        requests.append(payload)
        return {
            "output": [
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": json.dumps({"items": [], "comments": "ok"}),
                        }
                    ],
                }
            ]
        }

    transcriber = OpenAIMelodyVLM(client=fake_client, model="gpt-5.5")
    transcriber.transcribe(
        MelodyVLMRequest(
            image_path=image,
            context={"clef_hint": "treble", "time_signature_hint": "3/4"},
            input_kind="pitch_ruler",
            transcription_mode="pitch",
        )
    )

    payload = requests[0]
    user_text = payload["input"][0]["content"][0]["text"]
    assert "pitch labels printed on the left side" in user_text
    assert "nearest labeled pitch guide" in user_text
    assert "The image includes pitch labels on the left" in payload["instructions"]


def test_openai_melody_vlm_uses_soft_pitch_ruler_prompt_and_schema(tmp_path: Path) -> None:
    image = tmp_path / "measure_pitch_ruler_soft.png"
    Image.new("RGB", (10, 10), color="white").save(image)
    requests = []

    def fake_client(payload):
        requests.append(payload)
        return {
            "output": [
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": json.dumps(
                                {
                                    "items": [
                                        {
                                            "kind": "note",
                                            "pitch": "A4",
                                            "duration": "1/8",
                                            "accidental": "none",
                                            "confidence": 0.72,
                                            "evidence": "black notehead nearest A4 guide",
                                        }
                                    ],
                                    "comments": "best effort",
                                    "overall_confidence": 0.62,
                                    "uncertainties": ["duration is partly obscured"],
                                }
                            ),
                        }
                    ],
                }
            ]
        }

    transcriber = OpenAIMelodyVLM(client=fake_client, model="gpt-5.5")
    transcription = transcriber.transcribe(
        MelodyVLMRequest(
            image_path=image,
            context={"clef_hint": "treble", "time_signature_hint": "3/4"},
            input_kind="pitch_ruler_soft",
            transcription_mode="pitch",
        )
    )

    payload = requests[0]
    user_text = payload["input"][0]["content"][0]["text"]
    schema = payload["text"]["format"]["schema"]
    assert "The faint gray horizontal guide marks" in user_text
    assert "best transcription you can" in user_text
    assert "soft pitch ruler" in payload["instructions"]
    assert "evidence" in schema["properties"]["items"]["items"]["properties"]
    assert "overall_confidence" in schema["required"]
    assert transcription.items[0].evidence == "black notehead nearest A4 guide"
    assert transcription.overall_confidence == 0.62
    assert transcription.uncertainties == ("duration is partly obscured",)


def test_openai_melody_vlm_uses_panel_pitch_ruler_prompt_and_schema(tmp_path: Path) -> None:
    image = tmp_path / "measure_pitch_ruler_panel.png"
    Image.new("RGB", (10, 10), color="white").save(image)
    requests = []

    def fake_client(payload):
        requests.append(payload)
        return {
            "output": [
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": json.dumps(
                                {
                                    "items": [
                                        {
                                            "kind": "rest",
                                            "pitch": "",
                                            "duration": "1/8",
                                            "accidental": "none",
                                            "confidence": 0.6,
                                            "evidence": "visible empty opening",
                                        }
                                    ],
                                    "comments": "best effort",
                                    "overall_confidence": 0.55,
                                    "uncertainties": ["opening rest is partly distorted"],
                                }
                            ),
                        }
                    ],
                }
            ]
        }

    transcriber = OpenAIMelodyVLM(client=fake_client, model="gpt-5.5")
    transcription = transcriber.transcribe(
        MelodyVLMRequest(
            image_path=image,
            context={"clef_hint": "treble", "time_signature_hint": "3/4"},
            input_kind="pitch_ruler_panel",
            transcription_mode="pitch",
        )
    )

    payload = requests[0]
    user_text = payload["input"][0]["content"][0]["text"]
    schema = payload["text"]["format"]["schema"]
    assert "left pitch reference gutter" in user_text
    assert "clean handwritten crop on the right" in user_text
    assert "two visual regions" in payload["instructions"]
    assert "evidence" in schema["properties"]["items"]["items"]["properties"]
    assert "overall_confidence" in schema["required"]
    assert transcription.items[0].kind == "rest"
    assert transcription.items[0].evidence == "visible empty opening"
    assert transcription.overall_confidence == 0.55


def test_fixture_model_id_preserves_existing_gemini_keys_and_namespaces_openai() -> None:
    assert fixture_model_id("gemini", "shared-model") == "shared-model"
    assert fixture_model_id("openai", "shared-model") == "openai:shared-model"
    assert (
        fixture_model_id("openai", "shared-model", openai_reasoning_effort="medium")
        == "openai:shared-model:reasoning-medium"
    )


def test_gemini_schema_strips_openai_additional_properties() -> None:
    schema_text = json.dumps(_gemini_response_schema("pitch"))

    assert "additionalProperties" not in schema_text


def _make_vlm_out(tmp_path: Path) -> Path:
    out_dir = tmp_path / "out"
    system_dir = out_dir / "demo" / "vlm_melody_inputs" / "system_001"
    system_dir.mkdir(parents=True)
    records = []
    for measure_index in (1, 2):
        paths = {}
        for input_kind, suffix in (
            ("measure_raw", "raw"),
            ("measure_staff", "staff"),
            ("measure_staff_overlay", "staff_overlay"),
        ):
            image_path = system_dir / f"measure_{measure_index:03d}_{suffix}.png"
            Image.new("RGB", (10, 10), color="white").save(image_path)
            paths[input_kind] = str(image_path)
        context_path = system_dir / f"measure_{measure_index:03d}_context.json"
        context = {
            "slug": "demo",
            "title": "Demo",
            "rhythm": "pasillo",
            "clef_hint": "treble",
            "time_signature_hint": "3/4",
            "key_hint": None,
            "system_index": 1,
            "system_measure_index": measure_index,
            "global_measure_index": measure_index - 1,
            "display_measure_number": measure_index,
            "allow_pickup": measure_index == 1,
            "expected_measure_beats": "3",
            "paths": {**paths, "context": str(context_path)},
            "staff_lines_y_px_in_system": [1, 2, 3, 4, 5],
            "staff_lines_y_px_in_staff_crop": [1, 2, 3, 4, 5],
        }
        context_path.write_text(json.dumps(context), encoding="utf-8")
        records.append(context)
    (out_dir / "vlm_melody_inputs_manifest.jsonl").write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    pitch_ruler_records = []
    for record in records:
        pitch_ruler_path = (
            system_dir / f"measure_{int(record['system_measure_index']):03d}_pitch_ruler.png"
        )
        Image.new("RGB", (20, 10), color="white").save(pitch_ruler_path)
        pitch_ruler_record = dict(record)
        pitch_ruler_paths = dict(record["paths"])
        pitch_ruler_paths["measure_pitch_ruler"] = str(pitch_ruler_path)
        pitch_ruler_soft_path = (
            system_dir / f"measure_{int(record['system_measure_index']):03d}_pitch_ruler_soft.png"
        )
        Image.new("RGB", (20, 10), color="white").save(pitch_ruler_soft_path)
        pitch_ruler_paths["measure_pitch_ruler_soft"] = str(pitch_ruler_soft_path)
        pitch_ruler_panel_path = (
            system_dir / f"measure_{int(record['system_measure_index']):03d}_pitch_ruler_panel.png"
        )
        Image.new("RGB", (20, 10), color="white").save(pitch_ruler_panel_path)
        pitch_ruler_paths["measure_pitch_ruler_panel"] = str(pitch_ruler_panel_path)
        pitch_ruler_record["paths"] = pitch_ruler_paths
        pitch_ruler_record["pitch_ruler"] = {
            "source_kind": "staff_overlay",
            "clef": "treble",
            "style": "standard",
            "staff_lines_y_px": [1, 2, 3, 4, 5],
            "label_width_px": 72,
        }
        pitch_ruler_records.append(pitch_ruler_record)
    (out_dir / "vlm_pitch_ruler_inputs_manifest.jsonl").write_text(
        "\n".join(json.dumps(record) for record in pitch_ruler_records) + "\n",
        encoding="utf-8",
    )
    return out_dir
