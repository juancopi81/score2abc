"""Ask a VLM to select notehead IDs from the blind cap-24 candidate gallery.

This bounded spike removes transcription, coordinate refinement, rest detection,
evidence prose, and confidence fields from the earlier localization task. The
provider sees only the raw measure detail and labeled candidate gallery, then
returns candidate IDs. Every request and response is journaled before coordinate
ground truth is loaded for evaluation.

The default is a no-network dry run. Use ``--max-calls`` intentionally for live
attempts after loading ``OPENAI_API_KEY`` into the environment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import eval_vlm_notehead_proposal_baselines as evaluator  # noqa: E402
from scripts.build_vlm_notehead_localization_inputs import (  # noqa: E402
    build_vlm_notehead_localization_inputs,
)
from scripts.run_vlm_notehead_localization_spike import (  # noqa: E402
    LocalizationRequest,
    OpenAILocalizationTransport,
    RoleImage,
)

DEFAULT_OUT_DIR = REPO_ROOT / "out"
DEFAULT_SLUG = "jaime-llanos_12_aviador_pasillo_fulgencio-garcia"
DEFAULT_MEASURES = (1, 2, 3, 4)
DEFAULT_MODEL = "gpt-5.6-sol"
DEFAULT_REASONING_EFFORT = "medium"
DEFAULT_MAX_OUTPUT_TOKENS = 1024
DEFAULT_TIMEOUT_SECONDS = 180
DEFAULT_RUN_ID = "openai-gpt56sol-medium-cap24-s1m1-4"
CAP = 24
GT_DIR = REPO_ROOT / "tests/fixtures/vlm_melody/notehead_ground_truth"
CACHE_DIR = REPO_ROOT / ".cache/vlm_candidate_id_selector"

SYSTEM_PROMPT = (
    "You classify visual proposals for handwritten monophonic music. Select notehead "
    "candidate IDs only; do not transcribe pitches or rhythm. A notehead is the compact filled "
    "or hollow oval where a note stem attaches. Reject staff/stem intersections, bare stems, "
    "flags, beams, barlines, accidentals, clefs, text, noise, and rests. Select at most one "
    "candidate for each visible note. Make a best effort even when handwriting is ambiguous. "
    "Return only the strict JSON requested."
)

USER_PROMPT = (
    "Image 1 is the full target measure at high resolution. Image 2 is a gallery of labeled "
    "candidate patches from that same measure; each patch is centered on its candidate. Use the "
    "full measure to follow symbol grammar left-to-right, then inspect the gallery. Return exactly "
    "the candidate IDs whose patch center is on a true notehead, ordered by the noteheads' visual "
    "left-to-right position. Reject duplicate patches from the same note and do not infer an "
    "expected note count."
)


@dataclass(frozen=True)
class PreparedRequest:
    measure: int
    request: LocalizationRequest
    candidate_artifact: dict[str, Any]
    cache_path: Path
    journal_dir: Path


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        report = run_experiment(
            args.out_dir,
            slug=args.slug,
            system_index=args.system,
            measures=args.measure or DEFAULT_MEASURES,
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            max_output_tokens=args.max_output_tokens,
            timeout_seconds=args.timeout_seconds,
            max_calls=args.max_calls,
            force=args.force,
            run_id=args.run_id,
        )
    except (FileNotFoundError, KeyError, OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(report["paths"]["report_json"])
    print(report["paths"]["report_markdown"])
    print(f"live calls: {report['live_calls']}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--slug", default=DEFAULT_SLUG)
    parser.add_argument("--system", type=int, default=1)
    parser.add_argument("--measure", action="append", type=int)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--reasoning-effort",
        choices=("none", "low", "medium", "high", "xhigh", "max"),
        default=DEFAULT_REASONING_EFFORT,
    )
    parser.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS)
    parser.add_argument("--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument(
        "--max-calls",
        type=int,
        default=0,
        help="Maximum live attempts; zero is a no-network dry run.",
    )
    parser.add_argument("--force", action="store_true", help="Ignore cached provider responses.")
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    return parser


def run_experiment(
    out_dir: Path,
    *,
    slug: str,
    system_index: int,
    measures: Sequence[int],
    model: str,
    reasoning_effort: str,
    max_output_tokens: int,
    timeout_seconds: int,
    max_calls: int,
    force: bool,
    run_id: str,
    provider_client: Any = None,
) -> dict[str, Any]:
    selected_measures = tuple(sorted(set(int(value) for value in measures)))
    if not selected_measures or any(value <= 0 for value in selected_measures):
        raise ValueError("measure values must be positive")
    if max_calls < 0:
        raise ValueError("max_calls cannot be negative")
    if max_output_tokens <= 0 or timeout_seconds <= 0:
        raise ValueError("max_output_tokens and timeout_seconds must be positive")

    run_dir = out_dir / "experiments/vlm_candidate_id_selector" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    records = build_vlm_notehead_localization_inputs(
        out_dir,
        selected_slugs={slug},
        selected_systems={system_index},
        selected_measures=set(selected_measures),
        task_kind="candidate-assisted-localization",
        max_candidates=CAP,
        overwrite=True,
    )
    by_measure = {int(record["system_measure_index"]): record for record in records}
    missing = sorted(set(selected_measures) - set(by_measure))
    if missing:
        raise FileNotFoundError(f"Missing localization input records for measures: {missing}")

    # Preparation, prompt snapshots, and request hashes are complete before any GT read.
    prepared = [
        _prepare_request(
            out_dir,
            record=by_measure[measure],
            measure=measure,
            model=model,
            reasoning_effort=reasoning_effort,
            max_output_tokens=max_output_tokens,
            journal_dir=run_dir / f"measure_{measure:03d}",
        )
        for measure in selected_measures
    ]

    live_calls = 0
    responses: dict[int, dict[str, Any]] = {}
    transport = None
    for item in prepared:
        if item.cache_path.exists() and not force:
            cached = _load_json(item.cache_path)
            responses[item.measure] = {
                "status": "cached",
                "raw_response": cached["raw_response"],
                "usage": cached.get("usage"),
                "response_id": cached.get("response_id"),
                "provider_response": cached.get("provider_response"),
            }
            continue
        if live_calls >= max_calls:
            responses[item.measure] = {"status": "dry_run", "raw_response": None}
            continue
        if transport is None:
            transport = OpenAILocalizationTransport(
                model=model,
                reasoning_effort=reasoning_effort,
                max_output_tokens=max_output_tokens,
                timeout_seconds=timeout_seconds,
                image_detail="original",
                client=provider_client,
            )
        live_calls += 1
        try:
            response = transport.localize(item.request)
        except Exception as exc:  # one provider failure must not abort the batch
            responses[item.measure] = {
                "status": "failed",
                "raw_response": None,
                "error": f"{type(exc).__name__}: {exc}",
            }
            continue
        cached = {
            "raw_response": response.raw_response,
            "response_id": response.response_id,
            "usage": response.usage,
            "provider_response": response.provider_response,
        }
        _write_json(item.cache_path, cached)
        responses[item.measure] = {"status": "called", **cached}

    # Evaluation starts only after every selected request has finished or been skipped.
    results = []
    for item in prepared:
        response = responses[item.measure]
        result = _evaluate_response(
            response,
            item=item,
            slug=slug,
            system_index=system_index,
        )
        _write_json(item.journal_dir / "result.json", result)
        results.append(result)

    aggregate = _aggregate(results)
    report = {
        "schema_version": 1,
        "kind": "vlm_candidate_id_selector_spike",
        "run_id": run_id,
        "slug": slug,
        "system_index": system_index,
        "measures": list(selected_measures),
        "candidate_cap": CAP,
        "provider": "openai",
        "model": model,
        "reasoning_effort": reasoning_effort,
        "image_detail": "original",
        "max_output_tokens": max_output_tokens,
        "max_calls": max_calls,
        "live_calls": live_calls,
        "force": force,
        "ground_truth_exposure": {
            "provider": False,
            "input_builder": False,
            "evaluation_order": "all provider attempts finish before coordinate GT is loaded",
        },
        "prompt": {"system": SYSTEM_PROMPT, "user": USER_PROMPT},
        "results": results,
        "aggregate": aggregate,
        "gate": {
            "required_f1": 0.70,
            "required_recall": 0.70,
            "passed": (aggregate.get("f1", 0.0) >= 0.70 and aggregate.get("recall", 0.0) >= 0.70),
        },
        "paths": {
            "run_dir": str(run_dir),
            "report_json": str(run_dir / "report.json"),
            "report_markdown": str(run_dir / "report.md"),
        },
    }
    _write_json(run_dir / "report.json", report)
    (run_dir / "report.md").write_text(_markdown(report), encoding="utf-8")
    return report


def _prepare_request(
    out_dir: Path,
    *,
    record: dict[str, Any],
    measure: int,
    model: str,
    reasoning_effort: str,
    max_output_tokens: int,
    journal_dir: Path,
) -> PreparedRequest:
    journal_dir.mkdir(parents=True, exist_ok=True)
    image_by_role = {item["role"]: _resolve_path(item["path"]) for item in record["images"]}
    images = (
        RoleImage(role="full_measure_detail", path=image_by_role["detail"]),
        RoleImage(role="candidate_gallery", path=image_by_role["candidate_gallery"]),
    )
    candidate_path = _resolve_path(record["candidate_artifact_path"])
    artifact = _load_json(candidate_path)
    candidate_ids = tuple(str(item["id"]) for item in artifact["candidates"])
    if len(candidate_ids) != CAP:
        raise ValueError(f"Measure {measure} has {len(candidate_ids)} candidates, expected {CAP}")
    schema = {
        "type": "object",
        "properties": {
            "selected_candidate_ids": {
                "type": "array",
                "items": {"type": "string", "enum": list(candidate_ids)},
            }
        },
        "required": ["selected_candidate_ids"],
        "additionalProperties": False,
    }
    fixture_key = _fixture_key(
        images,
        schema=schema,
        model=model,
        reasoning_effort=reasoning_effort,
        max_output_tokens=max_output_tokens,
    )
    request = LocalizationRequest(
        record={"experiment_id": f"candidate-id-selector:s001:m{measure:03d}"},
        images=images,
        context_path=_resolve_path(record["context_path"]),
        context={},
        candidate_artifact_path=candidate_path,
        candidate_artifact=artifact,
        candidate_ids=candidate_ids,
        coordinate_ground_truth_path=None,
        canonical_ground_truth_path=_resolve_path(record["canonical_ground_truth_path"]),
        system_prompt=SYSTEM_PROMPT,
        user_prompt=USER_PROMPT,
        schema=schema,
        provider="openai",
        model=model,
        reasoning_effort=reasoning_effort,
        image_detail="original",
        max_output_tokens=max_output_tokens,
        fixture_key=fixture_key,
    )
    (journal_dir / "system_prompt.txt").write_text(SYSTEM_PROMPT + "\n", encoding="utf-8")
    (journal_dir / "user_prompt.txt").write_text(USER_PROMPT + "\n", encoding="utf-8")
    _write_json(
        journal_dir / "request.json",
        {
            "measure": measure,
            "model": model,
            "reasoning_effort": reasoning_effort,
            "max_output_tokens": max_output_tokens,
            "image_detail": "original",
            "images": [{"role": image.role, "path": str(image.path)} for image in images],
            "candidate_artifact": str(candidate_path),
            "candidate_ids": list(candidate_ids),
            "schema": schema,
            "fixture_key": fixture_key,
        },
    )
    cache_path = CACHE_DIR / f"{fixture_key}.json"
    return PreparedRequest(
        measure=measure,
        request=request,
        candidate_artifact=artifact,
        cache_path=cache_path,
        journal_dir=journal_dir,
    )


def _evaluate_response(
    response: dict[str, Any],
    *,
    item: PreparedRequest,
    slug: str,
    system_index: int,
) -> dict[str, Any]:
    base = {
        "measure": item.measure,
        "status": response["status"],
        "usage": response.get("usage"),
        "response_id": response.get("response_id"),
    }
    raw = response.get("raw_response")
    if not isinstance(raw, str):
        return {**base, "error": response.get("error"), "gt_status": "not_evaluated"}
    try:
        payload = json.loads(raw)
        selected_ids = payload["selected_candidate_ids"]
        if not isinstance(selected_ids, list) or not all(
            isinstance(value, str) for value in selected_ids
        ):
            raise ValueError("selected_candidate_ids must be an array of strings")
        if len(selected_ids) != len(set(selected_ids)):
            raise ValueError("selected_candidate_ids contains duplicates")
        candidate_by_id = {
            str(candidate["id"]): candidate for candidate in item.candidate_artifact["candidates"]
        }
        unknown = sorted(set(selected_ids) - set(candidate_by_id))
        if unknown:
            raise ValueError(f"Unknown candidate IDs: {unknown}")
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        return {
            **base,
            "gt_status": "not_evaluated",
            "raw_response": raw,
            "error": f"{type(exc).__name__}: {exc}",
        }

    points = [
        {
            "id": candidate_id,
            "center": candidate_by_id[candidate_id]["center"],
        }
        for candidate_id in selected_ids
    ]
    gt_path = evaluator._ground_truth_path(
        GT_DIR,
        slug=slug,
        system_index=system_index,
        measure=item.measure,
    )
    ground_truth = evaluator._load_ground_truth_fixture(gt_path)
    staff_lines = [int(value) for value in item.candidate_artifact["staff_lines_y_px"]]
    region = evaluator.match_region_points(
        points,
        ground_truth,
        staff_lines=staff_lines,
        margin=evaluator.ANNOTATION_REGION_MARGIN,
    )
    return {
        **base,
        "status": "evaluated",
        "gt_status": "evaluated",
        "raw_response": raw,
        "selected_candidate_ids": selected_ids,
        "selected_count": len(selected_ids),
        "gt_count": len(ground_truth),
        "annotation_region": region,
    }


def _aggregate(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    evaluated = [row for row in results if row.get("gt_status") == "evaluated"]
    if not evaluated:
        return {"evaluated_measure_count": 0}
    tp = sum(row["annotation_region"]["tp"] for row in evaluated)
    fp = sum(row["annotation_region"]["fp"] for row in evaluated)
    fn = sum(row["annotation_region"]["fn"] for row in evaluated)
    precision = evaluator._ratio(tp, tp + fp)
    recall = evaluator._ratio(tp, tp + fn)
    return {
        "evaluated_measure_count": len(evaluated),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": evaluator._f1(tp, fp, fn),
        "exact_measure_count": sum(row["annotation_region"]["exact_coverage"] for row in evaluated),
    }


def _fixture_key(
    images: Sequence[RoleImage],
    *,
    schema: dict[str, Any],
    model: str,
    reasoning_effort: str,
    max_output_tokens: int,
) -> str:
    digest = hashlib.sha256()
    for image in images:
        digest.update(image.role.encode("utf-8"))
        digest.update(hashlib.sha256(image.path.read_bytes()).digest())
    for value in (
        SYSTEM_PROMPT,
        USER_PROMPT,
        json.dumps(schema, sort_keys=True),
        model,
        reasoning_effort,
        str(max_output_tokens),
        "original",
    ):
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _markdown(report: dict[str, Any]) -> str:
    aggregate = report["aggregate"]
    lines = [
        "# VLM Candidate ID Selector",
        "",
        f"Model: `{report['model']}` with `{report['reasoning_effort']}` reasoning.",
        f"Live attempts: `{report['live_calls']}` / cap `{report['max_calls']}`.",
        "",
        "The provider saw no coordinate or canonical GT. All attempts completed before "
        "coordinate evaluation began.",
        "",
        "| Measure | Status | Selected | GT | TP/FP/FN | P/R/F1 |",
        "| ---: | --- | ---: | ---: | --- | --- |",
    ]
    for row in report["results"]:
        metric = row.get("annotation_region")
        if metric:
            score = f"{metric['precision']:.3f}/{metric['recall']:.3f}/{metric['f1']:.3f}"
            counts = f"{metric['tp']}/{metric['fp']}/{metric['fn']}"
        else:
            score = "-"
            counts = "-"
        lines.append(
            f"| {row['measure']} | {row['status']} | {row.get('selected_count', '-')} | "
            f"{row.get('gt_count', '-')} | {counts} | {score} |"
        )
    if aggregate.get("evaluated_measure_count"):
        lines.extend(
            [
                "",
                "## Aggregate",
                "",
                f"- TP/FP/FN: `{aggregate['tp']}/{aggregate['fp']}/{aggregate['fn']}`",
                f"- Precision/recall/F1: `{aggregate['precision']:.3f}` / "
                f"`{aggregate['recall']:.3f}` / `{aggregate['f1']:.3f}`",
                f"- Gate passed: `{'yes' if report['gate']['passed'] else 'no'}`",
            ]
        )
    lines.append("")
    return "\n".join(lines)


def _resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    candidate = REPO_ROOT / path
    if candidate.exists():
        return candidate
    return path


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
