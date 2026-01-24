from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from score2abc.manifest import load_manifest_jsonl
from score2abc.metrics import compare_events
from score2abc.utils import get_logger


def evaluate(out_dir: Path, ground_truth_dir: Path) -> int:
    logger = get_logger("score2abc.eval")
    manifest_path = out_dir / "manifest.jsonl"
    if not manifest_path.exists():
        logger.error("Manifest not found: %s", manifest_path)
        return 1
    if not ground_truth_dir.exists():
        logger.error("Ground-truth directory not found: %s", ground_truth_dir)
        return 1

    work_items = load_manifest_jsonl(manifest_path)
    results: List[Dict[str, object]] = []

    works_total = len(work_items)
    works_with_truth = 0
    works_with_predictions = 0

    for item in work_items:
        truth_path = ground_truth_dir / f"{item.slug}.json"
        if not truth_path.exists():
            logger.warning("Ground truth missing: %s", truth_path)
            continue
        works_with_truth += 1

        pred_path = out_dir / item.slug / "intermediate" / "events.json"
        if not pred_path.exists():
            logger.warning("Predicted events missing: %s", pred_path)
            continue
        works_with_predictions += 1

        try:
            pred_events = json.loads(pred_path.read_text(encoding="utf-8"))
            truth_events = json.loads(truth_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            logger.error("Failed to parse JSON for %s: %s", item.slug, exc)
            continue

        metrics = compare_events(pred_events, truth_events)
        results.append(
            {
                "slug": item.slug,
                "title": item.metadata.title,
                "metrics": metrics,
            }
        )

    summary = _summarize_results(
        results,
        works_total=works_total,
        works_with_truth=works_with_truth,
        works_with_predictions=works_with_predictions,
    )
    report = {"summary": summary, "works": results}

    eval_dir = out_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    report_path = eval_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote evaluation report: %s", report_path)
    return 0


def _summarize_results(
    results: List[Dict[str, object]],
    *,
    works_total: int,
    works_with_truth: int,
    works_with_predictions: int,
) -> Dict[str, object]:
    evaluated = len(results)

    def _rate(key: str) -> float:
        if evaluated == 0:
            return 0.0
        matches = sum(1 for item in results if item["metrics"].get(key))
        return matches / evaluated

    return {
        "works_total": works_total,
        "works_with_truth": works_with_truth,
        "works_with_predictions": works_with_predictions,
        "works_evaluated": evaluated,
        "note_count_match_rate": _rate("note_count_match"),
        "chord_count_match_rate": _rate("chord_count_match"),
        "time_signature_match_rate": _rate("time_signature_match"),
    }
