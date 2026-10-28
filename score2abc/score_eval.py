"""Offline paired evaluation; reviewed-edition agreement is not source accuracy."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from score2abc.eval_labels import verify_labels
from score2abc.reference_freeze import verify_freeze
from score2abc.score_eval_abc import normalize_abc

MODELS = [
    {"id": "astra", "label": "GPT-6 Astra", "model": "gpt-6-astra", "reasoning_effort": "xhigh"},
    {
        "id": "terra",
        "label": "GPT-6.1 Terra",
        "model": "gpt-6.1-terra",
        "reasoning_effort": "xhigh",
    },
]
METRICS = (
    "edition_note_f1",
    "edition_exact_bar_rate",
    "source_note_f1",
    "chord_f1",
    "form_exact_match",
    "meter_valid_rate",
    "bar_coverage",
    "cost_usd",
    "elapsed_seconds",
    "human_review_minutes",
    "editorial_acceptance",
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _events(bar: dict, kind: str) -> Counter:
    """Retain duplicate heads; voice names and engraving layout are not musical IDs."""
    if kind == "notes":
        return Counter(
            (
                n["onset_ticks"],
                n["duration_ticks"],
                n["pitch_midi"],
                bool(n.get("tie_start")),
                bool(n.get("tie_end")),
            )
            for n in bar.get(kind, [])
        )
    if kind == "rests":
        return Counter(
            (n["onset_ticks"], n["duration_ticks"])
            for n in bar.get(kind, [])
            if n.get("visible", True)
        )
    return Counter(
        (n["onset_ticks"], n["symbol"].replace("♭", "b").replace("♯", "#").strip())
        for n in bar.get("chords", [])
    )


def _f1(matched: int, predicted: int, expected: int) -> float | None:
    return 2 * matched / (predicted + expected) if predicted + expected else None


def _similarity(left: dict, right: dict) -> float:
    a, b = _events(left, "notes"), _events(right, "notes")
    return _f1(sum((a & b).values()), sum(a.values()), sum(b.values())) or 0.0


def align_measures(
    reference: list[dict], candidate: list[dict]
) -> list[tuple[int | None, int | None]]:
    """Global bar alignment prevents one missing bar shifting every later comparison.

    This is consumed edition alignment, not source-image segmentation. Indices are
    zero based; gaps remain explicit. Exact repetitions may have ambiguous alignment.
    """
    n, m = len(reference), len(candidate)
    if max(n, m) > 512:
        raise ValueError("Evaluation supports at most 512 written measures")
    cost = [[0.0] * (m + 1) for _ in range(n + 1)]
    steps = {}
    for i in range(1, n + 1):
        cost[i][0], steps[i, 0] = float(i), "missing"
    for j in range(1, m + 1):
        cost[0][j], steps[0, j] = float(j), "extra"
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            similarity = _similarity(reference[i - 1], candidate[j - 1])
            choices = [
                (cost[i - 1][j - 1] + 1 - similarity, "pair"),
                (cost[i - 1][j] + 1, "missing"),
                (cost[i][j - 1] + 1, "extra"),
            ]
            cost[i][j], steps[i, j] = min(choices, key=lambda x: x[0])
    result = []
    i, j = n, m
    while i or j:
        step = steps[i, j]
        if step == "pair":
            i, j = i - 1, j - 1
            result.append((i, j))
        elif step == "missing":
            i -= 1
            result.append((i, None))
        else:
            j -= 1
            result.append((None, j))
    return list(reversed(result))


def compare_normalized(reference: dict, candidate: dict) -> dict:
    """Measure descriptive edition agreement, leaving source/editorial judgment null."""
    metrics = dict.fromkeys(METRICS)
    statuses = {
        "source_note_f1": "pending_source_labels",
        "editorial_acceptance": "requires_musical_adjudication",
        "form_exact_match": "edition_route_only",
    }
    if not reference.get("valid"):
        raise ValueError("Reference cannot be parsed; evaluation must not proceed")
    if not candidate.get("valid"):
        return {
            "metrics": metrics,
            "metric_status": statuses,
            "mismatches": [],
            "coverage": {
                "reference_measures": len(reference["measures"]),
                "candidate_measures": 0,
                "source_scored_measures": 0,
            },
        }
    if reference["ticks_per_quarter"] != candidate["ticks_per_quarter"]:
        raise ValueError("Parser time units differ")
    refs, preds = reference["measures"], candidate["measures"]
    alignment = align_measures(refs, preds)
    totals = {kind: [0, 0, 0] for kind in ("notes", "rests", "chords")}
    mismatches, mapping = [], {}
    exact, paired = 0, 0
    for ri, pi in alignment:
        ref, pred = refs[ri] if ri is not None else {}, preds[pi] if pi is not None else {}
        exact_bar = ri is not None and pi is not None
        if exact_bar:
            paired += 1
            mapping[pred["number"]] = ref["number"]
        for kind, values in totals.items():
            a, b = _events(ref, kind), _events(pred, kind)
            values[0] += sum((a & b).values())
            values[1] += sum(b.values())
            values[2] += sum(a.values())
            exact_bar = exact_bar and a == b
        exact_bar = exact_bar and ref.get("meter") == pred.get("meter")
        exact += int(exact_bar)
        if not exact_bar:
            kind = "extra" if ri is None else "missing" if pi is None else "notation"
            detail = (
                "Extra bar relative to the reviewed edition."
                if ri is None
                else (
                    "Reviewed-edition bar is missing; it may contain editorial material."
                    if pi is None
                    else "Notes, durations, rests, ties, chords or meter differ from the edition."
                )
            )
            mismatches.append(
                {
                    "measure": ref.get("number"),
                    "candidate_measure": pred.get("number"),
                    "kind": kind,
                    "detail": detail,
                }
            )
    metrics["edition_note_f1"] = _f1(*totals["notes"])
    metrics["chord_f1"] = _f1(*totals["chords"])
    # Include extra bars in the denominator so appending music cannot score perfectly.
    metrics["edition_exact_bar_rate"] = exact / len(alignment) if alignment else None
    metrics["bar_coverage"] = paired / len(refs) if refs else None
    ordinary = [
        bar for bar in preds if bar.get("meter_valid") is not None and not bar.get("partial_reason")
    ]
    metrics["meter_valid_rate"] = (
        sum(bar["meter_valid"] is True for bar in ordinary) / len(ordinary) if ordinary else None
    )
    if reference.get("route_status") == candidate.get("route_status") == "resolved":
        mapped_route = [mapping.get(number) for number in candidate["route"]]
        metrics["form_exact_match"] = mapped_route == reference["route"]
        if not metrics["form_exact_match"]:
            mismatches.append(
                {
                    "measure": None,
                    "kind": "form",
                    "detail": "Performed bar/ending route differs from the selected edition.",
                }
            )
    else:
        statuses["form_exact_match"] = "unsupported_route"
    return {
        "metrics": metrics,
        "metric_status": statuses,
        "mismatches": mismatches,
        "coverage": {
            "reference_measures": len(refs),
            "candidate_measures": len(preds),
            "source_scored_measures": 0,
            "paired_measures": paired,
            "exact_measures": exact,
            "alignment_columns": len(alignment),
            "meter_scored_measures": len(ordinary),
            "meter_partial_measures": sum(bool(x.get("partial_reason")) for x in preds),
        },
        "event_counts": {
            kind: {"matched": counts[0], "predicted": counts[1], "expected": counts[2]}
            for kind, counts in totals.items()
        },
        "alignment": [
            {
                "reference": refs[i]["number"] if i is not None else None,
                "candidate": preds[j]["number"] if j is not None else None,
            }
            for i, j in alignment
        ],
    }


def validate_candidate(abc: str, decisions: list | None = None) -> dict:
    """Truth-free feedback suitable for an author; never opens references or labels."""
    score = normalize_abc(abc)
    issues = [{"kind": "parse", "detail": message} for message in score["errors"]]
    for bar in score["measures"]:
        if bar.get("meter_valid") is not True:
            issues.append(
                {
                    "kind": "meter",
                    "measure": bar["number"],
                    "observed_ticks": bar["duration_ticks"],
                    "expected_ticks": bar["expected_ticks"],
                    "partial_reason": bar.get("partial_reason"),
                    "detail": "Inspect source and voices; do not automatically pad with rests.",
                }
            )
    if score["route_status"] != "resolved":
        issues.append({"kind": "form", "detail": "Provide a supported explicit playback route."})
    if decisions is not None:
        if not isinstance(decisions, list):
            raise ValueError("Decision log must be a list")
        for decision in decisions:
            if not isinstance(decision, dict) or any(
                not decision.get(k)
                for k in ("id", "location", "category", "adopted_choice", "reason", "uncertainty")
            ):
                issues.append(
                    {
                        "kind": "decision_log",
                        "detail": (
                            "Every decision needs id, location, category, adopted choice, "
                            "reason and uncertainty."
                        ),
                    }
                )
    return {
        "valid": score["valid"],
        "counts": score["counts"],
        "issues": issues,
        "warnings": score["warnings"],
        "route_status": score["route_status"],
        "musical_accuracy": "not_measured_by_validation",
    }


def _pinned_file(record: dict, key: str, base: Path) -> tuple[Path, bytes]:
    path = Path(record[key])
    path = path if path.is_absolute() else base / path
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("Run artifacts cannot use symlinks")
    data = path.read_bytes()
    if len(data) > 256_000 or sha256(data) != record.get(key.removesuffix("_path") + "_sha256"):
        raise ValueError(f"Run artifact hash/size mismatch: {key}")
    return path.resolve(), data


def _usage_metrics(record: dict, base: Path) -> tuple[dict, dict, str | None]:
    """Only expose measured values backed by a hash-pinned measurement record."""
    allowed = {
        "cost_usd": {"provider_reported"},
        "elapsed_seconds": {"monotonic_clock", "timestamp_log"},
        "human_review_minutes": {"active_review_timer", "manual_timing"},
    }
    values = {key: record.get(key) for key in allowed}
    for key, value in values.items():
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, (float, int))
            or not math.isfinite(value)
            or value < 0
        ):
            raise ValueError(f"Invalid measured usage: {key}")
    if not any(value is not None for value in values.values()):
        return values, {}, None
    if not record.get("measurement_path"):
        raise ValueError("Measured usage requires a pinned measurement record")
    path, data = _pinned_file(record, "measurement_path", base)
    measurements = json.loads(data)
    statuses = {}
    for key, value in values.items():
        if value is None:
            continue
        item = measurements.get(key, {})
        if (
            item.get("value") != value
            or item.get("basis") not in allowed[key]
            or not isinstance(item.get("evidence"), str)
            or not item["evidence"].strip()
        ):
            raise ValueError(f"Missing or inconsistent measurement evidence: {key}")
        statuses[key] = item["basis"]
    return values, statuses, str(path)


def _verify_run_record(record: dict, base: Path) -> Path:
    """Check a pinned runner attestation, not a model's assertion of its own isolation."""
    from score2abc.score_eval_harness import AUTHOR_INSTRUCTIONS

    if not record.get("run_record_path"):
        raise ValueError("Actual runs require a pinned runner record")
    path, data = _pinned_file(record, "run_record_path", base)
    trace = json.loads(data)
    for key in ("model", "reasoning_effort", "input_pdf_sha256", "status"):
        if trace.get(key) != record.get(key):
            raise ValueError(f"Runner record differs from registered output: {key}")
    if (
        trace.get("schema_version") != 1
        or trace.get("instructions_sha256") != sha256(AUTHOR_INSTRUCTIONS.encode())
        or trace.get("isolation") != "input_only"
        or trace.get("fresh_context") is not True
        or trace.get("network_access") is not False
        or not trace.get("runner_name")
        or type(trace.get("revision_passes")) is not int
        or not 0 <= trace["revision_passes"] <= 2
    ):
        raise ValueError("Runner record does not satisfy the PDF-only paired contract")
    if record["status"] != "failed":
        for key in ("abc_sha256", "decision_log_sha256"):
            if trace.get(key) != record.get(key) or not trace.get(key):
                raise ValueError(f"Runner output hash is missing or inconsistent: {key}")
    return path


def build_report(frozen_root: Path, labels_root: Path, registry: Path | None = None) -> dict:
    """Evaluate registered outputs or produce an explicit six-by-two unrun matrix."""
    frozen_root, labels_root = Path(frozen_root).resolve(), Path(labels_root).resolve()
    frozen = verify_freeze(frozen_root)
    labels = verify_labels(labels_root, frozen_root)
    label_works = {work["slug"]: work for work in labels["works"]}
    records, evidence_kind, registry_base = [], "actual", Path.cwd()
    if registry is not None:
        registry = Path(registry).resolve()
        raw = json.loads(registry.read_text())
        if raw.get("schema_version") != 1 or raw.get("evidence_kind") != "actual":
            raise ValueError(
                "Run registries require actual evidence; use self-test for synthetic data"
            )
        records, evidence_kind, registry_base = raw["runs"], raw["evidence_kind"], registry.parent
    slugs, model_ids = {w["slug"] for w in frozen["works"]}, {m["id"] for m in MODELS}
    by_key = {}
    for record in records:
        key = record["slug"], record["model_id"]
        if key in by_key or key[0] not in slugs or key[1] not in model_ids:
            raise ValueError("Duplicate or unknown run identity")
        model = next(m for m in MODELS if m["id"] == key[1])
        if (
            record.get("model") != model["model"]
            or record.get("reasoning_effort") != model["reasoning_effort"]
        ):
            raise ValueError("Run model/settings differ from the paired contract")
        if record.get("status") not in ("complete", "partial", "failed"):
            raise ValueError("Registered runs must have an explicit outcome")
        by_key[key] = record
    cases = []
    for work in frozen["works"]:
        abc = (frozen_root / work["answer_abc_path"]).read_text()
        reference = normalize_abc(abc)
        if not reference["valid"]:
            raise ValueError(
                f"Reference parser unavailable/invalid for {work['title']}: {reference['errors']}"
            )
        results = {}
        for model in MODELS:
            record = by_key.get((work["slug"], model["id"]))
            result = {
                "status": "not_run",
                "metrics": dict.fromkeys(METRICS),
                "metric_status": {
                    "source_note_f1": "pending_source_labels",
                    "editorial_acceptance": "requires_musical_adjudication",
                },
                "coverage": {
                    "reference_measures": len(reference["measures"]),
                    "candidate_measures": None,
                    "source_scored_measures": 0,
                },
                "mismatches": [],
                "decisions": [],
                "error": None,
                "links": {"source_pdf": str(frozen_root / work["source_pdf_path"])},
            }
            if record is not None:
                expected_pdf = frozen["files"][work["source_pdf_path"]]["sha256"]
                if record.get("input_pdf_sha256") != expected_pdf:
                    raise ValueError("Run input PDF differs from frozen source")
                if evidence_kind == "actual" and record.get("authorship") != "model":
                    raise ValueError("Actual comparison requires declared model authorship")
                run_path = _verify_run_record(record, registry_base)
                result["links"]["run_record"] = str(run_path)
                result["status"] = record["status"]
                result["error"] = record.get("error")
                if record["status"] != "failed":
                    path, candidate_bytes = _pinned_file(record, "abc_path", registry_base)
                    if evidence_kind == "actual" and path.is_relative_to(frozen_root):
                        raise ValueError(
                            "A reference artifact cannot be registered as a model output"
                        )
                    candidate = normalize_abc(candidate_bytes.decode("utf-8"))
                    result.update(compare_normalized(reference, candidate))
                    result["links"]["candidate_abc"] = str(path)
                    result["candidate_sha256"] = sha256(candidate_bytes)
                    if not candidate["valid"]:
                        result["status"], result["error"] = "failed", "; ".join(candidate["errors"])
                    elif candidate["route_status"] != "resolved":
                        result["status"] = "partial"
                    elif result["coverage"].get("paired_measures") != len(reference["measures"]):
                        result["status"] = "partial"
                    if candidate_bytes == (frozen_root / work["answer_abc_path"]).read_bytes():
                        result["status"] = "partial"
                        result["mismatches"].append(
                            {
                                "measure": None,
                                "kind": "provenance",
                                "detail": (
                                    "Output is byte-identical to the reference. "
                                    "Audit runner isolation before interpreting agreement."
                                ),
                            }
                        )
                    if record.get("decision_log_path"):
                        path, decision_bytes = _pinned_file(
                            record, "decision_log_path", registry_base
                        )
                        decisions = json.loads(decision_bytes)
                        if not isinstance(decisions, list) or any(
                            not isinstance(d, dict) for d in decisions
                        ):
                            raise ValueError("Decision log must contain an array of objects")
                        result["decisions"] = decisions
                        result["links"]["decision_log"] = str(path)
                    else:
                        result["status"] = "partial"
                        result["metric_status"]["editorial_acceptance"] = "missing_decision_log"
                    result["validation"] = validate_candidate(
                        candidate_bytes.decode(), result["decisions"]
                    )
                    if result["validation"]["issues"]:
                        result["status"] = "partial" if candidate["valid"] else "failed"
                values, statuses, measurement_path = _usage_metrics(record, registry_base)
                result["metrics"].update(values)
                result["metric_status"].update(statuses)
                if measurement_path:
                    result["links"]["measurement_record"] = measurement_path
                result["usage"] = record.get("usage", {})
            results[model["id"]] = result
        cases.append(
            {
                "slug": work["slug"],
                "title": work["title"],
                "results": results,
                "reference_decisions": label_works[work["slug"]]["decisions"],
                "source_label_status": "pending_full_source_audit",
            }
        )
    return {
        "schema_version": 1,
        "evidence_kind": evidence_kind,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "frozen_manifest_sha256": sha256((frozen_root / "manifest.json").read_bytes()),
        "labels_sha256": sha256((labels_root / "manifest.json").read_bytes()),
        "models": MODELS,
        "cases": cases,
        "summary": {
            "recommendation": (
                "Insufficient measured source and editorial evidence to choose a model."
            ),
            "planned_runs": len(cases) * len(MODELS),
            "registered_runs": len(records),
            "source_accuracy_available": False,
        },
        "limitations": [
            "Edition agreement measures the chosen edition, including its editorial changes.",
            "Source fidelity and editorial acceptance await adjudicated labels.",
            "Repeated phrases can make bar alignment ambiguous; inspect mismatches.",
            "Pitch spelling, voice assignment, slurs, articulation and expression are ungraded.",
            "Written ties are compared; sounding attack/hold equivalence is not separately graded.",
            "Pinned runner attestations require independent enforcement, not model self-report.",
        ],
    }
