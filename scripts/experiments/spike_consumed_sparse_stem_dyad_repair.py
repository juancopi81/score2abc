"""Evaluate a sparse shared-stem dyad repair on consumed score evidence.

The A medio palo gate exposed a failure that pitch/count scoring alone can hide:
in sparse dotted-half measures, the fallback selector can choose an augmentation
dot or chord text instead of the second hollow notehead.  This postmortem looks
for exactly one close vertical candidate pair sharing a stem, replaces only a
two-candidate/two-onset lane whose displaced candidates are visually weak, and
then scores the fixed proposal against already-consumed truth.

This is model-selection evidence.  It does not modify any frozen prediction or
the default inference path.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiments import (  # noqa: E402
    evaluate_independent_multihead_recovery_gate as evaluator,
)
from scripts.experiments import run_independent_multihead_recovery_gate as runner  # noqa: E402
from scripts.experiments import spike_consumed_polyphonic_pitch_repair as recovery  # noqa: E402

SCHEMA_VERSION = 1
OUTPUT_VERSION = "consumed_sparse_shared_stem_dyad_repair_v3"
CONFIG_ID = "sparse_dotted_shared_stem_dyad__dx_0.5__dy_0.8_1.2__score_0__stem_0.55_0.25"
DEFAULT_OUTPUT = REPO_ROOT / "out/vlm_melody_consumed_training/sparse_stem_dyad_repair_v3"

PARAMETERS = {
    "maximum_center_dx_staff_spaces": 0.5,
    "minimum_center_dy_staff_spaces": 0.8,
    "maximum_center_dy_staff_spaces": 1.2,
    "minimum_candidate_score": 0.0,
    "minimum_shared_stem_score": 0.55,
    "minimum_secondary_stem_score": 0.25,
    "stem_bbox_margin_staff_spaces": 0.4,
    "minimum_pair_x_staff_spaces": 1.0,
    "staff_band_margin_staff_spaces": 0.5,
    "minimum_dot_dx_staff_spaces": 0.75,
    "maximum_dot_dx_staff_spaces": 2.5,
    "maximum_dot_y_error_staff_spaces": 0.2,
    "maximum_dot_pair_dx_staff_spaces": 0.5,
    "maximum_dot_candidate_score": 0.0,
    "maximum_dot_stem_score": 0.35,
    "required_current_candidate_count": 2,
    "required_current_onset_group_count": 2,
}

A_MEDIO_ROOT = Path(
    "out/local_restricted/jaime-llanos_7_a-medio-palo_pasillo_m-garavito-w/"
    "vlm_melody_independent_multihead_recovery_gate/v1/system_007"
)
NO_LO_ROOT = Path(
    "out/local_restricted/jaime-llanos_73_no-lo-creas_pasillo_a-vasquez-pedrero/"
    "vlm_melody_independent_dyad_recovery_gate/v1/system_008"
)
ALCIRA_ROOT = Path(
    "out/local_restricted/jaime-llanos_5_alcira_bambuco_oriol-rangel/"
    "vlm_melody_independent_key_gate/v1_alcira_system_entry_change_s6/system_006"
)
LA_CHATA_ROOT = Path(
    "out/jaime-llanos_64_la-chata_pasillo_luis-a-calvo/"
    "vlm_melody_third_score_heldout/v2/system_007"
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", nargs="?", type=Path, default=REPO_ROOT / "out")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    try:
        report = run_spike(out_dir=args.out_dir, output_dir=args.output_dir)
    except (FileExistsError, FileNotFoundError, KeyError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(report["artifacts"]["report_json"])
    return 0


def propose_sparse_shared_stem_dyad(
    row: Mapping[str, Any],
    selector: Mapping[str, Any],
    current_selected: Sequence[Mapping[str, Any]],
    stem_features: Mapping[str, Mapping[str, Any]],
    *,
    parameters: Mapping[str, Any] = PARAMETERS,
) -> dict[str, Any]:
    """Return a bounded replacement decision without reading musical truth."""
    spacing = recovery._staff_spacing(row)
    x_radius = spacing * float(selector["nms_x_spaces"])
    current = [dict(item) for item in current_selected]
    current_ids = [str(item["candidate_id"]) for item in current]
    if len(current) != int(parameters["required_current_candidate_count"]):
        return _decision("rejected_current_candidate_count", current_ids=current_ids)
    if recovery._onset_group_count(current, x_radius) != int(
        parameters["required_current_onset_group_count"]
    ):
        return _decision("rejected_current_onset_group_count", current_ids=current_ids)

    candidates = _normalized_candidates(row)
    staff_lines = [float(value) for value in row["staff_geometry"]["raw_staff_lines_y_px"]]
    pair_candidates = []
    for index, first in enumerate(candidates):
        for second in candidates[index + 1 :]:
            proposal = _pair_proposal(
                first,
                second,
                spacing=spacing,
                staff_lines=staff_lines,
                stem_features=stem_features,
                parameters=parameters,
            )
            if proposal is None:
                continue
            dot_pairs = _augmentation_dot_pairs(
                proposal,
                candidates=candidates,
                spacing=spacing,
                stem_features=stem_features,
                parameters=parameters,
            )
            if not dot_pairs:
                continue
            proposal["augmentation_dot_pairs"] = dot_pairs
            pair_candidates.append(proposal)
    pair_clusters = _pair_clusters(pair_candidates, x_radius=x_radius)
    if len(pair_clusters) != 1:
        return _decision(
            "rejected_sparse_pair_cluster_count",
            current_ids=current_ids,
            pair_cluster_count=len(pair_clusters),
            considered_pairs=pair_candidates,
        )
    chosen = max(pair_clusters[0], key=_proposal_rank)
    proposed_ids = list(chosen["candidate_ids"])
    if float(chosen["minimum_center_x"]) < spacing * float(
        parameters["minimum_pair_x_staff_spaces"]
    ):
        return _decision(
            "rejected_leading_edge_pair",
            current_ids=current_ids,
            proposed_ids=proposed_ids,
            considered_pairs=pair_candidates,
        )
    if set(proposed_ids) == set(current_ids):
        return _decision(
            "already_selected",
            current_ids=current_ids,
            proposed_ids=proposed_ids,
            considered_pairs=pair_candidates,
        )

    candidate_by_id = {str(item["candidate_id"]): item for item in candidates}
    displaced_ids = [
        candidate_id for candidate_id in current_ids if candidate_id not in proposed_ids
    ]
    if not displaced_ids:
        return _decision(
            "rejected_not_a_replacement",
            current_ids=current_ids,
            proposed_ids=proposed_ids,
            considered_pairs=pair_candidates,
        )
    weak_displaced = []
    for candidate_id in displaced_ids:
        candidate = candidate_by_id[candidate_id]
        stem = stem_features[candidate_id]
        inside_staff_band = _inside_staff_band(
            float(candidate["center"]["y"]),
            staff_lines=staff_lines,
            spacing=spacing,
            margin_spaces=float(parameters["staff_band_margin_staff_spaces"]),
        )
        is_weak = (
            float(stem["score"]) < float(parameters["minimum_shared_stem_score"])
            or not inside_staff_band
        )
        weak_displaced.append(
            {
                "candidate_id": candidate_id,
                "stem_score": float(stem["score"]),
                "inside_staff_band": inside_staff_band,
                "visually_weak": is_weak,
            }
        )
    if not all(item["visually_weak"] for item in weak_displaced):
        return _decision(
            "rejected_strong_displaced_candidate",
            current_ids=current_ids,
            proposed_ids=proposed_ids,
            displaced=weak_displaced,
            considered_pairs=pair_candidates,
        )

    proposed = [candidate_by_id[candidate_id] for candidate_id in proposed_ids]
    if recovery._onset_group_count(proposed, x_radius) != 1:
        raise ValueError("Sparse shared-stem dyad proposal did not form one onset group")
    return _decision(
        "accepted",
        accepted=True,
        current_ids=current_ids,
        proposed_ids=proposed_ids,
        displaced=weak_displaced,
        chosen_pair=chosen,
        considered_pairs=pair_candidates,
    )


def _pair_proposal(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    *,
    spacing: float,
    staff_lines: Sequence[float],
    stem_features: Mapping[str, Mapping[str, Any]],
    parameters: Mapping[str, Any],
) -> dict[str, Any] | None:
    first_x = float(first["center"]["x"])
    first_y = float(first["center"]["y"])
    second_x = float(second["center"]["x"])
    second_y = float(second["center"]["y"])
    dx = abs(first_x - second_x) / spacing
    dy = abs(first_y - second_y) / spacing
    if dx > float(parameters["maximum_center_dx_staff_spaces"]):
        return None
    if (
        not float(parameters["minimum_center_dy_staff_spaces"])
        <= dy
        <= float(parameters["maximum_center_dy_staff_spaces"])
    ):
        return None
    if min(float(first["score"]), float(second["score"])) < float(
        parameters["minimum_candidate_score"]
    ):
        return None
    if not all(
        _inside_staff_band(
            y,
            staff_lines=staff_lines,
            spacing=spacing,
            margin_spaces=float(parameters["staff_band_margin_staff_spaces"]),
        )
        for y in (first_y, second_y)
    ):
        return None

    first_id = str(first["candidate_id"])
    second_id = str(second["candidate_id"])
    first_stem = stem_features[first_id]
    second_stem = stem_features[second_id]
    if min(float(first_stem["score"]), float(second_stem["score"])) < float(
        parameters["minimum_secondary_stem_score"]
    ):
        return None
    shared_stems = []
    for source_id, feature in ((first_id, first_stem), (second_id, second_stem)):
        if feature.get("x") is None or float(feature["score"]) < float(
            parameters["minimum_shared_stem_score"]
        ):
            continue
        stem_x = float(feature["x"])
        if all(
            _bbox_near_x(
                candidate["source"]["bbox"],
                stem_x,
                margin=spacing * float(parameters["stem_bbox_margin_staff_spaces"]),
            )
            for candidate in (first, second)
        ):
            shared_stems.append(
                {"source_candidate_id": source_id, "x": stem_x, "score": float(feature["score"])}
            )
    if not shared_stems:
        return None
    ordered = sorted((first, second), key=_candidate_sort_key)
    return {
        "candidate_ids": [str(item["candidate_id"]) for item in ordered],
        "minimum_center_x": round(min(first_x, second_x), 6),
        "center_x": round((first_x + second_x) / 2.0, 6),
        "center_dx_staff_spaces": round(dx, 6),
        "center_dy_staff_spaces": round(dy, 6),
        "combined_candidate_score": round(float(first["score"]) + float(second["score"]), 9),
        "minimum_stem_score": round(
            min(float(first_stem["score"]), float(second_stem["score"])), 6
        ),
        "shared_stems": shared_stems,
    }


def _pair_clusters(
    proposals: Sequence[Mapping[str, Any]], *, x_radius: float
) -> list[list[dict[str, Any]]]:
    clusters: list[list[dict[str, Any]]] = []
    for proposal in sorted(proposals, key=lambda item: float(item["center_x"])):
        if (
            not clusters
            or float(proposal["center_x"]) - float(clusters[-1][-1]["center_x"]) >= x_radius
        ):
            clusters.append([])
        clusters[-1].append(dict(proposal))
    return clusters


def _augmentation_dot_pairs(
    proposal: Mapping[str, Any],
    *,
    candidates: Sequence[Mapping[str, Any]],
    spacing: float,
    stem_features: Mapping[str, Mapping[str, Any]],
    parameters: Mapping[str, Any],
) -> list[dict[str, Any]]:
    head_ids = {str(value) for value in proposal["candidate_ids"]}
    candidate_by_id = {str(item["candidate_id"]): item for item in candidates}
    heads = sorted(
        (candidate_by_id[candidate_id] for candidate_id in head_ids),
        key=lambda item: float(item["center"]["y"]),
    )
    maximum_head_x = max(float(item["center"]["x"]) for item in heads)
    matches_by_head: list[list[Mapping[str, Any]]] = []
    for head in heads:
        matches = []
        head_y = float(head["center"]["y"])
        for candidate in candidates:
            candidate_id = str(candidate["candidate_id"])
            if candidate_id in head_ids:
                continue
            candidate_x = float(candidate["center"]["x"])
            dx_spaces = (candidate_x - maximum_head_x) / spacing
            if (
                not float(parameters["minimum_dot_dx_staff_spaces"])
                <= dx_spaces
                <= float(parameters["maximum_dot_dx_staff_spaces"])
            ):
                continue
            if abs(float(candidate["center"]["y"]) - head_y) / spacing > float(
                parameters["maximum_dot_y_error_staff_spaces"]
            ):
                continue
            if float(candidate["score"]) > float(parameters["maximum_dot_candidate_score"]):
                continue
            if float(stem_features[candidate_id]["score"]) > float(
                parameters["maximum_dot_stem_score"]
            ):
                continue
            matches.append(candidate)
        matches_by_head.append(matches)
    if len(matches_by_head) != 2:
        return []
    result = []
    for upper in matches_by_head[0]:
        for lower in matches_by_head[1]:
            dx_spaces = abs(float(upper["center"]["x"]) - float(lower["center"]["x"])) / spacing
            if dx_spaces > float(parameters["maximum_dot_pair_dx_staff_spaces"]):
                continue
            result.append(
                {
                    "candidate_ids": [
                        str(upper["candidate_id"]),
                        str(lower["candidate_id"]),
                    ],
                    "center_dx_staff_spaces": round(dx_spaces, 6),
                    "stem_scores": [
                        float(stem_features[str(upper["candidate_id"])]["score"]),
                        float(stem_features[str(lower["candidate_id"])]["score"]),
                    ],
                }
            )
    return result


def _proposal_rank(proposal: Mapping[str, Any]) -> tuple[float, float, float, tuple[str, ...]]:
    return (
        float(proposal["combined_candidate_score"]),
        float(proposal["minimum_stem_score"]),
        -float(proposal["center_dx_staff_spaces"]),
        tuple(str(value) for value in proposal["candidate_ids"]),
    )


def _normalized_candidates(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    result = []
    for rank, source in enumerate(row.get("candidate_predictions") or [], start=1):
        if not isinstance(source, Mapping):
            raise ValueError("Candidate prediction must be an object")
        candidate_id = recovery._candidate_id(source)
        center_x, center_y = recovery._candidate_center(source)
        result.append(
            {
                "candidate_id": candidate_id,
                "center": {"x": center_x, "y": center_y},
                "score": recovery._candidate_score(source),
                "detector_rank": int(source.get("detector_rank", rank)),
                "source": dict(source),
            }
        )
    return sorted(result, key=_candidate_sort_key)


def _candidate_sort_key(candidate: Mapping[str, Any]) -> tuple[float, float, str]:
    return (
        float(candidate["center"]["x"]),
        float(candidate["center"]["y"]),
        str(candidate["candidate_id"]),
    )


def _inside_staff_band(
    y: float, *, staff_lines: Sequence[float], spacing: float, margin_spaces: float
) -> bool:
    return (
        staff_lines[0] - margin_spaces * spacing <= y <= staff_lines[-1] + margin_spaces * spacing
    )


def _bbox_near_x(bbox: Mapping[str, Any], x: float, *, margin: float) -> bool:
    return float(bbox["left"]) - margin <= x <= float(bbox["right"]) + margin


def _decision(reason: str, *, accepted: bool = False, **details: Any) -> dict[str, Any]:
    return {"accepted": accepted, "reason": reason, **details}


def run_spike(*, out_dir: Path, output_dir: Path) -> dict[str, Any]:
    out_dir = out_dir.resolve()
    destination = output_dir.resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite sparse dyad spike: {destination}")
    stage = destination.with_name(f".{destination.name}.tmp")
    if stage.exists():
        raise FileExistsError(f"Refusing stale sparse dyad spike temp dir: {stage}")
    stage.mkdir(parents=True)
    try:
        cases = {
            "a_medio_palo": _case_paths(out_dir, A_MEDIO_ROOT),
            "no_lo_creas": _case_paths(out_dir, NO_LO_ROOT),
            "alcira": _case_paths(out_dir, ALCIRA_ROOT),
            "la_chata": _case_paths(out_dir, LA_CHATA_ROOT),
        }
        results = {}
        provenance = {}
        for name, paths in cases.items():
            result, pins = _evaluate_case(name, paths=paths, output_dir=stage / name)
            results[name] = result
            provenance[name] = pins
        a_medio = results["a_medio_palo"]
        no_lo = results["no_lo_creas"]
        gate = {
            "a_medio_pitch_improved": (
                a_medio["sparse_repair"]["exact_diatonic_staff_position_matches"]
                > a_medio["multihead_recovery"]["exact_diatonic_staff_position_matches"]
            ),
            "a_medio_structure_improved": (
                a_medio["sparse_repair"]["exact_chord_size_matches"]
                > a_medio["multihead_recovery"]["exact_chord_size_matches"]
                and a_medio["sparse_repair"]["exact_structure_crops"]
                > a_medio["multihead_recovery"]["exact_structure_crops"]
            ),
            "no_lo_creas_unchanged": no_lo["accepted_repair_count"] == 0,
            "alcira_unchanged": results["alcira"]["accepted_repair_count"] == 0,
            "la_chata_unchanged": results["la_chata"]["accepted_repair_count"] == 0,
        }
        gate["passed_consumed_model_selection"] = all(gate.values())
        report = {
            "schema_version": SCHEMA_VERSION,
            "kind": "vlm_melody_consumed_sparse_shared_stem_dyad_repair_report",
            "version": OUTPUT_VERSION,
            "status": "consumed_model_selection_only",
            "truth_used_for_model_selection": True,
            "eligible_for_heldout_claim": False,
            "config_id": CONFIG_ID,
            "parameters": PARAMETERS,
            "gate": gate,
            "results": results,
            "provenance": provenance,
            "decision": (
                "eligible_for_future_frozen_gate"
                if gate["passed_consumed_model_selection"]
                else "not_selected"
            ),
            "caveat": (
                "The prior A medio MusicXML evaluation scored staff position, not pixel identity. "
                "This postmortem explicitly replaces weak dot/text anchors with a shared-stem pair."
            ),
        }
        _write_json(stage / "report.json", report)
        (stage / "report.md").write_text(_markdown(report), encoding="utf-8")
        stage.rename(destination)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    report["artifacts"] = {
        "report_json": str(destination / "report.json"),
        "report_markdown": str(destination / "report.md"),
    }
    return report


def _case_paths(out_dir: Path, relative_root: Path) -> dict[str, Path]:
    root = out_dir / relative_root.relative_to("out")
    if "a-medio-palo" in root.as_posix():
        return {
            "inference": root / "baseline_inference_v1/inference.jsonl",
            "model": root / "frozen/artifacts/model_and_training/002_model.json",
            "truth": root / "evaluation_v1/truth.jsonl",
            "mapping": root / "evaluation_v1/mapping.json",
            "musicxml": root / "evaluation_v1/source.musicxml",
            "expected_report": root / "evaluation_v1/report.json",
        }
    if "no-lo-creas" in root.as_posix():
        return {
            "inference": root / "baseline_inference_v1/inference.jsonl",
            "model": root / "frozen/artifacts/model_and_training/002_model.json",
            "truth": root / "evaluation_v2_mapping_erratum/truth.jsonl",
            "mapping": root / "evaluation_v2_mapping_erratum/mapping.json",
            "musicxml": root / "evaluation_v2_mapping_erratum/source.musicxml",
        }
    if "alcira" in root.as_posix():
        return {
            "inference": root / "inference_v1/inference.jsonl",
            "model": root / "frozen/artifacts/model/001_model.json",
        }
    return {
        "inference": root / "inference_v2/inference.jsonl",
        "model": root / "frozen/artifacts/model/001_model.json",
    }


def _evaluate_case(
    name: str, *, paths: Mapping[str, Path], output_dir: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    rows = _read_jsonl(paths["inference"])
    model = _read_json(paths["model"])
    selector = recovery.selector_config_from_model(model)
    output_dir.mkdir(parents=True)
    paired_by_crop = {}
    records = []
    for row in rows:
        crop = int(row["identity"]["automatic_measure_index"])
        current_candidates, current_notes, stem_features = _multihead_lane(row, selector)
        decision = propose_sparse_shared_stem_dyad(row, selector, current_candidates, stem_features)
        repaired_candidates = current_candidates
        repaired_notes = current_notes
        if decision["accepted"]:
            candidate_by_id = {
                str(item["candidate_id"]): item for item in _normalized_candidates(row)
            }
            repaired_candidates = [
                candidate_by_id[candidate_id] for candidate_id in decision["proposed_ids"]
            ]
            repaired_notes = [
                runner._normalized_note(
                    row,
                    candidate,
                    onset_group_index=1,
                    recovered=False,
                )
                for candidate in repaired_candidates
            ]
            repaired_notes.sort(key=runner._note_sort_key)
            _render_overlay(
                row,
                current_candidates=current_candidates,
                repaired_candidates=repaired_candidates,
                path=output_dir / f"measure_{crop:03d}.png",
            )
        paired_by_crop[crop] = {
            "identity": dict(row["identity"]),
            "lanes": {
                "multihead_recovery": {"notes": current_notes},
                CONFIG_ID: {"notes": repaired_notes},
            },
        }
        records.append(
            {
                "automatic_crop_index": crop,
                "decision": decision,
                "current_candidate_ids": [str(item["candidate_id"]) for item in current_candidates],
                "repaired_candidate_ids": [
                    str(item["candidate_id"]) for item in repaired_candidates
                ],
            }
        )
    accepted = sum(bool(record["decision"]["accepted"]) for record in records)
    result: dict[str, Any] = {
        "accepted_repair_count": accepted,
        "records": records,
    }
    if {"truth", "mapping", "musicxml"} <= paths.keys():
        truth_rows = _read_jsonl(paths["truth"])
        mapping = _read_json(paths["mapping"])
        truth = evaluator.heldout.load_visible_musicxml_truth(paths["musicxml"])
        support = evaluator._mapping_structure_support(mapping, truth)
        current_score = evaluator._score_lane(
            truth_rows,
            paired_by_crop,
            lane="multihead_recovery",
            structure_support=support,
        )
        repaired_score = evaluator._score_lane(
            truth_rows,
            paired_by_crop,
            lane=CONFIG_ID,
            structure_support=support,
        )
        result["multihead_recovery"] = current_score["summary"]
        result["sparse_repair"] = repaired_score["summary"]
        result["delta"] = _summary_delta(repaired_score["summary"], current_score["summary"])
        result["crops"] = repaired_score["crops"]
        if name == "a_medio_palo":
            expected = _read_json(paths["expected_report"])["lanes"]["multihead_recovery"][
                "summary"
            ]
            if current_score["summary"] != expected:
                raise ValueError("A medio multihead lane no longer reproduces sealed evaluation")
    pins = {key: _pin(path) for key, path in paths.items()}
    return result, pins


def _multihead_lane(
    row: Mapping[str, Any], selector: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    baseline = recovery.select_candidates(row, selector)
    canonical = row.get("canonical_prediction")
    if not isinstance(canonical, Mapping):
        raise ValueError("Inference row has no canonical prediction")
    runner._verify_generic_baseline(row, baseline, canonical)
    stem_features, _ = recovery.candidate_local_stem_features(row)
    recovered = recovery.recover_edge_safe_stem_aware_multihead_candidates(
        row,
        selector,
        baseline,
        stem_features=stem_features,
        **recovery.EDGE_SAFE_STEM_MULTIHEAD_PARAMETERS,
    )
    runner._verify_additive_recovery(row, selector=selector, baseline=baseline, recovered=recovered)
    group_by_id = runner._baseline_group_indices(row, selector=selector, baseline=baseline)
    generic_by_id = {str(note["candidate_id"]): note for note in canonical.get("notes") or []}
    notes = [
        runner._normalized_note(
            row,
            candidate,
            onset_group_index=group_by_id[str(candidate["candidate_id"])],
            recovered=False,
            generic_note=generic_by_id[str(candidate["candidate_id"])],
        )
        for candidate in baseline
    ]
    notes.extend(
        runner._normalized_note(
            row,
            candidate,
            onset_group_index=int(candidate["recovery_group_index"]),
            recovered=True,
        )
        for candidate in recovered
    )
    notes.sort(key=runner._note_sort_key)
    return [*baseline, *recovered], notes, stem_features


def _summary_delta(repaired: Mapping[str, Any], current: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "predicted_note_count",
        "exact_diatonic_staff_position_matches",
        "predicted_onset_group_count",
        "exact_chord_size_matches",
        "exact_structure_crops",
    )
    return {key: repaired[key] - current[key] for key in keys}


def _render_overlay(
    row: Mapping[str, Any],
    *,
    current_candidates: Sequence[Mapping[str, Any]],
    repaired_candidates: Sequence[Mapping[str, Any]],
    path: Path,
) -> None:
    source_path, _ = recovery._source_image_path_and_pin(row)
    with Image.open(source_path) as opened:
        image = opened.convert("RGB")
    draw = ImageDraw.Draw(image)
    repaired_ids = {str(item["candidate_id"]) for item in repaired_candidates}
    for candidate in current_candidates:
        bbox = candidate["source"]["bbox"]
        candidate_id = str(candidate["candidate_id"])
        color = "#f59e0b" if candidate_id in repaired_ids else "#dc2626"
        draw.rectangle(
            (bbox["left"], bbox["top"], bbox["right"], bbox["bottom"]),
            outline=color,
            width=2,
        )
        draw.text((bbox["left"], max(0, bbox["top"] - 10)), candidate_id, fill=color)
    current_ids = {str(item["candidate_id"]) for item in current_candidates}
    for candidate in repaired_candidates:
        if str(candidate["candidate_id"]) in current_ids:
            continue
        bbox = candidate["source"]["bbox"]
        draw.rectangle(
            (bbox["left"], bbox["top"], bbox["right"], bbox["bottom"]),
            outline="#16a34a",
            width=2,
        )
        draw.text(
            (bbox["left"], max(0, bbox["top"] - 10)),
            str(candidate["candidate_id"]),
            fill="#16a34a",
        )
    image.save(path)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.resolve().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line) for line in path.resolve().read_text(encoding="utf-8").splitlines() if line
    ]
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"Expected non-empty JSONL objects: {path}")
    return rows


def _pin(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    data = resolved.read_bytes()
    return {"path": str(resolved), "sha256": recovery._sha256_bytes(data), "bytes": len(data)}


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Sparse shared-stem dyad repair",
        "",
        f"Decision: `{report['decision']}`.",
        "",
        "| Case | Repairs | Pitch exact delta | Onset-group delta | Chord-size delta |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, result in report["results"].items():
        delta = result.get("delta") or {}
        lines.append(
            f"| {name} | {result['accepted_repair_count']} | "
            f"{delta.get('exact_diatonic_staff_position_matches', 0):+} | "
            f"{delta.get('predicted_onset_group_count', 0):+} | "
            f"{delta.get('exact_chord_size_matches', 0):+} |"
        )
    lines.extend(
        [
            "",
            "The rule is consumed model-selection evidence only. It must pass a new frozen, "
            "score-disjoint sparse-dyad gate before runtime integration.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
