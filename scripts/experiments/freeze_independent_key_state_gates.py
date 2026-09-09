"""Prepare two truth-blind independent gates for automatic visual key state.

The targets and policies are fixed before transcription. Preparation reads only
pipeline metadata and system images, pins the visual key detector result, and
creates measure crops for the existing score-disjoint selector.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiments import freeze_third_score_heldout as base  # noqa: E402
from scripts.experiments import spike_consumed_key_signature_detector as detector  # noqa: E402

OUTPUT_SUBDIR = "vlm_melody_independent_key_gate"
DEFAULT_NAMESPACE = "v1"
EVALUATOR_VERSION = "independent-key-state-gate-v1"

ESTRELLA_SLUG = "jaime-llanos_41_estrella-del-caribe_danza_luis-a-calvo"
SOBRE_EL_HUMO_SLUG = "jaime-llanos_92_sobre-el-humo_bambuco_fulgencio-garcia"


@dataclass(frozen=True)
class KeyGateCase:
    case_id: str
    slug: str
    target_system_index: int
    key_source_system_index: int
    detector_mode: str
    expected_crop_count: int
    policy: base.LayoutPolicy
    allow_pickup: bool
    boundary_hint_x: int | None = None
    key_event_x_px: int | None = None
    allow_inconclusive_diagnostic: bool = False

    @property
    def gate(self) -> base.HeldoutGateSpec:
        return base.HeldoutGateSpec(
            key=f"independent_key_{self.case_id}",
            output_subdir=OUTPUT_SUBDIR,
            evaluator_version=EVALUATOR_VERSION,
            implementation_path=Path(__file__),
        )

    @property
    def namespace(self) -> str:
        return f"{DEFAULT_NAMESPACE}_{self.case_id}"


CASES = {
    "estrella_initial_s3": KeyGateCase(
        case_id="estrella_initial_s3",
        slug=ESTRELLA_SLUG,
        target_system_index=3,
        key_source_system_index=1,
        detector_mode=detector.MODE_INITIAL,
        expected_crop_count=6,
        # This interior system is visually clean; its spacing CV is only 0.014435
        # above the older generic held-out threshold.
        policy=base.LayoutPolicy(
            min_measure_count=6,
            max_measure_count=6,
            max_spacing_cv=0.37,
        ),
        allow_pickup=False,
    ),
    "sobre_change": KeyGateCase(
        case_id="sobre_change",
        slug=SOBRE_EL_HUMO_SLUG,
        target_system_index=7,
        key_source_system_index=7,
        detector_mode=detector.MODE_CHANGE,
        expected_crop_count=7,
        policy=base.LayoutPolicy(min_measure_count=7, max_measure_count=7),
        allow_pickup=False,
    ),
}


CHALLENGE_CASES = {
    "chispazo_internal_change_s3": KeyGateCase(
        case_id="chispazo_internal_change_s3",
        slug="jaime-llanos_25_chispazo_pasillo_pedro-morales-pino",
        target_system_index=3,
        key_source_system_index=3,
        detector_mode=detector.MODE_CHANGE,
        expected_crop_count=8,
        policy=base.LayoutPolicy(
            min_measure_count=8,
            max_measure_count=8,
            max_spacing_cv=0.31,
        ),
        allow_pickup=False,
        # Preregistered from the truth-blind full-system barline scan. The
        # detector currently fails closed at this visually credible change.
        boundary_hint_x=1281,
        key_event_x_px=1284,
        allow_inconclusive_diagnostic=True,
    ),
    "alcira_system_entry_change_s6": KeyGateCase(
        case_id="alcira_system_entry_change_s6",
        slug="jaime-llanos_5_alcira_bambuco_oriol-rangel",
        target_system_index=6,
        key_source_system_index=6,
        # The changed signature starts immediately after the system's leading
        # structural bar, so it uses the left-edge signature detector while
        # preserving a stateful event x for pitch replay.
        detector_mode=detector.MODE_INITIAL,
        expected_crop_count=6,
        policy=base.LayoutPolicy(min_measure_count=6, max_measure_count=6),
        allow_pickup=False,
        # Truth-blind detector search region ends at x=328; both selected
        # sharp glyphs are fully to its left and sounding notes begin after it.
        key_event_x_px=328,
    ),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", nargs="?", type=Path, default=Path("out"))
    available_cases = {**CASES, **CHALLENGE_CASES}
    parser.add_argument("--case", choices=["all", *available_cases], default="all")
    args = parser.parse_args(argv)
    # Preserve the historical meaning of ``all`` so the two sealed v1 gates
    # are never implicitly mixed with a later challenge.
    selected = CASES.values() if args.case == "all" else (available_cases[args.case],)
    try:
        reports = [prepare_case(args.out_dir, case=case) for case in selected]
    except (FileExistsError, FileNotFoundError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(reports, indent=2, sort_keys=True))
    return 0


def prepare_case(out_dir: Path, *, case: KeyGateCase) -> dict[str, Any]:
    """Prepare one preregistered case and pin its automatic visual key state."""
    out_dir = out_dir.resolve()
    report = base.prepare_heldout_score(
        out_dir,
        namespace=case.namespace,
        candidate_pool=(
            base.Candidate(case.slug, case.target_system_index, "preregistered_key_gate_target"),
        ),
        policy=case.policy,
        gate=case.gate,
    )
    prepared_path = Path(report["prepared_manifest"])
    prepared = base._read_json(prepared_path)
    expected_target = {"slug": case.slug, "system_index": case.target_system_index}
    if prepared.get("target") != expected_target:
        raise ValueError(f"Unexpected independent-key target: {prepared.get('target')}")
    if len(prepared["artifacts"]["crops"]) != case.expected_crop_count:
        raise ValueError(
            f"Expected {case.expected_crop_count} crops for {case.case_id}, "
            f"got {len(prepared['artifacts']['crops'])}"
        )

    context_records, key_state = _write_context(
        out_dir,
        namespace_root=prepared_path.parent,
        case=case,
    )
    prepared["artifacts"]["context"] = context_records
    prepared["independent_key_gate"] = {
        "case_id": case.case_id,
        "baseline_fifths": None,
        "automatic_fifths": key_state["lane_fifths"],
        "strict_detector_fifths": key_state["strict_detector_fifths"],
        "automatic_lane_kind": key_state["lane_kind"],
        "key_event_x_px": key_state["key_event_x_px"],
        "localization_contract": "one selector pass shared by both pitch lanes",
        "truth_used": False,
    }
    base._write_json(prepared_path, prepared)
    report.update(
        {
            "case_id": case.case_id,
            "prepared_manifest_sha256": base._sha256(prepared_path),
            "context": context_records,
            "automatic_fifths": key_state["lane_fifths"],
            "strict_detector_fifths": key_state["strict_detector_fifths"],
            "automatic_lane_kind": key_state["lane_kind"],
        }
    )
    return report


def _write_context(
    out_dir: Path,
    *,
    namespace_root: Path,
    case: KeyGateCase,
) -> tuple[dict[str, dict[str, str]], dict[str, Any]]:
    metadata_path = out_dir / case.slug / "metadata.json"
    key_image_path = (
        out_dir / case.slug / "systems" / f"system_{case.key_source_system_index:03d}.png"
    )
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing pipeline metadata: {metadata_path}")
    if not key_image_path.is_file():
        raise FileNotFoundError(f"Missing key detector image: {key_image_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError(f"Expected metadata object: {metadata_path}")

    context_dir = namespace_root / "context"
    context_dir.mkdir(parents=False, exist_ok=False)
    detector_kwargs: dict[str, Any] = {"mode": case.detector_mode}
    if case.boundary_hint_x is not None:
        detector_kwargs["boundary_hint_x"] = case.boundary_hint_x
    prediction = detector.detect_signature(key_image_path, **detector_kwargs)
    strict_fifths = prediction.get("fifths")
    lane_kind = "strict_automatic_key"
    if prediction.get("gate_passed") is not True:
        raise ValueError(f"Visual key detector structural gate failed for {case.case_id}")
    if strict_fifths is None:
        if not case.allow_inconclusive_diagnostic:
            raise ValueError(f"Visual key detector was inconclusive for {case.case_id}")
        candidates = prediction.get("signature_candidates")
        if not isinstance(candidates, list) or not candidates:
            raise ValueError(
                f"Visual key detector produced no diagnostic candidate for {case.case_id}"
            )
        lane_fifths = int(candidates[0]["fifths"])
        lane_kind = "inconclusive_top_candidate_diagnostic"
    else:
        strict_fifths = int(strict_fifths)
        lane_fifths = strict_fifths

    prediction_path = context_dir / "visual_key_state.json"
    base._write_json(prediction_path, prediction)
    overlay_path = context_dir / "visual_key_state_overlay.png"
    detector._draw_overlay(prediction, overlay_path)

    allowed_path = context_dir / "allowed_context.json"
    key_hint = _key_hint(lane_fifths) if lane_kind == "strict_automatic_key" else None
    warnings = [
        "Rhythm, onset, duration, and rest decoding are intentionally disabled.",
        "The later gate compares pitch only with candidate identities and coordinates fixed.",
    ]
    if lane_kind != "strict_automatic_key":
        warnings.append(
            "The strict detector returned unknown; the keyed lane is a diagnostic top-candidate "
            "upper bound and is not promotion-eligible automatic output."
        )
    allowed = {
        "schema_version": 1,
        "kind": "independent_key_state_truth_blind_allowed_context",
        "truth_accessed": False,
        "truth_used": False,
        "target_slug": case.slug,
        "allowed_context": {
            "clef": "treble",
            "time_signature": None,
            "key_hint": key_hint,
            "expected_measure_beats": None,
            "allow_pickup": case.allow_pickup,
        },
        "provenance": {
            "clef": "melody-spike fixed treble-clef contract; not target truth",
            "time_signature": "withheld to isolate key-state mapping",
            "key_hint": (
                f"automatic visual {case.detector_mode} key-signature detector"
                if key_hint is not None
                else "withheld because the strict visual detector returned unknown"
            ),
            "expected_measure_beats": "withheld to keep localization independent of meter",
            "allow_pickup": "preregistered from target-system position, not target truth",
            "metadata": _file_record(metadata_path),
            "key_detector_image": _file_record(key_image_path),
        },
        "baseline_fifths": None,
        "automatic_fifths": lane_fifths,
        "strict_detector_fifths": strict_fifths,
        "automatic_lane_kind": lane_kind,
        "key_event_x_px": case.key_event_x_px,
        "warnings": warnings,
    }
    base._write_json(allowed_path, allowed)
    return (
        {
            "allowed_context": _relative_record(allowed_path, namespace_root),
            "visual_key_state": _relative_record(prediction_path, namespace_root),
            "visual_key_state_overlay": _relative_record(overlay_path, namespace_root),
        },
        {
            "strict_detector_fifths": strict_fifths,
            "lane_fifths": lane_fifths,
            "lane_kind": lane_kind,
            "key_event_x_px": case.key_event_x_px,
        },
    )


def _key_hint(fifths: int) -> str:
    sharps = ("F#", "C#", "G#", "D#", "A#", "E#", "B#")
    flats = ("Bb", "Eb", "Ab", "Db", "Gb", "Cb", "Fb")
    if fifths > 0:
        return f"{fifths} sharp(s): {', '.join(sharps[:fifths])}"
    if fifths < 0:
        return f"{abs(fifths)} flat(s): {', '.join(flats[: abs(fifths)])}"
    return "no sharps or flats"


def _file_record(path: Path) -> dict[str, str]:
    return {"path": base._repo_display_path(path), "sha256": base._sha256(path)}


def _relative_record(path: Path, root: Path) -> dict[str, str]:
    return {"path": path.relative_to(root).as_posix(), "sha256": base._sha256(path)}


def gates() -> Sequence[base.HeldoutGateSpec]:
    return tuple(case.gate for case in (*CASES.values(), *CHALLENGE_CASES.values()))


if __name__ == "__main__":
    raise SystemExit(main())
