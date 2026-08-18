"""Scan full system images for internal key-signature changes.

This is consumed spike evidence. It preserves full-system staff geometry and
uses detected barlines only as x-coordinate hints; it does not read MusicXML or
human labels while producing predictions.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiments import spike_consumed_key_signature_detector as detector  # noqa: E402


def scan_systems(system_paths: Sequence[Path], out_dir: Path) -> dict[str, Any]:
    out_dir = out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=False)
    systems = []
    for system_path in sorted({path.expanduser().resolve() for path in system_paths}):
        if not system_path.is_file():
            raise FileNotFoundError(system_path)
        slug = detector._work_slug(system_path) or system_path.parent.name
        system_id = f"{slug}_{system_path.stem}"
        try:
            scan = detector.scan_internal_change_signatures(system_path)
        except (OSError, ValueError) as exc:
            systems.append(
                {
                    "system_id": system_id,
                    "input": str(system_path),
                    "error": str(exc),
                    "overlays": [],
                }
            )
            continue
        overlays = []
        for index, prediction in enumerate(scan["predictions"], start=1):
            if prediction["structural_boundary"]["style"] != "double_bar":
                continue
            overlay_path = out_dir / "overlays" / f"{system_id}_{index:03d}.png"
            detector._draw_overlay(prediction, overlay_path)
            overlays.append(
                {
                    "boundary_x_px": prediction["structural_boundary"]["x_px"],
                    "fifths": prediction["fifths"],
                    "path": str(overlay_path),
                }
            )
        systems.append({"system_id": system_id, "scan": scan, "overlays": overlays})
    report = {
        "schema_version": 1,
        "kind": "consumed_full_system_internal_key_change_scan",
        "truth_used_for_prediction": False,
        "system_count": len(systems),
        "error_count": sum("error" in item for item in systems),
        "double_bar_count": sum(
            item["scan"]["double_bar_count"] for item in systems if "scan" in item
        ),
        "hit_count": sum(item["scan"]["hit_count"] for item in systems if "scan" in item),
        "systems": systems,
    }
    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {**report, "report_path": str(report_path)}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("system", nargs="+", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        report = scan_systems(args.system, args.out_dir)
    except (FileExistsError, FileNotFoundError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(report["report_path"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
