"""Build spike-only VLM melody image variants and experiment manifests.

This script does not call a VLM or any network API. It reads measure crops from
`scripts/build_vlm_melody_inputs.py`, writes optional derived pitch-ruler images,
and records each tested image variant with enough recipe metadata to reproduce
later fixture/eval/journal runs.

If the base VLM melody input manifest or selected base crop files are missing,
this script imports and calls `build_vlm_melody_inputs(...)` for the selected
slug/system filters with `overwrite=False`. The `--overwrite` flag here only
controls derived variant images; variant manifests are always rewritten.

Example:
    uv run python scripts/build_vlm_melody_variants.py out \\
        --slug jaime-llanos_12_aviador_pasillo_fulgencio-garcia \\
        --system 1 --measure 3 --variant all --overwrite
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from score2abc.melody.vlm import InputKind  # noqa: E402
from score2abc.utils import get_logger  # noqa: E402
from scripts.build_vlm_melody_inputs import build_vlm_melody_inputs  # noqa: E402
from scripts.build_vlm_notehead_localization_inputs import make_context_image  # noqa: E402
from scripts.build_vlm_pitch_ruler_inputs import (  # noqa: E402
    DEFAULT_LABEL_WIDTH_PX,
    SOURCE_PATH_KEYS,
    RulerStyle,
    SourceKind,
    _staff_lines_for_source,
    make_pitch_ruler_image,
)

BASE_MANIFEST_NAME = "vlm_melody_inputs_manifest.jsonl"
VARIANTS_MANIFEST_NAME = "vlm_melody_variants_manifest.jsonl"
PER_WORK_VARIANTS_MANIFEST_NAME = "variants_manifest.jsonl"


@dataclass(frozen=True)
class VariantSpec:
    variant_id: str
    input_kind: InputKind
    path_key: str | None = None
    source_kind: SourceKind | None = None
    style: RulerStyle | None = None

    @property
    def is_derived(self) -> bool:
        return self.source_kind is not None and self.style is not None


VARIANT_REGISTRY: tuple[VariantSpec, ...] = (
    VariantSpec("raw", "raw", path_key="measure_raw"),
    VariantSpec("staff", "staff", path_key="measure_staff"),
    VariantSpec("staff_overlay", "staff_overlay", path_key="measure_staff_overlay"),
    VariantSpec("neighbor_context", "neighbor_context", path_key="neighbor_context"),
    VariantSpec(
        "pitch_ruler_standard_from_staff",
        "pitch_ruler",
        source_kind="staff",
        style="standard",
    ),
    VariantSpec(
        "pitch_ruler_standard_from_staff_overlay",
        "pitch_ruler",
        source_kind="staff_overlay",
        style="standard",
    ),
    VariantSpec(
        "pitch_ruler_soft_from_staff",
        "pitch_ruler_soft",
        source_kind="staff",
        style="soft",
    ),
    VariantSpec(
        "pitch_ruler_soft_from_staff_overlay",
        "pitch_ruler_soft",
        source_kind="staff_overlay",
        style="soft",
    ),
    VariantSpec(
        "pitch_ruler_panel_from_staff",
        "pitch_ruler_panel",
        source_kind="staff",
        style="panel",
    ),
    VariantSpec(
        "pitch_ruler_panel_from_staff_overlay",
        "pitch_ruler_panel",
        source_kind="staff_overlay",
        style="panel",
    ),
)
VARIANT_IDS = tuple(spec.variant_id for spec in VARIANT_REGISTRY)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    logger = get_logger("score2abc.build_vlm_melody_variants")

    try:
        records = build_vlm_melody_variants(
            args.out_dir,
            selected_slugs=set(args.slug) if args.slug else None,
            selected_systems=set(args.system) if args.system else None,
            selected_measures=set(args.measure) if args.measure else None,
            variant_ids=tuple(args.variant) if args.variant else None,
            overwrite=args.overwrite,
        )
    except (FileNotFoundError, ValueError, KeyError) as exc:
        logger.error("%s", exc)
        return 1

    logger.info("Wrote %d VLM melody variant records", len(records))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", type=Path, help="Pipeline output directory.")
    parser.add_argument("--slug", action="append", default=None, help="Limit to a work slug.")
    parser.add_argument(
        "--system",
        action="append",
        type=int,
        default=None,
        help="Limit to a 1-based system index.",
    )
    parser.add_argument(
        "--measure",
        action="append",
        type=int,
        default=None,
        help="Limit to a system-local measure index.",
    )
    parser.add_argument(
        "--variant",
        action="append",
        choices=(*VARIANT_IDS, "all"),
        default=None,
        help="Variant id to build. Repeat for subsets; defaults to all.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite derived variant images. Manifests are always rewritten.",
    )
    return parser


def build_vlm_melody_variants(
    out_dir: Path,
    *,
    selected_slugs: set[str] | None = None,
    selected_systems: set[int] | None = None,
    selected_measures: set[int] | None = None,
    variant_ids: tuple[str, ...] | None = None,
    overwrite: bool = False,
) -> list[dict[str, Any]]:
    """Create selected local image variants and write variant manifests."""
    specs = _selected_variant_specs(variant_ids)
    base_records = _load_or_build_base_records(
        out_dir,
        selected_slugs=selected_slugs,
        selected_systems=selected_systems,
        selected_measures=selected_measures,
    )

    records: list[dict[str, Any]] = []
    for base_record in base_records:
        for spec in specs:
            records.append(
                _build_variant_record(
                    out_dir,
                    base_record,
                    spec,
                    overwrite=overwrite,
                )
            )

    _write_variant_manifests(out_dir, records)
    return records


def _selected_variant_specs(variant_ids: tuple[str, ...] | None) -> tuple[VariantSpec, ...]:
    if not variant_ids or "all" in variant_ids:
        return VARIANT_REGISTRY

    requested = set(variant_ids)
    unknown = sorted(requested - set(VARIANT_IDS))
    if unknown:
        raise ValueError(f"Unknown variant id(s): {', '.join(unknown)}")
    return tuple(spec for spec in VARIANT_REGISTRY if spec.variant_id in requested)


def _load_or_build_base_records(
    out_dir: Path,
    *,
    selected_slugs: set[str] | None,
    selected_systems: set[int] | None,
    selected_measures: set[int] | None,
) -> list[dict[str, Any]]:
    manifest_path = out_dir / BASE_MANIFEST_NAME
    if not manifest_path.exists():
        build_vlm_melody_inputs(
            out_dir,
            selected_slugs=selected_slugs,
            selected_systems=selected_systems,
            overwrite=False,
        )
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"VLM melody input manifest not found: {manifest_path}. "
            "Run scripts/build_vlm_melody_inputs.py first."
        )

    records = _read_jsonl(manifest_path)
    selected = list(
        _selected_base_records(
            records,
            selected_slugs=selected_slugs,
            selected_systems=selected_systems,
            selected_measures=selected_measures,
        )
    )
    if (not selected or _has_missing_base_paths(out_dir, selected)) and (
        out_dir / "manifest.jsonl"
    ).exists():
        build_vlm_melody_inputs(
            out_dir,
            selected_slugs=selected_slugs,
            selected_systems=selected_systems,
            overwrite=False,
        )
        selected = list(
            _selected_base_records(
                _read_jsonl(manifest_path),
                selected_slugs=selected_slugs,
                selected_systems=selected_systems,
                selected_measures=selected_measures,
            )
        )
    return selected


def _has_missing_base_paths(out_dir: Path, records: list[dict[str, Any]]) -> bool:
    for record in records:
        paths = record.get("paths", {})
        for key in ("context", "measure_raw", "measure_staff", "measure_staff_overlay"):
            value = paths.get(key)
            if not isinstance(value, str) or not _resolve_path(out_dir, value).exists():
                return True
    return False


def _build_variant_record(
    out_dir: Path,
    base_record: dict[str, Any],
    spec: VariantSpec,
    *,
    overwrite: bool,
) -> dict[str, Any]:
    context_path = _resolve_path(out_dir, base_record["paths"]["context"])
    context = json.loads(context_path.read_text(encoding="utf-8"))

    if not spec.is_derived:
        if spec.path_key is None:
            raise ValueError(f"Base variant {spec.variant_id} has no path key")
        if spec.path_key == "neighbor_context":
            image_path, recipe = _build_neighbor_context_variant(
                out_dir,
                base_record,
                context,
                overwrite=overwrite,
            )
        else:
            image_path = _resolve_path(out_dir, base_record["paths"][spec.path_key])
            recipe = {
                "kind": "base",
                "path_key": spec.path_key,
            }
    else:
        image_path, recipe = _build_derived_variant(
            out_dir,
            base_record,
            context,
            spec,
            overwrite=overwrite,
        )

    return {
        "variant_id": spec.variant_id,
        "input_kind": spec.input_kind,
        "slug": str(base_record["slug"]),
        "system_index": int(base_record["system_index"]),
        "system_measure_index": int(base_record["system_measure_index"]),
        "global_measure_index": int(base_record["global_measure_index"]),
        "display_measure_number": int(base_record["display_measure_number"]),
        "image_path": str(image_path),
        "context_path": str(context_path),
        "recipe": recipe,
    }


def _build_neighbor_context_variant(
    out_dir: Path,
    base_record: dict[str, Any],
    context: dict[str, Any],
    *,
    overwrite: bool,
) -> tuple[Path, dict[str, Any]]:
    manifest_path = out_dir / BASE_MANIFEST_NAME
    all_records = _read_jsonl(manifest_path)
    system_records = [
        record
        for record in all_records
        if str(record["slug"]) == str(base_record["slug"])
        and int(record["system_index"]) == int(base_record["system_index"])
    ]
    if not system_records:
        raise ValueError(
            "No source-system records available for neighbor context: "
            f"{base_record['slug']} system {base_record['system_index']}"
        )

    source_system_path = _resolve_path(out_dir, base_record["paths"]["source_system"])
    source_raw_path = _resolve_path(out_dir, base_record["paths"]["measure_raw"])
    output_path = source_raw_path.with_name(
        source_raw_path.name.replace("_raw.png", "_neighbor_context.png")
    )
    staff_lines = _staff_lines_for_source(context, "staff")
    staff_spacing = sum(
        right - left for left, right in zip(staff_lines, staff_lines[1:], strict=False)
    ) / (len(staff_lines) - 1)
    with Image.open(source_system_path) as opened:
        context_image, geometry = make_context_image(
            opened.convert("RGB"),
            system_records=system_records,
            target_record=base_record,
            staff_spacing=staff_spacing,
        )
    if overwrite or not output_path.exists():
        context_image.save(output_path)

    return output_path, {
        "kind": "neighbor_context",
        "source_system_path": str(source_system_path),
        "neighbor_measure_indices": geometry["neighbor_measure_indices"],
        "source_crop_x_bounds_px": geometry["source_crop_x_bounds_px"],
        "target_bounds_x_px_in_context": geometry["target_bounds_x_px_in_context"],
        "white_margin_px": geometry["white_margin_px"],
        "target_markers_touch_music_pixels": False,
    }


def _build_derived_variant(
    out_dir: Path,
    base_record: dict[str, Any],
    context: dict[str, Any],
    spec: VariantSpec,
    *,
    overwrite: bool,
) -> tuple[Path, dict[str, Any]]:
    source_kind = _require_value(spec.source_kind, "source_kind", spec.variant_id)
    style = _require_value(spec.style, "style", spec.variant_id)
    source_path = _resolve_path(out_dir, base_record["paths"][SOURCE_PATH_KEYS[source_kind]])
    output_path = _derived_output_path(source_path, source_kind, style)

    staff_lines = _staff_lines_for_source(context, source_kind)
    if overwrite or not output_path.exists():
        source_image = Image.open(source_path).convert("RGB")
        make_pitch_ruler_image(source_image, staff_lines, style=style).save(output_path)

    recipe = {
        "kind": "pitch_ruler",
        "source_kind": source_kind,
        "style": style,
        "source_path": str(source_path),
        "staff_lines_y_px": staff_lines,
        "label_width_px": DEFAULT_LABEL_WIDTH_PX,
    }
    return output_path, recipe


def _derived_output_path(source_path: Path, source_kind: SourceKind, style: RulerStyle) -> Path:
    suffix = f"_{source_kind}"
    if not source_path.stem.endswith(suffix):
        raise ValueError(f"Unexpected {source_kind} crop filename: {source_path}")
    base_stem = source_path.stem[: -len(suffix)]
    return source_path.with_name(f"{base_stem}_pitch_ruler_{style}_from_{source_kind}.png")


def _require_value(value: str | None, name: str, variant_id: str) -> str:
    if value is None:
        raise ValueError(f"Variant {variant_id} is missing {name}")
    return value


def _selected_base_records(
    records: list[dict[str, Any]],
    *,
    selected_slugs: set[str] | None,
    selected_systems: set[int] | None,
    selected_measures: set[int] | None,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for record in records:
        if selected_slugs is not None and record.get("slug") not in selected_slugs:
            continue
        if selected_systems is not None and int(record["system_index"]) not in selected_systems:
            continue
        if (
            selected_measures is not None
            and int(record["system_measure_index"]) not in selected_measures
        ):
            continue
        selected.append(record)
    return selected


def _write_variant_manifests(out_dir: Path, records: list[dict[str, Any]]) -> None:
    _write_jsonl(out_dir / VARIANTS_MANIFEST_NAME, records)

    by_slug: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_slug.setdefault(str(record["slug"]), []).append(record)

    for slug, slug_records in by_slug.items():
        manifest_path = out_dir / slug / "vlm_melody_inputs" / PER_WORK_VARIANTS_MANIFEST_NAME
        _write_jsonl(manifest_path, slug_records)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _resolve_path(out_dir: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    if path.exists():
        return path
    return out_dir / path


if __name__ == "__main__":
    raise SystemExit(main())
