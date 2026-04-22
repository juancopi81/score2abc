from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

from score2abc.chord_ocr import (
    CachedChordOCR,
    ChordDetection,
    ChordExtractionRequest,
    ChordOCR,
    FixtureChordOCR,
    FixtureNotFoundError,
)
from score2abc.chord_ocr.alignment import (
    assign_measures,
    detect_barlines,
    measures_in_system,
)
from score2abc.schemas import WorkMetadata


def build_chord_ocr(
    use_vlm: bool,
    *,
    fixtures_dir: Path,
    cache_dir: Path,
    model: str | None = None,
) -> ChordOCR:
    """Pick a ChordOCR backend based on the pipeline mode.

    `use_vlm=False` returns a replay-only fixture backend so runs stay hermetic;
    `use_vlm=True` returns a disk-cached Gemini backend so repeated calls on the
    same crop never re-hit the API.
    """
    if not use_vlm:
        return FixtureChordOCR(fixtures_dir)

    from score2abc.chord_ocr import GeminiChordOCR

    inner = GeminiChordOCR(model=model) if model else GeminiChordOCR()
    return CachedChordOCR(inner, cache_dir)


def extract_chords_for_systems(
    *,
    ocr: ChordOCR,
    system_crops: Sequence[Path],
    chord_crops_above: Sequence[Path],
    chord_crops_below: Sequence[Path],
    metadata: WorkMetadata,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """Run chord OCR across every system and return the chords.json payload.

    For each system we detect barlines on the system crop, call the OCR backend
    on both chord bands, and map each detection's x-fraction to a system-local
    measure. Measures are then offset by the cumulative measure count of prior
    systems so the canonical `chords` list uses global measure indices.
    """
    log = logger or logging.getLogger(__name__)

    if not (len(system_crops) == len(chord_crops_above) == len(chord_crops_below)):
        raise ValueError(
            "system_crops, chord_crops_above, and chord_crops_below must have equal length"
        )

    systems_payload: list[dict[str, Any]] = []
    canonical_chords: list[dict[str, Any]] = []
    cumulative_prior_measures = 0

    for system_index, (system_crop, above_crop, below_crop) in enumerate(
        zip(system_crops, chord_crops_above, chord_crops_below, strict=True),
        start=1,
    ):
        barlines = detect_barlines(system_crop)
        measure_count = measures_in_system(barlines)

        above_detections = _extract_safely(
            ocr,
            image_path=above_crop,
            band="above",
            system_index=system_index,
            metadata=metadata,
            logger=log,
        )
        below_detections = _extract_safely(
            ocr,
            image_path=below_crop,
            band="below",
            system_index=system_index,
            metadata=metadata,
            logger=log,
        )

        system_detections: list[dict[str, Any]] = []
        for detection, local_measure in zip(
            above_detections, assign_measures(above_detections, barlines), strict=True
        ):
            canonical_chords.append(
                _canonical_chord(detection, cumulative_prior_measures + local_measure)
            )
            system_detections.append(
                _system_detection(detection, local_measure, cumulative_prior_measures)
            )
        for detection, local_measure in zip(
            below_detections, assign_measures(below_detections, barlines), strict=True
        ):
            canonical_chords.append(
                _canonical_chord(detection, cumulative_prior_measures + local_measure)
            )
            system_detections.append(
                _system_detection(detection, local_measure, cumulative_prior_measures)
            )

        systems_payload.append(
            {
                "system_index": system_index,
                "barlines": list(barlines),
                "measure_count": measure_count,
                "measure_offset": cumulative_prior_measures,
                "detections": system_detections,
            }
        )
        cumulative_prior_measures += measure_count

    canonical_chords.sort(
        key=lambda entry: (entry["measure"], entry["source_band"], entry["symbol"])
    )

    return {
        "provider": type(ocr).__name__,
        "model_id": ocr.model_id,
        "prompt_version": ocr.prompt_version,
        "total_measures": cumulative_prior_measures,
        "systems": systems_payload,
        "chords": canonical_chords,
    }


def _extract_safely(
    ocr: ChordOCR,
    *,
    image_path: Path,
    band: str,
    system_index: int,
    metadata: WorkMetadata,
    logger: logging.Logger,
) -> list[ChordDetection]:
    request = ChordExtractionRequest(
        image_path=image_path,
        band=band,  # type: ignore[arg-type]
        system_index=system_index,
        rhythm_hint=metadata.rhythm,
        key_hint=metadata.key_hint,
        time_signature_hint=metadata.time_signature,
    )
    try:
        return list(ocr.extract(request))
    except FixtureNotFoundError as exc:
        logger.warning("Chord OCR fixture missing for %s: %s", image_path, exc)
        return []


def _canonical_chord(detection: ChordDetection, measure: int) -> dict[str, Any]:
    return {
        "measure": measure,
        "onset_beats": 0.0,
        "symbol": detection.symbol,
        "confidence": detection.confidence,
        "source_band": detection.band,
    }


def _system_detection(
    detection: ChordDetection, local_measure: int, measure_offset: int
) -> dict[str, Any]:
    return {
        "symbol": detection.symbol,
        "symbol_raw": detection.symbol_raw,
        "x_fraction": detection.x_fraction,
        "confidence": detection.confidence,
        "band": detection.band,
        "system_local_measure": local_measure,
        "measure": measure_offset + local_measure,
    }
