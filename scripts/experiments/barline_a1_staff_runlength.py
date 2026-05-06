"""Thin wrapper: delegates to the production detector.

A1 (staff-aware vertical run length) was folded into
`score2abc.chord_ocr.alignment.detect_barlines` after winning the bake-off.
Kept here so `eval_a1_all_systems.py` keeps working as a regression harness
when tweaking detector parameters via this script.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from score2abc.chord_ocr.alignment import detect_barlines


def detect(image_path: Path) -> list[float]:
    return detect_barlines(image_path)


if __name__ == "__main__":
    print(json.dumps(detect(Path(sys.argv[1]))))
