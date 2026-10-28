"""Normalize ABC for complete-score evaluation with the local review renderer."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path


def normalize_abc(text: str) -> dict:
    """Return written bars and the separately resolved playback route.

    The optional local abc2svg/Node installation is the same one used by the
    review desk. No network service, fixture truth, or score file is read.
    """
    if not isinstance(text, str):
        raise TypeError("ABC must be a string")
    empty = {
        "schema_version": 1,
        "valid": False,
        "errors": [],
        "warnings": [],
        "ticks_per_quarter": 384,
        "measures": [],
        "route": None,
        "route_status": "unavailable",
        "counts": {
            "measures": 0,
            "noteheads": 0,
            "rests": 0,
            "chord_labels": 0,
            "performed_measures": None,
        },
    }
    node, renderer = shutil.which("node"), shutil.which("abc2svg")
    renderer_dir = Path(renderer).resolve().parent if renderer else None
    if (
        not node
        or not renderer_dir
        or not all((renderer_dir / name).is_file() for name in ("abc2svg-1.js", "toaudio-1.js"))
    ):
        return {**empty, "errors": ["Local abc2svg and Node renderer unavailable."]}
    try:
        result = subprocess.run(
            [
                node,
                "--max-old-space-size=256",
                str(Path(__file__).with_name("score_eval_parse.js")),
                str(renderer_dir),
            ],
            input=json.dumps({"abc": text}),
            text=True,
            capture_output=True,
            timeout=8,
            check=True,
        )
        parsed = json.loads(result.stdout)
        if not isinstance(parsed, dict) or parsed.get("schema_version") != 1:
            raise ValueError("Invalid parser output")
        return parsed
    except (subprocess.SubprocessError, ValueError, OSError):
        return {**empty, "errors": ["ABC evaluation parsing failed or timed out."]}
