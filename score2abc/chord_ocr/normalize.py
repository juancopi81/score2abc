from __future__ import annotations

import re

_UNICODE_PRE: list[tuple[str, str]] = [
    ("Δ7", "maj7"),
    ("Δ", "maj7"),
    ("♭", "b"),
    ("♯", "#"),
    ("°", "dim"),
    ("º", "dim"),
    ("ø", "m7b5"),
    ("–", "-"),
    ("—", "-"),
]

_ROOT_RE = re.compile(r"([A-Ga-g])(##|bb|[#b])?")
_BASS_RE = re.compile(r"^([A-Ga-g])(##|bb|[#b])?$")

_QUALITY_CANON: dict[str, str] = {
    "": "",
    "maj": "",
    "major": "",
    "m": "m",
    "min": "m",
    "minor": "m",
    "-": "m",
    "7": "7",
    "maj7": "maj7",
    "major7": "maj7",
    "m7": "m7",
    "min7": "m7",
    "minor7": "m7",
    "6": "6",
    "m6": "m6",
    "min6": "m6",
    "dim": "dim",
    "dim7": "dim7",
    "m7b5": "m7b5",
    "aug": "aug",
    "+": "aug",
    "sus": "sus",
    "sus2": "sus2",
    "sus4": "sus4",
    "add9": "add9",
    "9": "9",
    "11": "11",
    "13": "13",
}


def normalize_chord_symbol(raw: str | None) -> str:
    """Canonicalize a handwritten/OCR chord symbol.

    Handles unicode accidentals (♭/♯), degree marks, slash chords, and common
    quality aliases (Emin/E-/E minor → Em; BM7/Bmaj7/BΔ → Bmaj7). Unknown
    suffixes pass through unchanged.
    """
    if raw is None:
        return ""
    cleaned = raw.strip()
    if not cleaned:
        return ""

    for src, dst in _UNICODE_PRE:
        cleaned = cleaned.replace(src, dst)
    cleaned = re.sub(r"\s+", "", cleaned)
    if not cleaned:
        return ""

    root_match = _ROOT_RE.match(cleaned)
    if not root_match:
        return cleaned

    root = root_match.group(1).upper()
    root_alter = root_match.group(2) or ""
    remainder = cleaned[root_match.end() :]

    if "/" in remainder:
        quality_raw, bass_raw = remainder.split("/", 1)
    else:
        quality_raw, bass_raw = remainder, None

    quality = _canonical_quality(quality_raw)

    result = f"{root}{root_alter}{quality}"
    if bass_raw is not None:
        bass_match = _BASS_RE.match(bass_raw)
        if bass_match:
            bass_alter = bass_match.group(2) or ""
            result += f"/{bass_match.group(1).upper()}{bass_alter}"
        else:
            result += f"/{bass_raw}"
    return result


def _canonical_quality(raw: str) -> str:
    # Capital 'M' is a major marker in jazz notation — handle before case folding.
    if raw == "M":
        return ""
    if raw == "M7":
        return "maj7"

    if raw in _QUALITY_CANON:
        return _QUALITY_CANON[raw]
    lowered = raw.lower()
    if lowered in _QUALITY_CANON:
        return _QUALITY_CANON[lowered]
    return raw
