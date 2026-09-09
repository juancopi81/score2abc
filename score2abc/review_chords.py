"""Conservative, notation-preserving upgrades of supplied chord labels."""

from __future__ import annotations

import re

CHORD_SOURCES = {"supplied_musicxml", "recognized_musicxml", "automatic_ocr", "none", "unknown"}
_TOKEN = re.compile(r'"(?:[^"\\]|\\.)*"|[\^_=]*[A-Ga-g]|.', re.DOTALL)
_NOTE = re.compile(r"(?:\^\^?|__?|=)?[A-Ga-g]")
_CHORD = re.compile(r'"(?:[^"\\]|\\.)*"')


def strip_chords(abc: str) -> str:
    header, body = _parts(abc)
    return header + _CHORD.sub("", body)


def _parts(abc: str) -> tuple[str, str]:
    match = re.search(r"^K:[^\n]*\n", abc, re.MULTILINE)
    if not match:
        raise ValueError("Cannot locate the generated ABC body")
    return abc[: match.end()], abc[match.end() :]


def transfer_accidentals(old_base: str, saved: str, new_base: str) -> str:
    """Keep only accidental edits at existing note positions; refuse every other edit."""
    old_header, old_body = _parts(old_base)
    saved_header, saved_body = _parts(saved)
    if old_header != saved_header or strip_chords(old_base) != strip_chords(new_base):
        raise ValueError("Chord refresh conflicts with header or notation structure")
    old_tokens, saved_tokens = _TOKEN.findall(old_body), _TOKEN.findall(saved_body)
    if len(old_tokens) != len(saved_tokens):
        raise ValueError("Chord refresh supports only existing-note accidental edits")
    edits = {}
    offset = 0
    for old, edited in zip(old_tokens, saved_tokens, strict=True):
        if old != edited:
            if not (_NOTE.fullmatch(old) and _NOTE.fullmatch(edited) and old[-1] == edited[-1]):
                raise ValueError("Chord refresh conflicts with chord or structural edits")
            edits[offset] = (old, edited)
        if not old.startswith('"'):
            offset += len(old)
    header, body = _parts(new_base)
    result = []
    offset = 0
    for token in _TOKEN.findall(body):
        if token.startswith('"'):
            result.append(token)
            continue
        if offset in edits:
            old, edited = edits.pop(offset)
            if token != old:
                raise ValueError("Chord refresh note positions do not match")
            result.append(edited)
        else:
            result.append(token)
        offset += len(token)
    if edits:
        raise ValueError("Chord refresh could not preserve all accidental edits")
    return header + "".join(result)
