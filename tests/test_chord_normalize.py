import pytest

from score2abc.chord_ocr import normalize_chord_symbol


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # Major
        ("C", "C"),
        ("C major", "C"),
        ("Cmaj", "C"),
        ("CMaj", "C"),
        ("CM", "C"),
        # Minor
        ("Em", "Em"),
        ("Emin", "Em"),
        ("E-", "Em"),
        ("E minor", "Em"),
        ("em", "Em"),
        ("EMIN", "Em"),
        # Dominant 7 and major 7 disambiguation
        ("B7", "B7"),
        ("B 7", "B7"),
        ("Bmaj7", "Bmaj7"),
        ("BM7", "Bmaj7"),
        ("Bmajor7", "Bmaj7"),
        ("Bm7", "Bm7"),
        ("BMIN7", "Bm7"),
        # Minor-sixth (present in ground truth)
        ("Gm6", "Gm6"),
        ("Gmin6", "Gm6"),
        # Diminished / half-diminished / augmented
        ("Cdim", "Cdim"),
        ("C°", "Cdim"),
        ("Cº", "Cdim"),
        ("Cm7b5", "Cm7b5"),
        ("Cø", "Cm7b5"),
        ("Caug", "Caug"),
        ("C+", "Caug"),
        # Accidentals + unicode
        ("Eb", "Eb"),
        ("E♭", "Eb"),
        ("F#7", "F#7"),
        ("F♯7", "F#7"),
        ("BΔ", "Bmaj7"),
        ("BΔ7", "Bmaj7"),
        # Slash chords
        ("D/F#", "D/F#"),
        ("Dmaj/F#", "D/F#"),
        ("D major/F#", "D/F#"),
        ("d/f#", "D/F#"),
        # Whitespace
        ("  Em  ", "Em"),
        ("B  7", "B7"),
        # Empty
        ("", ""),
        ("   ", ""),
    ],
)
def test_normalize_chord_symbol(raw: str, expected: str) -> None:
    assert normalize_chord_symbol(raw) == expected


def test_normalize_chord_symbol_passes_through_unknown_suffix() -> None:
    assert normalize_chord_symbol("C7b9") == "C7b9"


def test_normalize_chord_symbol_none_returns_empty_string() -> None:
    assert normalize_chord_symbol(None) == ""
