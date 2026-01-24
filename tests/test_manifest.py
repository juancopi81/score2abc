import textwrap
from pathlib import Path

import pytest

from score2abc.manifest import _slugify, load_metadata_csv


def _write_csv(path: Path, contents: str) -> None:
    path.write_text(textwrap.dedent(contents).lstrip(), encoding="utf-8")


def test_slugify_strips_accents_and_punctuation() -> None:
    assert _slugify("Acuatá (Pasillo)") == "acuata-pasillo"
    assert _slugify("Gato'e / Joropo") == "gatoe-joropo"
    assert _slugify("   ") == "untitled"


def test_load_metadata_csv_with_pdf_file(tmp_path: Path) -> None:
    input_dir = tmp_path / "dataset"
    input_dir.mkdir()
    csv_path = tmp_path / "metadata.csv"
    _write_csv(
        csv_path,
        """
        pdf_file,title,composer,rhythm,time_signature,key_hint
        acuata.pdf,Acuata,Fulgencio Garcia,Pasillo,3/4,Em
        """,
    )

    items = load_metadata_csv(csv_path, input_dir)
    assert len(items) == 1
    item = items[0]
    assert item.slug == "acuata"
    assert item.pdf_path == input_dir / "acuata.pdf"
    assert item.metadata.title == "Acuata"
    assert item.metadata.time_signature == "3/4"


def test_load_metadata_csv_without_pdf_file_uses_slug(tmp_path: Path) -> None:
    input_dir = tmp_path / "dataset"
    input_dir.mkdir()
    csv_path = tmp_path / "metadata.csv"
    _write_csv(
        csv_path,
        """
        title,composer,rhythm
        El Pajarillo,Simon Diaz,Joropo
        """,
    )

    items = load_metadata_csv(csv_path, input_dir)
    assert len(items) == 1
    item = items[0]
    assert item.slug == "el-pajarillo-joropo-simon-diaz"
    assert item.pdf_path == input_dir / "el-pajarillo-joropo-simon-diaz.pdf"


def test_load_metadata_csv_requires_core_fields(tmp_path: Path) -> None:
    input_dir = tmp_path / "dataset"
    input_dir.mkdir()
    csv_path = tmp_path / "metadata.csv"
    _write_csv(
        csv_path,
        """
        title,composer,rhythm
        ,Fulgencio Garcia,Pasillo
        """,
    )

    with pytest.raises(ValueError, match="missing title"):
        load_metadata_csv(csv_path, input_dir)
