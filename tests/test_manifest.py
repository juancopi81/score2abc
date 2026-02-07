import textwrap
from pathlib import Path

import pytest

from score2abc.manifest import (
    _slugify,
    load_manifest_jsonl,
    load_metadata_csv,
    write_manifest_jsonl,
)


def _write_csv(path: Path, contents: str) -> None:
    path.write_text(textwrap.dedent(contents).lstrip(), encoding="utf-8")


def test_slugify_strips_accents_and_punctuation() -> None:
    assert _slugify("Acuatá (Pasillo)") == "acuata-pasillo"
    assert _slugify("Gato'e / Joropo") == "gatoe-joropo"
    assert _slugify("   ") == "untitled"


def test_load_metadata_csv_with_pdf_file(tmp_path: Path) -> None:
    input_dir = tmp_path / "dataset"
    input_dir.mkdir()
    (input_dir / "acuata.pdf").write_bytes(b"%PDF-1.4\n%stub\n")
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
    (input_dir / "el-pajarillo-joropo-simon-diaz.pdf").write_bytes(b"%PDF-1.4\n%stub\n")
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


def test_load_metadata_csv_requires_existing_pdf(tmp_path: Path) -> None:
    input_dir = tmp_path / "dataset"
    input_dir.mkdir()
    csv_path = tmp_path / "metadata.csv"
    _write_csv(
        csv_path,
        """
        pdf_file,title,composer,rhythm
        missing.pdf,Acuata,Fulgencio Garcia,Pasillo
        """,
    )

    with pytest.raises(FileNotFoundError, match="Source PDF not found"):
        load_metadata_csv(csv_path, input_dir)


def test_load_metadata_csv_rejects_duplicate_slug(tmp_path: Path) -> None:
    input_dir = tmp_path / "dataset"
    input_dir.mkdir()
    (input_dir / "dup.pdf").write_bytes(b"%PDF-1.4\n%stub\n")
    csv_path = tmp_path / "metadata.csv"
    _write_csv(
        csv_path,
        """
        pdf_file,title,composer,rhythm
        dup.pdf,Acuata,Fulgencio Garcia,Pasillo
        dup.pdf,Carrizal,Emilio Murillo,Pasillo
        """,
    )

    with pytest.raises(ValueError, match="Duplicate slug"):
        load_metadata_csv(csv_path, input_dir)


def test_manifest_paths_round_trip_relative(tmp_path: Path) -> None:
    input_dir = tmp_path / "dataset"
    input_dir.mkdir()
    pdf_path = input_dir / "acuata.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%stub\n")
    csv_path = tmp_path / "metadata.csv"
    _write_csv(
        csv_path,
        """
        pdf_file,title,composer,rhythm
        acuata.pdf,Acuata,Fulgencio Garcia,Pasillo
        """,
    )
    items = load_metadata_csv(csv_path, input_dir)
    manifest_path = tmp_path / "out" / "manifest.jsonl"
    write_manifest_jsonl(items, manifest_path)

    content = manifest_path.read_text(encoding="utf-8")
    assert str(pdf_path) not in content

    loaded = load_manifest_jsonl(manifest_path)
    assert loaded[0].pdf_path == pdf_path.resolve()
