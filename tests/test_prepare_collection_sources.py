"""Verify safe, exact source-packet extraction with real Poppler tools."""

from __future__ import annotations

import copy
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from score2abc.manifest import load_metadata_csv
from scripts.build_collection_inventory import DEFAULT_CATALOG
from scripts.prepare_collection_sources import MANIFEST_NAME, prepare_packet

POPPLER = all(shutil.which(name) for name in ("pdfseparate", "pdfunite", "pdfinfo"))


def _write_blank_pdf(path: Path, pages: int) -> None:
    objects: list[bytes] = [b"<< /Type /Catalog /Pages 2 0 R >>"]
    kids = " ".join(f"{3 + index * 2} 0 R" for index in range(pages))
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {pages} >>".encode())
    for index in range(pages):
        page_id = 3 + index * 2
        content_id = page_id + 1
        width = 72 + index
        objects.append(
            (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {width} 72] "
                f"/Contents {content_id} 0 R >>"
            ).encode()
        )
        objects.append(b"<< /Length 0 >>\nstream\n\nendstream")

    data = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for number, body in enumerate(objects, 1):
        offsets.append(len(data))
        data.extend(f"{number} 0 obj\n".encode() + body + b"\nendobj\n")
    xref = len(data)
    data.extend(f"xref\n0 {len(objects) + 1}\n".encode())
    data.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        data.extend(f"{offset:010d} 00000 n \n".encode())
    data.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    )
    path.write_bytes(data)


def _page_widths(path: Path, pages: int) -> list[int]:
    output = subprocess.run(
        ["pdfinfo", "-f", "1", "-l", str(pages), "-box", str(path)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return [int(width) for width in re.findall(r"Page\s+\d+ size:\s+(\d+) x", output)]


@pytest.fixture
def synthetic_collection(tmp_path: Path) -> tuple[Path, Path, dict]:
    source = tmp_path / "source.pdf"
    _write_blank_pdf(source, 111)
    catalog = json.loads(DEFAULT_CATALOG.read_text())
    catalog["source"]["sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    catalog["source"]["size_bytes"] = source.stat().st_size
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
    return source, catalog_path, catalog


def test_wrong_source_leaves_no_output(tmp_path: Path):
    source = tmp_path / "wrong.pdf"
    source.write_bytes(b"wrong source")
    output = tmp_path / "packet"
    with pytest.raises(ValueError, match="hash does not match"):
        prepare_packet(source, output)
    assert not output.exists()


@pytest.mark.skipif(not POPPLER, reason="Poppler tools are required")
def test_preserves_explicit_gaps_and_two_page_work(synthetic_collection, tmp_path: Path):
    source, catalog_path, _ = synthetic_collection
    output = tmp_path / "packet"
    report = prepare_packet(source, output, catalog_path, [35, 36, 39])

    assert [work["original_pdf_pages"] for work in report["works"]] == [
        [50, 51],
        [52],
        [53],
    ]
    assert [work["output_page_count"] for work in report["works"]] == [2, 1, 1]
    assert _page_widths(output / report["works"][0]["output_pdf"], 2) == [121, 122]
    assert _page_widths(output / report["works"][1]["output_pdf"], 1) == [123]
    assert _page_widths(output / report["works"][2]["output_pdf"], 1) == [124]
    remaining_numbers = [
        row["song_number"] for row in report["remaining_missing_or_unresolved_catalog_rows"]
    ]
    assert remaining_numbers == [
        37,
        38,
        44,
        99,
        100,
    ]
    assert report["content_boundaries"]["musicxml_or_reference_content_opened"] is False
    assert (output / MANIFEST_NAME).is_file()
    assert len(list((output / "pdfs").glob("*.pdf"))) == 3
    ingest_items = load_metadata_csv(output / "metadata.csv", output)
    assert [item.slug for item in ingest_items] == [work["slug"] for work in report["works"]]


@pytest.mark.skipif(not POPPLER, reason="Poppler tools are required")
def test_rejects_overwrite_and_unsafe_or_duplicate_slugs(synthetic_collection, tmp_path: Path):
    source, catalog_path, catalog = synthetic_collection
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        prepare_packet(source, existing, catalog_path, [1])

    unsafe_or_duplicate = (
        ("../escape", "safe filename"),
        (catalog["entries"][0]["slug"], "unique"),
    )
    for slug, message in unsafe_or_duplicate:
        changed = copy.deepcopy(catalog)
        changed["entries"][1]["slug"] = slug
        changed_path = tmp_path / f"changed-{message.split()[0]}.json"
        changed_path.write_text(json.dumps(changed), encoding="utf-8")
        output = tmp_path / f"packet-{message.split()[0]}"
        with pytest.raises(ValueError, match=message):
            prepare_packet(source, output, changed_path, [1])
        assert not output.exists()

    for number in (37, 99):
        unavailable = tmp_path / f"packet-unavailable-{number}"
        with pytest.raises(ValueError, match="not verified mapped"):
            prepare_packet(source, unavailable, catalog_path, [number])
        assert not unavailable.exists()


@pytest.mark.skipif(not POPPLER, reason="Poppler tools are required")
def test_default_selects_all_95_verified_works(synthetic_collection, tmp_path: Path):
    source, catalog_path, _ = synthetic_collection
    report = prepare_packet(source, tmp_path / "all", catalog_path)
    assert report["selection"] == "all_verified"
    assert len(report["works"]) == 95
