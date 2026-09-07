"""Protect the reviewed collection's page ownership and source/truth boundaries."""

import copy
import json

import pytest

from scripts.build_collection_inventory import DEFAULT_CATALOG, build_report, validate_catalog


@pytest.fixture
def catalog():
    return json.loads(DEFAULT_CATALOG.read_text())


def test_reviewed_page_gaps_and_two_page_work(catalog):
    summary = validate_catalog(catalog)
    assert summary["verified_present_works"] == 95
    assert summary["missing_manuscript_pages"] == [38, 39, 45]
    assert [e["song_number"] for e in catalog["entries"] if e["mapping_status"] == "missing"] == [
        37,
        38,
        44,
    ]
    assert catalog["entries"][34]["pdf_pages"] == [50, 51]
    assert catalog["entries"][34]["manuscript_pages"] == [35, 36]
    assert all(e["mapping_status"] == "unresolved" for e in catalog["entries"][98:])


def test_rejects_duplicate_physical_ownership(catalog):
    catalog["entries"][1]["pdf_pages"] = catalog["entries"][0]["pdf_pages"]
    with pytest.raises(ValueError, match="exhaustive and unique"):
        validate_catalog(catalog)


def test_rejects_silent_loss_of_continuation_page(catalog):
    catalog["entries"][34]["pdf_pages"] = [50]
    with pytest.raises(ValueError, match="every manuscript page"):
        validate_catalog(catalog)


def test_rejects_promoting_placeholder_to_named_work(catalog):
    catalog["entries"][98]["literal_source"]["title"] = "Invented title"
    with pytest.raises(ValueError, match="remain unnamed"):
        validate_catalog(catalog)


def test_rejects_wrong_source_without_opening_references(catalog, tmp_path):
    source = tmp_path / "wrong.pdf"
    source.write_bytes(b"not the inspected source")
    with pytest.raises(ValueError, match="hash does not match"):
        build_report(catalog, source, tmp_path)


def test_linked_truth_is_only_a_path_inventory(catalog, tmp_path):
    entry = next(e for e in catalog["entries"] if e["references"])
    reference = next(r for r in entry["references"] if r["kind"] == "musicxml_reference")
    target = tmp_path / reference["path"]
    target.parent.mkdir(parents=True)
    target.write_bytes(b"intentionally invalid MusicXML; never parse this")
    before = copy.deepcopy(catalog)
    report = build_report(catalog, None, tmp_path)
    found = next(r for r in report["references"] if r["path"] == reference["path"])
    assert found["exists"] is True
    assert found["content_opened"] is False
    assert report["source_sha256_verified"] is False
    assert catalog == before
