"""Real approved-score identity and form regressions; no network or score mutation."""

import copy
import json

import pytest

from scripts import build_reference_mapping_checks as builder
from scripts.build_reference_mapping_checks import DEFAULT_SPEC, compile_contract, dashboard
from scripts.check_score_coverage_form import check_normalized


@pytest.fixture
def works():
    spec = json.loads(DEFAULT_SPEC.read_text())
    return {work["key"]: work for work in spec["works"]}


def normalized(work):
    pin = work["normalization"]
    return {
        "valid": True,
        "measures": [{} for _ in range(pin["written_bar_count"])],
        "route_status": pin["route_status"],
        "route": copy.deepcopy(pin["route"]),
    }


def test_all_six_declared_routes_agree_with_preserved_edition_pins(works):
    assert len(works) == 6
    assert sum(len(work["source_bars"]) for work in works.values()) == 312
    assert sum(work["normalization"]["written_bar_count"] for work in works.values()) == 314
    for work in works.values():
        checked = check_normalized(normalized(work), compile_contract(work))
        assert checked["coverage"]["status"] == checked["form"]["status"] == "pass"


def test_aviador_insertions_are_not_source_bars(works):
    contract = compile_contract(works["aviador"])
    additions = [x for x in contract["assignments"] if x["kind"] == "editorial_insertion"]
    assert [x["written_bar"] for x in additions] == [29, 45]
    assert all("source_bar_id" not in x for x in additions)
    assert len(contract["expected_source_bar_ids"]) == 62


@pytest.mark.parametrize("junction", [(16, 36), (33, 36)])
def test_coqueteos_destination_specific_endings_cannot_replace_coda_entry(works, junction):
    work = works["coqueteos"]
    score, contract = normalized(work), compile_contract(work)
    route = score["route"]
    index = next(i + 1 for i in range(len(route) - 1) if tuple(route[i : i + 2]) == junction)
    route[index] = junction[0] + 1  # Wrong casilla, same number of visits.
    contract["normalization"] = {**contract["normalization"], "route": route}
    checked = check_normalized(score, contract)
    assert checked["normalization_regression"]["status"] == "pass"
    assert checked["coverage"]["status"] == "pass"
    assert checked["form"]["status"] == "fail"


def test_coqueteos_final_stop_excludes_first_ending(works):
    work = works["coqueteos"]
    score, contract = normalized(work), compile_contract(work)
    assert score["route"][-1] == 51
    score["route"].append(52)
    contract["normalization"] = {**contract["normalization"], "route": score["route"]}
    assert check_normalized(score, contract)["form"]["status"] == "fail"


def test_carrizal_accidental_is_not_an_extra_source_boundary(works):
    bars = [bar for bar in works["carrizal"]["source_bars"] if bar["system"] == 5]
    assert [bar["written_bar"] for bar in bars] == [30, 31, 32, 33, 34]


def test_reconstruction_inside_a_bar_keeps_its_source_identity(works):
    for key, written_bar in [("amalia", 40), ("rumichaca", 16), ("atardecer", 4)]:
        contract = compile_contract(works[key])
        assignment = next(x for x in contract["assignments"] if x["written_bar"] == written_bar)
        assert assignment["kind"] == "primary"
        assert assignment["source_bar_id"] in contract["expected_source_bar_ids"]


@pytest.mark.parametrize("change", ["row_missing", "wrong_location", "wrong_primary", "pending"])
def test_visual_declaration_inconsistency_or_pending_questions_fail_closed(works, change):
    work = copy.deepcopy(works["amalia"])
    if change == "row_missing":
        work["source_rows"][0]["source_ids"].pop()
    elif change == "wrong_location":
        work["source_bars"][0]["system"] = 99
    elif change == "wrong_primary":
        work["source_bars"][0]["written_bar"] = 2
    else:
        work["unresolved_mapping_questions"].append("Unmapped source music")
    with pytest.raises(ValueError):
        compile_contract(work)


def test_dashboard_escapes_evidence_text(works):
    work = copy.deepcopy(works["amalia"])
    work["source_vs_edition_route"] = '<script>alert("unsafe")</script>'
    case = check_normalized(normalized(work), compile_contract(work))
    page = dashboard([work], [case], 2)
    assert "<script>" not in page
    assert "&lt;script&gt;" in page
    assert "not heldout model results" in page


def test_each_mapping_has_matching_portable_review_evidence(works):
    for work in works.values():
        assert builder.verify_audit_evidence(work, DEFAULT_SPEC.parent)


def test_review_status_cannot_replace_actual_evidence(works):
    work = copy.deepcopy(works["amalia"])
    work["provenance"]["review"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="evidence hash"):
        builder.verify_audit_evidence(work, DEFAULT_SPEC.parent)


@pytest.mark.parametrize("kind", ["absolute", "parent_segment"])
def test_evidence_paths_cannot_escape_publication_root(works, kind):
    work = copy.deepcopy(works["amalia"])
    pin = work["provenance"]["author"]
    if kind == "absolute":
        pin["path"] = str(DEFAULT_SPEC.parent / pin["path"])
    else:
        pin["path"] = "../eval/" + pin["path"]
    with pytest.raises(ValueError, match="relative without parent"):
        builder.verify_audit_evidence(work, DEFAULT_SPEC.parent)


def test_published_map_cannot_drift_from_separately_reviewed_author(works):
    work = copy.deepcopy(works["amalia"])
    work["route_segments"][0]["written_bars"].reverse()
    with pytest.raises(ValueError, match="differs from reviewed author"):
        builder.verify_audit_evidence(work, DEFAULT_SPEC.parent)


def test_builder_refuses_to_write_inside_frozen_reference_before_reading(tmp_path):
    frozen = tmp_path / "frozen"
    frozen.mkdir()
    for target in (frozen, frozen / "new-sidecar"):
        with pytest.raises(ValueError, match="outside the frozen reference"):
            builder.build(frozen, target, tmp_path / "absent-spec.json")
    assert not list(frozen.iterdir())


def test_build_preserves_verified_bytes_if_inputs_change_during_check(tmp_path, monkeypatch):
    """Exercise publication, not only the in-memory map checker, with tiny fake inputs."""
    import hashlib

    frozen = tmp_path / "frozen"
    frozen.mkdir()
    (frozen / "manifest.json").write_bytes(b"fixture manifest")
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    keys = ("amalia", "atardecer", "aviador", "carrizal", "coqueteos", "rumichaca")
    works = []
    for key in keys:
        for folder, filename, payload in (
            ("answers", "review.abc", b"fixed abc"),
            ("inputs", "source.pdf", b"fixed pdf"),
        ):
            parent = frozen / folder / key
            parent.mkdir(parents=True)
            (parent / filename).write_bytes(payload)
        work = {
            "key": key,
            "title": key,
            "slug": key,
            "abc_sha256": hashlib.sha256(b"fixed abc").hexdigest(),
            "source_pdf_sha256": hashlib.sha256(b"fixed pdf").hexdigest(),
            "audit_status": "reused_prior_mapping",
            "unresolved_mapping_questions": [],
            "normalization": {"written_bar_count": 2, "route_status": "resolved", "route": [1, 2]},
            "source_bars": [{"id": "1", "written_bar": 1}, {"id": "2", "written_bar": 2}],
            "assignments": [
                {"written_bar": n, "kind": "primary", "source_bar_id": str(n)} for n in (1, 2)
            ],
            "route_segments": [{"label": "A", "written_bars": [1, 2], "evidence": "fixture"}],
            "decisions": [],
            "source_vs_edition_route": "Fixture only",
        }
        path = evidence / f"{key}.json"
        path.write_text(json.dumps(compile_contract(work)))
        work["provenance"] = {
            "prior_contract": {
                "path": str(path.relative_to(tmp_path)),
                "sha256": builder.digest(path),
            }
        }
        works.append(work)
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "reference_manifest_sha256": builder.digest(frozen / "manifest.json"),
                "works": works,
            }
        )
    )
    spec_bytes = spec_path.read_bytes()
    monkeypatch.setattr(builder, "verify_freeze", lambda _: {"files": []})
    monkeypatch.setattr(builder, "normalize_abc", lambda _: normalized(works[0]))

    def checked_then_changed(abc, pdf, contract):
        result = check_normalized(normalized(works[0]), contract)
        abc.write_bytes(b"concurrent changed input")
        spec_path.write_text("concurrent changed spec")
        return {"binding": {"status": "pass"}, **result}

    monkeypatch.setattr(builder, "check_paths", checked_then_changed)
    output = tmp_path / "output"
    builder.build(frozen, output, spec_path)
    assert (output / "mapping_spec.json").read_bytes() == spec_bytes
    for key in keys:
        assert (output / key / "approved.abc").read_bytes() == b"fixed abc"
    with pytest.raises(FileExistsError):
        builder.build(frozen, output, spec_path)
