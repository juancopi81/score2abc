"""Declared source identity is independent of bar count and encoded playback form."""

import copy
import json
import shutil

import pytest

from scripts import check_score_coverage_form as checks


def normalized(count=3, route=None, route_status="resolved", valid=True):
    return {
        "valid": valid,
        "measures": [{"number": number} for number in range(1, count + 1)],
        "route": list(range(1, count + 1)) if route is None else route,
        "route_status": route_status,
    }


def contract(score):
    count = len(score["measures"])
    return {
        "schema_version": 1,
        "normalization": checks.normalization_pin(score),
        "expected_source_bar_ids": [str(i) for i in range(1, count + 1)],
        "assignments": [
            {"written_bar": i, "kind": "primary", "source_bar_id": str(i)}
            for i in range(1, count + 1)
        ],
        "expected_performed_route": [f"source:{i}" for i in range(1, count + 1)] + ["STOP"],
    }


def test_missing_bar_is_reported_despite_unchanged_written_count():
    score = normalized()
    mapping = contract(score)
    mapping["assignments"].pop()
    result = checks.check_normalized(score, mapping)
    assert result["coverage"]["status"] == "fail"
    assert result["coverage"]["missing_source_ids"] == ["3"]
    assert result["coverage"]["unclassified_written_bars"] == [3]
    assert result["form"]["status"] == "not_checked"


def test_duplicate_primary_cannot_hide_missing_source_with_same_count():
    score = normalized()
    mapping = contract(score)
    mapping["assignments"][2]["source_bar_id"] = "2"
    result = checks.check_normalized(score, mapping)["coverage"]
    assert result["status"] == "fail"
    assert result["missing_source_ids"] == ["3"]
    assert result["duplicate_primary_source_ids"] == ["2"]


def test_return_copy_and_editorial_insertion_do_not_inflate_source_coverage():
    score = normalized(4, [1, 2, 3, 4])
    mapping = contract(score)
    mapping["expected_source_bar_ids"] = ["1", "2"]
    mapping["assignments"][2] = {
        "written_bar": 3,
        "kind": "return_copy",
        "source_bar_id": "1",
        "reason": "Explicit written D.C. realization",
    }
    mapping["assignments"][3] = {
        "written_bar": 4,
        "kind": "editorial_insertion",
        "editorial_id": "cadence",
        "reason": "Declared editorial cadence",
    }
    mapping["expected_performed_route"] = [
        "source:1",
        "source:2",
        "source:1",
        "editorial:cadence",
        "STOP",
    ]
    result = checks.check_normalized(score, mapping)
    assert result["coverage"]["status"] == result["form"]["status"] == "pass"
    assert result["coverage"]["expected_source_bars"] == 2
    assert result["coverage"]["primary_assignments"] == 2
    assert result["coverage"]["return_copy_written_bars"] == [3]


@pytest.mark.parametrize("route", [[1, 3, 2], [1, 2, 3, 1]])
def test_wrong_order_or_extra_final_visit_fails_even_with_complete_coverage(route):
    score = normalized(route=route)
    mapping = contract(score)
    result = checks.check_normalized(score, mapping)
    assert result["coverage"]["status"] == "pass"
    assert result["form"]["status"] == "fail"
    assert result["form"]["actual_performed_route"][-1] == "STOP"
    assert result["form"]["first_mismatch_visit"] == (2 if len(route) == 3 else 4)


def test_unknown_unclassified_duplicate_and_out_of_range_assignments_are_reported():
    score = normalized()
    mapping = contract(score)
    mapping["assignments"] = [
        {"written_bar": 1, "kind": "primary", "source_bar_id": "unknown"},
        {"written_bar": 1, "kind": "primary", "source_bar_id": "1"},
        {"written_bar": 2, "kind": "guess", "source_bar_id": "2"},
        {"written_bar": 4, "kind": "primary", "source_bar_id": "3"},
    ]
    result = checks.check_normalized(score, mapping)["coverage"]
    assert result["status"] == "fail"
    assert result["unknown_source_ids"] == ["unknown"]
    assert result["unclassified_written_bars"] == [2, 3]
    assert result["duplicate_written_bar_assignments"] == [1]
    assert result["out_of_range_assignments"] == [4]


def test_copy_without_primary_or_explanation_fails():
    score = normalized()
    mapping = contract(score)
    mapping["assignments"][0]["kind"] = "return_copy"
    result = checks.check_normalized(score, mapping)["coverage"]
    assert result["status"] == "fail"
    assert result["missing_source_ids"] == ["1"]
    assert result["unclassified_written_bars"] == [1]


@pytest.mark.parametrize("has_contract", [False, True])
def test_no_mapping_or_route_is_never_a_coverage_or_form_pass(has_contract):
    score = normalized()
    mapping = {"schema_version": 1, "normalization": checks.normalization_pin(score)}
    result = checks.check_normalized(score, mapping if has_contract else None)
    assert result["coverage"]["status"] == result["form"]["status"] == "not_checked"
    assert result["normalization_regression"]["status"] == (
        "pass" if has_contract else "not_checked"
    )


def test_mapping_without_expected_route_checks_only_coverage():
    score = normalized()
    mapping = contract(score)
    del mapping["expected_performed_route"]
    result = checks.check_normalized(score, mapping)
    assert result["coverage"]["status"] == "pass"
    assert result["form"]["status"] == "not_checked"


def test_unsupported_route_cannot_pass_form():
    score = normalized(route_status="unsupported")
    score["route"] = None
    result = checks.check_normalized(score, contract(score))
    assert result["coverage"]["status"] == "pass"
    assert result["form"] == {
        "status": "not_checked",
        "reason": "unsupported_or_unresolved_parser_route",
    }


@pytest.mark.parametrize("change", ["route", "valid", "count"])
def test_parser_drift_fails_closed(change):
    score = normalized()
    mapping = copy.deepcopy(contract(score))
    if change == "route":
        score["route"] = [1, 3, 2]
    elif change == "valid":
        score["valid"] = False
    else:
        score["measures"].pop()
    result = checks.check_normalized(score, mapping)
    assert result["normalization_regression"]["status"] == "fail"
    assert result["coverage"]["status"] == result["form"]["status"] == "not_checked"


@pytest.mark.parametrize("field", ["abc_sha256", "source_pdf_sha256"])
def test_stale_hash_prevents_parser_or_mapping_checks(tmp_path, monkeypatch, field):
    abc, pdf = tmp_path / "score.abc", tmp_path / "source.pdf"
    abc.write_text("X:1\nK:C\nC |]")
    pdf.write_bytes(b"source placeholder")
    mapping = contract(normalized())
    mapping.update(abc_sha256=checks.digest(abc), source_pdf_sha256=checks.digest(pdf))
    mapping[field] = "0" * 64

    def forbidden(_):
        pytest.fail("Hash mismatch must prevent parser invocation")

    monkeypatch.setattr(checks, "normalize_abc", forbidden)
    result = checks.check_paths(abc, pdf, mapping)
    assert result["binding"]["status"] == "fail"
    assert result["binding"]["mismatched_fields"] == [field]
    assert result["coverage"]["status"] == result["form"]["status"] == "not_checked"


def test_expected_route_requires_final_stop():
    score = normalized()
    mapping = contract(score)
    mapping["expected_performed_route"].pop()
    with pytest.raises(ValueError, match="final STOP"):
        checks.check_normalized(score, mapping)


def test_report_is_create_once(tmp_path):
    report = {"cases": []}
    output = tmp_path / "evidence"
    checks.write_report(report, output)
    before = (output / "report.json").read_bytes()
    with pytest.raises(FileExistsError):
        checks.write_report({"cases": [], "replacement": True}, output)
    assert (output / "report.json").read_bytes() == before


def test_inventory_refuses_stale_evidence_before_cases(tmp_path):
    evidence = tmp_path / "evidence.json"
    evidence.write_text("changed")
    inventory = tmp_path / "inventory.json"
    inventory.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "cases": [{"id": "unused"}],
                "evidence_pins": [{"path": "evidence.json", "sha256": "0" * 64}],
            }
        )
    )
    with pytest.raises(ValueError, match="Evidence hash mismatch"):
        checks.check_inventory(inventory)


@pytest.mark.skipif(
    not shutil.which("node") or not shutil.which("abc2svg"), reason="Local renderer unavailable"
)
def test_real_parser_treats_header_part_order_as_complete_route():
    score = checks.normalize_abc(
        "X:1\nT:Explicit route\nP:ABA\nM:3/4\nL:1/8\nK:C\n" "[P:A] |: C6 :|\n[P:B] D6 |]\n"
    )
    assert score["valid"]
    assert checks.normalization_pin(score) == {
        "written_bar_count": 2,
        "route_status": "resolved",
        "route": [1, 2, 1],
    }
    mapping = contract(score)
    mapping["expected_performed_route"] = ["source:1", "source:2", "source:1", "STOP"]
    assert checks.check_normalized(score, mapping)["form"]["status"] == "pass"
