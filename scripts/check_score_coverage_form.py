"""Offline, create-once checks of declared source coverage and complete playback form."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from score2abc.reference_freeze import verify_freeze
from score2abc.score_eval_abc import normalize_abc

LIMIT = (
    "A pass validates declared mapping bookkeeping and encoded playback order, "
    "not visual PDF correctness, note accuracy, or human acceptance."
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _status(status: str, reason: str) -> dict:
    return {"status": status, "reason": reason}


def normalization_pin(normalized: dict) -> dict:
    """The exact written-bar and playback-route contract, independent of note scoring."""
    return {
        "written_bar_count": len(normalized["measures"]),
        "route_status": normalized["route_status"],
        "route": normalized["route"],
    }


def _identifiers(value: object, field: str) -> list[str]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item.strip() for item in value)
        or len(set(value)) != len(value)
    ):
        raise ValueError(f"{field} must contain distinct nonempty string IDs")
    return value


def _coverage(contract: dict, count: int) -> tuple[dict, dict[int, str]]:
    expected = _identifiers(contract["expected_source_bar_ids"], "expected_source_bar_ids")
    assignments = contract["assignments"]
    if not isinstance(assignments, list):
        raise ValueError("assignments must be a list")
    assigned, primaries, copies, editorials = Counter(), Counter(), [], []
    identities, unknown, unclassified, invalid_bars = {}, set(), set(), []
    for row in assignments:
        if not isinstance(row, dict):
            raise ValueError("Each assignment must be an object")
        number, kind = row.get("written_bar"), row.get("kind")
        if type(number) is not int or not 1 <= number <= count:
            invalid_bars.append(number)
            continue
        assigned[number] += 1
        source = row.get("source_bar_id")
        if kind in {"primary", "return_copy"} and isinstance(source, str) and source.strip():
            if source not in expected:
                unknown.add(source)
            if kind == "primary":
                primaries[source] += 1
            elif isinstance(row.get("reason"), str) and row["reason"].strip():
                copies.append(number)
            else:
                unclassified.add(number)
            if "editorial_id" in row:
                unclassified.add(number)
            identities[number] = f"source:{source}"
        elif (
            kind == "editorial_insertion"
            and isinstance(row.get("editorial_id"), str)
            and row["editorial_id"].strip()
            and isinstance(row.get("reason"), str)
            and row["reason"].strip()
            and "source_bar_id" not in row
        ):
            editorials.append(row["editorial_id"])
            identities[number] = f"editorial:{row['editorial_id']}"
        else:
            unclassified.add(number)
    unclassified.update(set(range(1, count + 1)) - set(assigned))
    diagnostics = {
        "missing_source_ids": sorted(set(expected) - set(primaries)),
        "duplicate_primary_source_ids": sorted(key for key, n in primaries.items() if n > 1),
        "unknown_source_ids": sorted(unknown),
        "unclassified_written_bars": sorted(unclassified),
        "duplicate_written_bar_assignments": sorted(key for key, n in assigned.items() if n > 1),
        "out_of_range_assignments": invalid_bars,
        "duplicate_editorial_ids": sorted(key for key, n in Counter(editorials).items() if n > 1),
    }
    failed = any(diagnostics.values())
    return {
        **_status("fail" if failed else "pass", "declared_mapping_bookkeeping"),
        "expected_source_bars": len(expected),
        "primary_assignments": sum(primaries.values()),
        "return_copy_written_bars": copies,
        "editorial_insertions": editorials,
        **diagnostics,
    }, identities


def check_normalized(normalized: dict, contract: dict | None) -> dict:
    """Check explicit identities only; never infer a source map from musical similarity."""
    result = {
        "normalization": normalization_pin(normalized),
        "parser_valid": normalized["valid"],
        "parser_errors": normalized.get("errors", []),
        "parser_warnings": normalized.get("warnings", []),
        "normalization_regression": _status("not_checked", "no_contract"),
        "coverage": _status("not_checked", "no_source_mapping"),
        "form": _status("not_checked", "no_expected_identity_route"),
    }
    if contract is None:
        return result
    if contract.get("schema_version") != 1:
        raise ValueError("Unsupported coverage/form contract")
    pin = contract.get("normalization")
    if not isinstance(pin, dict) or set(pin) != {"written_bar_count", "route_status", "route"}:
        raise ValueError("Contract requires an exact normalization pin")
    matched = normalized["valid"] and result["normalization"] == pin
    result["normalization_regression"] = {
        **_status("pass" if matched else "fail", "exact_written_count_and_route_pin"),
        "expected": pin,
    }
    if not matched:
        result["coverage"] = result["form"] = _status("not_checked", "normalization_pin_failed")
        return result
    has_mapping = "assignments" in contract or "expected_source_bar_ids" in contract
    identities = {}
    if has_mapping:
        if not {"assignments", "expected_source_bar_ids"} <= contract.keys():
            raise ValueError("Coverage requires both assignments and expected_source_bar_ids")
        result["coverage"], identities = _coverage(contract, len(normalized["measures"]))
    expected = contract.get("expected_performed_route")
    if expected is None:
        return result
    if result["coverage"]["status"] != "pass":
        reason = "coverage_mapping_failed" if has_mapping else "no_source_mapping"
        result["form"] = _status("not_checked", reason)
        return result
    allowed = set(identities.values())
    if (
        not isinstance(expected, list)
        or len(expected) < 2
        or expected[-1] != "STOP"
        or any(not isinstance(item, str) or item not in allowed for item in expected[:-1])
    ):
        raise ValueError("Expected performed route requires mapped identities and one final STOP")
    route = normalized["route"]
    if (
        normalized["route_status"] != "resolved"
        or not isinstance(route, list)
        or not route
        or any(type(bar) is not int or bar not in identities for bar in route)
    ):
        result["form"] = _status("not_checked", "unsupported_or_unresolved_parser_route")
        return result
    actual = [identities[bar] for bar in route] + ["STOP"]
    mismatch = next(
        (
            i
            for i in range(max(len(actual), len(expected)))
            if actual[i : i + 1] != expected[i : i + 1]
        ),
        None,
    )
    result["form"] = {
        **_status("pass" if actual == expected else "fail", "exact_identity_route_and_final_stop"),
        "expected_performed_route": expected,
        "actual_performed_route": actual,
        "expected_performed_bars": len(expected) - 1,
        "actual_performed_bars": len(actual) - 1,
        "first_mismatch_visit": None if mismatch is None else mismatch + 1,
    }
    return result


def check_paths(abc: Path, pdf: Path, contract: dict | None = None) -> dict:
    abc_bytes, pdf_bytes = abc.read_bytes(), pdf.read_bytes()
    hashes = {
        "abc_sha256": hashlib.sha256(abc_bytes).hexdigest(),
        "source_pdf_sha256": hashlib.sha256(pdf_bytes).hexdigest(),
    }
    result = {
        "abc_path": str(abc),
        "source_pdf_path": str(pdf),
        **hashes,
        "contract": contract,
        "binding": _status("not_checked", "no_contract"),
        "limitation": LIMIT,
    }
    if contract is not None:
        mismatches = [key for key, value in hashes.items() if contract.get(key) != value]
        result["binding"] = {
            **_status("fail" if mismatches else "pass", "exact_input_hashes"),
            "mismatched_fields": mismatches,
        }
        if mismatches:
            for field in ("normalization_regression", "coverage", "form"):
                result[field] = _status("not_checked", "input_hash_mismatch")
            return result
    return {**result, **check_normalized(normalize_abc(abc_bytes.decode("utf-8")), contract)}


def check_inventory(path: Path) -> dict:
    """Read an offline inventory; paths are relative to the inventory directory."""
    inventory = json.loads(path.read_text())
    if inventory.get("schema_version") != 1 or not inventory.get("cases"):
        raise ValueError("Unsupported or empty inventory")
    base = path.parent
    pins = inventory.get("evidence_pins", [])
    for pin in pins:
        if digest(base / pin["path"]) != pin["sha256"]:
            raise ValueError(f"Evidence hash mismatch: {pin['path']}")
    frozen = inventory.get("frozen_reference")
    verification = None
    if frozen:
        root = base / frozen["path"]
        if digest(root / "manifest.json") != frozen["manifest_sha256"]:
            raise ValueError("Frozen reference manifest hash mismatch")
        manifest = verify_freeze(root)
        verification = {"status": "pass", "files": len(manifest["files"])}
    cases = []
    for case in inventory["cases"]:
        checked = check_paths(
            base / case["abc_path"], base / case["pdf_path"], case.get("contract")
        )
        cases.append({"id": case["id"], "note": case.get("note", ""), **checked})
    if len({case["id"] for case in cases}) != len(cases):
        raise ValueError("Duplicate inventory case IDs")
    return {
        "schema_version": 1,
        "inventory_sha256": digest(path),
        "inventory": inventory,
        "frozen_reference_verification": verification,
        "cases": cases,
        "limitation": LIMIT,
    }


def write_report(report: dict, output: Path) -> None:
    """Reserve a new output directory; never replace prior evidence."""
    output.mkdir(parents=True, exist_ok=False)
    (output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    lines = [
        "# Offline coverage and form checks",
        "",
        LIMIT,
        "",
        "`not_checked` is an evidence gap, never a pass. A normalized route pin is an edition",
        "regression check; it does not establish source coverage or an adopted performance route.",
        "",
        "| Case | Written / performed | Hashes | Route pin | Coverage | Adopted form |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for case in report["cases"]:
        norm = case.get("normalization", {})
        route = norm.get("route")
        count = f"{norm.get('written_bar_count', '?')} / {len(route) if route else '?'}"
        fields = ("binding", "normalization_regression", "coverage", "form")
        statuses = " | ".join(case[field]["status"] for field in fields)
        lines.append(f"| {case['id']} | {count} | {statuses} |")
    lines += ["", "## Interpretation", ""]
    for case in report["cases"]:
        if case.get("note"):
            lines.append(f"- **{case['id']}**: {case['note']}")
        form = case["form"]
        if form["status"] == "fail":
            lines.append(
                f"- **{case['id']}**: first route mismatch at visit "
                f"{form['first_mismatch_visit']}; "
                f"{form['actual_performed_bars']} performed bars versus "
                f"{form['expected_performed_bars']} expected."
            )
    (output / "report.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check", help="check saved ABC with an optional explicit contract")
    check.add_argument("abc", type=Path)
    check.add_argument("pdf", type=Path)
    check.add_argument("--contract", type=Path)
    check.add_argument("--output-dir", required=True, type=Path)
    inventory = sub.add_parser("inventory", help="check a pinned local regression inventory")
    inventory.add_argument("inventory", type=Path)
    inventory.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    try:
        if args.command == "inventory":
            report = check_inventory(args.inventory)
        else:
            contract = json.loads(args.contract.read_text()) if args.contract else None
            report = {
                "schema_version": 1,
                "cases": [{"id": args.abc.stem, **check_paths(args.abc, args.pdf, contract)}],
                "limitation": LIMIT,
            }
        write_report(report, args.output_dir)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(2, f"Coverage/form check refused: {exc}\n")
    failed = any(
        case[key]["status"] == "fail"
        for case in report["cases"]
        for key in ("binding", "normalization_regression", "coverage", "form")
    )
    print(json.dumps({"report": str(args.output_dir / "report.json"), "has_failures": failed}))
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
