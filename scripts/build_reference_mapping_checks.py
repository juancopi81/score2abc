"""Build offline reviewed-edition source-map regressions and a local HTML summary."""

from __future__ import annotations

import argparse
import copy
import hashlib
import html
import json
from pathlib import Path

from score2abc.reference_freeze import verify_freeze
from score2abc.score_eval_abc import normalize_abc
from scripts.check_score_coverage_form import check_normalized, check_paths, digest, write_report

DEFAULT_SPEC = Path(__file__).resolve().parents[1] / "docs/eval/reviewed_six_source_maps_v1.json"
LIMIT = (
    "Source identities and approved playback routes are mapped; note accuracy is not newly "
    "certified. The existing source-note audit remains 67 of 312 bars. These are consumed "
    "regression references, not heldout model results or uniquely correct manuscript routes."
)


def verify_audit_evidence(work: dict, spec_root: Path) -> dict[str, bytes]:
    """Bind the published map to actual author and separate-reader artifacts."""
    provenance = work["provenance"]
    payloads = {}

    def read_pin(name):
        pin = provenance[name]
        relative = Path(pin["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Mapping evidence path must be relative without parent segments")
        path = (spec_root / relative).resolve()
        if not path.is_relative_to(spec_root.resolve()):
            raise ValueError("Mapping evidence escapes specification directory")
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != pin["sha256"]:
            raise ValueError(f"Mapping evidence hash/path mismatch: {work['key']} {name}")
        payloads[pin["path"]] = payload
        return json.loads(payload)

    if work["audit_status"] == "reused_prior_mapping":
        prior, current = read_pin("prior_contract"), compile_contract(work)
        required = {
            "schema_version",
            "abc_sha256",
            "source_pdf_sha256",
            "normalization",
            "expected_source_bar_ids",
            "assignments",
        }
        if not required <= prior.keys() or any(current.get(k) != v for k, v in prior.items()):
            raise ValueError("Reused mapping differs from pinned prior contract")
        return payloads
    author, review = read_pin("author"), read_pin("review")
    if provenance["author_agent"] == provenance["reviewer_agent"]:
        raise ValueError("Mapping requires a separate reader")
    fields = (
        "source_rows",
        "source_bars",
        "assignments",
        "route_segments",
        "decisions",
        "source_vs_edition_route",
        "unresolved_mapping_questions",
        "abc_sha256",
        "source_pdf_sha256",
    )
    if any(author[field] != work[field] for field in fields):
        raise ValueError("Published map differs from reviewed author evidence")
    if (
        review["reviewed_author_sha256"] != provenance["author"]["sha256"]
        or review["abc_sha256"] != work["abc_sha256"]
        or review["source_pdf_sha256"] != work["source_pdf_sha256"]
        or review.get("work") != work["key"]
        or review.get("mapping_review_pass") is not True
        or review.get("form_review_pass") is not True
        or review.get("visual_audit", {}).get("all_rows_inspected") is not True
        or review.get("unresolved_mapping_questions") != []
    ):
        raise ValueError("Missing, stale or unsuccessful independent mapping review")
    return payloads


def compile_contract(work: dict) -> dict:
    """Compile independently declared spans, never derive expectations from parser output."""
    if work["unresolved_mapping_questions"]:
        raise ValueError(f"Unresolved mapping questions: {work['key']}")
    if work["audit_status"] not in {"independently_verified", "reused_prior_mapping"}:
        raise ValueError(f"Unverified mapping: {work['key']}")
    source = work["source_bars"]
    ids = [bar["id"] for bar in source]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Source identities must be unique and nonempty")
    if work["audit_status"] == "independently_verified":
        locations = [(bar["page"], bar["system"], bar["measure"]) for bar in source]
        if len(set(locations)) != len(locations):
            raise ValueError("Duplicate source location")
        if any(type(n) is not int or n < 1 for loc in locations for n in loc):
            raise ValueError("Invalid source location")
        row_ids = [sid for row in work["source_rows"] for sid in row["source_ids"]]
        if row_ids != ids:
            raise ValueError("Visual row inventory disagrees with source identities")
        for row in work["source_rows"]:
            located = [
                bar["id"]
                for bar in source
                if (bar["page"], bar["system"]) == (row["page"], row["system"])
            ]
            if located != row["source_ids"]:
                raise ValueError("Source row location disagrees with bar locations")
        primary = {
            item["source_bar_id"]: item["written_bar"]
            for item in work["assignments"]
            if item["kind"] == "primary"
        }
        if any(primary.get(bar["id"]) != bar["written_bar"] for bar in source):
            raise ValueError("Visual source table disagrees with primary assignments")
    identities = {}
    for item in work["assignments"]:
        number = item["written_bar"]
        if number in identities:
            raise ValueError("Duplicate written assignment")
        identities[number] = (
            f"editorial:{item['editorial_id']}"
            if item["kind"] == "editorial_insertion"
            else f"source:{item['source_bar_id']}"
        )
    visits = []
    for segment in work["route_segments"]:
        if not segment["label"] or not segment["evidence"] or not segment["written_bars"]:
            raise ValueError("Route segment needs a label, evidence and visits")
        if any(type(bar) is not int or bar not in identities for bar in segment["written_bars"]):
            raise ValueError("Route segment references an unclassified written bar")
        visits.extend(identities[bar] for bar in segment["written_bars"])
    if not visits:
        raise ValueError("Adopted route is empty")
    return {
        "schema_version": 1,
        "abc_sha256": work["abc_sha256"],
        "source_pdf_sha256": work["source_pdf_sha256"],
        "normalization": work["normalization"],
        "expected_source_bar_ids": ids,
        "assignments": work["assignments"],
        "expected_performed_route": visits + ["STOP"],
    }


def mutation_controls(works: list[dict], scores: dict[str, dict]) -> list[dict]:
    """Prove known classes of mistakes are detected using the actual edition identities."""
    results = []
    for work in works:
        key = work["key"]
        contract = compile_contract(work)
        score = copy.deepcopy(scores[key])
        # Preserve the candidate's own parser pin so the adopted route, not pin drift,
        # detects a wrong visit. Reordering preserves total length.
        route = score["route"]
        pair = next((i for i in range(len(route) - 1) if route[i] != route[i + 1]), None)
        if pair is None:
            raise ValueError("Cannot construct order-sensitive regression control")
        route[pair], route[pair + 1] = route[pair + 1], route[pair]
        contract["normalization"] = {**contract["normalization"], "route": route}
        checked = check_normalized(score, contract)
        results.append(
            {"id": f"{key}_same_count_wrong_order", "expected_failure": "form", **checked}
        )
        contract = compile_contract(work)
        # Duplicate one primary while keeping the number of written bars fixed.
        primaries = [x for x in contract["assignments"] if x["kind"] == "primary"]
        contract = copy.deepcopy(contract)
        target = next(x for x in contract["assignments"] if x == primaries[1])
        target["source_bar_id"] = primaries[0]["source_bar_id"]
        checked = check_normalized(scores[key], contract)
        results.append(
            {"id": f"{key}_duplicate_hides_omission", "expected_failure": "coverage", **checked}
        )
    if any(result[result["expected_failure"]]["status"] != "fail" for result in results):
        raise ValueError("A negative control was not detected")
    return results


def _escape(value: object) -> str:
    return html.escape(str(value), quote=True)


def dashboard(works: list[dict], cases: list[dict], control_count: int) -> str:
    source_count = sum(len(work["source_bars"]) for work in works)
    new_count = sum(work["audit_status"] == "independently_verified" for work in works)
    cards = []
    for work, case in zip(works, cases, strict=True):
        key = work["key"]
        coverage, form = case["coverage"]["status"], case["form"]["status"]
        rows = "".join(
            f"<tr><td>{_escape(bar['id'])}</td><td>{bar.get('written_bar', '—')}</td>"
            f"<td>{_escape(bar.get('evidence', 'Prior explicit source identity'))}</td></tr>"
            for bar in work["source_bars"]
        )
        decisions = "".join(
            f"<li><strong>{_escape(item['category'])}</strong>: " f"{_escape(item['summary'])}</li>"
            for item in work["decisions"]
        )
        route = " → ".join(_escape(s["label"]) for s in work["route_segments"])
        count = len(work["normalization"]["route"])
        audit = (
            "New map · separate reader verification"
            if work["audit_status"] == "independently_verified"
            else "Existing map reused · no new visual audit"
        )
        cards.append(
            f'<article id="{key}"><h2>{_escape(work["title"])}</h2>'
            f'<p class="tag">{audit}</p><p><b>{len(work["source_bars"])} source bars</b>'
            f" · {work['normalization']['written_bar_count']} written · {count} playback visits</p>"
            f"<p>Coverage: <b>{coverage}</b> · Approved route: <b>{form}</b></p>"
            f'<p><a href="{key}/source.pdf">Manuscript PDF</a> · '
            f'<a href="{key}/approved.abc">Approved ABC</a> · '
            f'<a href="{key}/mapping.json">Full mapping</a></p>'
            f"<p class='route'>{route} → STOP</p>"
            f"<p>{_escape(work['source_vs_edition_route'])}</p>"
            f"<details><summary>Editorial choices and source ambiguities</summary>"
            f"<ul>{decisions}</ul></details>"
            f"<details><summary>Every mapped source bar</summary><table>"
            f"<thead><tr><th>Source identity</th><th>Written bar</th><th>Evidence</th>"
            f"</tr></thead><tbody>{rows}</tbody></table></details></article>"
        )
    return (
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>Six approved scores · source maps</title><style>"
        "body{font:17px/1.6 system-ui,sans-serif;background:#f5f2ea;color:#202c31;"
        "margin:0}main{max-width:1100px;margin:auto;padding:36px 22px}"
        "h1{font-size:clamp(32px,5vw,52px);line-height:1.12;max-width:850px}"
        "h2{margin:0}.intro{max-width:850px}.grid{display:grid;"
        "grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:14px}"
        ".metric,article{background:white;border:1px solid #d7dfdc;border-radius:14px;"
        "padding:24px}.metric b{display:block;font-size:32px}article{margin-top:22px}"
        ".tag{color:#39715a;font-size:14px}.route{background:#eef4ef;padding:14px;"
        "border-radius:8px}a{color:#075e68}summary{cursor:pointer;font-weight:600;"
        "padding:12px 0}table{width:100%;border-collapse:collapse;font-size:14px}"
        "td,th{text-align:left;padding:9px;border-bottom:1px solid #ddd;vertical-align:top}"
        "details{overflow:auto}.note{border-left:4px solid #c18c34;padding:12px 20px}"
        "</style><main><p>REFERENCE MAPPING · v1 · 5 November 2026</p>"
        "<h1>Can we account for every bar—and every return?</h1>"
        '<p class="intro">Five new manuscript maps join Aviador’s existing map. '
        "All six approved editions remain unchanged. These checks make future omissions, "
        "duplicate passages and wrong return orders easier to detect.</p>"
        f'<div class="grid"><div class="metric"><b>{len(works)} / {len(works)}</b>'
        "approved editions checked</div>"
        f'<div class="metric"><b>{source_count}</b>source identities mapped</div>'
        f'<div class="metric"><b>{new_count} new + {len(works) - new_count} reused</b>'
        "mapping provenance</div>"
        f'<div class="metric"><b>{control_count} / {control_count}</b>'
        "deliberate errors detected</div></div>"
        f'<p class="note">{LIMIT} No new API/model comparison was run. '
        "No action is required to reapprove already accepted editorial choices.</p>"
        '<p><a href="report.json">Machine-readable results</a> · '
        '<a href="mutation_controls.json">Deliberate-error checks</a> · '
        '<a href="manifest.json">Version and file hashes</a></p>'
        + "".join(cards)
        + "</main></html>"
    )


def build(frozen_root: Path, output: Path, spec_path: Path = DEFAULT_SPEC) -> dict:
    if output.resolve().is_relative_to(frozen_root.resolve()):
        raise ValueError("Output must be outside the frozen reference tree")
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    spec_bytes = spec_path.read_bytes()
    spec = json.loads(spec_bytes)
    if spec["schema_version"] != 1:
        raise ValueError("Unsupported mapping specification")
    if digest(frozen_root / "manifest.json") != spec["reference_manifest_sha256"]:
        raise ValueError("Frozen reference identity changed")
    frozen = verify_freeze(frozen_root)
    works, cases, scores = spec["works"], [], {}
    evidence_payloads, source_payloads = {}, {}
    expected = {"amalia", "atardecer", "aviador", "carrizal", "coqueteos", "rumichaca"}
    if len(works) != 6 or {work["key"] for work in works} != expected:
        raise ValueError("Expected the six distinct reviewed works")
    for work in works:
        evidence_payloads.update(verify_audit_evidence(work, spec_path.parent))
        abc = frozen_root / "answers" / work["slug"] / "review.abc"
        pdf = frozen_root / "inputs" / work["slug"] / "source.pdf"
        abc_bytes, pdf_bytes = abc.read_bytes(), pdf.read_bytes()
        if (
            hashlib.sha256(abc_bytes).hexdigest() != work["abc_sha256"]
            or hashlib.sha256(pdf_bytes).hexdigest() != work["source_pdf_sha256"]
        ):
            raise ValueError("Source bytes differ from reviewed map")
        source_payloads[work["key"]] = (abc_bytes, pdf_bytes)
        checked = check_paths(abc, pdf, compile_contract(work))
        if any(
            checked[field]["status"] != "pass"
            for field in ("binding", "normalization_regression", "coverage", "form")
        ):
            raise ValueError(f"Reference checks did not pass: {work['key']}")
        cases.append({"id": work["key"], **checked})
        scores[work["key"]] = normalize_abc(abc_bytes.decode("utf-8"))
    controls = mutation_controls(works, scores)
    report = {"schema_version": 1, "cases": cases, "limitation": LIMIT}
    write_report(report, output)
    (output / "mapping_spec.json").write_bytes(spec_bytes)
    for work in works:
        root = output / work["key"]
        root.mkdir()
        abc_bytes, pdf_bytes = source_payloads[work["key"]]
        (root / "approved.abc").write_bytes(abc_bytes)
        (root / "source.pdf").write_bytes(pdf_bytes)
        (root / "mapping.json").write_text(json.dumps(work, indent=2, ensure_ascii=False) + "\n")
    for relative, payload in evidence_payloads.items():
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
    (output / "mutation_controls.json").write_text(json.dumps(controls, indent=2) + "\n")
    (output / "index.html").write_text(dashboard(works, cases, len(controls)))
    files = {str(p.relative_to(output)): digest(p) for p in output.rglob("*") if p.is_file()}
    manifest = {
        "schema_version": 1,
        "spec_sha256": hashlib.sha256(spec_bytes).hexdigest(),
        "builder_sha256": digest(Path(__file__)),
        "checker_sha256": digest(Path(__file__).with_name("check_score_coverage_form.py")),
        "reference_manifest_sha256": spec["reference_manifest_sha256"],
        "verified_reference_files": len(frozen["files"]),
        "files": files,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "manifest.sha256").write_text(digest(output / "manifest.json") + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frozen_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    args = parser.parse_args()
    manifest = build(args.frozen_root, args.output, args.spec)
    print(json.dumps({"output": str(args.output), "files": len(manifest["files"])}))


if __name__ == "__main__":
    main()
