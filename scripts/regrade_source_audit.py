"""Regrade pinned saved candidates offline under old and expanded source labels."""

from __future__ import annotations

import argparse
import copy
import html
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory

from score2abc.eval_labels import (
    INVENTORY_PATH,
    V4_INVENTORY_PATH,
    WORK_LIMITS,
    _source_audit,
    create_labels,
    verify_labels,
)
from score2abc.reference_freeze import verify_freeze
from score2abc.score_eval import compare_normalized
from score2abc.score_eval_abc import normalize_abc
from score2abc.score_eval_source import score_source
from score2abc.source_note_audit import EVIDENCE_PATH, bundle_evidence, expand_inventory, sha

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "out/eval/references/reviewed_six_20261028_v1"
REPRO = ROOT / "out/eval/reproduction"
LABELS = ROOT / "out/eval/labels"
ARCHIVES = (
    (
        "Astra subscription",
        "assisted-subscription-v1",
        REPRO / "astra_workflow_20261028_v1/rumichaca_full/evaluation/report.json",
    ),
    (
        "Terra subscription",
        "assisted-subscription-v1 (retry)",
        REPRO
        / "assisted_astra_terra_20261028_v1/retries/rumichaca_20261029_v1/evaluation/report.json",
    ),
    (
        "Astra API + review v2",
        "api-review-v2",
        REPRO / "astra_api_rumichaca_20261029_v1/evaluation/report.json",
    ),
    (
        "Astra API + completion gate",
        "api-completion-gate-v1",
        REPRO / "astra_api_rumichaca_gate_20261029_v1/evaluation/report.json",
    ),
)
METRICS = (
    "edition_note_f1",
    "edition_exact_bar_rate",
    "chord_f1",
    "form_exact_match",
    "meter_valid_rate",
    "bar_coverage",
)


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def pin(path: Path) -> dict:
    return {"path": str(path.relative_to(ROOT)), "sha256": sha(path.read_bytes())}


def verified(pin_record: dict) -> bytes:
    path = Path(pin_record["path"])
    path = path if path.is_absolute() else ROOT / path
    if not path.resolve().is_relative_to(ROOT) or path.is_symlink():
        raise ValueError("Saved artifact is outside the repository or a symlink")
    data = path.read_bytes()
    if sha(data) != pin_record["sha256"]:
        raise ValueError(f"Saved artifact changed: {path}")
    return data


def inventory_saved() -> list[dict]:
    """All terminal outputs in four assisted Rumichaca pairs and the original suite.

    Check the archive's original ABC pins, including failures with no output.
    Intermediate tool drafts and seven-bar-only tasks are outside this full-score
    mask expansion; their original reports are preserved.
    """
    candidates = []
    for label, cohort, report_path in ARCHIVES:
        report = read(report_path)
        for stage, previous in report["stages"].items():
            if report.get("stage_artifacts"):
                abc = report["stage_artifacts"][stage]["abc"]
            else:
                matches = [
                    p
                    for p in report["pins"]
                    if p["path"].endswith(f"/rumichaca_full/{stage}/score.abc")
                ]
                if len(matches) != 1:
                    raise ValueError(f"No unique terminal ABC pin: {report_path}/{stage}")
                abc = matches[0]
            verified(abc)
            candidates.append(
                {
                    "id": f"{report_path.parent.parent.name}/{stage}",
                    "work": "rumichaca",
                    "model": report["model"],
                    "label": label,
                    "cohort": cohort,
                    "stage": stage,
                    "abc": abc,
                    "prior_report": pin(report_path),
                    "prior_metrics": previous["metrics"],
                    "status": previous["status"],
                }
            )
    registry_path = ROOT / "out/eval/runs/astra_terra_20261028_v1/registry.json"
    report_path = ROOT / "out/eval/comparison/astra_terra_20261028_v1/report.json"
    registry, report = read(registry_path), read(report_path)
    for run in registry["runs"]:
        key = next(k for k in read(INVENTORY_PATH)["works"] if f"_{k}_" in run["slug"])
        previous = next(c for c in report["cases"] if c["slug"] == run["slug"])["results"][
            run["model_id"]
        ]
        abc = (
            {"path": run["abc_path"], "sha256": run["abc_sha256"]} if run.get("abc_path") else None
        )
        if abc:
            verified(abc)
        candidates.append(
            {
                "id": run["run_id"],
                "work": key,
                "model": run["model"],
                "label": f"{run['model_id'].title()} subscription",
                "cohort": "tool-free-900s-v1",
                "stage": "terminal",
                "abc": abc,
                "registry": pin(registry_path),
                "prior_report": pin(report_path),
                "prior_metrics": previous["metrics"],
                "status": previous["status"],
            }
        )
    return candidates


def grade_candidate(reference: dict, candidate: dict, old: dict, new: dict) -> dict:
    edition = compare_normalized(reference, candidate)
    alignment = edition.get("alignment", [])
    return {
        "edition": edition,
        "v3": score_source(reference, candidate, alignment, old),
        "v4": score_source(reference, candidate, alignment, new),
    }


def assert_reproduced(result: dict, prior_metrics: dict) -> None:
    current = {**result["edition"]["metrics"], **result["v3"]["metrics"]}
    for name in (*METRICS, "source_note_f1"):
        a, b = current[name], prior_metrics[name]
        equal = a == b or (
            isinstance(a, (int, float))
            and isinstance(b, (int, float))
            and math.isclose(a, b, abs_tol=1e-12)
        )
        if not equal:
            raise ValueError(f"Historical metric does not reproduce: {name}: {b} -> {a}")


def preview_labels(old: dict, frozen: dict, inventory: dict) -> dict:
    """Validate proposed masks in memory before writing any canonical artifacts."""
    proposed = copy.deepcopy(old)
    for work in proposed["works"]:
        key = next(k for k in WORK_LIMITS if f"_{k}_" in work["slug"])
        frozen_work = next(w for w in frozen["works"] if w["slug"] == work["slug"])
        work["source_audit"] = _source_audit(
            inventory["works"][key]["source_audit"],
            frozen_root=FROZEN,
            reference=frozen,
            frozen_work=frozen_work,
            limits=WORK_LIMITS[key],
            mapping=work["source_to_edition_mapping"],
            exclusions=work["source_metric_exclusions"],
            decisions={d["id"]: d for d in work["decisions"]},
        )
    return proposed


def publish_labels(evidence_bytes: bytes, inventory: dict) -> None:
    """Called only after all historical grades reproduce; roll back failed creation."""
    label_root = LABELS / "reviewed_six_v4"
    if EVIDENCE_PATH.exists() or V4_INVENTORY_PATH.exists() or label_root.exists():
        raise FileExistsError("Expanded labels already materialized")
    created = []
    try:
        for path, payload in (
            (EVIDENCE_PATH, evidence_bytes),
            (
                V4_INVENTORY_PATH,
                (json.dumps(inventory, ensure_ascii=False, indent=2) + "\n").encode(),
            ),
        ):
            with path.open("xb") as output:
                created.append(path)
                output.write(payload)
        with TemporaryDirectory(prefix=".v4-labels-", dir=LABELS) as tmp:
            staged = Path(tmp) / "labels"
            create_labels(staged, FROZEN, V4_INVENTORY_PATH)
            staged.rename(label_root)
    except BaseException:
        for path in reversed(created):
            path.unlink()
        raise


def render(report: dict) -> str:
    esc = html.escape

    def percent(value):
        if value is None:
            return "—"
        return f"{value:.1%}"

    tables = []
    for cohort in dict.fromkeys(r["cohort"] for r in report["results"]):
        rows = []
        for row in report["results"]:
            if row["cohort"] != cohort:
                continue
            result = row.get("grading")
            if not result:
                cells = [
                    row["label"],
                    row["work"].title(),
                    row["stage"],
                    "No saved score",
                    "—",
                    "—",
                    "—",
                    "—",
                ]
            else:
                old, new = result["v3"], result["v4"]
                form = result["edition"]["metrics"]["form_exact_match"]
                cells = [
                    row["label"],
                    row["work"].title(),
                    row["stage"],
                    row["status"],
                    percent(old["metrics"]["source_note_f1"]),
                    percent(new["metrics"]["source_note_f1"]),
                    percent(result["edition"]["metrics"]["edition_note_f1"]),
                    "Match" if form is True else "Differs" if form is False else "Unresolved",
                ]
            rows.append("<tr>" + "".join(f"<td>{esc(c)}</td>" for c in cells) + "</tr>")
        tables.append(
            f"<h3>{esc(cohort)}</h3><div class='scroll'><table><thead><tr>"
            "<th>System</th><th>Piece</th><th>Stage</th><th>Saved status</th>"
            "<th>Source v3</th><th>Source v4</th><th>Edition notes</th><th>Edition form</th>"
            f"</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
        )
    audit = report["audit"]
    previous = sum(w["previous_bars"] for w in audit["works"].values())
    total = sum(w["total_bars"] for w in audit["works"].values())
    rejected = sum(len(w["rejected"]) for w in audit["works"].values())
    coverage = []
    for key, w in audit["works"].items():
        reasons = (
            "; ".join(f"bar {r['edition_bar']}: {', '.join(r['reasons'])}" for r in w["rejected"])
            or "All selected bars accepted"
        )
        coverage.append(
            f"<tr><td>{key.title()}</td><td>{w['previous_bars']} → {w['total_bars']}</td>"
            f"<td>{esc(reasons)}</td></tr>"
        )
    return f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport"
content="width=device-width,initial-scale=1"><title>Expanded source-note audit</title>
<style>body{{font:16px/1.55 system-ui;margin:0;background:#f6f4ef;color:#192b33}}
main{{max-width:1200px;margin:40px auto;padding:0 24px}}
h1{{font-size:38px;line-height:1.2}}
.cards{{display:flex;gap:16px;flex-wrap:wrap}}
.card{{background:#fff;padding:18px 25px;border-top:4px solid #177c73;flex:1}}
.card strong{{display:block;font-size:30px}}
table{{border-collapse:collapse;width:100%;background:white;margin:12px 0 28px}}
td,th{{padding:10px 12px;text-align:left;border-bottom:1px solid #dce1de}}
th{{font-size:13px;background:#e3ece8}}
.scroll{{overflow-x:auto}}
small{{color:#50616a}}
a{{color:#126a65}}
.note{{padding:16px;background:#e3ece8}}
details{{margin:22px 0}}
code{{overflow-wrap:anywhere}}</style>
<main><small>OFFLINE REGRADING · CONSUMED SIX-SCORE COLLECTION · 2026-11-06</small><h1>A broader
check of the same saved scores</h1>
<p>No new transcription runs. Only the source-note label coverage changed. Lower or higher scores
below reflect a different set of checked bars, not a change in model performance.</p>
<div class="cards"><div class="card"><strong>{previous} → {total}/312</strong>source bars
scored</div><div class="card"><strong>{rejected}/60</strong>selected bars withheld</div><div
class="card"><strong>0</strong>new paid API calls</div></div>
<p class="note">Source F1 requires exact pitch, onset, duration, ties and duplicate noteheads in
audited bars. It excludes chord symbols, separate rest scoring and form. Edition agreement includes
your adopted editorial choices. These are partial model-assisted labels, not new human
certification.</p>
<h2>Saved model results</h2><p>Compare systems within their generation conditions. API review v2,
the completion gate and the earlier tool-free harness are different experiments. All four assisted
experiments concern Rumichaca; the six-piece tool-free cohort is shown separately. Both author and
reviewed outputs remain visible.</p>
{''.join(tables)}<h2>What entered the expanded labels</h2><table><tr><th>Piece</th><th>Audited
bars</th><th>Withheld selections</th></tr>{''.join(coverage)}</table>
<p>The original {previous} bars were retained without re-audit. Sixty additional bars were
selected before looking at candidate metrics. Two separate readers checked each; disagreements were
excluded without replacement. Unscored regions are not automatically errors.</p>
<details><summary>Reproducibility and limits</summary><p>All available full-score terminal outputs
in the four recorded Rumichaca pairs and original six-piece experiment were regraded. Six failed
tool-free Astra attempts still have no score. Aviador’s seven-bar experiments and intermediate tool
drafts retain their separate historical reports.</p><p>Original v3 metrics reproduced exactly.
Candidate bytes, old labels, approved references and historical reports are unchanged. Bar
alignment remains the existing edition-based alignment. No claim about unseen-score accuracy
follows from this
audit.</p><pre>{esc(json.dumps(report['versions'],indent=2))}</pre></details><p><a
href="report.json">Full metrics and per-bar mismatches (JSON)</a> · <a
href="audit_summary.json">Audit decisions</a> · <a href="inputs.json">Pinned saved
inputs</a></p></main></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--materialize-labels", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    bundle = bundle_evidence(args.audit_root)
    inventory, summary = expand_inventory(read(INVENTORY_PATH), bundle)
    evidence_bytes = (json.dumps(bundle, ensure_ascii=False, indent=2) + "\n").encode()
    inventory["source_audit_evidence_sha256"] = sha(evidence_bytes)
    if args.materialize_labels:
        if (
            EVIDENCE_PATH.exists()
            or V4_INVENTORY_PATH.exists()
            or (LABELS / "reviewed_six_v4").exists()
        ):
            raise FileExistsError("Expanded labels already materialized")
    elif read(V4_INVENTORY_PATH) != inventory or EVIDENCE_PATH.read_bytes() != evidence_bytes:
        raise ValueError("Audit evidence differs from published labels")
    candidates = inventory_saved()
    # Verify every archived candidate before opening frozen reference content.
    candidate_bytes = {r["id"]: verified(r["abc"]) for r in candidates if r["abc"]}
    plan = read(args.audit_root / "audit_plan.json")
    if sha((LABELS / "reviewed_six_v3/manifest.json").read_bytes()) != plan["prior_labels_sha256"]:
        raise ValueError("Prior labels differ from the preregistered audit plan")
    frozen = verify_freeze(FROZEN)
    old = verify_labels(LABELS / "reviewed_six_v3", FROZEN)
    new = (
        preview_labels(old, frozen, inventory)
        if args.materialize_labels
        else verify_labels(LABELS / "reviewed_six_v4", FROZEN)
    )
    normalized = {}
    for work in frozen["works"]:
        normalized[work["slug"]] = normalize_abc((FROZEN / work["answer_abc_path"]).read_text())
    results = []
    for record in candidates:
        row = dict(record)
        if row["abc"]:
            work_old = next(w for w in old["works"] if f"_{row['work']}_" in w["slug"])
            work_new = next(w for w in new["works"] if w["slug"] == work_old["slug"])
            result = grade_candidate(
                normalized[work_old["slug"]],
                normalize_abc(candidate_bytes[row["id"]].decode()),
                work_old,
                work_new,
            )
            assert_reproduced(result, row["prior_metrics"])
            row["grading"] = result
        results.append(row)
    if args.materialize_labels:
        publish_labels(evidence_bytes, inventory)
    report = {
        "schema_version": 1,
        "kind": "offline_source_label_expansion",
        "api_calls": 0,
        "audit": summary,
        "results": results,
        "versions": {
            "prior_labels": pin(LABELS / "reviewed_six_v3/manifest.json"),
            "expanded_labels": pin(LABELS / "reviewed_six_v4/manifest.json"),
            "reference": pin(FROZEN / "manifest.json"),
            "implementations": [
                pin(ROOT / p)
                for p in (
                    "score2abc/score_eval.py",
                    "score2abc/score_eval_abc.py",
                    "score2abc/score_eval_parse.js",
                    "score2abc/score_eval_source.py",
                    "score2abc/source_note_audit.py",
                    "scripts/regrade_source_audit.py",
                )
            ],
        },
        "historical_metrics_reproduced": True,
    }
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output / "inputs.json", {"candidates": candidates})
    write(args.output / "audit_summary.json", summary)
    write(args.output / "report.json", report)
    (args.output / "index.html").write_text(render(report))
    write(
        args.output / "manifest.json",
        {p.name: sha(p.read_bytes()) for p in sorted(args.output.iterdir()) if p.is_file()},
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "saved_candidates": len(candidate_bytes),
                "no_output": len(candidates) - len(candidate_bytes),
                "audit": summary,
            }
        )
    )


if __name__ == "__main__":
    main()
