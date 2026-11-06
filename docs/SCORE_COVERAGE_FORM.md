# Offline source coverage and form checks

The later [reviewed-six source-map packet](REVIEWED_SOURCE_MAPS.md) adds separately
verified manuscript identities for the five editions without maps in the initial
sprint inventory below. The initial inventory and reports remain unchanged.

`scripts/check_score_coverage_form.py` checks a saved ABC against an explicit,
hash-bound declaration. It writes a new sidecar directory and never changes the
ABC, source PDF, review state, subscription workflow, API completion gate, parser,
or frozen evaluations. It makes no network calls and needs no new dependencies.
ABC normalization uses the existing local Node/abc2svg renderer.

A **coverage pass** means every declared source-bar identity has exactly one
primary written-bar assignment, and every parsed written bar is classified.
Explicit return copies and editorial insertions are allowed. A **form pass**
means the entire performed identity sequence, including its final stop, equals
the supplied adopted route. Equal written or performed counts alone cannot pass
either check. These are bookkeeping and encoding checks: they do not certify
that the declared map matches the PDF, that notes are correct, or that a human
has accepted the score.

## Running a check

```sh
uv run python scripts/check_score_coverage_form.py check path/to/score.abc path/to/source.pdf \
  --contract path/to/coverage_form.json --output-dir out/new_coverage_form_check
```

The output directory must not already exist. It contains `report.json` with full
pins, identities and diagnostics, plus `report.md` for reading. Exit code `0`
means no asserted check failed, `1` means the report contains a failed check, and
`2` means the invocation or evidence was refused. A `0` with `not_checked` fields
does **not** mean coverage or form passed. Without a contract, inputs are hashed
and parsed, while coverage/form remain `not_checked`.

The minimal contract has `schema_version: 1`, exact `abc_sha256` and
`source_pdf_sha256` hashes, and a normalization pin. Add coverage and form only
when independently declared evidence is available:

```json
{
  "schema_version": 1,
  "abc_sha256": "<SHA256 of exact ABC bytes>",
  "source_pdf_sha256": "<SHA256 of exact PDF bytes>",
  "normalization": {
    "written_bar_count": 4,
    "route_status": "resolved",
    "route": [1, 2, 3, 4]
  },
  "expected_source_bar_ids": ["1", "2"],
  "assignments": [
    {"written_bar": 1, "kind": "primary", "source_bar_id": "1"},
    {"written_bar": 2, "kind": "primary", "source_bar_id": "2"},
    {"written_bar": 3, "kind": "return_copy", "source_bar_id": "1",
      "reason": "Explicit written realization of the return"},
    {"written_bar": 4, "kind": "editorial_insertion", "editorial_id": "cadence",
      "reason": "Declared editorial cadence"}
  ],
  "expected_performed_route": [
    "source:1", "source:2", "source:1", "editorial:cadence", "STOP"
  ]
}
```

Written-bar numbers are the normalizer's one-based physical ABC bar positions.
Source IDs are opaque, unique strings; use a stable manuscript bar identifier,
not a count guessed from ABC segmentation. Each written bar needs one assignment.
A primary claims first coverage of its source ID; a return copy references an
already covered source ID and requires a reason. An editorial insertion has its
own unique ID and reason, never a source-bar ID. The checker reports missing
source IDs, duplicate primaries, unknown IDs, unclassified written bars,
duplicate assignments, out-of-range assignments, and duplicate editorial IDs.

`expected_performed_route` uses `source:<id>` and `editorial:<id>` tokens followed
by exactly one final `STOP`. Return copies resolve to their original source
identity. Every visit and ending choice matters, including the bar before STOP.
The existing parser implements [FORM_NOTATION.md](FORM_NOTATION.md): a header
`P:` order is the complete route and printed repeats are not expanded again.
The checker does not independently interpret manuscript navigation text.

The normalization pin is a separate regression expectation for this exact ABC:
written count, route status and the full written-bar route. It can intentionally
pin a known wrong candidate route while the adopted identity route detects its
form failure. Changing parser behavior requires explicit investigation and a
new contract; the checker never silently refreshes the pin. Hash mismatches
prevent parsing, and pin drift or invalid parsing prevents coverage/form passes.
Unsupported playback remains `not_checked`. Missing maps or adopted routes also
remain `not_checked`, regardless of bar-count agreement.

## Local saved regression inventory

The sprint inventory is local, ignored evidence under
`out/collection/subscription_sprint_20261105_v1/checks/inventory.json`. Replay it
to a **new** directory:

```sh
uv run python scripts/check_score_coverage_form.py inventory \
  out/collection/subscription_sprint_20261105_v1/checks/inventory.json \
  --output-dir out/collection/subscription_sprint_20261105_v1/checks/replay_new
```

Inventory paths are relative to the inventory file. Each `cases` entry has
`id`, `abc_path`, `pdf_path`, optional `note`, and optional inline `contract`.
Optional `evidence_pins` contain `{path, sha256}`; optional `frozen_reference`
contains `{path, manifest_sha256}` and invokes the existing full freeze verifier.
These pins fail closed before any case runs. The output snapshots the inventory
and its hash; cases include their exact contracts and input hashes.

The initial report covers six frozen reviewed editions plus three saved
Rumichaca drafts. All six have exact edition count/route regression pins.
Aviador additionally reuses its existing explicit 62-source-bar mapping and
declares editorial bars 29 and 45; the other five receive no invented source
mapping. Their route pins are regression expectations, not a new PDF audit.

The Rumichaca postmortem explicitly documents 45 source bars, with initial ABC
bars 46–74 copying 1–29. The saved initial and final gated drafts therefore pass
declared coverage bookkeeping while their 117- and 116-bar routes fail the
adopted 132-bar route. The earlier 45-written/132-performed draft passes that
form comparison. The inventory intentionally returns exit `1` for these known
form failures. This is consumed diagnostic evidence and does not establish an
accuracy improvement, new heldout result, or visual correctness of every mapped
bar. The 903-file reference freeze and all saved drafts remain unchanged.
