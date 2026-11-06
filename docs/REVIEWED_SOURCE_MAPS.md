# Reviewed-six source identities and approved routes

`docs/eval/reviewed_six_source_maps_v1.json` adds complete **bar-identity maps** for
Amalia, Atardecer, Carrizal, Coqueteos and Rumichaca. Each received a visual PDF
pass and a different Astra reader's verification. Aviador retains its prior
explicit mapping, including two editorial cadence bars. All approved editions
and the October 28 reference freeze remain unchanged.

This accounts for 312 source identities and 314 written edition bars. It does
**not** expand source-note truth: [labels v3](EVAL_REFERENCE_LABELS.md) still cover
67/312 bars and 268 noteheads at the mapping checkpoint. The later
[v4 note audit](SOURCE_NOTE_AUDIT_V4.md) expands that separate label set to 122/312
bars; it does not change these maps. A mapped source bar can contain an accepted
reconstruction. These consumed references are not heldout accuracy evidence.

## What is recorded

- Manuscript page, musical row and bar for the five new maps. Musical rows differ
  from automatic desk selectors. Aviador retains its earlier numeric source IDs.
- Exact written assignments: ditto signs occupy distinct source bars; blank
  ruling and navigation-only endings do not invent bars.
- Declared approved route segments, destination-specific casillas and final stop.
  Expectations come from the preserved approved form, then are checked against
  the parser; a candidate's output never generates its own expected route.
- Source notation, accepted editorial choices and source ambiguities separately.
  Historical recording-informed human choices remain editorial. No new recording
  or paid API generation call was used for this mapping sprint.

Agreement between readers concerns identity and approved-route encoding, not
per-note correctness, new musician approval or a uniquely correct source route.
Existing approved choices do not need to be approved again.

## Offline replay

```sh
uv run python -m scripts.build_reference_mapping_checks \
  out/eval/references/reviewed_six_20261028_v1 \
  out/eval/source_maps/reviewed_six_20261105_v1_verified
```

Use a new destination for each replay. The builder verifies the full freeze,
exact PDF/ABC hashes, row/bar declarations, coverage, parser pins and adopted
identity routes. Unresolved mapping questions or inconsistent declarations
prevent publication. It reuses [the coverage/form checker](SCORE_COVERAGE_FORM.md)
without changing normalization, subscription/API workflows or model scores.

The delivered local dashboard is `http://127.0.0.1:8781/`. To serve it again:

```sh
python3 -m http.server 8781 --bind 127.0.0.1 \
  --directory out/eval/source_maps/reviewed_six_20261105_v1_verified
```

Outputs include HTML, exact local source/ABC copies, maps/results, deliberate-error
controls and a file-hash manifest. Twelve controls detect duplicate primaries
hiding omitted identities and wrong visit order with unchanged total counts.
Tests also cover Coqueteos's destination-specific endings/final stop, Carrizal's
false accidental boundary and reconstructions within mapped source bars. These
are checker controls, not model results.

Source/ABC copies stay in ignored local output. The tracked specification holds
identities and decision annotations, not complete scores. It pins the original
author/reviewer evidence under
`out/collection/reference_mapping_20261105_v1/`. The separate reader caught a
Carrizal image-locator boundary error; the original map and correction are
preserved, with no change to identities, approved music or route.

## Future comparisons

Approved ABC hashes bind exact regressions. A future candidate requires its own
explicit, checked source-to-written mapping and hashes before comparing performed
identities. Do not infer alignment from bar counts or substitute reference written
positions for candidate positions.

A defensible different route is an **edition disagreement**, not automatically a
source error; retain it for the editorial rubric and musician adjudication. When
identities, adopted form or accepted alternatives change, create a new map version,
retain old reports, and regrade retained outputs under the same version.
