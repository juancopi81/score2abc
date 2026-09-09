# Spike integration review

Reviewed 2026-09-07: original `spike/vlm-transcription` **4fe44dc** against
`main` **b0d1b8d**. This review excludes subsequent collection-review work.

**Recommendation: proceed with a draft integration PR for the research branch.**
No demonstrated regression blocks source integration in this bounded review.
Keep melody recognition opt-in; this is not approval to promote the composed
recognizer into default runtime or claim collection-wide transcription accuracy.

## Scope and default behavior

The diff contains 19 commits, 306 files, 126,402 additions and 60 deletions:
97 script files, 198 test/fixture files, and four package files. No production
dependency, lockfile, CLI, or MusicXML backend-selection changes occur.
`pipeline.run` still defaults to `use_vlm=False` and fixture MusicXML.
The melody VLM module is not connected to default pipeline execution.

Default preprocessing **does change**: `render.py` adds staff validation,
splitting, weak-preamble recovery, trailing-blank rejection and diagnostics;
`chord_ocr/alignment.py` changes barline detection and measure cleanup.
`pipeline.py` calls this renderer normally and persists new diagnostics.
These changes should be explicit in the PR, even though recognition sidecars
remain opt-in.

## Isolated validation

Used clean `git archive` extractions, with no original ignored output, caches,
restricted PDFs or frozen models. The existing virtual environment supplies
dependencies; this is clean-source validation, not a fresh dependency installation.

```sh
git archive 4fe44dc | tar -x -C /tmp/score2abc-spike-integration-aNjV9i
git archive b0d1b8d | tar -x -C /tmp/score2abc-spike-integration-aNjV9i-main
```

From the spike archive, using the existing project virtual environment:

```sh
PYTHONPATH=. /Users/juanpineros/juancopi81/score2abc/.venv/bin/python -m pytest -q
/Users/juanpineros/juancopi81/score2abc/.venv/bin/ruff check .
/Users/juanpineros/juancopi81/score2abc/.venv/bin/black --check score2abc scripts tests main.py
```

Results: **708 passed, 2 skipped in 71.61 seconds**; Ruff passes; Black leaves
228 files unchanged. The two skips are localhost socket-binding tests blocked
by the sandbox. The branch diff also passes `git diff --check`.
Relevant tests cover connected staffs, non-staff rejection and source numbering,
weak preambles, sparse staffs, trailing blank staffs, false note stems, and
Carrizal/Coqueteos boundary regressions. A disposable cleanup probe seeded stale
system/chord/rejected-candidate/segment files, then processed a blank page:
all seeded generated files were removed, unrelated content survived, and no
systems were accepted.

## Golden default-pipeline comparison

Ran these commands independently from **both** archive roots, using the same
Python executable above and `PYTHONPATH=.`:

```sh
python main.py ingest dataset dataset/metadata.csv integration-out
python main.py run integration-out
python main.py eval integration-out --ground-truth dataset/ground_truth
```

All commands exit successfully. All ten tracked PDFs complete using fixture
chord OCR and fixture/stub melody; no live model APIs are called. Artifacts and
logs remain only in the disposable archive directories.

| Score | Systems main → spike | Detected measures main → spike |
| --- | ---: | ---: |
| Aviador | 10 → 8 | 64 → 60 |
| Carrizal | 12 → 8 | 57 → 49 |
| Coqueteos | 11 → 9 | 57 → 64 |
| Chispazo | 10 → 8 | 64 → 59 |
| Entusiasmo | 12 → 11 | 136 → 125 |
| Estrella del Caribe | 12 → 9 | 88 → 74 |
| Gato'e Fique | 11 → 7 | 56 → 50 |
| La Chata | 11 → 8 | 72 → 65 |
| Rumichaca | 11 → 6 | 67 → 47 |
| Sobre el Humo | 11 → 9 | 69 → 59 |
| Total | 111 → 83 | 730 → 652 |

These are measured output changes, not asserted physical-count accuracy for
every score. This review did not manually label every accepted/rejected staff.

Fixture compatibility survives: **the same six fixture keys match** on both
refs, yielding the same 13 Aviador chord detections. Total crop lookups change
from 222 to 166; the other nine works have zero hits on both refs. Their missing
fixtures are pre-existing coverage gaps, not a new crop-hash regression.

### Apparent chord regression resolved at the pixel/label level

Whole-work evaluation drops Aviador chord true positives from 1 to 0
(F1 0.051282 → 0), despite unchanged detections. Investigation localized the
change to **system 4**, whose image bytes are identical on both refs. The old
detector includes x=0.285336856, approximately pixel 648; the spike removes it,
reducing that system from ten to nine automatic measures and shifting later
chord indices back one.

The committed `tests/fixtures/barlines/jaime-llanos_12_aviador_pasillo_fulgencio-garcia/system_004_ground_truth.json`
contains ten physical barlines and **does not label pixel 648**. Direct source
image inspection confirms that this stroke belongs to a note stem. Reusing
`barline_harness._parse_via` and `_match`, with its median-label-width tolerance,
on the same ten main-rendered Aviador crops gives:

| Detector | True positives | False positives | False negatives |
| --- | ---: | ---: | ---: |
| main | 69 | 7 | 1 |
| spike | 69 | 6 | 1 |

System 4 alone improves from 10/1/0 to 10/0/0. Other fixed-crop results are
unchanged. Thus the lower whole-work chord score does **not** establish an
incorrect new barline change: it loses an incidental match while removing a
labeled false positive. Do not reintroduce that stem to recover the metric.
The existing `build_vlm_melody_event_benchmark.py:62-69` already documents old
global-index errors: physical systems 7/8 start after measures 46/53 while old
manifests used 47/54. The new default offsets are 46/53. A proper end-to-end
chord claim still needs physical mapping and fixture quality reviewed together.

## Remaining limits and merge conditions

- Only Aviador has tracked whole-work event truth: truth coverage is 1/10.
  Its perfect note score replays committed MusicXML; it is not recognition
  accuracy. Other works retain the existing stub path.
- Runtime preprocessing changes are substantial and deserve reviewer inspection
  of the generated overlays, particularly removed systems. Passing tests and
  the table above do not establish every changed boundary as correct.
- `out/`, `.cache/`, and `dataset/local_restricted/` remain untracked. Portable
  consumed evidence is committed under tests, with partial MusicXML kept outside
  production `dataset/musicxml/`. A source merge alone does not archive every
  original frozen prediction/model needed for historical replay.
- Fresh full-event evaluation is still required before default melody
  integration (`docs/ROADMAP.md:99-105`). Tío Clímaco remains a pre-truth failure;
  no frozen experiment was rerun or altered in this review.

Keep the full research history in the integration PR rather than selectively
cherry-picking dependent experiment commits. Resolve review comments against
the explicit preprocessing delta; retain fresh-recognizer promotion as a
separate decision. This report authorizes neither merge nor deployment.
