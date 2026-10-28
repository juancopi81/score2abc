# Reviewed collection checkpoint — October 28, 2026

Six complete performing editions are musician-reviewed, with no open questions
and valid notation. This checkpoint preserves their exact saved bytes; it does
not revise music or grant model approval.

| Work | Current revision | Surviving archived revisions |
| --- | ---: | --- |
| Amalia | 24 | 23 |
| Atardecer | 18 | 3–17 |
| Aviador | 18 | 8–17 |
| Carrizal | 5 | 4 |
| Coqueteos | 15 | 9–14 |
| Rumichaca | 10 | 3–9 |

The snapshot contains the six current reviews and all 40 available archived
revisions. Earlier overwritten revisions are not recoverable from this checkpoint.
Historical notes and model provenance are retained verbatim, including superseded
Draft descriptions; the current saved review state and revision govern approval.

## Separate data preservation

The local, create-once freeze is
`out/eval/references/reviewed_six_20261028_v1/`. Its 903 hashed files include the six
source PDFs, exact review JSON/ABC, every available override/history file and
explicitly referenced decision/provenance artifacts. Redundant generated
`preview_out/`, `verification_out/` and pipeline-check trees are excluded. There
are no remaining context-cap or missing-reference omissions in this freeze.

Manifest SHA-256, independently recorded here:

```text
ad1995ca51050202770f8ceeb6fc6fe03ee2d622cbbf042b5ad84e3ca27f722f
```

A separately stored archive is at
`/Users/juanpineros/Downloads/jaime-llanos/backups/reviewed-six-20261028-v1.tar.gz`.
Its adjacent `.receipt.json` records hashes, preserved revisions and verification.
The archive was restored to a disposable directory, all frozen contents verified,
and all 18 dashboard file links checked. Archive SHA-256:

```text
8ee7c9411b9c6a36f127f5dc0af5b8827c92804b38ce5d579109676ee6ee1ce1
```

This is a second **local** copy on the same computer, not an off-machine backup.
The PDFs, reviews, freeze and archive are not included in GitHub PRs. A code push
does not back them up remotely. A checksum detects changed content; it is not an
authenticated signature.

## Correction-desk checkpoint

PR #13 now includes the pending selected-playback controls and continuous written
measure numbering. Clicking a note/rest selects its ABC without starting audio;
**Play from here** begins at its first performed visit and preserves the remaining
route, ties, rest timing, bar-limited chords and highlighting. Stop, tempo changes,
saving and resize preserve the selection; editing or changing work clears it.

Renderer-enabled Node tests cover 51 playback/audio cases. The existing Python
suite passed 793 tests under the sandbox, and all three skipped localhost tests
passed when binding was enabled (58 tests in that focused rerun). Ruff, Black and
diff checks passed. Disposable-browser checks on Aviador and Coqueteos confirmed
note/rest/bar selection, editor caret scrolling, selected playback highlighting,
tempo handling, selection after save, work switching and visible measure numbers.
These checks exercised copied reviews; all six live review hashes remain unchanged.

## Evaluation boundary and next work

The frozen references are consumed editorial editions, not heldout test data and
not yet event-level source-fidelity truth. Aviador includes an editorial opening
rest, added cadence bars and Trío harmony. Coqueteos includes a musician-selected
route whose endings depend on the destination. Those choices must not be confused
with uniquely readable manuscript facts.

The initial HTML page at `out/eval/index.html` shows reference readiness and links
to preserved evidence. No Astra–Terra comparison has run. The next PR introduces
the reproducible freezer, verifier, dashboard builder and agreed evaluation plan.
Then add versioned source/editorial labels and the shared grader/harness before
launching model comparisons. New labels must reference the frozen hashes without
rewriting this snapshot. Juan's next immediate action is to merge PR #13 once its
updated head passes CI; further musical questions should be narrowly scoped.
