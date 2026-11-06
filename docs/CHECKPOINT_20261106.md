# Seven reviewed editions and Gato playback checkpoint — November 6, 2026

Juan approved Gato’e Fique at revision 14 after correcting the grace-note playback
and choosing the octave endpoint. Seven editions are now reviewed, with nine
remaining drafts. Estrella del Caribe is the next human review; this checkpoint
does not save, reload or change its draft.

| Reviewed edition | Revision |
| --- | ---: |
| Amalia | 24 |
| Atardecer | 18 |
| Aviador | 18 |
| Carrizal | 5 |
| Coqueteos | 15 |
| Gato’e Fique | 14 |
| Rumichaca | 10 |

All seven current review hashes were pinned before preservation and verified
unchanged afterward. Historical notes are retained verbatim, including obsolete
Draft wording; the saved review state and revision govern approval.

## New reference, existing benchmark preserved

Gato is frozen separately at `out/eval/references/gato_reviewed_20261106_v1/`.
The 134 hashed files include its original PDF, exact ABC/JSON, all surviving
revisions 3–14, and referenced decision/provenance evidence. No referenced context
was omitted. Manifest SHA-256:

```text
ed6f5cb31a46f6dd06a6d6f90ec21d6e5120c483a9e5bee73711ad1507df0c2c
```

The adopted performance order is `AABBCXCYAB`: 42 written bars and 112 performed
bars. The 8va covers manuscript S7m2–5, including the first bar of the second
casilla; the final bar S7m6 returns to written pitch. The accepted S3m5 quarter-note
reconstruction, octave span, and D.C./Fine route remain editorial decisions, not
newly established source ink.

This is a consumed, musician-approved performing edition. Source-fidelity labels
are pending. Gato is not silently added to the six-score model comparison or its
label masks. The original six-score freeze and its 903 files remain unchanged,
with manifest SHA-256
`ad1995ca51050202770f8ceeb6fc6fe03ee2d622cbbf042b5ad84e3ca27f722f`.

The freezer now accepts an explicit, repeatable catalog selection. Explicit
selection uses a version-2 manifest that pins its catalog numbers; omitting the
option preserves the legacy six-score selection and version-1 format. Both
formats retain create-once publication, review/history checks and exact file
verification.

```sh
# Already created; do not rerun this create-once command at this destination.
uv run python -m scripts.freeze_reviewed_references freeze out \
  out/eval/references/gato_reviewed_20261106_v1 --catalog-number 49

uv run python -m scripts.freeze_reviewed_references verify \
  out/eval/references/gato_reviewed_20261106_v1 --out-dir out
```

## Separate backup

The verified archive and adjacent receipt are stored at:

`/Users/juanpineros/Downloads/jaime-llanos/backups/reviewed-seven-20261106-v1.tar.gz`

The archive contains 1,089 files: both sealed reference bundles and the three
Gato correction-delivery packets. It was restored into a disposable directory;
every file hash and both restored freeze manifests passed verification. Archive
SHA-256:

```text
b1543ab514597ecb6c97028c761fd7a29b4919b3d991b6d1238d3ea18df970b5
```

This is a second local copy on the same computer, not an off-machine backup.
Music, PDFs, review histories and the archive remain excluded from Git. Only code,
tests, documentation and portable audit results are included in the PR.
Preservation receipts are under `out/collection/gato_reviewed_checkpoint_20261106_v1/`.

## Playback fix and offline audit

With an explicit part order, grace-note children previously retained their linear
source times instead of following their owning section. Gato’s ornaments could
sound during A and disappear from repeated B sections. The player now maps their
source offsets to the parent section before audio conversion. The two ornaments
sound on all three B visits, including the final return; omitted sections do not
leak ornaments. Ordinary playback without a part order remains unchanged.

The PR also includes the completed [expanded source-note audit](SOURCE_NOTE_AUDIT_V4.md):
122/312 source bars and 547 noteheads, plus offline regrading of 14 saved outputs.
Five ambiguous new bars remain excluded and v3 remains the default. No new paid
model generation or changes to the subscription workflow were needed.

Validation: 1,362 tests passed in the sandbox plus all three localhost tests in a
separate socket-enabled run (1,365 total). Ruff, Black, renderer-enabled playback regressions,
Gato save/reopen/export, and backup restore checks pass. The prior listening pass
and human approval are preserved; technical checks do not replace that approval.
