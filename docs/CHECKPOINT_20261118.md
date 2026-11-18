# Collection and model checkpoint — November 18, 2026

The collection checkpoint and the new model/delivery tooling ship separately.
The [review plan](REVIEW_PLAN_20261118.md) starts with La Chata, then Estrella del
Caribe, then Celina. All three were still Draft when this checkpoint was captured;
no musician approval has been added by an agent.

## Reviewed data and recovery

`out/backups/review_queue_20261118_v1` captures all 19 saved works, including seven
Reviewed editions and twelve Drafts. Its 2,468 files include current review JSON,
available histories, desk inputs, and explicitly linked source/proposal evidence.
All 87 protected review/override files match the before-state hashes. The snapshot
was captured before the new three-score API production pilot.

The sibling `review_queue_20261118_v1.tar.gz` is 224,424,432 bytes, SHA256
`790b57841424054ec31e7ee80e13b7dd42a8f743e5dc300ed31fba2c9c839d99`.
It was restored into an empty temporary directory and every file was verified.
This is a second local copy, not an off-machine backup or a GitHub music backup.
The [release plan](DATASET_RELEASE_PLAN.md) specifies exclusions and the separately
protected evaluation-reference archive.

`out/collection/releases/reviewed_20261118_v1` is a private seven-edition preview:
931 files, exact approved ABC/review bytes, original notes, histories, source
linkage and copied evidence. It does not certify all source readings or clear
distribution rights. Rumichaca's known chord/form classification questions attach
only to the exact investigated edition hash; its approval and music are preserved.

## Engineering and model evidence

- [Collection PR #19](https://github.com/juancopi81/score2abc/pull/19) checkpoints
  Gato labels, focused passage work and the three November 6 model drafts.
- GPT-6.2 support keeps API and subscription execution explicit, with source-only
  author/reviewer contexts, saved traces, completion checks and versioned results.
  [Aviador](ASTRA62_API_AVIADOR_PILOT.md) and
  [Rumichaca](ASTRA62_API_RUMICHACA_PILOT.md) record the completed matched pilots.
- The review queue is read-only and rebuilds from saved review states. Backup and
  export are create-once, hash-verified operations; concurrent manifest/review
  changes are rejected. Raw model/API traces and environment files are excluded
  from the musical release preview.

The next release milestone is ten musician-approved editions, with portable
metadata and documented editorial choices. Public Hugging Face publication and
the public landing page remain separate from this private preparation.

## Validation

The combined local suite passes **1,492 tests**, with three localhost tests skipped
under the sandbox. Ruff, Black and diff checks pass. Independent review found and
resolved concurrent manifest/review capture races and trace filtering in the
exporter. The served review-plan page was checked in the in-app browser; its links
point to the running local correction desk. Both the 19-work backup and seven-work
preview pass independent inventory/hash verification.
