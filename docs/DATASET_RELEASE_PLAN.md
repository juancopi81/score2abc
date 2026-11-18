# Reviewed collection: private preview and first-ten release plan

The November 18 checkpoint contains **seven musician-reviewed adopted editions**:
Amalia, Atardecer, Aviador, Carrizal, Coqueteos, Gato'e Fique and Rumichaca.
After the three-score API pilot, the review queue contains 22 saved works,
including 15 drafts. The earlier backup described below captured 19 works before
that pilot. Seven is the
current export count; ten is a publication milestone, not a completed dataset.

The first preview is local and private. Rights to distribute the manuscripts,
compositions, arrangements and derived editions have **not been cleared**. No
license is assigned and nothing is uploaded to Hugging Face or another service.
The original review JSON and notes can contain local paths and historical
provenance, so this exact-preservation package is not ready for public sharing.

## What the export means

`scripts/export_reviewed_collection.py` creates a queue backup, then derives a
private preview from that immutable snapshot. Each exported work includes:

- Exact saved adopted ABC, exact review JSON, revision and current review state.
- Exact free-text editorial notes, unresolved field, save metadata and available
  history. Older history can start after revision 1; missing saves are not invented.
- Source PDF and SHA256, original manifest metadata and source linkage.
- Model and drafting provenance as recorded, including stale historical statements
  inside that provenance. The current `review_state` is the saved approval signal;
  an old model statement that human review was required is not rewritten.
- Explicitly linked source readings, proposals, decisions and images. Original
  absolute paths remain verbatim in archival JSON; portable `evidence_files` and
  source paths identify the copied counterparts.

The adopted edition is distinct from a literal source transcription. Existing
source-reading files retain their identity inside evidence packets. Neither the
free-text notes nor an older model's structured decision is automatically promoted
to a certified source fact or individually approved editorial decision. The
package does not replace the frozen evaluation references or their labels.

Rumichaca keeps its Reviewed state and exact ABC. Its export metadata separately
flags **pending classification** for chords at written bars 1, 6 and 30–32, and the
final D.C. interpretation (adopted route 132 performed bars, model alternative
117). These are recorded disagreements, not confirmed mistakes or a withdrawal
of approval. They attach only to the exact score slug and investigated ABC hash;
a changed edition does not inherit these old findings automatically. The
[API pilot report](ASTRA62_API_RUMICHACA_PILOT.md) explains them.

## Snapshot and restore scope

The queue backup includes all 19 current review records, their available automatic
histories and extra override snapshots, the exact collection manifest, each
reviewed or draft work's source PDF and local desk inputs (`pages`, `review_sources`,
metadata), and explicitly linked proposal/source evidence found in current and
historical review notes/provenance. All 87 protected review/override JSON files
must match the coordinator's before-state pins. Originals are checked again
before publication; a changed file or new history save aborts the snapshot.

Copies have repository-relative paths under `payload/`. The manifest inventories
every file with size and SHA256, describes each work and lists omitted context.
Excluded material includes disposable desk copies, caches, code, raw API/model
traces, private response directories and environment files. Linked context is
bounded to 5,000 files / 500 MB; exceeding this fails rather than silently dropping
musical evidence. The backup is not a whole-repository or experiment archive.

Existing frozen references remain protected by the separate November 6 reviewed
seven archive:
`out/collection/gato_reviewed_checkpoint_20261106_v1/reviewed-seven-20261106-v1.tar.gz`.
Its 61,795,781-byte archive was rechecked against its existing receipt on November
18: SHA256 `b1543ab514597ecb6c97028c761fd7a29b4919b3d991b6d1238d3ea18df970b5`.
The existing receipt records a successful restore of 1,089 files, including the
six-score and Gato reference freezes. That previous restore was not rerun during
this export; the archive hash was reverified. Frozen evaluation inputs, labels,
and historical experiment outputs are not duplicated in the new queue backup.

The new archive is restored into an empty disposable directory and every copied
file is checked against the manifest before success is reported. All copies are
on this computer: this is **not an off-machine backup**.

The created `review_queue_20261118_v1` snapshot contains **2,468 files**
(282,675,003 payload bytes); its restored-and-verified archive is 224,424,432 bytes.
The `reviewed_20261118_v1` private preview contains **931 files** (90,671,840
payload bytes), including seven exact reviewed editions. All 87 protected original
review/override JSON files were rechecked unchanged, and captured review states,
revisions and source records were checked against their copied bytes.

A second immutable backup, `review_queue_20261118_postpilot_v1`, captures all
**22 works / 3,264 files** after installation. Its 250,862,218-byte archive was
restored and hash-verified, then the backup was independently verified again.
Archive SHA256:
`712ae396d74760cd6d6cecbe8a21e53f414badf4f1f5dbc1ec282e0e45fb5208`.
The seven-edition private preview stays unchanged: the new three are Drafts.

## Commands

Use a new destination for every snapshot or export. Existing destinations are
never overwritten.

```bash
uv run python scripts/export_reviewed_collection.py backup out \
  out/backups/review_queue_20261118_v1 \
  --protected-manifest out/collection/checkpoint_20261118_v1/protected_reviews_before.json

uv run python scripts/export_reviewed_collection.py export \
  out/backups/review_queue_20261118_v1 \
  out/collection/releases/reviewed_20261118_v1

uv run python scripts/export_reviewed_collection.py verify \
  out/collection/releases/reviewed_20261118_v1
```

The backup command also writes sibling `.tar.gz` and `.receipt.json` files. To
inspect a restore, extract the archive into a new empty directory and run the
`verify` command there. Its `payload/` preserves repository-relative locations;
review these against current files before any manual restoration. The tool never
restores over the live review desk automatically. Some manifest rows describe
works outside this 19-work backup and do not imply their data were copied.

## From seven to the first ten

1. Finish the current musician reviews without overwriting WIP. Add only works
   the musician explicitly marks Reviewed; select three additional editions from
   the existing queue rather than treating model completion as approval.
2. Resolve or explicitly retain publication-relevant source/editorial alternatives.
   A defensible alternative may remain documented; review should distinguish
   visible ink, reconstruction, performed form and chord interpretation.
3. Freeze a new ten-edition preview and verify rendering, playback, exact exports,
   source links, histories and the package inventory. Keep earlier releases intact.
4. Establish distribution permissions and the appropriate license per asset.
   Then prepare a public manifest that omits personal paths/private histories as
   needed, while retaining approved provenance and clear limits on source truth.
5. Ask for publication approval on the concrete package. Publish only after that
   approval; record a version, immutable hashes, changes and any evaluation split.

These seven pieces have already informed development. They are useful regression
references, not an unseen test set. Future model evaluation should version the
edition, source labels, prompts, tools and scoring separately; human editorial
acceptance and source fidelity remain separate measurements.
