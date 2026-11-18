# Three new subscription drafts — 6 November 2026

Celina, Blanca Inés and Barbatuscas are installed as complete, independently
reviewed **model drafts**. The desk now contains **19 works: seven reviewed and
twelve drafts**. All 78 pre-existing review/history files remain byte-identical,
including Estrella del Caribe. No paid API call was made.

Each work had a fresh Astra author and a different fresh Astra reader, configured
as `gpt-6-astra` / `xhigh` through subscription agents. Each reader recorded an
initial manuscript-only inspection before opening the author's hypothesis, then
checked every musical row. The supplied PDF was the only musical source; no
recordings, external scores or reference answers were supplied.

## Review order and findings

Continue **Estrella del Caribe** first. Then use this order:

| Work | Manuscript rows / mapped spans | Performed visits | Draft playback | First things to check |
| --- | --- | --- | --- | --- |
| Celina | 8 / 52 | 122 | 2:32.5 | Row-one Fine/return interpretation; reconstructed dots; row-seven bass |
| Blanca Inés | 8 / 53 | 150 | 3:01.3 | Damaged S3 pickup/arpeggio; S7 natural and duration; final return |
| Barbatuscas | 9 / 47 | 133 | 3:19.5 | Closing bass, chromatic dyads, selected high heads, ties and coda route |

The [local review guide](http://127.0.0.1:8777/subscription_batch_20261106_v1/index.html)
links the desk, original PDFs, engraved previews, ABC downloads, decisions and
reader notes. Six, seven and eight focused questions respectively are also saved
in each draft. Listen through the complete score, compare the manuscript, save
corrections, and move answered questions into Review notes before marking Reviewed.

Celina's reader retained all normalized pitches, durations, rests, chord labels
and performed order. It restored visible beaming and `leg.`, clarified the first
return instruction, and distinguished the final `dim.` from a possible Fine.
Three uncertain duration dots and one inferred boundary remain explicit editorial
choices; the source-oriented ABC preserves the shorter duration readings.

Blanca Inés received six pitch-head changes, C7 → Cm, and more precise placement
of the Fine direction. An initial row-count disagreement was resolved from the
scan: S2 has eight spans, while S8 has two musical spans and a direction-only
ending. The 1+2-beat split and one-beat pickup are preserved with source-backed
meter exceptions, without padded rests. The damaged S3 reading and cramped S7
accidental remain musician questions.

Barbatuscas needed the most substantial second pass: 16 written bars changed in
normalized musical content, with 349 final heads versus the author's 352.
Corrections include chromatic voicings, rhythmic placement, accidental scope,
four-head final chords, and cross-bar phrasing. Its low closing bass, some high
heads and the exact D.C./coda interpretation remain provisional.

These are author-to-reader differences, not a ground-truth accuracy measurement.
All three works remain Draft at revision 2, with **zero saved human review time**.
None has been promoted to a reference, label set or training example.

## Evidence and checks

The local packet is `out/collection/subscription_batch_20261106_v1/`. It preserves
early drafts, frozen author handoffs, each independent source-first account,
source/proposal ABCs, decisions, questions, source crops and audit seals. Exact
source identities and declared performance routes are checked against fresh
normalization. Spans include pickups, endings and explicitly inferred boundaries;
coverage bookkeeping alone does not establish musical truth.

All three pass notation, declared coverage, executable form and finite audio-event
checks. Reviewers verified that tied continuations do not reattack, including
simultaneous heads. Rendered endings, navigation and key changes were inspected.
The coordinator verified exact save/reopen/export and history in a disposable
desk before publishing only the three new work directories and manifest rows.

The live browser loads all three manuscripts and rendered scores, with playback
and download controls enabled. The temporary inspection tab was closed without
saving; only the static guide was retained, so agent inspection does not become
saved human review time. The existing desk was not restarted or reloaded.

Twenty-one coverage/form and playback tests pass. The original six-reference
freeze verifies all 903 files and Gato's separate freeze all 134 files, with their
prior manifest hashes unchanged. Full musical data remain local; portable counts,
configuration and hashes are tracked in
[the batch summary](eval/subscription_batch_20261106_summary.json).

## Checkpoint and resume

The local checkpoint is on `codex/gato-labels-passage-audit`, after the preceding
Gato-label and focused-passage checkpoint. Score data and histories are excluded
from Git. Their separate archive is:

`/Users/juanpineros/Downloads/jaime-llanos/backups/subscription-batch-20261106-v1.tar.gz`

The archive inventory, restore verification and Git commit are recorded in
`out/collection/subscription_batch_checkpoint_20261106_v1/receipt.json`.
This is a second local copy on the same computer. The prior focused-review archive
also remains intact, including the Gato reference/label artifacts.

To reopen the saved guide and the new desk after the servers stop:

```sh
python3 -m http.server 8777 --bind 127.0.0.1 --directory out/eval
uv run python main.py review out --port 8767
```

No transcription rerun is needed. Continue the existing human review queue;
reviewed states, source/editorial distinctions and history must survive future
edits. The existing six-score evaluation and subscription/API workflows were not
changed by this production batch.
