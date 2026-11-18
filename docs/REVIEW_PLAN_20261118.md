# Review plan — November 18, 2026

The current saved collection contains **7 reviewed scores and 15 drafts**. This is
a snapshot, not a new musical approval. The local dashboard reads current review
states, revisions and questions each time it is rebuilt.

Open the [review queue](http://127.0.0.1:8777/collection_review_20261118_v1/index.html).
Its links open the editable desk on port 8766. The HTML is saved under
`out/eval/collection_review_20261118_v1/index.html`, alongside the other pages served
by the existing port-8777 server. The static page never saves a score or changes
review history.

## One score at a time

1. **La Chata** — resume revision 6. Start with the crowded system 2 passage in the
   [focused packet](http://127.0.0.1:8777/diagnostics/focused_review_sprint_20261106_v1/index.html).
   Decide voice/rhythm structure and accidentals separately. The packet's alternate
   auditions are proposals; none is automatically installed. Then finish all rows
   and the complete playback route.
2. **Estrella del Caribe** — resume revision 4 and the saved work in progress.
   Its current questions concern an accidental comparison, dense chromatic chords,
   independent held voices/endings, and expressive text.
3. **Celina** — revision 2. Begin with the saved Fine/return interpretation and
   uncertain duration dots; then check the lower voices and the complete score.

The dashboard lists the remaining prepared drafts without requiring another model
run. Entusiasmo's terminal A/B/C groups are already human-adjudicated navigation
lettering; do not reopen that classification. Its exact return policy and other
current saved musical questions remain separate.

The new API drafts — **Beatriz Eugenia, Fiesta de la montaña and Cúcuta 75** — are
installed in the backlog. They have independent model reviews and explicit
musician questions, but no human approval. Keep the three priorities above first.
Their [batch handoff](http://127.0.0.1:8777/collection_api_20261118_v1/index.html)
shows source/engraving previews, costs and playback limitations. In Beatriz, report
corrections to A once; the coordinator will synchronize its written-out returns.

## What to do in the desk

- Compare **every source row**, including unflagged passages. Check melody, lower
  voices, rhythm, accidentals, chords, ties and endings.
- Listen through the **complete performed route**: repeats, casillas, transitions,
  returns and the final stopping point.
- Save each decision in **Review notes and resolutions**, naming the source
  row/bar, adopted reading, evidence and uncertainty. Distinguish **source
  correction**, **editorial reconstruction**, and **editorial interpretation**.
- Move answered questions into those notes and remove only the resolved item from
  **Questions to resolve**. Keep genuinely unresolved items open. An explicitly
  accepted provisional editorial choice can resolve a review question without
  becoming proven source truth.
- **Save and reopen** to confirm the correction and playback persist. Mark
  **Reviewed** only after the full pass and after all open questions are settled.
  Otherwise save as Draft and send the difficult location to the coordinator.

AI reconstruction uses only the supplied PDF. Human additions informed by another
score, recording or musical knowledge are welcome as documented editorial choices;
record the external source/link and passage when available. Do not silently relabel
those additions as visible manuscript evidence. The [editorial protocol](EDITORIAL_TRANSCRIPTION_PROTOCOL.md)
describes the distinction.

When finished, send: **“Done — [score], revision [number], marked reviewed.”**
Mention editorial choices or remaining concerns. The coordinator will inspect the
saved result, preserve the checkpoint and identify the next score.

## Refresh the local dashboard

```sh
uv run python scripts/build_collection_review_dashboard.py out
```

Only the HTML page is written. Existing score drafts, review histories, frozen
references and desk sessions remain untouched. The builder includes registered
works with a current saved review and excludes nested preview copies. Once La
Chata becomes Reviewed, the next rebuild advances Estrella automatically.

The page also links the [latest model comparison](http://127.0.0.1:8777/reproduction/astra62_api_rumichaca_20261118_v1/index.html).
Model-evaluation outcomes do not alter the musician's saved review state.
