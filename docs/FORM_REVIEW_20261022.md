# Collection form checks — October 22, 2026

Carrizal already contains its complete supported repeat structure. Aviador now
has a provisional performance route awaiting two targeted musical decisions.
Coqueteos now incorporates Juan's first pass and selected editorial route; its
next check is a targeted audition. Aviador needs a form check afterwards, not a
fresh note-by-note review.

## Carrizal: preserve the reviewed version

The existing revision 4 repeats written bars 1–16, 17–32 and 33–48: **AA BB CC**.
All eight manuscript rows support those boundaries; no secure casillas or
D.C./D.S./Fine instruction was found. The saved note “Repetitions look good to
me!” remains intact. No live ABC, notes, review state or review time changed.
Playback visits 96 bars in 144 seconds at the default 120 quarter-note BPM.

## Aviador: provisional return before the Trio

The selected route is **A → B → A → Trio**, encoded as `P:ABAC`:

- A: written bars 1–16, ending at the Fine mark in manuscript row 2.
- B: bars 17–46, ending at the D.C. instruction in row 6.
- C: Trio, bars 47–62, manuscript rows 7–9.

This interprets the D.C. literally as a return to the beginning through Fine,
then the Trio once. Two separate Astra/Extra High readers checked the form signs.
The destination abbreviation after “hasta” is unclear and the opening segno is
before bar 2, so this remains an editorial proposal rather than settled source
truth. No casillas or secure Trio repeat dots were identified.

Two questions are saved in the desk:

1. Do the irregular marks after bar 16 require an extra first-pass A repeat?
2. Should the return begin at bar 1 or at the segno before bar 2?

All 62 written bars, 250 noteheads, 52 rests and 26 chord markings are preserved,
including the original 2.5-beat opening bar. The proposed route plays 78 bar
visits in 116.5 seconds at default tempo. Melody, chords and following use the
same order. An unselected `AABAC` alternative is retained locally; the segno
alternative would require its own section boundary.

The prior reviewed revision 7 is backed up byte-for-byte. Live revision 8 is a
draft only because the new form needs approval; no human review time was added.
Record the two decisions in Review notes, remove their resolved questions, and
mark Reviewed once the performance route is accepted.

## Coqueteos: first-pass corrections and selected form

Human revision 9 is preserved exactly in automatic history. Its edits were
mapped to manuscript measures; both added ties work and all 52 bars contain
three beats. Some descriptive measure numbers differed from the actual edited
bars, so the local correction report keeps the original notes alongside the
mapped changes. The old A5-quarter annotation was stale after the human B5 edit.

Juan selected **A1–A2, B1–B2, returned A, C1–C2**, followed by a final
**D.C. A–B–C** without repeats and stopping at manuscript system 9, measure 4,
before C's first ending. His revision 11 audition clarified the connections:
the casillas contain lead-ins for particular destinations, so they are not
automatically played whenever repeats are suppressed. On the returned A, play
through S3M4 and skip both A endings into C at S7M1. On the final D.C., A's
second ending still leads into B; B ends at S5M6 and skips both endings into C.
The initial A/B repetitions retain their endings. C's second ending is a
direction, not an extra musical bar. This remains an adopted editorial route;
uncertain manuscript signs have not become confirmed source readings.

Live revision 12 is a draft with one targeted audition question. It preserves
all human musical choices: 52 written bars, 246 noteheads, 16 rests, 28 chord
symbols, 41 phrase slurs and 12 tied-pitch links. The corrected connections remove
two ending visits: the route now plays 160 bar visits in 240 seconds at 120 BPM.
Independent checks verify the full melody, chords,
highlight order and key resets, including the return from D major to D minor.

Refresh the desk and listen for S3M4 directly into the coda, the final D.C.'s
S5M6 directly into the coda, and the stop at system 9 measure 4. If accepted, record the resolution,
remove the audition question, and mark Reviewed after completing the source
review. Current coordinator notes appear first; Juan's exact notes and earlier
AI preparation notes are retained separately. No human review time was added.

The first-pass comparison, form proposal, independent audit and preview/live
receipts are under `out/collection/coqueteos_first_pass_20261022/`. The live
save/reopen/export and automatic history were verified; the other twelve saved
reviews remain unchanged. The later connection correction and exact human
revision 11 are preserved under `out/collection/coqueteos_connections_20261022/`
and in automatic review history.

## Evidence and scope

Local snapshots, source crops, measure maps, independent audit, validation and
preview/live delivery receipts are under `out/collection/form_review_20261022/`.
Rendering and Play/Stop were checked in an isolated desk; live save/reopen/export
matched exactly. During that earlier Aviador/Carrizal pass, Coqueteos was not
edited or reloaded. The subsequent Coqueteos pass is documented above. Canonical
data and frozen research artifacts were not changed.

These local score artifacts are excluded from Git. The PR records application
code and this handoff; it does not publish the scores. Automatic history was
subsequently added and all thirteen saved reviews checkpointed without changes;
it cannot recover earlier overwritten revisions. See
[what human saves preserve](COLLECTION_WORKFLOW.md#what-human-saves-preserve).
