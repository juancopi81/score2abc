# Aviador and Carrizal form check — October 22, 2026

Carrizal already contains its complete supported repeat structure. Aviador now
has a provisional performance route awaiting two targeted musical decisions.
Continue reviewing Coqueteos first; Aviador needs a form check afterwards, not a
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

## Evidence and scope

Local snapshots, source crops, measure maps, independent audit, validation and
preview/live delivery receipts are under `out/collection/form_review_20261022/`.
Rendering and Play/Stop were checked in an isolated desk; live save/reopen/export
matched exactly. Coqueteos was not edited or reloaded, and canonical data and
frozen research artifacts were not changed.

These local score artifacts are excluded from Git. The PR records application
code and this handoff; it does not publish the scores or create automatic human
revision history at that handoff. Automatic history was subsequently added for
future saves; it cannot recover earlier overwritten revisions. See
[what human saves preserve](COLLECTION_WORKFLOW.md#what-human-saves-preserve).
