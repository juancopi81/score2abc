# Collection form notation

Use one engraving vocabulary while preserving each musician-approved performance
route. This applies to future editorial drafts and presentation-only maintenance
of reviewed works.

## Visible notation

- Use boxed **A**, **B**, **C** at the starts of real musical sections. An
  introduction, bridge, Trío or Coda may also have a short descriptive label.
  Internal playback spans and individual endings are not extra sections.
- Use opening and closing repeat barlines (`|:` and `:|`) for repeated passages.
  A plain double bar (`||`) divides sections; it does not mean repeat.
- Use numbered volta brackets for alternate endings: **1.**, **2.**, or an
  explicit list of visits where necessary. Cover exactly the relevant bars.
  Do not print prose such as “Coda first ending” instead of a bracket.
- A shared end/start repeat is appropriate where the next repeated passage
  starts in a second-ending lead-in. Do not add a silent measure to make it fit.
- Use concise Spanish navigation text and conventional `D.C.`, `D.S.`, `Fine`,
  `Coda` or `Trío` where their meaning matches the adopted route. Specify
  **sin repeticiones**, the chosen ending, destination and stopping point when
  an unqualified return would be ambiguous. Never replace a reviewed route with
  the usual interpretation of an abbreviation.
- Keep a short explicit direction for a destination-specific bridge or a
  partial final ending. Numbered brackets alone do not explain every return.
  A second ending containing only a navigation instruction need not invent music.
- Put segno/coda destination markers at the passage entrance and return/stop
  instructions at the end of the relevant bar. If a direction-only ending has
  no musical span, print a separate numbered instruction after the repeat
  instead of stretching its bracket over unrelated bars. Coqueteos uses this
  exception for its final **2. D.C. al Fine, sin repetir**.
- Distinguish section labels from chords: boxed **A** is a section; **A7** over
  a note is harmony. Avoid duplicate unboxed section letters.
- Show actual key changes, without restarting the key signature at every
  technical split. Retain necessary accidental/key instructions when removing
  one would change a pitch. Avoid duplicate tempo markings; technical resets
  must retain approved expressive timing.
- Retire stale “pending”, “unclear” and provisional form labels after resolution.
  Accepted reconstructions remain identified in review notes and provenance;
  acceptance does not turn missing ink into a source fact.

Volta brackets select endings on specified visits. Return instructions identify
the beginning, a segno, a coda destination or a Fine stop. See the primary
[MuseScore volta reference](https://handbook.musescore.org/notation/repeats/voltas)
and [jump/marker reference](https://handbook.musescore.org/notation/repeats/jumps-and-markers).
Boxed letters, Spanish explanatory text and hidden technical spans are collection
choices, rather than requirements of those references.

## ABC and playback

Ordinary ABC repeats may control a simple form without a header `P:` order.
More involved approved returns may use the desk's bounded explicit `P:` order.
The visible vocabulary is the same in either case.

With a header `P:` order, that order is the complete playback route in this desk:
include every intended repeat and ending selection. Printed repeat signs remain
visible and are not expanded a second time. This is the desk's contract, not a
claim that every ABC player handles mixed part orders and repeats identically.

Hide the technical banner and internal ending identifiers. For example:

```abc
X:1
T:Ejemplo de forma
%%writefields P false
P:AXAYB
%%partsbox true
M:3/4
L:1/8
K:C
[I:writefields P true][P:A][I:writefields P false] |: C2 E2 G2 |1
[P:X] G2 E2 C2 :| [2
[P:Y] C6 ||
[I:writefields P true][P:B][I:writefields P false] E2 G2 c2 |]
```

Here X/Y identify internal first/second-ending spans; only sections A/B are
visible. Choose identifiers and routes for the actual work. Use `[1`/`[2` or
`|1`/`|2` as appropriate; keep conventional repeat syntax in exported ABC.

Never remove a key change, tie, lower voice, rest or fermata to simplify labels.
In the installed abc2svg, `!caesura!` attached directly to a bar hides that bar.
Attach it to the preceding note when the pause follows that note and the repeat
barline must remain visible. Preserve its musical position and inspect the render.

## Maintaining reviewed scores

1. Snapshot the exact saved JSON/ABC, revision and hashes.
2. Change presentation only. Map visible sections/endings to the unchanged
   physical bars and performed route.
3. Compare written pitches, per-head durations, rests, harmonies, dynamics,
   slurs, ties and tempo. Compare unfolded attacks/holds, following and bar-limited
   accompaniment by physical event, rather than shifting character offsets.
   Check every special return and final stop.
4. Inspect the engraving in a disposable desk, including repeat dots, bracket
   extents, collisions and real key changes. Check playback and exact
   save/reopen/export.
5. Save only if the human revision and full-file hash still match. Preserve the
   predecessor, all prior notes/provenance and accumulated human review time.

A presentation-only save may preserve an existing **Reviewed** state when those
comparisons pass and no new musical question is introduced. It carries forward
the musician's approval, rather than granting model approval. An already compliant
score may remain untouched. Any change to music, performed order or unresolved
source interpretation belongs in a separate **Draft** proposal for human approval.
Never mark a previously unreviewed score Reviewed as part of formatting work.

The local five-score packet is `out/collection/form_standard_20261027_v1/`.
Its scores, snapshots and receipts are excluded from Git; pushing code/docs does
not back up that review data to GitHub.
