# Rumichaca: offline trace diagnosis, October 29, 2026

This is a consumed postmortem of the saved API pair, with reference access. No
new model calls, changes to frozen results, or changes to the live score were
made. The images are direct clips of the original PDF, without enhancement.

## What happened

Meter warnings were computed correctly and returned in draft feedback. They
were not lost. The author retained 67 divisions and 47 warnings. Review repaired
the grouping to 45 bars, but explicitly retained three warnings. The latest
checkpoint says `complete_claimed: true`; its notation parses, so the API marks
the session complete. The offline evaluator correctly labels the music partial.

The source tool records meter issues but does not require each issue to have a
structured disposition (`scripts/experiments/astra_source_tools.py:260`). The
API completion check uses `complete_claimed` and parser validity, without
consulting the issue list (`scripts/experiments/assisted_api.py:644`). This is a
completion-policy gap, not a missing beat counter. Musical approval remains
separate from both checks.

## Three bars and existing human decisions

Numbers below count musical rows, excluding the title band. Global measures
2, 16 and 44 are respectively row 1 bar 2, row 3 bar 1 and row 7 bar 3. ABC
examples use L:1/8; 3/4 expects six eighth-note units.

| Location | API reviewed reading | Earlier subscription review | Frozen human edition |
| --- | --- | --- | --- |
| Row 1, bar 2 | `z ^F A2 G e2` = 7 units | `^F A2 G e2` = 6 | Same as subscription |
| Row 3, bar 1 | `^C4` = 4 units | `z2 ^C4` = 6 | `D2 ^C4` = 6 |
| Row 7, bar 3 | `C2 C2 C E G B & C4 C4` = 8 units per voice | `C C xEGB & x2 C4` = 6 per voice | `D _D C E G B` = 6 |

**Bar 2: the strongest immediate source question.** The first reviewer
checkpoint omitted the leading rest and had no meter warnings across its seven
entered bars. Later checkpoints restored the rest; decision R03 explicitly
withdraws the omission because the glyph seemed separate from the accidental.
It acknowledges seven eighths and gives a lost flag or ambiguous glyph as
alternatives. The similar A7 figure at row 2 bar 3 (global bar 10) is a useful
within-PDF comparison. This supports inspecting the extra rest; it does not
prove that repeated phrases must be identical. The earlier subscription and
human edition both omit it. Ask whether there is a genuine rest, or ink belonging
to the accidental/another sign; if it is a rest, which written duration differs?

**Bar 16: a known reconstruction, not an undiscovered human task.** The PDF
clearly presents an open low head and stem, while the beginning remains hard
to interpret. The API retains a C-sharp half note and explicitly declines to
supply a dot. The earlier subscription fills the missing beat with a quarter
rest. Juan's edition instead has D quarter then C-sharp half, and his saved
note says: “S3M1 changed notes, not visible in PDF.” That approved edition remains
intact. The optional question is why D was chosen—musical inference, another
passage, or another source—not whether Juan must approve it again. Under the
PDF-only experiment policy, a performance completion must remain labeled as
reconstruction; exact agreement with the human choice is not source fidelity.

**Bar 44: voice interpretation and rhythm are entangled.** Both API voices
last four beats, so this is not the validator mistakenly adding simultaneous
voices together. The reader explicitly calls its two held low Cs an uncertain
polyphonic reconstruction. The subscription used shorter initial notes with
a delayed held C, fitting three beats. Juan previously answered: “Solved, it
was D then D flat then C”; the edition uses a single six-eighth melodic line.
The useful optional clarification is whether the D/D-flat reading was clear ink
or an adopted reconstruction, and whether the open low shape under the ascending
run represents a held voice. Preserve the adopted edition either way.

## Evaluation consequence

The nine-bar source-note metric includes bar 2, but excludes bars 16 and 44.
Bar 16 is already documented as a reconstruction. Neither excluded bar caused
the reported nine-bar source F1 deficit. Source-audited disagreements also remain
at global bars 3, 4, 5 and 42, so resolving the three rhythm warnings will not by
itself establish transcription quality. Those source labels are model-audited,
not a new human-confirmed whole-score reference.

In particular, a beat-perfect repair can still be musically wrong: `z2 ^C4` and
`D2 ^C4` both fill bar 16, but make different musical choices. Keep source
agreement, editorial choices and structural completion as separate measurements.

## Proposed next harness version

Preserve draft saves and the original time limits. A final submission should
require every structural warning to be corrected or have a structured,
source-linked disposition tied to that exact ABC hash and issue. The same
readiness check must drive tool feedback and API/subscription final status.

- **Corrected reading:** fix the notation and revalidate; the warning disappears.
- **Provisional reconstruction:** supply the best supported complete performance
  reading, record which event was inferred, the evidence, alternatives and
  uncertainty, and preserve the literal source reading separately.
- **Source exception:** retain a genuinely supported pickup, irregular bar or
  other intentional exception, explaining the evidence. “Unclear” alone should
  not silently waive a warning or turn it into a verified source exception.
- **Unresolved at deadline:** preserve the full best-effort draft and report
  `needs_review`; do not discard progress or retry indefinitely.

Reject stale, missing, duplicate and unmatched dispositions. Structural coverage
of an exception is machine-checkable; its musical correctness is not. Never
insert rests automatically merely to reach three beats. A warning disposition
does not grant human approval and must stay visible in the dashboard.

This policy is a recommendation, not implemented by this investigation. Future
tests should exercise unresolved meter warnings, actual corrections, supported
exceptions, stale hashes, pickup handling and partial-draft preservation. Freeze
the policy/prompt version before another paid trial; retain the current run as-is.

## Review packet

Local visual packet:
`http://127.0.0.1:8777/diagnostics/rumichaca_api_offline_20261029_v1/index.html`.
It includes the three crops, the comparison phrase, full-row context, prior
decisions and the original PDF. The frozen run, reference labels and 13 live
review files are hash-checked before and after this investigation.
