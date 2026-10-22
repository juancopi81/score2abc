# Astra three-score diagnostic — 2026-09-09

This follow-up uses Carrizal system 4, Gato'e Fique system 3 and Coqueteos
system 2. All three already have independent human transcriptions and have
previously contributed development evidence. These are excerpts, not three
completed collection scores.

The useful result is a viable assisted-drafting path with specific remaining
musical errors. Carrizal matches 6 of 8 physical measures exactly; Coqueteos
matches 5 of 7. Gato'e Fique improves the pitch sequence substantially, while
overlapping voices require a richer full-event scoring contract. The context
limitation below prevents interpreting this as a blind accuracy benchmark.

## Results

| Excerpt | Strict written-note F1 | Exact physical measures, including rests | Remaining issue |
| --- | ---: | ---: | --- |
| Carrizal S4 | 90.0% | 6/8 | Two pitch disagreements; all note timing and rests match |
| Gato'e Fique S3 | Not supported | Not supported | Overlapping voices/rests in the reference; draft rhythm and voice readings remain uncertain |
| Coqueteos S2 | 91.8% | 5/7 | Rhythm in measure 4; missing final note and an extra rest in measure 7 |

F1 requires pitch, onset and duration to match together; it is not generally the
percentage of symbols read correctly. Carrizal matches 18 of 20 written notes.
Coqueteos matches 28 of 31 reference notes exactly, with 30 notes submitted.
There is **no pooled three-score accuracy figure** because Gato'e Fique's
full-event reference contains a rest in one voice while another voice sounds,
which the existing strict event validator does not support. This is an evaluator
limitation, not evidence that the manuscript is wrong.

Gato'e Fique's supported secondary pitch/count comparison improves ordered-pitch
alignment from 22.2% to 85.2%, and exact pitch sequences from 0/6 to 4/6 measures.
Its note count remains 25 predicted versus 26 reference notes; equal counts with
the old baseline do not imply equal note identities. Neither pitch-only result
validates rhythm, rests, polyphonic timing or grace-note interpretation.

The frozen Coqueteos comparison is directly reproducible under its six-crop
mapping: strict note F1 rises from 36.4% to 91.8%, and exact old crops from 0/6 to
4/6. Native physical-measure scoring above keeps seven actual bars. Carrizal's
old baseline reproduces at 32.6% F1, but it merges a tie across bars while the new
contract retains written tie segments. Its numerical side-by-side comparison is
therefore withheld. All three historical baselines reproduce without changing
their stored artifacts.

The remaining disagreements are located for later source review:

- Carrizal m2, beat 0.5: Bb4 in the draft versus A4 in the reference; m4, beat
  2: F#4 versus G#4. The first error was not clearly flagged by the model.
- Coqueteos m4: dotted-eighth/sixteenth versus two ordinary eighths; m7:
  dotted-eighth rest versus a Bb4 dotted half. The model marked both readings
  uncertain. Keep the ambiguous last glyph attached to its source crop for
  adjudication; reference agreement alone does not prove a manuscript reading.
- Gato'e Fique m5: an extra inferred grace note; m6: a wrong lower opening pitch
  and two missing upper notes. These fall in its stated rhythm/voice uncertainties.

Self-reported uncertainty helps prioritize review but misses at least one
Carrizal error. It cannot replace a complete source check of the delivered score.

## Interpretation limit

The intended source-only protocol failed its initial-context audit. Each fresh,
no-history subagent received its assigned original PDF and no reference files,
but the host also supplied this repository's `AGENTS.md`. Those instructions
contain target measure counts, key context and previous experiment outcomes.
The agents did not access forbidden files through tools; that narrower check
does not establish that their complete input context was source-only.

Keep these outputs as **consumed assisted diagnostics**. Their numerical scores
can describe agreement with the references, but they cannot establish blind
recognition accuracy or an input-only causal advantage over the old pipeline.
The original protocol is preserved alongside `protocol_deviation.json`; no
reference-inspired corrections or replacement attempts are made.

The earlier Aviador original-PDF task ran outside the repository and passed
this additional context check. Its historical-crop subagent had the same
limitation found here. The [earlier report](ASTRA_BENCHMARK.md) now includes that
erratum without changing its frozen outputs or scores.

## Outputs and independent checks

| Excerpt | Physical measures submitted | Worker elapsed time |
| --- | ---: | ---: |
| Carrizal, system 4 | 8 | 8 min 26 sec |
| Gato'e Fique, system 3 | 6 | 9 min 30 sec |
| Coqueteos, system 2 | 7 | 10 min 56 sec |

All workers used `gpt-6-astra` with extra-high reasoning and ran in parallel.
Times exclude coordinator work and human review. They consumed Codex account
usage; no external paid API calls were made. Cumulative cached/uncached usage
is recorded in each access-audit file; it is not a dollar-cost estimate.

An independent abc2svg parse matches all ordinary written note/rest events
between each ABC and its structured submission. Gato'e Fique's grace note also
matches pitch and attachment position, but its nominal grace duration is not
verified by that check. Invisible voice spacers are excluded. There are no ABC
parser errors. The coordinator visually checked all three target-system images
for source identity and retained context; that is not musical approval.

## Scoring contract

All original inputs, submissions, inspection artifacts and access logs are
archived under `out/collection/astra_pilot_v2/`. Scoring verifies every submission
and reference hash before opening reference events, then preserves written
durations, explicit rests and tied continuations as separate written events.

Native metrics use physical measures. Carrizal and Coqueteos also have fixed
historical mappings that combine two physical measures into one old crop. Their
old baseline summaries must reproduce exactly; comparisons are withheld if
written and historical canonical tie semantics differ. Gato'e Fique's historical
benchmark supports only note count and ordered pitch alignment, so its full-event
result is never presented as an improvement over an unavailable baseline.

Malformed, extra or duplicate measures invalidate that target's headline result.
Missing measures count as empty predictions. Unsupported reference geometry is
reported separately from an invalid draft. A failed target is never silently
dropped to create a favorable pooled three-score result. Chords, spelling,
slurs, ties, form and editorial correctness still need their own source review.

The evaluator is `scripts/experiments/evaluate_astra_collection_pilot.py` with
synthetic checks for written ties/rests/chords, fixed physical mappings,
malformed submissions and verification before truth access.

Two recorded evaluator corrections were required before any report was
published: apply the original Coqueteos measure-ID adapter, then reproduce the
five meter fields added by its historical evaluator. The original evaluator,
both correction records and the first corrected implementation are preserved.
No inference output, reference event or metric definition was changed.

## Delivery decision

Proceed to a complete Carrizal draft using the original page plus system crops,
then perform a source-based form pass for repeats, casillas, ties and chord
symbols before handing it to Juan. Carry disagreements as located questions;
do not replace source readings with plausible music. Follow with the poorer
Rumichaca page to measure how much review effort remains on degraded material.

The next complete-page blind attempt should run in a separate projectless task
whose actual initial context is checked before inference. Keep that evaluation
separate from drafting consumed material in the repository. The practical
delivery criterion remains a source-checked complete score and measured human
correction effort, rather than another isolated excerpt score.

Juan's current action is only the repaired B-natural in Aviador system 8,
measure 6. These three diagnostics do not create a new ground-truth assignment
or replace the incomplete drafts in his review desk.
