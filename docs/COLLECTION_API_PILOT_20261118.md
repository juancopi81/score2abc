# Three-score API production pilot — November 18, 2026

This pilot prepares new complete drafts for musician review with **GPT-6.2 Astra,
Extra High**, through the API. Each score has a fresh author and an independent
reviewer. It is a production delivery trial, not an accuracy benchmark: these
three works do not yet have musician-approved references.

| Work | Author + independent review | Estimated API cost | Delivery |
| --- | --- | --- | --- |
| Beatriz Eugenia | 28 min 12 sec | $5.68 | Installed Draft; 4 questions |
| Fiesta de la montaña | 38 min 37 sec | $7.93 | Installed Draft; 5 questions |
| Cúcuta 75 | 54 min 00 sec | $10.15 | Installed Draft; 6 questions |

Times measure the two model stages, excluding offline preparation, coordination,
presentation checks and human review. Costs include both model stages and their
in-stage revisions; these are estimates, not billed amounts. All six model stages
completed. Total estimated API cost is **$23.76**, within the $60 batch ceiling;
no automatic retry or additional paid run was launched.
All 378 responses retained the full 25,000-token output allowance; no stage hit
its time or budget limit.

The immutable local plan and raw runs are under
`out/collection/api_batch_20261118_v1`. The
[delivery dashboard](http://127.0.0.1:8777/collection_api_20261118_v1/index.html)
shows each work's readiness and remaining questions. The
[ordered musician review plan](REVIEW_PLAN_20261118.md) remains La Chata, Estrella
del Caribe, then Celina; new drafts join the backlog.

## Fixed execution policy

- Model `gpt-6.2-astra`, reasoning `xhigh` on every turn; source-only tools and
  `completion_gate_v1`, followed by the v2 independent review instructions.
- A fresh reviewer sees its own PDF and sealed author draft/decisions. Neither
  stage receives ground truth, prior-score answers, external recordings or
  coordinator source adjudications.
- Maximum 60 minutes for the author and 30 for the reviewer; up to 25,000 output
  tokens per response. The existing adapter may lower that allowance near its
  spending ceiling; any reduction or early stop must be reported.
- Three sequential author/reviewer pairs, no automatic retries. The ceiling is
  $20 per pair ($12 author + $8 reviewer), $60 total. Estimates use the pinned
  runner pricing and measured usage; they are not an invoice or account balance.
- Generic full-score tasks request row coverage and an explicit performed route.
  The task wording includes collection handoff requirements, so this is not a
  matched rerun of the earlier Rumichaca benchmark.

The source inventory found no prior transcription/evaluation output or existing
live draft for these selected works. This does not claim pretraining novelty.
All three original source PDFs, prompts, tools and runtime outputs are hash-pinned.

## Delivery distinctions

**Beatriz Eugenia:** eight source rows, 53 source bars, 83 engraved bars and 128
performed bar visits. Two explicit 15-bar A return copies remain in the engraving.
The independent source count initially treated two non-boundaries as bars; visual
inspection corrected the count from 55 to 53. The reviewer's route array omitted
one C pass although its narrative and ABC agreed on both endings. A separate
bookkeeping erratum reconciles the array; the raw sealed draft is unchanged.
Boxed sections and Spanish directions are a separate presentation derivative,
verified to preserve the complete musical content, playback and highlighting.
When a musician corrects an A passage, the coordinator must check all three
written copies and synchronize the correction where appropriate. They are not
automatically linked in the editor.

**Fiesta de la montaña:** eight source rows, 47 source/written bars and 141
performed visits. Its proposed two-D.C. route remains an explicit musician
question. A presentation derivative preserves the music and raw chord text.
The adopted `C7dim/Eb` hypothesis at written bar 38 requires a separate encoding
normalization to the supported `Cdim7/Eb`; the root reading remains uncertain.
Unlike the presentation changes, this makes previously silent accompaniment
audible on three visits. Its proof must isolate that change and preserve the
melody, duration, route, timing and every other accompaniment event.

**Cúcuta 75:** the author and independent reviewer both completed. The raw score
has 57 written bars and 144 performed visits, including four passes through the
16-bar closing Vivo. Rendering and audio have no parser/accompaniment warnings.
A conditional G tie across bars 8–9 is represented as a slur plus prose, so the
current player reattacks it; this is a flagged articulation limitation, not exact
source playback. The unmeasured G7 arpeggio remains a text instruction, and the
Moderato/Lento/Vivo words do not yet change the numeric playback tempo. Final
source-only crop checks distinguish a lower note stem from a barline in row 7
(five measured spaces) and an accidental from a barline in row 8 (three measured
spaces followed by two instruction panels). The provisional 63/61 source counts
were revised to 57 with the evidence and earlier uncertainty retained. No missing
musical interval was identified. A display-only correction removes an empty
second-ending bracket from the Vivo, retaining its numbered D.S. instruction.
The proof explicitly records that one ending-label metadata change while
preserving all musical events, route, timing, highlighting and accompaniment.

All three final selections passed guarded installation, exact ABC export and
available-history checks. They are now in the live desk as Drafts with zero human
review time. The original 87 review/history files and manifest prefix are
byte-identical. The post-pilot 22-work backup was restored and verified; see the
[checkpoint](CHECKPOINT_20261118.md). No musical accuracy score is claimed.

## What passing the checks establishes

Delivery requires intact raw seals, source-row coverage, a declared source-to-score
map, a finite executable route, valid notation and recorded meter exceptions,
rendering/audio checks, and a successful disposable save/reopen/export round trip.
Presentation changes must preserve full musical events and playback. Any encoding
change gets its own explicit proof and provenance.

These checks establish a usable, traceable draft; they do not prove every pitch,
chord or editorial choice matches the manuscript. Material uncertainties remain
in the desk's review questions. New scores stay **Draft**, with zero credited human
review time. Existing musician drafts and histories must remain byte-identical.
Raw API traces stay private and separate from the served handoff and future public
dataset. No further paid batch follows automatically.
