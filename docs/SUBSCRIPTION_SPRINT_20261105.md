# Subscription preparation sprint — November 5, 2026

Status: seven existing drafts strengthened and three fresh drafts independently
audited, validated and saved. The live desk now contains six musician-approved
editions and ten pending drafts. No new musical approval is granted by this sprint.

The musician-approved editions remain Amalia, Atardecer, Aviador, Carrizal,
Coqueteos and Rumichaca. The seven existing drafts receive targeted source and
form work, followed by a separate Astra reader. Work uses subscription agents;
no paid API generation requests are part of this sprint.

## Delivery contract

- Preserve exact current drafts, histories, source readings and alternatives.
- Record source corrections separately from editorial reconstructions and
  proposed performance routes. Compare every source row during independent review.
- Test the actual encoded route and bounded playback, including ties and lower
  voices, then validate rendering and save/reopen/export in a disposable desk.
- Replace only an untouched model draft whose revision and full-file hash still
  match the starting snapshot. Archive through the normal review save path,
  retain earlier notes, and add no human review time.
- Keep all seven scores Draft until Juan accepts their complete music and form.
  An unresolved source region remains a coverage question even when its known
  bars and provisional route pass structural checks.

The local packet is
`out/collection/subscription_sprint_20261105_v1/`. It contains the starting
13-score preservation manifest, exact snapshots and a verified ZIP of 61 saved
review/history files. These local score artifacts are excluded from Git;
committing code and this report does not publish or back them up remotely.

## Existing drafts delivered

All 64 manuscript rows were independently reread. All seven candidates passed
syntax, bounded route-aware playback, exact performed-route comparison,
coordinator engraving inspection and disposable save/reopen/export. Installation
used the normal review history path after checking each full baseline hash and
revision. The six accepted reviews remain byte-identical. Human review time added:
zero. The local `installation_receipts.json` records exact delivery hashes.

| Score | Saved revision | Written / performed spans | Main remaining decision |
| --- | --- | --- | --- |
| Gato'e Fique | 4 | 42 / 112 | Ottava scope and editorial final stop |
| Estrella del Caribe | 3 | 52 / 97 | Chromatic chords and short ending; count includes pickup |
| Chispazo | 4 | 55 / 128 | Final return grammar and ending connections |
| Sobre el Humo | 3 | 49 / 112 | Final stop; 23 lower heads restored from source |
| Acuata | 3 | 54 / 120 | Five irregular spans and return/bridge choices |
| La Chata | 4 | 51 / 141 | Overwritten S2m1; terminal text reading, with 52-span alternative preserved |
| Entusiasmo | 3 | 85 / 143 | Unmapped terminal cluster may contain music; complete recovery is blocked |

Sobre el Humo's audit rejected the author's earlier route: the manuscript's
"1–2 / 3 Trío" endings require two B visits before the Trío. The restored
E-position lower heads sound E-flat under the active two-flat signature.
Estrella S3m1's third note is changed from C-natural to C-sharp after comparing
written accidental shapes. Acuata gains the visible D7 at S4m1. Chispazo's S7m2
slur ends after the third note. These are model-proposed source corrections,
pending musician review; editorial returns remain explicitly labeled.

The seven declared maps/routes also pass the new offline checker. Entusiasmo's
unknown terminal region remains a completeness blocker despite that bookkeeping
pass. La Chata's excluded tail remains a provisional text-versus-music decision.
Neither is silently promoted to complete, verified source recovery.

Open the local handoff at `http://127.0.0.1:8780/`. If servers need restarting:

```sh
uv run python main.py review out --port 8766
uv run python -m http.server 8780 --bind 127.0.0.1 \
  --directory out/collection/subscription_sprint_20261105_v1
```

Run these in separate terminals. Start with Gato'e Fique, then Estrella. Answer
the focused questions in review notes, remove only resolved open questions,
save partial work as Draft, and mark Reviewed after accepting the full score
and performed form. Entusiasmo's final cluster deserves a targeted source decision
before a full listening pass.

## Reusable offline checks

The new [coverage/form sidecar](SCORE_COVERAGE_FORM.md) checks explicit mappings
and complete performed sequences using the existing parser. It does not infer
source identities from similar music and does not change the subscription
workflow or completion gate.

The nine-case regression preserves all six accepted edition count/route pins.
It additionally verifies Aviador's declared 62 source bars plus two editorial
insertions. The other five editions have no invented source mapping. Both saved
gated Rumichaca drafts pass their declared coverage bookkeeping but fail the
adopted performance route: 117 and 116 visits instead of 132. The earlier
132-visit API draft passes that route comparison. All 903 frozen reference files
were verified before the report. These are consumed regressions, not new model
accuracy results.

Focused validation: 69 tests passed, Ruff and Black passed, and a separate
read-only code review reproduced all nine report cases without findings.

## Resource decision

API-credit expiry was checked separately from subscription-usage limits. This
sprint uses the subscription; API credit is reserved for a later bounded
experiment. Account-specific billing observations remain in the local packet.

After the seven-draft delivery passed, the optional three-score batch started:
Adentro veteranos, Albores de junio and Baby. Existing source-packet hashes were
verified; each receives a fresh Astra author and separate source reviewer.
Candidate selection and an author draft alone are not delivery or accuracy results.

## Three new drafts delivered

Each complete author draft received a separate GPT-6 Astra / Extra High source
pass across the full page and all nine musical rows. All three passed source-map
bookkeeping, exact adopted-route comparison, bounded per-head audio checks,
coordinator browser engraving inspection, and disposable save/reopen/export.
New live reviews were seeded and saved through the normal review-history path;
all are revision 2, Draft, with zero human time. The existing 13 manifest entries
were retained unchanged and three new entries appended. Source PDFs are exact
copies of the hash-verified collection packets. No canonical inference or frozen
evaluation artifact was changed.

| New score | Written / performed spans | Main review focus |
| --- | --- | --- |
| Adentro veteranos | 50 / 97 | Lower C5 correction at row 2 bar 5; accidental, held chords and final cadence |
| Baby | 61 / 128 | Corrected row-7/8 pitches; two ambiguous chords, canceled bridge and editorial final stop |
| Albores de junio | 68 / 224 | Dot reconstructions, long ties and interaction of D.S., D.C. and Trío |

Adentro's audit changes one lower head from B4 to C5. Baby's audit records nine
per-head pitch changes, including accidental carry effects; its row-2 final
C-sharp and row-8 E-natural/A chord remain uncertain visible-ink readings with
the author's exact alternatives preserved. Baby has no source chord labels and
receives no invented accompaniment.

Albores retains the author's 224-visit route as the primary provisional edition.
The reviewer proposed a 209-visit route that omits the second D.S. detour; both
are fully executable and independently checked. The coordinator retained the
longer version because the explicit segno instruction and B-ending pickup support
returning to A. The source does not settle jump reactivation, so this disagreement
remains a musician decision, not a claimed source correction. Its two-beat opening
pickup is not padded. Audition tempos are editorial, not newly recovered markings.

Together, the sprint audits cover 91 manuscript rows. The local handoff links
each live draft, original PDF, audited notation, focused questions, close-ups,
decision record and earlier candidate. `execution_manifest.json` pins model,
instructions, renderer and playback code; `new_installation_receipts.json`
records the three new saved-review hashes. `delivery_checkpoint.zip` and its
manifest preserve the final reviews/histories and text/JSON decision packets
locally, separately from Git. These are production drafts, not three new ground
truths or a heldout model benchmark.
