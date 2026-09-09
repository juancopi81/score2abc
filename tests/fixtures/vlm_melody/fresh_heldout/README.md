# Carrizal Fresh Heldout Evidence

These files preserve the consumed Carrizal system-4 one-shot gate outside
`dataset/musicxml/`. They are benchmark evidence and training/diagnostic material;
the fixture MusicXML backend must not treat this partial-system transcription as a
complete work.

- `carrizal_system_004.musicxml`: independently authored eight-measure transcription.
- `carrizal_system_004_truth.jsonl`: seven automatic-crop truth rows; crop 2 maps to
  physical measures 2 and 3.
- `carrizal_system_004_truth_import.json`: source, freeze, mapping, truth, and evaluator
  hashes captured before scoring.
- `carrizal_system_004_evaluation.json`: immutable one-shot metrics and per-crop results.

The predictions remain in the original gitignored `out/` freeze. Do not regenerate
them from these truth files or report Carrizal system 4 as heldout again.
