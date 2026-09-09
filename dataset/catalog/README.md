# Jaime Llanos collection inventory

`jaime_llanos_collection.json` records the inspected source identity and page ownership without distributing the source PDF. All PDF page numbers are physical, one-based page numbers. Manuscript page numbers and table song numbers are separate identifiers.

The 111-page source contains 15 pages of front matter and 96 score pages representing **95 named works**. Its typed table on PDF pages 10–12 has 98 named entries and two unnamed placeholder rows (99–100). The cover's “100 Melodías” is not an available-work count.

| Table IDs | Source titles | Manuscript pages | Finding |
| --- | --- | --- | --- |
| 35 | El Tato | 35–36 | Both present, physical PDF pages 50–51 |
| 37–38 | El vallenato; Entre amigos | 38–39 | Missing between physical pages 52 and 53 |
| 44 | Fita chiquita pasillo | 45 | Missing between physical pages 57 and 58; also explicitly marked missing in table |
| 99–100 | Dashes | None | Unresolved placeholders, not named missing works |

Every score-page header/footer was visually reviewed on 2026-09-07; title identity, manuscript labels, and sequence were reconciled together. Some labels are faint, so a mapping does not claim an independent high-confidence reading of every footer. The identified gaps are absent from this exact PDF; this says nothing about copies elsewhere. The original scanned index on physical page 15 covers manuscript pages through 63; the typed three-page table supplies the full named inventory.

`literal_source` preserves table spellings, punctuation, uncertainty markers, and wrapped cell line breaks. Embedded table text was extracted without OCR. Rows 31 and 66, above the ruled continuation-table region, were recovered from embedded text and checked visually. For existing sources, normalized titles and slugs come from the existing metadata; otherwise normalization is typographic only. Author claims and starred genres remain unverified. For example, literal “Santanás” remains separate from the existing `satanas` slug. The title/author on a manuscript may differ from its table entry and is not silently substituted.

`mapping_status` means `verified` (all listed manuscript pages have visually reconciled PDF pages), `missing` (named table work absent in the complete audited sequence), or `unresolved` (unnamed table placeholder). This is source-page coverage, not a claim of musical legibility, correct transcription, or production readiness.

References cover 16 existing individual PDF sources and 12 works with MusicXML paths. Paths are repository-relative and may be absent in another checkout because local and ignored artifacts are deliberately not copied. References were inventoried by path only: no protected transcription content was opened, no freeze was modified, and an XML link does not imply complete-song ground truth. This inventory grants no distribution clearance and does not change existing dataset/local-restricted classifications.

Validate the catalog and optionally verify the original PDF identity:

```sh
uv run python scripts/build_collection_inventory.py --source /path/to/all_melodies.pdf --output out/collection/inventory_report.json
```

The output is create-once; choose a new output name for a later report. Omit `--source` to check portable mapping consistency only, and omit `--output` to print the report. The helper uses the standard library and checks linked-file existence without reading their contents. It does not extract scores, transcribe music, or run evaluation. Splitting or review tools must consume the explicit `pdf_pages` list rather than infer an offset from song number.
