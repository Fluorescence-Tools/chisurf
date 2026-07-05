---
type: PRD
prd: "02c"
title: "PRD-02c: Aligning ChiSurf MFDB Export to flrCIF"
description: Map ChiSurf's internal parameter short names to canonical flrCIF dictionary items on export
status: done
phase: "foundation"
resource: overhaul/PRD-02c-flrcif-alignment.md
tags: [prd, mfdb]
timestamp: '2026-07-05T00:00:00Z'
---

# Summary
Ensures parameters exported from ChiSurf to MFDB strictly adhere to the flrCIF
standard, with the flrCIF dictionaries (plus the local `mfdb_flr_ext.dic`
extension) as the canonical source of truth for parameter definitions. ChiSurf's
internal short abbreviations (e.g. `E_FRET`, `bg`) are mapped to canonical
dictionary item IDs via a `flrcif_item_id` field in a renamed internal parameter
registry, and parameters missing from the standard dictionary are added to the
extension `.dic`. Export logic then emits each parameter under its canonical
flrCIF identifier and category rather than the internal short name.

# Status
Done. The internal registry maps to dictionary items; the extension dictionary
carries ChiSurf-specific parameters absent from the standard.

# Relationships
- Builds on the dictionary API from [PRD-02a](prd-02a.md) and the sample model from [PRD-02](prd-02.md).
- Enforces dictionary-as-authority for export in the [MFDB (current)](/architecture/mfdb.md) store toward the [MFDB target](/specs/mfdb.md).

# Source
- Primary: `overhaul/PRD-02c-flrcif-alignment.md`
