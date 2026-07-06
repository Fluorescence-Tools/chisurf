---
type: PRD
prd: "02c"
title: "PRD-02c: Aligning ChiSurf MFDB Export to flrCIF"
description: Map ChiSurf's internal parameter short names to canonical flrCIF dictionary items on export
status: done
phase: "foundation"
resource: modules/mfdb/src/mfdb/
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

# Goal
Ensure that parameters exported from ChiSurf to `mfdb` strictly adhere to the
flrCIF standard. The flrCIF dictionaries (and the local `mfdb_flr_ext.dic`
extension) are the **canonical source of truth** for parameter definitions;
ChiSurf's internal short abbreviations must be mapped to these standard
definitions so that stored data matches flrCIF exactly.

# Background
ChiSurf internally uses short abbreviations (e.g. `E_FRET`, `bg`) defined in
`chisurf/core/settings/constants/fitting_parameters.json` with their
descriptions. Since these are not exclusively "fitting" parameters, the registry
should be renamed to `parameter_registry.json`. flrCIF is the public standard for
archiving fluorescence data, and MFDB's schema mirrors it, so exports must match
strict flrCIF naming. To avoid duplication, the `.dic` files remain canonical:
the JSON registry is internal-only and maps internal short names to canonical
`.dic` parameter IDs rather than dictating `.dic` contents.

# Requirements
1. **`.dic` as canonical source** — the standard flrCIF dictionary (and
   `mfdb_flr_ext.dic` for extensions) is the absolute source of truth for
   parameter definitions, names, and descriptions; the JSON registry maps
   internal short names to flrCIF parameter IDs.
2. **Add missing parameters to `.dic`** — identify ChiSurf parameters absent
   from the standard dictionary and add them to
   `modules/mfdb/src/mfdb/data/mfdb_flr_ext.dic`, strictly following flrCIF
   formatting and naming. The short names/descriptions in the JSON registry help
   draft the initial entries, but the `.dic` then becomes canonical.
3. **Map internal short names to flrCIF** — add a `"flrcif_item_id"` field to
   the JSON registry (e.g. `"flrcif_item_id": "_flr_chisurf_parameter.E_FRET"`).
4. **Update export logic** — when a short-named parameter is exported from
   ChiSurf to MFDB, write it under its canonical flrCIF identifier and category.

# Implementation steps
**Step 1 — Rename and audit the registry.** Rename
`fitting_parameters.json` → `parameter_registry.json` and update all references:
`chisurf/core/settings/__init__.py` (loaded filename + `fitting_parameters` →
`parameter_registry` variable); `chisurf/core/parameter.py` (~line 372,
`getattr(chisurf.core.settings, "parameter_registry", {})`); the
`build_tools/dev_utils/` export/fill scripts; and
`docs/parameter_registry_tools.rst`.

**Step 2 — Extend `mfdb_flr_ext.dic`.** Write
`build_tools/dev_utils/align_flrcif_parameters.py` to read
`parameter_registry.json`, check each short name against standard flrCIF, and
auto-generate + append `.dic` entries for the missing ones. Example:

```text
save__flr_chisurf_parameter.E_FRET
   _item.name                "_flr_chisurf_parameter.E_FRET"
   _item.category_id         flr_chisurf_parameter
   _item_type.code           float
   _chisurf_schema.table_name  flr_chisurf_parameter
   _chisurf_schema.column_name e_fret
   _item_description.description
;     Apparent FRET efficiency parameter E_FRET (0e00..1).
;
```

Ensure a matching category definition `save_flr_chisurf_parameter` exists.

**Step 3 — Create the mapping in `parameter_registry.json`.** The same script
modifies the JSON in place, adding `"flrcif_item_id"` to each parameter (e.g.
`E_FRET` → `_flr_chisurf_parameter.E_FRET`). To avoid duplication, prefer pulling
descriptions dynamically from the loaded `.dic` schema; the JSON `"description"`
field can become a fallback or be removed.

**Step 4 — Update export logic.** In the export mechanisms that save fitting
parameters and model results to MFDB (typically `pdbx_metadata.py`,
`dictionary_schema_map.py`, or the MFDB SQLAlchemy models), write each model
parameter using its canonical `flrcif_item_id` so the exported database mirrors
flrCIF.

**Step 5 — Tests (`test/fio/`).** Verify that all mapped parameters translate to
their `flrcif_item_id`, that the extended `mfdb_flr_ext.dic` parses with the CIF
parser, and that exported MFDB data validates against the combined flrCIF
dictionaries.

# Acceptance criteria
- [ ] `fitting_parameters.json` renamed to `parameter_registry.json` and all references updated
- [ ] `mfdb_flr_ext.dic` contains standard-compliant definitions for all ChiSurf parameters missing from core flrCIF
- [ ] `parameter_registry.json` has a `"flrcif_item_id"` field linking each internal short name to the canonical `.dic` item
- [ ] Parameter-description duplication minimized/eliminated by treating `.dic` files as canonical
- [ ] The ChiSurf → MFDB export correctly translates internal short names into standard flrCIF identifiers

As built, the alignment injected `flrcif_item_id` mappings from ChiSurf's
internal abbreviations to 219 generated standard-compliant entries in
`mfdb_flr_ext.dic`, and export routes through the `chinet_adapter.py` lookup so
internal abbreviations do not pollute external archives.
`test/fio/test_flrcif_alignment.py` passes (13/13).

# Relationships
- Builds on the dictionary API from [PRD-02a](prd-02a.md) and the sample model from [PRD-02](prd-02.md).
- Enforces dictionary-as-authority for export in the [MFDB (current)](/architecture/mfdb.md) store toward the [MFDB target](/specs/mfdb.md).
