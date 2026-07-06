---
type: PRD
prd: "44"
title: "PRD-44: Vendor-Neutral Dictionary Schema Namespace"
description: Renames the MFDB dictionary's local extension tags from an application-branded namespace to a store-keyed vendor-neutral one, behind a backward-compatible parser, so MFDB is usable by software beyond ChiSurf.
status: done
phase: "unassigned"
resource: chisurf/core/mfdb/
tags: [prd, mfdb]
timestamp: '2026-07-05T00:00:00Z'
---

# Summary
PRD-44 de-brands the MFDB dictionary's local extension tags so MFDB is a general-purpose metadata/provenance store rather than one carrying a consuming application's name in its schema authority. The schema-mapping tags (`_chisurf_schema.*`) and the branded `_chisurf_parameter` category are renamed to a namespace keyed on the store itself (`_mfdb_schema.*` / `_mfdb_parameter`), which is already the table prefix everywhere and introduces no new brand. The change is mechanical but wide (~1000 call sites) and ships behind a backward-compatible parser that dual-recognizes both spellings for one cycle, so it lands in one pass without breaking existing databases; the materialized SQL is unchanged, so no DB migration is needed. The flrCIF table `flr_chisurf_parameter` and its item ids are intentionally out of scope, since renaming those would change generated SQL.

# Status
Done (unassigned phase, STATUS TABLE authoritative). All four Definition-of-Done items met (2026-06-27): parser dual-recognizes old and new tags, the dictionary is rewritten (0 legacy / 985 new), consumers reference only the new namespace, and a back-compat test is added (`test/fio/test_dictionary_namespace_backcompat.py`). 11 schema tests + prerequisites gate green. Out of scope (correctly untouched): the flrCIF *table/category* `flr_chisurf_parameter` and its `_flr_chisurf_parameter.*` item ids — renaming those would change materialized SQL, which this PRD forbids.

# Goal
The MFDB schema is generated from mmCIF dictionaries via local extension tags that carry the application name in their namespace:

- `_chisurf_schema.table_name` / `.column_name` / `.status` / `.foreign_key` — the schema-mapping tags read by `pdbx_metadata.py` and consumed by `schema_from_dictionary.py` + `dictionary_schema_map.py`.
- `_chisurf_parameter` — a separate branded category.

MFDB is intended to be a **shareable, software-agnostic** store (see PRD-41's dissemination strategy and the rule that `.dic` *entries* must be software-agnostic). A `chisurf`-branded tag namespace contradicts that: any other tool adopting MFDB inherits ChiSurf's name in its schema authority. The goal is to rename the local extension namespace to a vendor-neutral one keyed on the **store** (MFDB), not the **application** (ChiSurf).

# Decision
Rename the local extension namespace `_chisurf_*` → `_mfdb_*`:

| Old tag | New tag |
|---|---|
| `_chisurf_schema.table_name` | `_mfdb_schema.table_name` |
| `_chisurf_schema.column_name` | `_mfdb_schema.column_name` |
| `_chisurf_schema.status` | `_mfdb_schema.status` |
| `_chisurf_schema.foreign_key` | `_mfdb_schema.foreign_key` |
| `_chisurf_parameter.*` | `_mfdb_parameter.*` |

`_mfdb_` is the store's own identity (already the table prefix everywhere), so it is the natural neutral namespace and introduces no new brand.

# Current State (scope)
- **`chisurf/core/mfdb/data/mfdb_flr_ext.dic`** — ~985 `_chisurf_schema` occurrences (the only `.dic` using these tags; the bundled mmCIF/PDBx/IHM dictionaries use none).
- **`chisurf/core/mfdb/pdbx_metadata.py`** — the parser; reads the four `_chisurf_schema.*` tags (≈lines 317–323) into `DictItem.schema_*` fields.
- **`chisurf/core/mfdb/schema_from_dictionary.py`** — DDL generator; consumes `DictItem.schema_table` / `schema_column` / `schema_status` / `schema_foreign_key`. These Python attribute names are internal and need **not** change (decoupled from the tag spelling) — only the tag-parsing strings move.
- **`chisurf/core/mfdb/dictionary_schema_map.py`** — references the tags.
- **`_dictionary_cache.json`** — regenerated automatically (mtime-invalidated).

# Approach (backward compatible, one pass)
1. **Parser accepts both, prefers new.** In `pdbx_metadata.py`, recognize both `_mfdb_schema.*` and the legacy `_chisurf_schema.*` (and `_mfdb_parameter` / `_chisurf_parameter`) for one release, so old `.dic` copies and any pickled / third-party dictionaries keep parsing. New writes use `_mfdb_schema.*`.
2. **Bulk-rewrite the dictionary.** Mechanically replace `_chisurf_schema` → `_mfdb_schema` and `_chisurf_parameter` → `_mfdb_parameter` in `mfdb_flr_ext.dic`. Regenerate `_dictionary_cache.json`.
3. **Update the two consumers** (`schema_from_dictionary.py`, `dictionary_schema_map.py`) to read the new tag if they match on tag strings anywhere (most logic is on `DictItem` attributes and is unaffected).
4. **No DB migration.** The generated table/column names are unchanged — only the *dictionary tag namespace* changes, not the materialized SQL. Existing databases reconcile identically.
5. **Deprecation.** Keep the legacy-tag fallback for one cycle with a parse-time note; remove in a later cleanup once no `.dic` in the wild uses it.

# Tasks
1. Parser: dual-recognize `_mfdb_schema.*` + legacy `_chisurf_schema.*`; same for `_mfdb_parameter` / `_chisurf_parameter`.
2. Rewrite `mfdb_flr_ext.dic` tags; regenerate cache.
3. Sweep `schema_from_dictionary.py` / `dictionary_schema_map.py` for literal tag strings.
4. Grep the whole tree for `_chisurf_schema` / `_chisurf_parameter` to catch stragglers (tests, docs).

# Definition of Done
- [x] `grep -r _chisurf_schema chisurf/` returns only the parser's legacy-fallback branch (4 lines, `pdbx_metadata.py:317–323`).
- [x] Fresh reconcile produces the unchanged schema (table/column/FK set) — verified by the 11 `test_schema_from_dictionary.py` tests + the `test_setup_prerequisites` live⊇declared gate.
- [x] A `.dic` fragment using the **legacy** `_chisurf_schema.*` tag still parses (`test/fio/test_dictionary_namespace_backcompat.py`).
- [x] `_mfdb_schema.*` is the spelling in `mfdb_flr_ext.dic` (0 legacy / 985 new).

# Definition of Clean
The local extension namespace carries the store's identity (`mfdb`), not any consuming application's. Adding a new dictionary item or a new consuming tool requires no ChiSurf-named tag. Description prose stays software-agnostic (consistent with the existing `.dic` authoring rule).

# Relationships
- A precondition for [PRD-41](prd-41.md) (dissemination): the schema namespace must be vendor-neutral to disseminate the dictionary as a product.
- Keeps the `.dic`-as-schema-authority of PRD-19 (single canonical `.dic`-driven schema) and PRD-26 (model-driven data layer) brand-neutral.
- Rewrote the `mfdb_event_log` tags introduced by [PRD-43](prd-43.md) in its bulk sweep.
- Cleans up the extension namespace of the [MFDB (current)](/architecture/mfdb.md) toward the [MFDB target](/specs/mfdb.md).
