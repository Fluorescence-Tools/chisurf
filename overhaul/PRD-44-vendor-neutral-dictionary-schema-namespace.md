# PRD-44: Vendor-Neutral Dictionary Schema Namespace

> **Scope:** De-brand the MFDB dictionary's local extension tags so MFDB is a
> general-purpose metadata/provenance store usable by software beyond ChiSurf.
> Mechanical but wide (≈1000 call sites); ships behind a backward-compatible
> parser so it can land in one pass without breaking existing databases.

## Goal

The MFDB schema is generated from mmCIF dictionaries via local extension tags
that carry the application name in their namespace:

- `_chisurf_schema.table_name` / `.column_name` / `.status` / `.foreign_key`
  — the schema-mapping tags read by `pdbx_metadata.py` and consumed by
  `schema_from_dictionary.py` + `dictionary_schema_map.py`.
- `_chisurf_parameter` — a separate branded category.

MFDB is intended to be a **shareable, software-agnostic** store (see PRD-41's
dissemination strategy and the rule that `.dic` *entries* must be software
agnostic). A `chisurf`-branded tag namespace contradicts that: any other tool
adopting MFDB inherits ChiSurf's name in its schema authority. Rename the local
extension namespace to a vendor-neutral one keyed on the **store** (MFDB), not
the **application** (ChiSurf).

## Decision

Rename the local extension namespace `_chisurf_*` → `_mfdb_*`:

| Old tag | New tag |
|---|---|
| `_chisurf_schema.table_name` | `_mfdb_schema.table_name` |
| `_chisurf_schema.column_name` | `_mfdb_schema.column_name` |
| `_chisurf_schema.status` | `_mfdb_schema.status` |
| `_chisurf_schema.foreign_key` | `_mfdb_schema.foreign_key` |
| `_chisurf_parameter.*` | `_mfdb_parameter.*` |

`_mfdb_` is the store's own identity (already the table prefix everywhere), so it
is the natural neutral namespace and introduces no new brand.

## Current State (scope)

- **`chisurf/core/mfdb/data/mfdb_flr_ext.dic`** — ~985 `_chisurf_schema`
  occurrences (the only `.dic` using these tags; the bundled mmCIF/PDBx/IHM
  dictionaries use none).
- **`chisurf/core/mfdb/pdbx_metadata.py`** — the parser; reads the four
  `_chisurf_schema.*` tags (≈lines 317–323) into `DictItem.schema_*` fields.
- **`chisurf/core/mfdb/schema_from_dictionary.py`** — DDL generator; consumes
  `DictItem.schema_table` / `schema_column` / `schema_status` /
  `schema_foreign_key`. These Python attribute names are internal and need **not**
  change (decoupled from the tag spelling) — only the tag-parsing strings move.
- **`chisurf/core/mfdb/dictionary_schema_map.py`** — references the tags.
- **`_dictionary_cache.json`** — regenerated automatically (mtime-invalidated).

## Approach (backward compatible, one pass)

1. **Parser accepts both, prefers new.** In `pdbx_metadata.py`, recognize both
   `_mfdb_schema.*` and the legacy `_chisurf_schema.*` (and `_mfdb_parameter` /
   `_chisurf_parameter`) for one release, so old `.dic` copies and any pickled /
   third-party dictionaries keep parsing. New writes use `_mfdb_schema.*`.
2. **Bulk-rewrite the dictionary.** Mechanically replace `_chisurf_schema` →
   `_mfdb_schema` and `_chisurf_parameter` → `_mfdb_parameter` in
   `mfdb_flr_ext.dic`. Regenerate `_dictionary_cache.json`.
3. **Update the two consumers** (`schema_from_dictionary.py`,
   `dictionary_schema_map.py`) to read the new tag if they match on tag strings
   anywhere (most logic is on `DictItem` attributes and is unaffected).
4. **No DB migration.** The generated table/column names are unchanged — only the
   *dictionary tag namespace* changes, not the materialized SQL. Existing
   databases reconcile identically.
5. **Deprecation.** Keep the legacy-tag fallback for one cycle with a parse-time
   note; remove in a later cleanup once no `.dic` in the wild uses it.

## Tasks

1. Parser: dual-recognize `_mfdb_schema.*` + legacy `_chisurf_schema.*`;
   same for `_mfdb_parameter` / `_chisurf_parameter`.
2. Rewrite `mfdb_flr_ext.dic` tags; regenerate cache.
3. Sweep `schema_from_dictionary.py` / `dictionary_schema_map.py` for literal
   tag strings.
4. Grep the whole tree for `_chisurf_schema` / `_chisurf_parameter` to catch
   stragglers (tests, docs).

## Definition of Done

- [ ] `grep -r _chisurf_schema chisurf/` returns only the parser's legacy-fallback
      branch (and its test).
- [ ] Fresh `:memory:` reconcile produces byte-identical schema to before the
      rename (table/column/FK set unchanged).
- [ ] A `.dic` fragment using the **legacy** `_chisurf_schema.*` tag still parses
      (back-compat test).
- [ ] `_mfdb_schema.*` is the spelling in `mfdb_flr_ext.dic` and all new entries.

## Definition of Clean

The local extension namespace carries the store's identity (`mfdb`), not any
consuming application's. Adding a new dictionary item or a new consuming tool
requires no ChiSurf-named tag. Description prose stays software-agnostic
(consistent with the existing `.dic` authoring rule).

## Relationship

- **PRD-41** (FDB4ChemBio access-layer / dissemination): a vendor-neutral schema
  namespace is a precondition for disseminating the dictionary as a product.
- **PRD-19** (single canonical `.dic`-driven schema) / **PRD-26** (model-driven
  data layer): both treat the `.dic` as schema authority; this keeps that
  authority brand-neutral.
- **PRD-43** (history/MFDB projection): introduced `mfdb_event_log` using the
  current tags; it is rewritten by this PRD's bulk sweep along with everything
  else.
