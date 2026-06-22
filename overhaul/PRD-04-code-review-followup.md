# PRD-04 Prerequisite — Follow-up Review (Round 2)

Re-review after the fixes for `PRD-04-code-review.md` / `PRD-04-punch-list.md`.
Governing rule: **the `.dic` dictates the schema — no hardcoded SQL.**

Reviewed (2026-06-20, second pass):

- `chisurf/core/mfdb/schema_from_dictionary.py` (new generator)
- `chisurf/core/mfdb/schema.py` (fresh-DB wiring, v30 migration, tttr_reading)
- `chisurf/core/mfdb/pdbx_metadata.py` (`schema_foreign_key` support)
- `chisurf/core/mfdb/data/mfdb_flr_ext.dic` (`_chisurf_schema.foreign_key`)
- `chisurf/gui/widgets/wizard/tttr_channeldefinition/tttr_detector_setups.py`
- `chisurf/plugins/core/mfdb_admin/gui/entity_schema.py`
- `chisurf/plugins/core/mfdb_admin/backend/setup_services.py`
- `test/fio/test_setup_prerequisites.py`

Tests: `test_setup_prerequisites.py` + `test_burst_pipeline_mfdb.py` → **19 passed**.

## Verdict

**All blockers from round 1 are resolved.** The `.dic` now genuinely dictates the
schema for the setup/detector/window tables: the DDL is generated from the
dictionary, the timing columns exist by construction, the validation gate is
total-coverage with no allow-list, the wizard reads the structured tables, and
the residual hardcodes are gone. Remaining items are minor/robustness only.

## Blocker status

### ✅ Blocker 0 — DDL generated from the dictionary

- New `schema_from_dictionary.py` builds `CREATE TABLE` / `ALTER TABLE` from
  `DictItem` metadata: column ← `_chisurf_schema.column_name`, type ←
  `_item_type.code` via `TYPE_CODE_SQL_MAP`, `NOT NULL` ← `mandatory`, PK ←
  `DictCategory.key_item`, FK ← `_chisurf_schema.foreign_key`, `DEFAULT` ←
  `_item_default.value`, audit columns by convention.
- Fresh-DB path uses **placeholder** entries (`__DICT_DDL__mfdb_setup_..._`,
  `schema.py:874-875`) replaced by generated DDL at import (`:1131-1142`).
- v30 migration calls the same generator (`:3452-3456`) — no pasted DDL.
- The hand-written child-table `CREATE TABLE` strings are gone; the column set
  now lives once, in the `.dic`. The three round-1 drift sites collapsed to one.

### ✅ Blocker 1 — timing-column divergence + weakened gate

- `_chisurf_schema.foreign_key` was added to `DictItem` (`pdbx_metadata.py:48`,
  parsed `:323-324`) and to the `.dic` (`mfdb_setup(setup_id) ON DELETE
  CASCADE`), so the FK is expressed in the dictionary, not hardcoded.
- `macro_time_resolution` / `micro_time_resolution` / `micro_time_binning` are
  real columns on `mfdb_setup` (`schema.py:863…`), added by a dedicated migration
  with `ALTER TABLE` + backfill from `timing_resolution_json` (`:3627-3657`).
- The gate (`test_setup_prerequisites.py:164-195`) now iterates
  `mapper.dictionary.get_category(cat).items` over `_SETUP_CATEGORIES` and asserts
  none are unmapped — **no `required_items` / `new_items` allow-list.** A newly
  added `.dic` item can no longer be silently excluded.

### ✅ Blocker 2 — structured tables are now the read path

- `_setup_row_data(row, detector_channels=..., pie_windows=...)` builds from the
  child rows when present (`tttr_detector_setups.py:55-97`), and the loader calls
  `db.get_setup()` and passes `detector_channels=dcs, pie_windows=pws`
  (`:165-168`). The blob is now fallback-only.

### ✅ Medium hardcodes removed

- `entity_schema.py` no longer has the `("laser_wavelengths","detector_channels")`
  widget special-case.
- `setup_services.validate_setup_config` no longer hardcodes the stale
  `laser_wavelengths` / `detector_channels` / `pie_window` vocabulary.

## Remaining (minor / optional — not blocking)

1. **Indices still hardcoded** in both `CREATE_INDICES_SQL` (`schema.py:1210-1211`)
   and the v30 migration (`:3463-3467`). PRD allowed a small explicit index list,
   so this is acceptable, but it is a small duplicated hardcode (`setup_id`) — the
   unused `generate_index_for_table()` helper could drive it from the generated
   tables to keep it single-sourced.
2. **Silent-omission failure mode:** if `MmcifDictionary.load_bundled()` lacks a
   category, `generate_create_table_for_category()` returns a `-- comment`
   string, which as a `CREATE_TABLES_SQL` entry is a no-op — the table would be
   silently skipped. Prefer raising on a missing category / empty generation so a
   broken dictionary fails loudly at import rather than producing a DB without the
   table.
3. **Enumerations not enforced in SQL:** `micro_time_binning` declares
   `_item_enumeration.value 1/2/4/8` but the generator emits no `CHECK`. This
   matches the existing codebase pattern (enums enforced at the app layer), so
   it's acceptable; emitting `CHECK (col IN (...))` from `item.enumerations` would
   be a nice future tightening.
4. **Column order is alphabetical** (PK first) rather than dictionary-authoring
   order. Deterministic and harmless, just cosmetic in `PRAGMA table_info`.

## Bottom line

The prerequisite now satisfies "the `.dic` dictates the schema." No code changes
are required to merge; consider addressing minor item 2 (fail-loud on missing
category) since a silently dropped table is the one failure mode that could pass
tests yet ship a broken fresh DB.
