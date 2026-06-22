# PRD-04 Code Review: Setup/Channel-Definition Prerequisite

Scope: the prerequisite from `PRD-04-burst-pipeline.md` ("Setup And Channel
Definitions In MFDB") — structured, dictionary-described, admin-visible
reading/processing setup, plus the burst operation setup link.

Reviewed (2026-06-20):

- `chisurf/core/mfdb/schema.py` (child tables, v30 migration)
- `chisurf/core/mfdb/data/mfdb_flr_ext.dic` (new save blocks)
- `chisurf/core/mfdb/repository.py` (`save_setup` / `get_setup` /
  `list_detector_channels` / `list_pie_windows`)
- `chisurf/gui/widgets/wizard/tttr_channeldefinition/tttr_detector_setups.py`
- `chisurf/plugins/core/mfdb_admin/gui/entity_schema.py` (new)
- `chisurf/plugins/core/mfdb_admin/gui/entity_registry.py` (new)
- `chisurf/plugins/core/mfdb_admin/gui/generic_form.py`
- `chisurf/plugins/core/mfdb_admin/backend/setup_services.py`
- `chisurf/plugins/burst/burst_selection/api/mfdb.py`
- `test/fio/test_setup_prerequisites.py`, `test/fio/test_burst_pipeline_mfdb.py`

## Verdict

The GUI/admin layer is genuinely dict-driven (`entity_schema.py` derives form
fields, `entity_registry.py` is wiring-only, the `.dic` blocks carry
`_chisurf_schema` bridges). **But the storage layer still hardcodes the SQL, and
under the governing directive — "the `.dic` dictates the schema; no hardcoded
SQL" — that is the lead blocker.** Compounding it: a dictionary↔schema divergence
that the validation test was written to hide. The DDL for the new tables must be
*generated from the dictionary*, not hand-written.

## 🔴 Blocker 0 — the SQL is hardcoded (and duplicated); the `.dic` must dictate it

The directive is that the dictionary generates the schema. Today the opposite is
true: every table definition is a hand-written SQL string, and for the new tables
it is written **twice**:

- `schema.py:869-892` — `CREATE TABLE mfdb_setup_detector_channel` /
  `mfdb_setup_pie_window` in `CREATE_TABLES_SQL` (fresh-DB path).
- `schema.py:3451-3474` — the **same** two `CREATE TABLE` statements pasted again
  inside the v30 migration.

So the column set for these tables now lives in three hand-maintained places
(`CREATE_TABLES_SQL`, the v30 migration, and the `.dic`) — the precise drift
surface the directive forbids, and the direct cause of Blocker 1 below.

Fix: implement the PRD-04 **Task P1a** generator
(`schema_from_dictionary.py`): read the bridged categories from the dictionary
and emit the `CREATE TABLE` / `ALTER TABLE` DDL (column names from
`_chisurf_schema.column_name`, types from `_item_type.code` via a single
`TYPE_CODE_SQL_MAP`, PK from `_category_key.name`, FK from item parent links,
defaults from `_item_default.value`). Wire both the fresh-DB build and the v30
migration to call the generator — never a pasted DDL copy. Existing hand-written
tables are grandfathered; only the setup/detector/window tables move to the
generated path in this PRD.

## 🔴 Blocker 1 — `.dic` declares columns that don't exist; the gate test was weakened to pass

## 🔴 Blocker 1 — `.dic` declares columns that don't exist; the gate test was weakened to pass

The dictionary declares three `mfdb_setup` items bound to real columns:

```
save__mfdb_setup.macro_time_resolution  → _chisurf_schema.column_name macro_time_resolution
save__mfdb_setup.micro_time_resolution  → _chisurf_schema.column_name micro_time_resolution
save__mfdb_setup.micro_time_binning     → _chisurf_schema.column_name micro_time_binning
```

But `mfdb_setup` (`schema.py`) has **no such columns** — `tttr_reading` is still
dumped into the `timing_resolution_json` blob. So `DictionarySchemaMap`
classifies all three as `UnmappedItem` (reason: column missing). This violates
the source-of-truth contract: the dictionary promises a column the schema does
not provide.

This should have been caught by Task P6 #3/#4, but the tests use **curated
allow-lists** (`test/fio/test_setup_prerequisites.py:166-181` and `:200-217`)
that omit those three items, with a comment rationalizing the omission:

```
# ...timing-resolution items are stored as JSON blob and have
# no dedicated column.
```

That defeats the gate. As written, the test passes precisely because the failing
items were excluded — a hand-maintained list silently re-introduced, papering
over a real divergence.

Resolution (per directive: the `.dic` dictates the SQL — no hardcoded DDL):

- The root cause is that schema and dictionary are two hand-maintained copies
  that drifted. Fix the architecture, not just the columns: **generate the
  `CREATE TABLE` DDL for these tables from the dictionary** (PRD-04 Task P1a).
  When schema and dictionary share one source, `macro_time_resolution` &
  friends cannot exist in the `.dic` without existing as columns.
- The three `tttr_reading` items become real generated columns on `mfdb_setup`
  (typed, units already in the `.dic`); the `timing_resolution_json` blob
  fallback is not acceptable.
- Rewrite the gate to **derive its own list with no allow-list**: iterate
  `mapper.dictionary.get_category(cat).items` for `mfdb_setup`,
  `mfdb_setup_detector_channel`, `mfdb_setup_pie_window` and assert
  `validate_mapping()` is True for **all** items — delete the literal
  `required_items` / `new_items` arrays so a newly added `.dic` item cannot be
  silently excluded.

## 🔴 Blocker 2 — structured tables are write-only; the GUI still reads the opaque blob

`save_setup()` dual-writes: it dumps `detectors_json` **and** populates
`mfdb_setup_detector_channel` / `mfdb_setup_pie_window`
(`repository.py:3970-4018`). But the wizard read path `_setup_row_data()`
(`tttr_detector_setups.py:54-71`) still reads from
`configuration_json.setup_data` / `detectors_json` — it **never reads the child
tables**. Consequences:

- The blob remains the de-facto source of truth; the structured tables are a
  derived copy nothing consumes.
- The two can silently diverge.
- PRD-04 Task P2 explicitly required "stops round-tripping through opaque
  `setup_data` blobs once structured storage exists." That did not happen.

Fix: make `get_setup()`'s structured `detector_channels` / `pie_windows` (already
implemented at `repository.py:4034-4044`) the read path for the wizard, and treat
the blob as legacy/back-compat only (or stop writing it). Otherwise the
"queryable, structured" goal is cosmetic.

## 🟠 Medium — residual hardcoding (the original concern)

1. **`entity_schema.py:234-235`** — hardcoded field-name special-case:

   ```python
   if name in ("laser_wavelengths", "detector_channels"):
       widget = "str"
   ```

   Widget selection is overridden by **column name** instead of by dictionary
   type code — the exact dict-bypass the PRD forbids. Fix: give those items the
   appropriate `_item_type.code` (e.g. `text`) so `TYPE_CODE_WIDGET_MAP` resolves
   the widget, then delete the special-case.

2. **`setup_services.py:31-78` `validate_setup_config`** hardcodes a parallel,
   stale vocabulary: `laser_wavelengths`, `detector_channels`, `pie_window`,
   `pie_enabled`. None match the new schema (`mfdb_setup_detector_channel.channels`,
   `mfdb_setup_pie_window`; no laser table in this PRD). Both hardcoded and out of
   sync with what is now stored. Fix: drive validation off the dictionary item
   types/enumerations, or at minimum align key names with the structured tables.

3. **Legacy `SCHEMAS` dict still live in `generic_form.py:8-10`**, used as the
   fallback at `:228` (`SCHEMAS.get(self.schema_type, [])`). The new entity types
   (`mfdb_setup_detector_channel`, `mfdb_setup_pie_window`) have no `SCHEMAS`
   entry, so if they hit the fallback path they render an empty form. Confirm they
   always go through `field_specs_for_category()`; schedule the hardcoded dict for
   removal rather than leaving it as a silent fallback.

## 🟡 Minor

4. **`repository.py save_setup`** enumerates detector fields (`channels`,
   `micro_time_ranges`, `g_factor`, `l1`, `l2`, `g_factor_channels`) a third time
   (after `schema.py` and the `.dic`). Inherent to a payload→column writer, but
   it can drift — add a comment pointing at the `.dic` as canonical, or drive the
   loop from the schema-map columns.

5. **Column naming:** child tables use generic `name` rather than `channel_name`
   / `window_name` (PRD suggested the latter). Harmless since the `.dic` matches,
   but less self-describing in joins.

6. **Test data smell:** `test_setup_prerequisites.py:104` uses `"chs"` instead of
   `"channels"` in one fixture — confirm intentional (tolerance of unknown keys)
   and not a silent drop.

## What's good (keep)

- `entity_schema.py` deriving `FieldSpec` (label/widget/choices/tooltip/readonly/
  FK) from `DictItem` + `DictionarySchemaMap` — correct pattern.
- `entity_registry.py` reduced to wiring-only.
- `.dic` blocks for the child tables are complete with bridges and column names
  that map cleanly.
- v30 migration with backfill + `MigrationReport`, indices, the P5
  setup-existence guard, and the burst `setup_id` NULL-on-missing behavior.

## Required Before Merge

- [ ] Blocker 0: generate the setup/detector/window `CREATE TABLE` DDL from the
      dictionary (Task P1a); remove the hand-written DDL from `CREATE_TABLES_SQL`
      and the v30 migration so the column set lives only in the `.dic`.
- [ ] Blocker 1: with the generated schema, the `mfdb_setup` timing columns exist
      by construction; make the validation gate iterate **all** items in the new
      categories (derived from the dictionary), not a curated allow-list.
- [ ] Blocker 2: make the wizard read the structured child tables so they are
      authoritative, not a write-only shadow of the blob.
- [ ] Medium 1: remove the `entity_schema.py` hardcoded widget special-case;
      drive it from the dictionary type code.
- [ ] Medium 2: align/retire the hardcoded `validate_setup_config` vocabulary.
- [ ] Medium 3: confirm the new entities never fall back to legacy `SCHEMAS`;
      plan its removal.
