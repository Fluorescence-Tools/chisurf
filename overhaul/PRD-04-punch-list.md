# PRD-04 Prerequisite — Coder Punch-List

Actionable fixes from `PRD-04-code-review.md`. Governing rule: **the `.dic`
dictates the schema — no hardcoded SQL, no second copy of field
names/labels/types/enums.** References: `PRD-04-burst-pipeline.md` (Task P1a,
Source Of Truth), `PRD-04-code-review.md` (findings).

Do in order. Each item lists the files and the done-when.

## 0. Generate DDL from the dictionary (Blocker 0, Task P1a)

- [ ] New `chisurf/core/mfdb/schema_from_dictionary.py`: read bridged categories
      from `MmcifDictionary`, emit `CREATE TABLE` / `ALTER TABLE`.
  - column name <- `_chisurf_schema.column_name` (fallback: item attribute)
  - SQL type <- `_item_type.code` via one `TYPE_CODE_SQL_MAP`
    (`int/uint/integer` -> INTEGER; `float/double/num` -> REAL; else TEXT)
  - `NOT NULL` <- `_item.mandatory_code == yes`
  - PK <- `DictCategory.key_item`; FK <- item parent link; DEFAULT <-
    `_item_default.value`
  - audit cols `created_at`/`updated_at`/`deleted_at` by fixed convention
  - deterministic, stable column order
- [ ] Categories generated: `mfdb_setup` (new `tttr_reading` cols only),
      `mfdb_setup_detector_channel`, `mfdb_setup_pie_window`.
- [ ] Wire generator into the fresh-DB build **and** the v30 migration.
- [ ] Remove hand-written DDL for these tables from `schema.py` —
      `CREATE_TABLES_SQL` (~`:869-892`) and the v30 migration (~`:3451-3474`).
- Done when: the column set for these tables exists in exactly one place
  (the `.dic`); the other two sites call the generator.

## 1. Kill the divergence + total-coverage gate (Blocker 1)

- [ ] `macro_time_resolution` / `micro_time_resolution` / `micro_time_binning`
      become real generated columns on `mfdb_setup` (migration adds via
      `ALTER TABLE`). Drop the `timing_resolution_json`-only fallback.
- [ ] Rewrite the validation gate in `test/fio/test_setup_prerequisites.py` to
      iterate **every** item from `dictionary.get_category(cat).items` for the
      three categories and assert `validate_mapping()` is True for all; assert
      `get_unmapped_flr_items()` contains none of them.
- [ ] Delete the literal `required_items` / `new_items` allow-lists.
- Done when: the gate passes with no curated list, and would fail if any `.dic`
  item lacked a column.

## 2. Make structured tables the read path (Blocker 2)

- [ ] `tttr_detector_setups.py:_setup_row_data()` (and callers) source
      detectors/windows from `get_setup()`'s `detector_channels` / `pie_windows`,
      not from `configuration_json.setup_data` / `detectors_json`.
- [ ] Treat the JSON blob as legacy/back-compat only (stop relying on it as the
      source of truth; optionally stop writing it).
- Done when: editing a setup in the wizard round-trips through the child tables;
      the blob can be stale without affecting the GUI.

## 3. Remove residual hardcoding (Medium)

- [ ] `entity_schema.py:234-235`: delete the
      `if name in ("laser_wavelengths","detector_channels")` widget special-case;
      drive the widget from the dict `_item_type.code`.
- [ ] `setup_services.py:31-78 validate_setup_config`: align/retire the stale
      hardcoded vocabulary (`laser_wavelengths`, `detector_channels`,
      `pie_window`, `pie_enabled`) — drive from dictionary item types/enums or
      match the structured table names.
- [ ] `generic_form.py`: confirm the new entities always use
      `field_specs_for_category()` and never fall back to the legacy hardcoded
      `SCHEMAS` dict; schedule `SCHEMAS` for removal.

## 4. Minor

- [ ] `repository.py save_setup`: comment the detector-field extraction list as
      derived-from-`.dic`, or drive it from the schema-map columns.
- [ ] Decide `name` vs `channel_name`/`window_name` (keep consistent with `.dic`).
- [ ] `test_setup_prerequisites.py:104` `"chs"` vs `"channels"` — confirm intent.

## Verify

```bash
PYTHONPATH="modules/chinet:modules/imp-tricks/src:." \
  /Users/tpeulen/mambaforge/envs/arm64/bin/python3 \
  -m pytest -p no:cov -o addopts='' \
  test/fio/test_setup_prerequisites.py \
  test/fio/test_burst_pipeline_mfdb.py
```

- [ ] Total-coverage gate passes with no allow-list.
- [ ] Fresh DB built via the generator has exactly the columns the `.dic`
      declares (round-trip with `introspect_sqlite_schema`).
- [ ] Report the three drift sites (CREATE_TABLES_SQL, v30 migration, `.dic`)
      collapsing to one.

## Watch-out

If FK `ON DELETE CASCADE` cannot be expressed from current parent-link metadata,
add a `_chisurf_schema` hint to the `.dic` rather than hardcoding the FK in
Python — surface this if hit.
