---
type: PRD
prd: "02a"
title: "PRD-02a: mmCIF Dictionary Infrastructure"
description: Parse bundled mmCIF dictionaries into a cached API for vocabulary validation and autocomplete
status: done
phase: "foundation"
resource: modules/mfdb/src/mfdb/
tags: [prd, mfdb]
timestamp: '2026-07-05T00:00:00Z'
---

# Summary
Parses all bundled mmCIF `.dic` dictionary files to extract categories, items,
descriptions, data types, and enumerated allowed values, exposing them through a
fast cached Python API. That API backs vocabulary validation, GUI autocomplete,
and flrCIF export-compliance checking, treating the dictionaries as the schema
authority. The prior parser only read one dictionary, returned empty
descriptions, and parsed no enumerations, data types, category metadata, or
parent-child links; this PRD covers the full FLR/IHM/ModelCIF category set
including non-enumerated required fields needed for export validation.

# Status
Done. The dictionary API is consumed by the sample/probe persistence boundary so
validation lives on one canonical path rather than duplicated across raw SQL
call sites.

# Goal
Parse all bundled mmCIF dictionary files (`.dic`) to extract categories, items,
descriptions, data types, and enumerated allowed values, and make this data
available as a fast, cached Python API for vocabulary validation, GUI
autocomplete, and flrCIF export-compliance checking. The parser must expose
complete field metadata, not only enumerated fields: non-enumerated required
fields such as `_flr_sample.id` matter as much as enum-bearing fields because
export validation depends on them.

**Execution order:** do not start until [PRD-020](prd-020.md) has established the
bounded MFDB SQLAlchemy mapping and fixed sample-scoped FRET-pair persistence.
This PRD can remain file/cache based, but its validation APIs are consumed by the
sample/probe ORM boundary; that boundary must exist first so dictionary
validation is wired into one canonical persistence path.

# Current state

**What exists:** the dictionary files under `data/*.dic` (7 files, ~20 MB) with
an `update_dictionaries.sh` downloader; a partial parser in `pdbx_metadata.py`; a
JSON cache `_dictionary_cache.json`; an external flrCIF/IHM Python library whose
FLR module (its `ihm.flr`-style class set) maps 1:1 to flrCIF categories; and the
MFDB ORM boundary from PRD-020.

**What the legacy parser got wrong:** only parsed one dictionary (hardcoded
`mmcif_pdbx_v50.dic`, so all FLR/IHM/ModelCIF categories were invisible);
descriptions always empty (multi-line `;`-delimited blocks not read); no
enumeration parsing; no data-type parsing (`_item_type.code`); no category
metadata (`_category.description`/`mandatory_code`); no parent-child
relationships (`_item_linked`). The rewritten parser then still had R14-review
blockers: it dropped item blocks without `loop_` data (so `_flr_sample.id`,
`_flr_sample.num_of_probes` were missing), had incomplete flrCIF coverage (only
11 of ~36 `flr_*` categories), fragile next-line `;`-descriptions, fragile
quoted-loop-value handling via `str.split()`, and a crashing `--stats` CLI path.

**Dictionary file structure (DDL2):** category definitions are `save_<name>`
blocks (no leading `_`); item definitions are `save_<_category.item>` blocks
(leading `_`) carrying `_item.name`, `_item.category_id`, `_item.mandatory_code`,
`_item_type.code`, a `;`-delimited `_item_description.description`, and an
optional `loop_ _item_enumeration.value / _item_enumeration.detail`.

**Bundled dictionaries:** `mmcif_pdbx_v50.dic` (~700 categories),
`mmcif_ihm_ext.dic` (~60, IHM), `mmcif_ihm_flr_ext.dic` (36 — the flrCIF
fluorescence/FRET extension, primary), `mmcif_ma.dic` (~40, ModelCIF),
`mmcif_std.dic` (~180), `mmcif_ddl.dic` (~30), `mmcif_pdbx_v5_next.dic` (~750).
The 36 flrCIF categories map directly onto ChiSurf's `flr_*` tables
(`flr_sample`, `flr_sample_condition`, `flr_sample_probe_details` →
`flr_sample_probe`, `flr_poly_probe_position`, `flr_probe_list` → `probes`,
`flr_fret_forster_radius`, `flr_fret_analysis`, `flr_fret_distance_restraint`,
`flr_fret_calibration_parameters`, `flr_instrument`, `flr_experiment`,
`flr_entity_assembly`, …), with several FPS/reference-measurement/lifetime-fit
categories not yet in the schema.

# Tasks

## Task 1: Repair and complete the parser (`pdbx_metadata.py`)
Extract per item a `DictItem` (`name`, `category`, `attribute`, `description`,
`type_code`, `mandatory`, `enumerations`, `enum_details`, `parent`) and per
category a `DictCategory` (`name`, `description`, `mandatory`, `key_item`,
`items`). Expose an `MmcifDictionary` class with `load_bundled()`,
`get_category`, `get_item`, `get_enumerations`, `get_description`,
`search_items`, `categories`, `flr_categories`, and `validate_value`.

Implementation notes: parse multi-line `;`-delimited descriptions; parse
`loop_ _item_enumeration` blocks; distinguish category `save_` blocks (no `_`)
from item `save_` blocks (leading `_`); **register the current `DictItem`
whenever an item block ends, whether or not it had a `loop_`** (mandatory for
`_flr_sample.id`); parse loop rows with a tokenizer that respects single/double
quotes, `;`-blocks, and `.`/`?` missing-value markers (never plain
`str.split()`); lazy-load via `functools.lru_cache`; parse ALL `.dic` files in
order ddl → std → pdbx_v50 → ihm_ext → ihm_flr_ext → ma (extensions overlay
base).

## Task 2: Cache the parsed dictionary
After parsing, serialize to `data/_dictionary_cache.json`. On load, use the
cache when it is newer than every `.dic` file; re-parse and regenerate
otherwise. JSON (not pickle) to avoid versioning issues. If PRD-020 adds
SQLAlchemy dictionary tables, keep JSON as the import-speed cache and use any
DB-backed tables only for query/index — the `.dic` files stay canonical.

## Task 3: Vocabulary validation functions
`validate_flr_sample(fields)` (required present, enums valid, types compatible);
`validate_flr_value(category, attribute, value)`; `suggest_values(category,
attribute, prefix)` returning enum values / `COMMON_PROBE_NAMES` / `ENTITY_TYPES`
for autocomplete.

## Task 4: Wire validation into the sample API boundary
Validate dictionary-enumerated fields at the ORM-backed sample-graph boundary
(`orm/sample_repository.py`), keeping `sample_manager.create_sample()` as a
public wrapper. Use **warn, don't reject** for most fields (labs use
non-standard names); hard-reject only truly closed vocabularies like
`entity_type` and `fluorophore_type`. Request-layer probe-name validation must
match `SampleDefinition`: unknown dye names warn, not reject.

## Task 4b: Dictionary field search in request objects
`SampleSearchRequest` must validate `vocabulary_field` against
`MmcifDictionary.load_bundled()` (not the older `chisurf.core.fio.mmcif` path),
accepting fields with or without a leading underscore (`flr_sample.id`,
`_flr_sample.id`, `entity.type`, `_entity.type`). If dictionary loading fails,
skip field validation with a warning rather than rejecting all searches.

## Task 5: Dictionary introspection CLI
A `__main__` block supporting `--list-categories`, `--category`, `--enums`,
`--search`, `--validate`, and `--stats` (the last must exit successfully).

## Task 6: Tests (`test/fio/test_pdbx_metadata.py`)
Load-bundled category counts (>500 PDBx, ≥30 flr), flr categories present, item
lookup (`_flr_sample.id`), fluorophore_type and entity_type enumerations,
descriptions parsed, non-loop items retained, `SampleSearchRequest` accepts valid
dictionary fields, valid/invalid value validation, item search, JSON cache
round-trip, and the `--stats` CLI running without crashing.

Update dictionaries with `bash …/data/update_dictionaries.sh` (downloads the
latest from wwPDB); the JSON cache auto-regenerates when any `.dic` is newer.

# Definition of Done
- [ ] PRD-020 complete enough that sample/probe/FRET persistence has one ORM-backed boundary
- [ ] `MmcifDictionary` parses all 7 bundled `.dic` files
- [ ] Item blocks without `loop_` retained (`_flr_sample.id` exists)
- [ ] Multi-line `;`-delimited descriptions parsed
- [ ] Enumerations extracted for all `_item_enumeration` fields
- [ ] Quoted loop values parsed without `str.split()` corruption
- [ ] Data types extracted (`_item_type.code`)
- [ ] JSON cache auto-regenerates when `.dic` files change
- [ ] `validate_value()` checks enumerations and types
- [ ] `suggest_values()` returns GUI-autocomplete completions
- [ ] `search_items()` keyword search across all items
- [ ] CLI introspection works
- [ ] ≥30 `flr_*` categories and ≥500 PDBx categories available
- [ ] `SampleSearchRequest(vocabulary_field="flr_sample.id", ...)` accepts valid fields
- [ ] Request-layer custom probe names warn rather than raise
- [ ] `--stats` exits successfully
- [ ] All tests pass; `update_dictionaries.sh` downloads all 7 files from wwPDB

# Relationships
- Depends on [PRD-020](prd-020.md); prerequisite for [PRD-02](prd-02.md) vocabulary validation.
- Feeds export alignment in [PRD-02c](prd-02c.md).
- Encodes the dictionary-as-authority principle of the [MFDB (current)](/architecture/mfdb.md) store and its [MFDB target](/specs/mfdb.md).
