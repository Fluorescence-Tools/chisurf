# PRD-04: Stable Burst Pipeline MFDB Integration

## Goal

When the Burst Selection plugin produces burst tables, those results are
registered in MFDB with stable, queryable provenance:

```text
raw TTTR artifact(s)
  -> operation: burst_selection
  -> burst_table artifact(s)
  -> optional packaged sidecar/output artifacts
```

The pipeline must be stable across GUI, CLI, RPC, and direct API callers. The
scientific analysis API remains the source of truth for computing burst results;
MFDB registration is a narrow archival side effect layered on top of the
analysis result.

## Status And Gate

PRD-04 is implemented. Burst Selection is the reference implementation for
workflow-ready plugins with MFDB archival.

- PRD-030 payload codecs exist and are used by PRD-03.
- PRD-03 result registry has been reviewed and approved in
  `CODE_REVIEW_PRD03.md`.
- `register_result()` supports `burst_table` and `burst_selection` typed payloads.
- Implementation reference: `chisurf/plugins/burst/burst_selection/docs/REFERENCE_IMPLEMENTATION.md`.
- Workflow contract: `chisurf/plugins/burst/burst_selection/docs/CONTRACT.md`.

PRD-04 must not reimplement object storage, operation recording, parameter
recording, or payload serialization. Those are PRD-03 responsibilities.

## Prerequisite: Setup And Channel Definitions In MFDB

Full provenance tracing of a burst run is only possible if the *setup* that
produced the photons is itself a first-class, queryable MFDB record. In chisurf,
the **setup is defined by its detector definitions and its PIE/microtime window
(channel) definitions** together with the TTTR reading routine. These are the
same objects the Detector Wizard edits and that the burst request carries as
`selected_setup`, `windows`, and `detectors`. Until they live in MFDB as typed,
dictionary-described rows, a `burst_table` artifact can record *which named
setup* was used but cannot be traced back to *what that setup actually was*.

This prerequisite must be satisfied before the burst operation can wire its
`mfdb_operation.setup_id` foreign key and before mfdb-admin can render setup
provenance.

> Note on scope: the setup/channel definition owned here is **only about the
> reading and processing of the data** — which detector channels exist, the
> PIE/microtime window definitions, and the TTTR reading routine. It is the base
> record. PRD-08 (Optical Configuration Schema) **extends** this same setup with
> additional *spectroscopic* information (light sources, filters, dichroics,
> objectives, detector hardware) describing the optical path that produced the
> photons. PRD-04 does not define or require that optical/spectroscopic layer; it
> only needs the reading/processing channel definition to be a queryable MFDB
> record so the burst chain is traceable. PRD-05 (Calibration Provenance) attaches
> calibration values to the same setup.

### Current Gap

- Detector setups are persisted to `mfdb_setup` (via
  `chisurf/gui/widgets/wizard/tttr_channeldefinition/tttr_detector_setups.py`)
  as **opaque JSON blobs** in `configuration_json` / `detectors_json` /
  `timing_resolution_json` / `burst_defaults_json`. They are not queryable and
  not validated against any dictionary.
- `mfdb_flr_ext.dic` has **no save block** describing `mfdb_setup`, the detector
  definitions, or the channel/window definitions, so admin cannot resolve field
  descriptions or enumerate valid values for them.
- mfdb-admin's "Setups" entity is wired to the `flr_inst_setting` category and
  shows only setup-level rows; the detector and channel/window definitions
  inside the JSON blobs are not displayed as structured records.
- The burst pipeline treats `selected_setup` as a free-text metadata label.
  Nothing guarantees a matching `mfdb_setup` row exists, and
  `mfdb_operation.setup_id` is not populated, so the chain
  `raw TTTR -> operation -> burst_table` has no setup node.

### Required Outcome

1. **Setup persisted as a resolvable record.** Each named setup is an
   `mfdb_setup` row addressed by the deterministic id from
   `setup_id_for_name()` (`tttr_detector_setup:<slug>`). The burst pipeline
   resolves `request.mfdb.setup_id` (or derives it from `selected_setup`) to a
   real row before registration; a missing setup row produces a warning, never
   an exception.
2. **Channel/detector definitions are structured, not opaque.** The detector
   definitions and PIE/microtime window definitions that govern reading and
   processing are stored so they are queryable, following a dictionary-described
   schema rather than an arbitrary blob. This reading/processing record is the
   base; PRD-08 later extends the same setup with the optical/spectroscopic path
   (`mfdb_optical_channel` and hardware-component tables). PRD-04 only requires
   the reading/processing layer to be structured and queryable.
3. **flrCIF / `.dic` alignment — the dictionary is the single source of truth.**
   `mfdb_flr_ext.dic` is authoritative for the field *vocabulary*: item names,
   types (`_item_type.code`), units (`_item_units.code`), enumerations
   (`_item_enumeration.value`), and descriptions. The SQL schema and the admin
   display must **derive from / be validated against** the dictionary, not carry
   a second hand-maintained copy of names, labels, or allowed values. The
   optical/spectroscopic extension categories are added later by PRD-08 and are
   out of scope here.
4. **Correct mfdb-admin display.** The setup, its detector definitions, and its
   channel/window definitions display correctly in mfdb-admin as structured
   rows, not as raw JSON. Selecting a burst `burst_table` artifact can navigate
   to the setup that produced it. (The optical/spectroscopic columns PRD-08 adds
   appear alongside once that PRD lands; they are not required here.)
5. **Operation links to setup.** `mfdb_operation.setup_id` is populated for the
   `burst_selection` operation so the burst table's provenance edges include the
   setup node and full tracing is possible end to end.

### Boundary

| Concern | Owner |
| --- | --- |
| Reading/processing setup record: detector channels, PIE/microtime windows, TTTR reading routine, persisted as a queryable, dictionary-described `mfdb_setup` row; admin display of those fields | **PRD-04 (this prerequisite)** |
| Wiring an existing setup record into the burst operation (resolve setup id, populate `mfdb_operation.setup_id`, reference setup in burst metadata) | **PRD-04 (main body)** |
| Optical/spectroscopic extension: light sources, filters, dichroics, objectives, detector hardware (`mfdb_optical_channel` + hardware tables, their `.dic` blocks, their admin entity-registry entries) | **PRD-08** |
| Calibration values attached to a setup | **PRD-05** |

PRD-04 must not implement the PRD-08 optical/spectroscopic tables or the PRD-05
calibration tables. It only needs the reading/processing layer of the setup to
be a structured, queryable, dictionary-described, admin-visible MFDB record, and
the burst operation to link to it.

### Source Of Truth: The `.dic` Dictionary (Read Before Coding)

The dictionary **dictates** the schema. For the tables introduced by this
prerequisite there is **no hardcoded SQL** — the `.dic` is the single authored
artifact, and the table definitions are *generated from it*. Do not hand-write
`CREATE TABLE` strings and do not hardcode a second copy of field names, labels,
types, or enumerations in `schema.py` or `entity_registry.py`. The required flow:

1. **Author the field vocabulary once in `mfdb_flr_ext.dic`** — one item block
   per stored field, with `_item_type.code`, `_item_units.code` where physical,
   `_item_enumeration.value` for closed value sets, `_item.mandatory_code`, the
   category `_category_key.name` (primary key), parent links for foreign keys,
   and a description. This block is the authority; everything else derives.
2. **Bind every item to its column with the `_chisurf_schema` bridge.** This is
   mandatory: the new tables (`mfdb_setup`, `mfdb_setup_detector_channel`,
   `mfdb_setup_pie_window`) are **not** `flr_`-prefixed, so
   `DictionarySchemaMap._mapped_categories()`
   (`chisurf/core/mfdb/dictionary_schema_map.py`) will only discover their
   category if each item declares `_chisurf_schema.table_name` and
   `_chisurf_schema.column_name`. Without the bridge the columns are invisible to
   the generator, the mapper, validation, and admin description-resolution.
3. **The SQL DDL is generated from the dictionary, not written by hand** (see
   Task P1a). The generator reads each bridged category and emits the
   `CREATE TABLE` with: column names from `_chisurf_schema.column_name`, SQL
   types from `_item_type.code`, `NOT NULL` from `_item.mandatory_code`, the
   primary key from `_category_key.name`, foreign keys from item parent links,
   and `DEFAULT` from `_item_default.value`. Because schema and dictionary come
   from the same source, `DictionarySchemaMap.validate_mapping()` cannot diverge.
4. **Admin pulls labels/descriptions/enumerations from the dictionary**, not from
   literals re-typed into `EntitySpec`. Resolve them through the existing
   dictionary path (`MmcifDictionary` / `DictionarySchemaMap`) so a label or
   enum is defined in exactly one place.

If a field cannot yet be expressed in the dictionary, that field is not ready to
be stored as a structured column — fix the dictionary first.

Scope of generation: this PRD applies the generated-DDL path to the setup,
detector-channel, and PIE-window tables only. Existing hand-written tables in
`schema.py` are grandfathered and are not rewritten here; the generator must be
reusable but is wired for the new categories in this PRD.

### Prerequisite Tasks (Detailed)

These tasks deliver the reading/processing setup record and wire it into the
burst chain. They are ordered; later main-body tasks assume these are done. Per
the source-of-truth rule above, author the `.dic` blocks (Task P3) **together
with** the schema (Task P1) so column names and dictionary item names are
chosen once and never diverge.

Current state (verified 2026-06-20):

- Main body is largely implemented: `chisurf/core/mfdb/pipeline.py` is deleted,
  `chisurf/plugins/burst/burst_selection/api/mfdb.py` exists with
  `BurstMFDBPipeline`, and `register_result()` / `register_raw_measurement()`
  already accept and forward `setup_id` to `mfdb_operation`.
- Outstanding for this prerequisite: P1-P4 (structured, dic-described,
  admin-visible reading/processing setup) and P6 (tests) are **not started**.
  P5 is **mostly wired**; only the setup-existence guard is missing.

#### Task P1: Define The Reading/Processing Setup Schema

Goal: replace the opaque `mfdb_setup` JSON blobs for TTTR detector setups with a
dictionary-described, queryable representation of the reading/processing layer.

- The setup payload edited by the Detector Wizard has exactly three parts:
  - `detectors`: map of detector-channel name -> `{channels, micro_time_ranges,
    g_factor, l1, l2, g_factor_channels}` (see
    `tttr_channel_definition.py:_initial_detectors`).
  - `windows`: map of PIE/microtime window name -> `[start, end]` (see
    `_initial_windows`).
  - `tttr_reading`: `{macro_time_resolution, micro_time_resolution,
    micro_time_binning}` (see `_initial_tttr_reading`).
**Decision (locked 2026-06-20): Option A — child tables.** Option B (pinned
JSON) is rejected because it cannot satisfy the admin-listable-structured-rows
and SQL-queryable requirements without an extra view layer, and because PRD-08
must be able to *extend* the detector channel with spectroscopic columns/links,
which requires it to be a real table rather than a blob.

The table *shapes* below are the design target, but they are **declared in the
`.dic` (Task P3) and generated into DDL (Task P1a)** — do not hand-write these
`CREATE TABLE` strings in `schema.py`.

- Tables (all columns declared as `.dic` items with `_chisurf_schema` bridges):
  - `mfdb_setup_detector_channel` — one row per detector channel, with
    `setup_id` (FK to `mfdb_setup(setup_id)`, `ON DELETE CASCADE`), a channel
    name column, and one typed column per detector field (`channels`,
    `micro_time_ranges`, `g_factor`, `l1`, `l2`, `g_factor_channels`), plus the
    standard `created_at`/`updated_at`/`deleted_at`. Use a JSON-encoded `text`
    item only for genuinely list-valued fields (`channels`,
    `micro_time_ranges`); keep scalars typed (`float`/`int`).
  - `mfdb_setup_pie_window` — one row per PIE/microtime window, with `setup_id`
    (FK, `ON DELETE CASCADE`), a window name column, `start` (`int`), `end`
    (`int`), plus the standard timestamps.
  - `tttr_reading` scalars (`macro_time_resolution`, `micro_time_resolution`,
    `micro_time_binning`) are typed columns on `mfdb_setup`. Because the `.dic`
    declares them (with units), they **must** be real generated columns — the
    `timing_resolution_json` blob fallback is not acceptable, since a declared
    item bound to a missing column is the exact divergence this PRD forbids.
  - Indexes on `setup_id` for both child tables (declare via the generator's
    index hook or a small explicit index list keyed off the generated tables).
- The canonical column names must be identical across the `.dic` items (Task P3,
  the authority), the generated columns (Task P1a), and the admin columns
  (Task P4) — by construction, since all three derive from the dictionary.

**PRD-08 reconciliation (forward note, do not implement here):** PRD-08's
`mfdb_optical_channel` is the *spectroscopic extension* of the detection channel
defined here, not a parallel concept. When PRD-08 lands it should reference this
base — e.g. `mfdb_optical_channel.detector_channel_id REFERENCES
mfdb_setup_detector_channel(...)` — rather than re-deriving channels from a blob.
Leave a corresponding note in PRD-08 so the two channel tables are linked, not
duplicated.
- File: `chisurf/core/mfdb/schema.py`. Bump `SCHEMA_VERSION` and add a migration
  that creates the generated tables (calling the Task P1a generator, not a
  pasted copy of the DDL) and backfills existing `tttr_detector_setup:*` rows in
  `mfdb_setup` from their current `detectors_json` / `configuration_json` /
  `timing_resolution_json` blobs into the new structure without data loss. Follow
  the existing `MigrationReport` pattern. The new `tttr_reading` columns on
  `mfdb_setup` are added via `ALTER TABLE` in the same migration.

#### Task P1a: Generate Table DDL From The Dictionary (No Hardcoded SQL)

Goal: the `.dic` dictates the schema. Add a generator that emits `CREATE TABLE`
(and `ALTER TABLE ADD COLUMN`) statements for the bridged setup categories so no
`CREATE TABLE` string for these tables is hand-written in `schema.py`.

File: `chisurf/core/mfdb/schema_from_dictionary.py` (new) or a function in
`dictionary_schema_map.py` (it already loads the dictionary and introspects the
live schema).

- Input: a `MmcifDictionary` and a list of category names to generate
  (`mfdb_setup_detector_channel`, `mfdb_setup_pie_window`, and the new
  `mfdb_setup` `tttr_reading` columns).
- For each category, build the table from its `DictCategory` / `DictItem`
  metadata (all already exposed by `pdbx_metadata.py`):
  - table name <- `_chisurf_schema.table_name` (consistent across the category).
  - column name <- `_chisurf_schema.column_name` (fall back to item attribute).
  - SQL type <- `_item_type.code` via a single `TYPE_CODE_SQL_MAP`
    (`int`/`uint`/`integer` -> `INTEGER`; `float`/`double`/`num` -> `REAL`;
    everything textual / JSON-encoded -> `TEXT`). This is the SQL counterpart of
    the existing `TYPE_CODE_WIDGET_MAP`; define it once.
  - `NOT NULL` <- `_item.mandatory_code == yes`.
  - primary key <- `DictCategory.key_item` (`_category_key.name`).
  - foreign key <- item parent link where present (e.g. `setup_id` -> 
    `mfdb_setup(setup_id) ON DELETE CASCADE`); encode the FK target and delete
    rule in the dictionary item (add a `_chisurf_schema.foreign_key` /
    on-delete hint if the existing parent link cannot express `ON DELETE
    CASCADE`).
  - `DEFAULT` <- `_item_default.value` when present.
  - audit columns: emit `created_at` / `updated_at` / `deleted_at` for every
    generated table (either declared as `.dic` items or appended by the
    generator as a fixed convention — pick one and document it).
- Emit deterministic, stable SQL (sorted columns by a declared order or by
  dictionary order) so migrations and fresh-DB creation match.
- Wire the generator into both creation paths: `FRESH_DB_TABLES_SQL` / the
  fresh-DB build and the v-bump migration. Existing hand-written tables are
  untouched (see "Scope of generation").
- Add a unit test asserting the generated DDL for each category contains the
  expected columns/types/PK/FK, and that a DB built via the generator has the
  same columns the dictionary declares (round-trip with
  `introspect_sqlite_schema`).

#### Task P2: Persist Through The Setup Repository, Not Loose JSON

File: `chisurf/core/mfdb/repository.py` and
`chisurf/gui/widgets/wizard/tttr_channeldefinition/tttr_detector_setups.py`.

- Extend `MFDatabase.save_setup()` / `get_setup()` / `list_setups()` so they read
  and write the structured detector-channel and PIE-window rows chosen in Task P1
  (or the pinned JSON schema), keeping the deterministic id contract
  `setup_id_for_name()` -> `tttr_detector_setup:<slug>`.
- `tttr_detector_setups.py` keeps its current MFDB-first behavior (`_use_mfdb`,
  `_migrate_json_setups_to_mfdb`) but stops round-tripping through opaque
  `setup_data` blobs once structured storage exists. Legacy JSON import must
  still work for first-run migration from `detector_setups.json`.
- Do not change the GUI editing surface in this PRD beyond what is needed to load
  and save the structured record.

#### Task P3: flrCIF `.dic` Save Blocks For The Reading/Processing Setup

File: `chisurf/core/mfdb/data/mfdb_flr_ext.dic`. This is the authoritative
artifact and must be authored **first** — Task P1a generates the SQL from it, so
the `.dic` is written before (not after) the schema exists.

- Add `save_` category and item blocks for the setup, detector-channel, and
  PIE-window fields, following the existing block style in this file (e.g. the
  `flr_sample` / `flr_poly_probe_position` blocks).
- **Every item must declare the `_chisurf_schema.table_name` and
  `_chisurf_schema.column_name` bridge** (mandatory — see the source-of-truth
  rule; the new tables are not `flr_`-prefixed so they are otherwise invisible to
  `DictionarySchemaMap`).
- Each item carries `_item_type.code`; add `_item_units.code` where physical
  (`ns`, `ps`, `MHz` for `tttr_reading`) and `_item_enumeration.value` where the
  value set is closed (e.g. `micro_time_binning`).
- Item column names must equal the schema column names from Task P1 exactly.
- Keep these blocks scoped to reading/processing fields; do not add the optical
  hardware categories (those are PRD-08).
- Verify the dictionary still parses and that `MmcifDictionary.load_bundled()`
  exposes the new categories; refresh `_dictionary_cache.json` via
  `update_dictionaries.sh` if the loader uses the cache.

#### Task P4: mfdb-admin Display Of The Setup And Its Channel Definitions

File: `chisurf/plugins/core/mfdb_admin/gui/entity_registry.py` (and the setup
service if a new query is needed:
`chisurf/plugins/core/mfdb_admin/backend/setup_services.py`).

- The existing `setup` `EntitySpec` (key `setup`, `schema_type="setup"`) must
  show the reading/processing setup as structured columns, not raw JSON.
- Add child `EntitySpec` entries (or a detail view) for the detector-channel and
  PIE-window rows from Task P1 so a coder/user can see what a named setup
  actually contains.
- **Column labels, descriptions, and enumerations must come from the dictionary**
  (resolve via `DictionarySchemaMap` / `MmcifDictionary`), not be re-typed as
  literals in `EntitySpec`. If the registry currently needs a label string,
  source it from the dict item description so it is defined once. Follow how the
  existing flr-backed entities resolve their field metadata.
- Confirm `mfdb.setups.list` and the setup detail RPC return the structured
  fields needed by these columns.

#### Task P5: Resolve And Link The Setup In The Burst Operation

Files: `chisurf/plugins/burst/burst_selection/api/mfdb.py` and
`chisurf/plugins/burst/burst_selection/gui/tool.py`.

Already in place (verify, do not rebuild):

- `MFDBContext.setup_id` / `setup_version` exist (`api/models.py`).
- `register_raw_measurement()` and `register_result()` already accept `setup_id`
  and pass it through to the `mfdb_operation` row (`result_registry.py`); so
  `mfdb_operation.setup_id` is populated whenever a setup id is supplied. No
  PRD-03 extension is needed.
- `api/mfdb.py` already forwards `request.mfdb.setup_id` into both register
  calls and records `selected_setup` / `setup_id` in metadata.
- `gui/tool.py` already derives `setup_id_for_name(selected_setup)`.

Remaining work (the missing guard):

- Before forwarding the setup id, look up the `mfdb_setup` row for the resolved
  id. If the row exists, forward it (current behavior). If it does **not** exist,
  append a warning and forward an empty setup id so `mfdb_operation.setup_id`
  stays NULL rather than referencing a dangling/absent setup. This keeps the
  operation insert valid and registration best-effort (never raises), per the
  Error Handling section.
- When `request.mfdb.setup_id` is empty but `request.selected_setup` is set,
  derive the id via `setup_id_for_name()` before the existence check, so a run
  driven only by a setup label still links when the row exists.
- Record `setup_version` in `build_burst_metadata()` alongside the existing
  `selected_setup` / `setup_id`.

#### Task P6: Prerequisite Tests

Add focused tests (alongside the main-body `test/fio/test_burst_pipeline_mfdb.py`
suite, or a sibling `test/fio/test_setup_definition.py`):

1. Saving a detector setup creates a queryable `mfdb_setup` row plus structured
   detector-channel and PIE-window records (Task P1/P2).
2. Legacy `detector_setups.json` import backfills into the structured form
   without losing detectors, windows, or `tttr_reading` values.
3. Dictionary dictates the schema (Tasks P1a/P3) — **total coverage, no
   allow-list**: build `DictionarySchemaMap(db_path)` against a fresh DB, then
   iterate **every** item in the new categories (`mfdb_setup`,
   `mfdb_setup_detector_channel`, `mfdb_setup_pie_window`) discovered from the
   dictionary itself and assert `validate_mapping()` is `True` for each, and that
   `get_unmapped_flr_items()` contains none of them. Do **not** hand-list the
   items being checked — derive the list from `dictionary.get_category(...).items`
   so a newly added `.dic` item cannot be silently excluded. This is the gate
   that catches a declared item bound to a missing/hardcoded column.
4. mfdb-admin setup list/detail RPC returns the structured fields (Task P4).
5. A burst run with a known `selected_setup` populates
   `mfdb_operation.setup_id` and records the setup in burst-table metadata
   (Task P5).
6. A burst run naming a non-existent setup still succeeds and emits a warning,
   leaving `mfdb_operation.setup_id` NULL.

### Sidequest: Per-User Detector-Setup Migration And Ownership

Independent of the burst chain, the detector setups a scientist defines are
*their* settings. Today they live in a single machine-wide
`detector_setups.json` and migrate into MFDB globally with no owner. The
sidequest: **auto-migrate each user's detector JSON into MFDB on first run and
associate the setups with that user, so different users keep different
settings.**

Files: `chisurf/gui/widgets/wizard/tttr_channeldefinition/tttr_detector_setups.py`,
`chisurf/core/mfdb/repository.py` (`save_setup` / `list_setups` / `get_setup`),
`chisurf/core/mfdb/schema.py`, `chisurf/core/mfdb/auth/__init__.py`
(`create_default_acl_for_object`), and the login flow that resolves the active
user (`current_user_id`).

What exists today:

- `_migrate_json_setups_to_mfdb()` already imports legacy `detector_setups.json`
  into `mfdb_setup`, but **globally** — no user is recorded, and it keys off
  "are there any setups at all" rather than "does this user have setups".
- `mfdb_setup` has **no owner column**; `save_setup()` takes no user/ACL argument.
- `setup_id_for_name()` returns `tttr_detector_setup:<slug>` — **global and
  user-agnostic**. Two users with a setup named "MFD" collide on the same
  `setup_id` and overwrite each other. This is the central problem to fix.

#### Sidequest Task S1: Record Setup Ownership

**Decision (locked): owner column, not ACL.** Setup ownership is a single
`created_by_user_id` column on `mfdb_setup`. Do **not** route setup ownership
through `mfdb_object_acl` / `create_default_acl_for_object` for this sidequest —
the column is the whole mechanism. (A future PRD may layer ACL-based *sharing* on
top; out of scope here.)

- Add `created_by_user_id TEXT REFERENCES flr_sample_users(user_id)` to
  `mfdb_setup`, declared in `mfdb_flr_ext.dic` with a `_chisurf_schema` bridge
  per the source-of-truth rule (so it is generated/validated, not hardcoded) and
  added via a `SCHEMA_VERSION` bump with `ALTER TABLE`.
- Add an `is_public INTEGER DEFAULT 0` column to `mfdb_setup` in the same way
  (dictionary-declared, generated/validated). `is_public = 1` means the setup is
  visible to **all** users; `0` means it is private to its owner.
- Extend `save_setup()` with `created_by_user_id` and `is_public` parameters and
  persist them.
- The total-coverage `validate_mapping()` gate must still pass with the new items
  mapping to live columns.

#### Sidequest Task S2: Namespace Setup IDs Per User (Avoid Collisions)

- Make setup identity user-scoped so same-named setups across users do not
  collide. Either:
  - include the user in the deterministic id, e.g.
    `tttr_detector_setup:<user_slug>:<name_slug>`, or
  - keep the name-slug id but add a `UNIQUE (owner_user_id, name)` constraint and
    resolve by (user, name) on read/write.
- Whichever is chosen, `setup_id_for_name()` and every caller
  (`tttr_detector_setups.py`, `gui/tool.py`, burst `api/mfdb.py` setup
  resolution) must pass and honor the active user. Builtin/shared setups (no
  owner) remain resolvable by all users.

#### Sidequest Task S3: Per-User Auto-Migration On First Run

- Resolve the active user from the login/session flow (the same
  `current_user_id` the GUI already determines); fall back to a defined local /
  default user when nobody is logged in, so single-user installs keep working.
- Change `_migrate_json_setups_to_mfdb()` to be **per-user idempotent**: import
  `detector_setups.json` into MFDB stamped with the active user's id only if that
  user has no setups yet. Running again, or for a different user, must not
  duplicate or overwrite another user's setups.
- After a successful per-user import, **delete the legacy JSON file from the
  settings folder** (superseding the earlier "do not delete" guidance). MFDB is
  the authoritative store; the auto-maintained `~/.chisurf/*_setups.json` must not
  linger or be re-created. See "Sidequest B addendum 2" for the full rule. JSON
  survives only as an explicit, user-initiated *export* to a chosen path.

> JSON stays a supported export/import format. MFDB is the authoritative store
> for a user's working setups, but the `detector_setups.json` shape remains a
> valid serialization for portability — exporting a setup, sharing it between
> machines/users, and embedding setups in ChiSurf project (`.csp`) files. So:
> keep `save_detector_setups()` / `load_detector_setups()` able to round-trip the
> JSON shape against an explicit `file_path`, and keep the MFDB<->JSON conversion
> (`_setup_row_data()` and the save path) lossless so an exported JSON re-imports
> into MFDB without dropping detector/window/tttr_reading fields. Do not couple
> the JSON serializer to MFDB ownership — an exported setup is owner-agnostic;
> ownership is applied on import under the active user.

#### Sidequest Task S4: Scope Reads/Writes To The Active User (Own + Public)

- `load_detector_setups()` returns **the active user's own setups plus all
  setups where `is_public = 1`** — not every user's private setups. The query is
  `created_by_user_id = <active_user> OR is_public = 1` (builtin setups with no
  owner are treated as public).
- `save_detector_setups()` / the Detector Wizard write under the active user's
  ownership and namespaced id, carrying the setup's `is_public` flag.
- Only the owner may edit or toggle the public flag of a setup. A non-owner who
  opens a public setup gets read-only / save-as-a-copy semantics (saving creates
  a setup owned by them); do not let one user mutate another user's row.
- mfdb-admin (Task P4) shows `Owner` and `Public` columns for setups so an admin
  can see per-user settings and which are shared.

#### Sidequest Task S4a: "Make Public" Checkbox In The Detector Wizard

File: `chisurf/gui/widgets/wizard/tttr_channeldefinition/tttr_channel_definition.py`
(the Detector Wizard) and `tttr_detector_setups.py`.

- Add a **"Public (visible to all users)"** checkbox to the setup save UI.
- Its state maps to the `is_public` column when saving via `save_setup()`.
- When loading an existing setup, reflect its current `is_public` value; disable
  the checkbox (and other edit controls) when the active user is not the owner.
- Default for a newly created setup is private (`is_public = 0`).

#### Sidequest Task S5: Tests

1. Two users importing different `detector_setups.json` files keep distinct,
   non-colliding setups (same setup *name* under two users does not overwrite).
2. Per-user migration is idempotent: re-running for the same user adds nothing.
3. `load_detector_setups()` for user A does not return user B's **private**
   setups, but **does** return user B's setups marked `is_public = 1`.
4. A setup saved by a logged-in user records that user as owner; toggling the
   public flag flips `is_public` and changes visibility for other users.
5. No-login / default-user path still migrates and loads setups (single-user
   install unaffected).
6. JSON round-trip is lossless: a setup exported to JSON (via `file_path`) and
   re-imported into MFDB preserves all detector/window/`tttr_reading` fields, and
   the imported copy is owned by the active user (owner-agnostic on disk).

#### Sidequest Definition Of Done

- [ ] `mfdb_setup` records setup ownership (dictionary-declared owner column,
      optional ACL), and `save_setup()` accepts/persists the owner.
- [ ] Setup ids are user-scoped; same-named setups across users no longer
      collide.
- [ ] Setups carry an `is_public` flag; users see their own setups plus all
      public ones, and a "Make public" checkbox in the Detector Wizard toggles it
      (owner-only).
- [ ] `detector_setups.json` auto-migrates per user, idempotently, stamped with
      the active (or default) user.
- [ ] Reads/writes are scoped to the active user; builtin/shared setups remain
      visible to all.
- [ ] Single-user / no-login installs keep working unchanged.
- [ ] JSON remains a supported, lossless export/import format (portability,
      `.csp` project files); MFDB<->JSON conversion drops no fields and the
      serializer is not coupled to ownership.
- [ ] Sidequest tests pass.

### Sidequest B: FCS Channel Definitions — Same Treatment

FCS channel-pair setups are the FCS analogue of detector setups and currently
suffer the identical problems: they live in a single machine-wide
`fcs_channel_setups.json` (`chisurf/core/fluorescence/fcs/channel_setups.py`,
`FCS_CHANNEL_SETUPS_FILE`), are edited by the `fcs_channel_preset` plugin, and
have no MFDB storage, no structure, no ownership, and no public/private
visibility. The code itself notes the file "mirrors the detector_setups.json
layout." Because the two are so similar, do this **now**, modelled exactly on the
detector-setup prerequisite + sidequest above.

Apply the same pattern, end to end:

- **MFDB storage as an `mfdb_setup` row** with `setup_type = "fcs_channel_setup"`,
  addressed by a deterministic, user-namespaced id
  (`fcs_channel_setup:<user_slug>:<name_slug>`), reusing `save_setup()` /
  `get_setup()` / `list_setups()`.
- **Structured child table for correlation pairs** —
  `mfdb_setup_fcs_pair` (one row per pair): `setup_id` (FK to
  `mfdb_setup(setup_id) ON DELETE CASCADE`), `name`, `channel_a`, `channel_b`,
  `kind`, plus standard audit columns. This is the FCS counterpart of
  `mfdb_setup_detector_channel`.
- **Correlator settings are per correlation pair (decision: locked).** Each pair
  has its own `n_bins`, `n_casc`, `make_fine`. They are **typed columns on
  `mfdb_setup_fcs_pair`**, dictionary-declared and generated — **not** stored in
  any `configuration_json` blob (storing structured, queryable settings in an
  opaque JSON column is rejected for the same source-of-truth reasons as the rest
  of PRD-04). See "Sidequest B addendum" below for the per-pair scheme and
  migration.
- **`.dic` dictates the schema (no hardcoded SQL).** Declare the new
  `mfdb_setup_fcs_pair` category and the correlator columns in `mfdb_flr_ext.dic`
  with `_chisurf_schema` bridges; generate their DDL via the Task P1a generator;
  the total-coverage `validate_mapping()` gate must extend to the FCS category
  (add it to `_SETUP_CATEGORIES`).
- **mfdb-admin** shows the FCS setup and its pair rows as structured records
  (entity-registry entries, dictionary-sourced labels) — same as Task P4.
- **Per-user ownership + public flag** via the same `created_by_user_id` /
  `is_public` columns already on `mfdb_setup`; reads scoped to own + public;
  defaults private; owner-only "Public" toggle in the FCS preset UI.
- **Per-user idempotent auto-migration** of `fcs_channel_setups.json`, stamped
  with the active (or default) user — same logic as `_migrate_json_setups_to_mfdb`.
- **JSON stays a lossless, owner-agnostic export/import format**
  (`load_fcs_channel_setups` / `save_fcs_channel_setups` keep round-tripping
  against an explicit `file_path`).
- **Tests** mirroring the detector sidequest: structured pair rows, total-coverage
  dict gate including the FCS category, per-user isolation + public visibility,
  idempotent migration, no-login path, lossless JSON round-trip.

Files: `chisurf/core/fluorescence/fcs/channel_setups.py`,
`chisurf/plugins/fcs/fcs_channel_preset/__init__.py`,
`chisurf/core/mfdb/{schema.py,repository.py,data/mfdb_flr_ext.dic}`,
`chisurf/plugins/core/mfdb_admin/gui/entity_registry.py`.

Boundary: factor the shared per-user/ownership/migration logic so detector and
FCS setups do not duplicate it — both are `mfdb_setup` rows distinguished by
`setup_type`, owned and shared through the same columns and the same load/save
scoping helpers. Do not fork a second ownership mechanism.

#### Sidequest B Definition Of Done

- [ ] FCS channel setups persist as `mfdb_setup` rows (`setup_type =
      fcs_channel_setup`) with a structured `mfdb_setup_fcs_pair` child table and
      dictionary-declared correlator columns; DDL generated from the `.dic`.
- [ ] FCS category is covered by the total-coverage `validate_mapping()` gate.
- [ ] FCS setups are user-scoped with `is_public` visibility and an owner-only
      public toggle in the FCS preset UI; reads return own + public.
- [ ] `fcs_channel_setups.json` auto-migrates per user, idempotently; JSON stays
      a lossless, owner-agnostic export format.
- [ ] mfdb-admin displays FCS setups and pair rows as structured records.
- [ ] Sidequest B tests pass; shared ownership/migration logic is not duplicated.

#### Sidequest B addendum: Per-Pair Correlator Settings + UI

Move the correlator settings from per-setup/global to **per correlation pair**,
stored as structured columns (not JSON). Current state: `n_bins`, `n_casc`,
`make_fine` are columns on the parent `mfdb_setup`; `mfdb_setup_fcs_pair` has only
`id, setup_id, name, channel_a, channel_b, kind`.

Schema (`schema.py`, `mfdb_flr_ext.dic`, `repository.py`):

- Add `n_bins INTEGER`, `n_casc INTEGER`, `make_fine INTEGER` to
  `mfdb_setup_fcs_pair`, each declared in `mfdb_flr_ext.dic` with
  `_chisurf_schema` bridges. The `mfdb_setup_fcs_pair` category is already in
  `_SETUP_CATEGORIES`, so the total-coverage gate covers them automatically.
  `SCHEMA_VERSION` bump; add the columns via the generator/`_ensure_column`.
- The existing `mfdb_setup.n_bins/n_casc/make_fine` become **defaults only** (seed
  values for a newly added pair); the authoritative per-pair values live on the
  `mfdb_setup_fcs_pair` row. (Keep the setup-level columns; do not remove them.)
- Extend the `save_setup` fcs-pair write path (which already writes
  `name`/`channel_a`/`channel_b`/`kind`) to persist the three correlator fields
  per row; `_fcs_row_to_data` reads them back per pair.

Migration:

- Backfill each existing `mfdb_setup_fcs_pair` row's `n_bins/n_casc/make_fine`
  from its parent `mfdb_setup` values so no setting is lost; non-lossy.

JSON:

- `fcs_channel_setups.json` remains a lossless export mirror and may carry the
  per-pair correlator values for portability, but the structured columns are the
  source of truth — never the blob.

UI (`chisurf/plugins/fcs/fcs_channel_preset/__init__.py`):

- Remove the global "Correlator settings" group box.
- Make the channel-pairs table's existing `Bins` / `Cascades` / `Fine` columns
  editable per row, bound to the per-pair structured columns; a newly added pair
  seeds them from the setup-level defaults.
- Add a delete tool-button at the end of each pair row.
- Add a construction smoke test for `FCSChannelDialog` (mirror
  `test/gui/test_detector_wizard_page.py`) — the FCS dialog's widget-build path
  currently has no coverage.

Addendum Definition Of Done:

- [ ] `mfdb_setup_fcs_pair` has dictionary-declared `n_bins`/`n_casc`/`make_fine`
      columns; gate passes; no correlator settings are stored in a JSON blob.
- [ ] save/load round-trips per-pair correlator settings through the structured
      columns; migration backfills existing pairs from the setup-level defaults.
- [ ] FCS dialog shows per-pair Bins/Cascades/Fine (global box removed) with a
      per-row delete button; a construction smoke test passes.

#### Sidequest B addendum 2: MFDB-default save; migrate-and-remove legacy JSON

Observed defect: saving an FCS setup pops "Saved FCS channel pairs for setup
'Test' to: `/Users/<user>/.chisurf/fcs_channel_setups.json`". The save is
MFDB-first under the hood, but (a) the confirmation message hard-codes the JSON
path (`FCS_CHANNEL_SETUPS_FILE`) regardless of where it actually saved, and (b)
when MFDB is unavailable it silently writes the user-folder JSON. The user-folder
JSON must stop being a default store.

Rules:

- **Default Save → MFDB only, owned by the active user.** `save_fcs_channel_setups`
  with no `file_path` persists to MFDB (`created_by_user_id` = active user) and
  does **not** write `~/.chisurf/fcs_channel_setups.json`. The confirmation
  message must reflect the real target (e.g. "Saved to MFDB for setup 'Test'"),
  not a JSON path.
- **Migrate and remove the legacy file.** On first access, after the existing
  `~/.chisurf/fcs_channel_setups.json` is migrated into MFDB (per-user idempotent,
  already implemented), **delete the file** from the settings folder. Do not
  re-create it. The same applies to `detector_setups.json` (parallel cleanup).
  Guard the delete: only remove after a verified successful import.
- **JSON becomes export-only.** JSON serialization survives solely as an explicit,
  user-initiated *Export to file…* action that writes to a user-chosen path (for
  portability and `.csp`). It is never the default Save target and never written
  to `~/.chisurf`. `save_fcs_channel_setups(..., file_path=<explicit path>)`
  remains the export entry point.
- **No-MFDB fallback (soft, with warning):** if MFDB genuinely cannot be opened,
  Save still succeeds by writing the JSON fallback, but the user is **warned** that
  MFDB was unavailable, so the setup was saved to a local JSON file and is **not**
  stored in the database or assigned to a user. The save must not fail or lose
  data — it degrades with a clear warning. (This is the only path that may touch
  the settings JSON; the normal MFDB path never does.)

Files: `chisurf/core/fluorescence/fcs/channel_setups.py`
(`save_fcs_channel_setups`, `_migrate_json_to_mfdb`),
`chisurf/plugins/fcs/fcs_channel_preset/__init__.py` (`_on_save` message,
optional Export action), and the shared `tttr_setup_utils` migration helper
(to perform the post-migration file removal for both FCS and detector setups).

Addendum 2 Definition Of Done:

- [ ] Default Save persists FCS setups to MFDB owned by the active user; no
      `~/.chisurf/fcs_channel_setups.json` is written, and the confirmation
      message names MFDB, not a JSON path.
- [ ] Legacy `fcs_channel_setups.json` (and `detector_setups.json`) are migrated
      into MFDB and then removed from the settings folder; not re-created.
- [ ] JSON export is available only via an explicit Export-to-file action to a
      user-chosen path; the canonical settings JSON is never the default store.
- [ ] No-MFDB case degrades softly: the save still succeeds via JSON fallback
      but warns the user that MFDB was unavailable and the setup is not stored in
      the database / not assigned to a user.
- [ ] Tests: default save writes MFDB and not the settings JSON; migration removes
      the legacy file; export-to-explicit-path still round-trips.

### Sidequest C: G-Factor Calibration Provenance (Reference Decay In MFDB)

The detector-setup work surfaced two related defects:

1. **The Jordi G-factor calculator does not archive what it computes.** The plugin
   (`chisurf/plugins/jordi_g_factor/`) computes a G-factor by tail-matching a
   polarization-resolved *reference decay* (VV/VH, a "Jordi" file), but only
   stores the raw file blob via `mfdb.objects.put` and exports a CSV. There is no
   typed result, no operation, no parameters, and no provenance edge. So a
   detector setup carries a `g_factor` / `l1` / `l2` with **no traceable link to
   the reference decay that produced it** — the analysis cannot be fully traced.
2. **The Jordi G-factor calc and FCS channel setup are partially broken** after
   the detector-setup refactor (changed setup API / read paths). The pure math
   (`core/calculations.py`) still passes its unit tests, so the breakage is in the
   GUI/MFDB integration and the detector-setup coupling, not the algorithm.

The reference decay is scientifically the relevant datum: it *is* the calibration
evidence. It must live in MFDB, linked to the G-factor it yields and to the
detector channel that consumes it.

> Relationship to PRD-05: PRD-05 (Calibration Provenance) owns the general
> calibration framework (g-factor, gamma, crosstalk, direct excitation, Förster
> radius) and already lists `chisurf/plugins/jordi_g_factor/`. Sidequest C pulls
> the **g-factor case** forward because it is tied to the detector-setup channel
> calibration shipped here. Reuse PRD-03's `register_calibration()` (already
> present in `result_registry.py`); do not hand-roll writes and do not
> re-implement PRD-05's broader framework.

#### Provenance chain to build

```text
reference decay (Jordi VV/VH file)        [raw_measurement / tcspc_decay artifact]
  -> operation: calibration (calibration_type = "g_factor")
  -> calibration_data result                [g_factor (+ stddev), l1, l2, region, bg, r_inf]
  -> consumed by mfdb_setup_detector_channel  [g_factor/l1/l2 + link to the calibration]
```

#### Sidequest C Task C1: Register The Reference Decay

File: `chisurf/plugins/jordi_g_factor/` (gui/client, backend/services).

- Register the polarization-resolved reference decay as an MFDB artifact (a
  `raw_measurement`, or a typed `tcspc_decay` payload carrying the VV and VH
  channels) instead of only `mfdb.objects.put`. Keep the object-store put for the
  raw bytes; add the typed artifact so the decay is queryable and parentable.
- Metadata: source filename, channel roles (`parallel`/VV, `perpendicular`/VH),
  time axis / micro-time resolution, and the active user.

#### Sidequest C Task C2: Register The G-Factor As A Calibration

- After a successful calculation, call
  `register_calibration(payload, calibration_type="g_factor",
  source_artifact_id=<reference-decay artifact>, ...)` so the result is a
  `calibration_data` artifact under a `calibration` operation, parented to the
  reference decay.
- Parameters (scalars in `mfdb_parameter`): `g_factor`, `g_factor_stddev`,
  `g_factor_uncorrected`, `g_factor_corrected`, `r_inf`, `region_min`,
  `region_max`, `decay_shift`, `flip`, `use_bg`, `bg_vv`, `bg_vh`, `l1`, `l2`.
- Best-effort: a failed registration must warn, never break the calculator UI.

#### Sidequest C Task C3: Link The Detector Channel To Its Calibration

Files: `chisurf/core/mfdb/{schema.py, data/mfdb_flr_ext.dic, repository.py}`,
`chisurf/gui/widgets/wizard/tttr_channeldefinition/`.

- Add a dictionary-declared `g_factor_calibration_id` column to
  `mfdb_setup_detector_channel` (the `.dic` dictates the schema; generate the
  column, total-coverage gate must still pass). It references the calibration
  artifact/operation that produced the channel's `g_factor` / `l1` / `l2`.
- When the Detector Wizard applies a computed G-factor to a channel, record the
  calibration id so the channel's calibration is traceable back to the reference
  decay. When entered manually, leave it null.

#### Sidequest C Task C4: Fix The Broken Integration

- Repair the Jordi G-factor plugin's GUI/MFDB path and its detector-setup
  coupling against the current setup API (`setup_id_for_name(name, user_id,
  prefix)`, the structured child tables, the per-user/`is_public` model,
  read-only construction). Root-cause and fix the "does not work" GUI flow; the
  `core/calculations.py` unit tests already pass, so focus on the service/GUI/
  MFDB seams.
- Resolve the FCS channel-setup regression introduced by the refactor (the load
  path migrating/writing on construction; align it with the read-only
  construction pattern used for detector setups).

#### Sidequest C Definition Of Done

- [ ] The reference decay (VV/VH) is registered as a queryable MFDB artifact, not
      only an opaque object-store blob.
- [ ] A computed G-factor is archived via `register_calibration()`
      (`calibration_type="g_factor"`) parented to the reference-decay artifact,
      with scalar parameters recorded.
- [ ] `mfdb_setup_detector_channel` carries a dictionary-declared
      `g_factor_calibration_id` linking the channel's g-factor to its calibration;
      gate still passes.
- [ ] The Jordi G-factor calculator works end to end against the current
      detector-setup API; the FCS channel-setup regression is resolved.
- [ ] MFDB-archival failures never break the calculator UI (best-effort, warns).
- [ ] Sidequest C tests pass: reference-decay artifact registered, calibration
      registered and parented, channel→calibration link recorded, calc runs
      without raising when MFDB is unavailable.

### Sidequest D: Time-Versioned Setup Calibration (Calibration Drift)

A setup drifts over time: detectors age, alignment shifts, so the calibration
factors (g-factor, l1, l2, …) determined today differ from last month's. The
current scheme **overwrites** a channel's calibration in place
(`save_setup` does `INSERT ... ON CONFLICT(setup_id) DO UPDATE`, and the
detector-channel factors are replaced), so the setup only ever holds the *latest*
calibration. Re-analyzing data measured under an earlier calibration is then
impossible — the historical factors are gone. Calibration is **date-dependent**
and must be versioned.

#### The better scheme: separate stable definition from dated calibration

Split the two concerns that are currently conflated in one mutable row:

- **Structural definition (stable):** what the channels *are* — routing
  (`channels`, `micro_time_ranges`, `g_factor_channels`), PIE windows, TTTR
  reading. Stays in `mfdb_setup` / `mfdb_setup_detector_channel` /
  `mfdb_setup_pie_window`. Changes rarely.
- **Calibration (time-varying):** the *factors* that drift — `g_factor`, `l1`,
  `l2` (room for `gamma`, crosstalk, etc. later). These move into a new
  **dated-snapshot** table, append-only, never overwritten.

```text
mfdb_setup (definition)
  └─ mfdb_setup_detector_channel (routing; "current" factor convenience columns)
       └─ mfdb_setup_calibration (one row per channel per calibration date)
            valid as of calibrated_at, linked to the reference-decay calibration
```

#### Task D1: `mfdb_setup_calibration` table

`.dic`-declared and generated (source-of-truth rule; add to `_SETUP_CATEGORIES`,
gate must pass). Columns:

- `calibration_snapshot_id` TEXT PK
- `setup_id` TEXT FK → `mfdb_setup(setup_id)` ON DELETE CASCADE
- `channel_name` TEXT — which detector channel these factors apply to
- `g_factor` REAL, `l1` REAL, `l2` REAL
- `g_factor_calibration_id` TEXT — link to the Sidequest C `calibration_data`
  artifact (the reference decay / evidence). Null for manual entry.
- `calibrated_at` TEXT NOT NULL — full ISO **date and time** the calibration was
  determined (not just a date; two calibrations on one day must be distinct)
- `method` TEXT (e.g. `jordi_g_factor`, `manual`), `notes` TEXT
- `created_by_user_id`, `created_at`, `updated_at`, `deleted_at`
- index on `(setup_id, channel_name, calibrated_at)`

#### Task D2: Append, don't overwrite

- When new calibration factors are determined for a setup channel (a G-factor
  calc applied, or manual edit committed as a calibration), **INSERT a new
  `mfdb_setup_calibration` row** stamped with the current date+time. Never update
  a prior snapshot.
- `mfdb_setup_detector_channel.g_factor`/`l1`/`l2` become a *cache of the latest*
  snapshot (kept for back-compat and quick reads); the snapshot table is the
  source of truth for history.
- `save_setup` stops mutating calibration factors in place; structural edits and
  calibration updates are distinct paths.

#### Task D3: Setup + date selection in the Detector menu

File: `chisurf/gui/widgets/wizard/tttr_channeldefinition/tttr_channel_definition.py`.

- Next to the existing **setup combobox**, add a **date/calibration combobox**.
- Populate it from `mfdb_setup_calibration` for the selected setup: distinct
  `calibrated_at` values, formatted as date+time, most-recent first, with a
  `Latest` entry at the top (the newest snapshot).
- Changing the setup repopulates the date combobox; changing the date loads that
  snapshot's `g_factor`/`l1`/`l2` into the channel table.
- A setup with no calibration snapshots shows only `Latest` (the cached factors),
  and migration (D5) backfills one snapshot so the picker is never empty for
  previously-calibrated setups.

#### Task D4: Repository + load/save API

File: `chisurf/core/mfdb/repository.py`, `tttr_setup_utils.py`.

- `add_setup_calibration(setup_id, channel_name, g_factor, l1, l2,
  g_factor_calibration_id, calibrated_at, method, notes, created_by_user_id)`.
- `list_setup_calibration_dates(setup_id)` → distinct `calibrated_at` desc.
- `get_setup_calibration(setup_id, calibrated_at|"latest")` → per-channel factors
  for that snapshot.
- `get_setup()` returns the latest snapshot per channel as the current factors.

#### Task D5: Migration

- Seed one `mfdb_setup_calibration` row per existing detector channel from its
  current `g_factor`/`l1`/`l2`, `calibrated_at = channel.created_at` (fallback
  now), `method = "migrated"`, carrying any existing `g_factor_calibration_id`.
- `SCHEMA_VERSION` bump; columns added via the generator/`_ensure_column`.

#### Task D6: Analysis reproducibility

- When an analysis/operation uses a setup, record **both** the `setup_id` and the
  calibration snapshot actually used (`calibration_snapshot_id` or
  `calibrated_at`) in the operation metadata, so a re-run reproduces the exact
  calibration. (Burst pipeline `setup_id` wiring from the main body extends to
  carry the calibration date.)

> Boundary / PRD-05: PRD-05 owns the general calibration framework; Sidequest D is
> the **temporal storage + selection** mechanism for setup calibration, reusing
> the Sidequest C g-factor evidence link. Do not fork a second calibration store.

#### Sidequest D Definition Of Done

- [ ] `mfdb_setup_calibration` exists (dict-declared, generated, gate passes),
      append-only, linked to the reference-decay calibration artifact.
- [ ] New calibration factors create a dated snapshot instead of overwriting;
      `mfdb_setup_detector_channel` factors cache the latest.
- [ ] The Detector menu has a date/calibration combobox next to the setup
      combobox; selecting a date loads that snapshot's factors (date+time shown).
- [ ] Migration backfills one snapshot per existing channel; no previously
      calibrated setup shows an empty date picker.
- [ ] Analyses record the calibration snapshot/date used for reproducibility.
- [ ] Sidequest D tests pass: snapshot append (no overwrite), date listing,
      load-by-date, latest resolution, migration backfill.

## Files To Read First

- `chisurf/core/mfdb/result_registry.py` - PRD-03 registration boundary.
- `chisurf/core/mfdb/payload_models.py` - `BurstTable` and `BurstSelection` payloads.
- `chisurf/core/mfdb/repository.py` - direct repository primitives for the few
  links that PRD-03 does not expose.
- `chisurf/core/mfdb/pipeline.py` - legacy/broken burst pipeline to replace or
  deprecate.
- `chisurf/plugins/burst/burst_selection/api/models.py` - `AnalysisRequest` and
  `AnalysisResult`.
- `chisurf/plugins/burst/burst_selection/api/contract.py` - JSON/RPC contract.
- `chisurf/plugins/burst/burst_selection/api/selection.py` - pure analysis code.
- `chisurf/plugins/burst/burst_selection/backend/services.py` - RPC/service adapter.
- `chisurf/plugins/burst/burst_selection/gui/client.py` and
  `chisurf/plugins/burst/burst_selection/gui/tool.py` - GUI request construction.
- `chisurf/gui/widgets/wizard/tttr_channeldefinition/tttr_detector_setups.py` -
  how detector/channel setups are persisted to `mfdb_setup` today (prerequisite).
- `chisurf/core/mfdb/data/mfdb_flr_ext.dic` - flrCIF extension dictionary that
  must describe setup/detector/channel fields (prerequisite).
- `chisurf/plugins/core/mfdb_admin/gui/entity_registry.py` - how setups appear
  in mfdb-admin (prerequisite).
- `overhaul/PRD-08-optical-configuration.md` - structured optical/channel schema
  that detector and window definitions map onto (prerequisite owner).

## Current Problems

1. `chisurf/core/mfdb/pipeline.py` uses stale repository vocabulary:
   - `artifact_type` should be `artifact_kind`.
   - `storage_mode="local"` should be `"local_file"`.
   - `status="success"` should be `"succeeded"`.
2. `chisurf/core/mfdb/pipeline.py` imports and executes the Burst Selection
   backend from inside `chisurf.core`. That makes core depend on a plugin and
   creates the wrong ownership boundary.
3. The current pipeline performs manual MFDB writes instead of using
   `register_result()`.
4. The Burst Selection request contract has no stable MFDB context for sample
   identity, pre-existing source artifact IDs, or opt-out behavior.
5. `AnalysisResult.output_paths` is currently a flat role map, so multi-file
   outputs can overwrite each other for keys such as `"bur"`.
6. Burst outputs do not consistently appear in mfdb-admin as typed
   `burst_table` artifacts with parent/source provenance.

## Design Principles

- Keep analysis pure. `analyze_request(request)` computes and writes requested
  files; it must not require MFDB and should remain directly testable without a
  database.
- Register after success. MFDB registration runs only after analysis succeeds.
- Use PRD-03 for writes. Store inputs and outputs with `register_raw_measurement()`
  and `register_result()`.
- Prefer per-file provenance. For each input TTTR file, register one raw input
  artifact and one primary burst-table artifact when output rows exist.
- Preserve optionality. MFDB failure must not break burst analysis in GUI/CLI/RPC
  flows; it should produce warnings and empty artifact IDs.
- Avoid core-to-plugin dependencies. Burst-specific orchestration belongs under
  `chisurf/plugins/burst/burst_selection/`. Core MFDB modules may expose generic
  helpers only.
- Keep the external contract explicit and canonical. MFDB context is nested
  under `mfdb`; legacy top-level provenance fields and legacy RPC aliases are
  intentionally not part of the reference implementation.

## Stable Pipeline Contract

### Ownership

Create the stable implementation in:

```text
chisurf/plugins/burst/burst_selection/api/mfdb.py
```

This module is plugin-owned and may import Burst Selection API models. It may
import MFDB/result-registry primitives. `chisurf/core/mfdb/pipeline.py` must no
longer execute burst analysis directly.

`chisurf/core/mfdb/pipeline.py` is removed. Core MFDB exports do not expose
`BurstPipeline`; plugin-specific registration belongs in
`chisurf.plugins.burst.burst_selection.api.mfdb`.

### Request Context

Add a stable MFDB context to `AnalysisRequest`. Use a nested dataclass instead of
more top-level loose fields so future provenance attributes can be added without
churning the request signature.

```python
@dataclass
class MFDBContext:
    """MFDB archival context for a burst-selection request."""

    enabled: bool = True
    sample_id: str = ""
    source_artifact_ids: dict[str, str] = field(default_factory=dict)
    register_missing_inputs: bool = True
```

Add to `AnalysisRequest`:

```python
mfdb: MFDBContext = field(default_factory=MFDBContext)
```

Semantics:

- `enabled=False`: skip MFDB registration entirely.
- `sample_id`: existing MFDB sample to link to all registered artifacts.
- `source_artifact_ids`: optional mapping from normalized input file path to an
  existing `raw_measurement` artifact ID. If present, reuse it instead of
  registering that input file again.
- `register_missing_inputs=True`: register input files that are not in
  `source_artifact_ids`.

Update `analysis_request_from_payload()`, `analysis_request_to_payload()`, and
`contract_descriptor()` to accept:

```json
{"mfdb": {"sample_id": "...", "source_artifact_ids": {...}}}
```

Top-level provenance fields are intentionally not supported.

### Result Contract

Extend `AnalysisResult` with:

```python
output_paths_by_file: dict[str, dict[str, str]] = field(default_factory=dict)
mfdb_artifacts: dict[str, Any] = field(default_factory=dict)
warnings: list[str] = field(default_factory=list)
```

Semantics:

- `output_paths` is a convenience role map. It may hold the last path for a role.
- `output_paths_by_file[input_path][role]` is the stable per-file output map.
  `analyze_file()` should populate it for one file, and `analyze_request()`
  should merge it without role collisions.
- `mfdb_artifacts` is filled by the service/registration layer, not by pure
  `analyze_request()` unless a new explicit `analyze_and_register_request()`
  helper is introduced.
- `warnings` carries non-fatal MFDB registration failures.

### Pipeline Result

In `api/mfdb.py`, define a small registration result object:

```python
@dataclass
class BurstRegistrationResult:
    input_artifacts: dict[str, str] = field(default_factory=dict)
    burst_table_artifacts: dict[str, str] = field(default_factory=dict)
    sidecar_artifacts: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
```

Keys for `input_artifacts` and `burst_table_artifacts` are normalized input file
paths. Keys for `sidecar_artifacts` are stable roles such as `"hdf5"`, `"zip"`,
or `"output_folder"`.

### Pipeline API

The stable pipeline class should look like this:

```python
class BurstMFDBPipeline:
    """Register Burst Selection inputs and outputs in MFDB."""

    def __init__(self, db=None):
        self.db = db

    def register_run(
        self,
        request: AnalysisRequest,
        result: AnalysisResult,
    ) -> BurstRegistrationResult:
        ...
```

`register_run()` must:

1. Return an empty `BurstRegistrationResult` immediately when
   `request.mfdb.enabled` is false.
2. Normalize input file paths with `Path(path).resolve()` for mapping keys.
3. Reuse `request.mfdb.source_artifact_ids[path]` when provided.
4. Register missing raw inputs with `register_raw_measurement()`.
5. Register one `burst_table` artifact per input file that produced burst rows or
   a `.bur` output.
6. Register optional sidecar artifacts only after primary burst tables are
   registered.
7. Never raise into GUI/CLI/RPC flow for MFDB failures; append warnings instead.

Direct unit tests may assert stricter behavior by passing an explicit `db` and
checking returned IDs.

## Artifact And Provenance Model

### Input Artifacts

Each input TTTR file is a `raw_measurement` artifact:

```python
register_raw_measurement(
    file_path=input_path,
    sample_id=request.mfdb.sample_id,
    metadata={
        "plugin": "burst_selection",
        "role": "raw_tttr",
        "filetype": request.filetype,
        "selected_setup": request.selected_setup,
    },
    db=db,
)
```

If `source_artifact_ids` provides an artifact ID for the file, do not register a
new raw artifact. Still use that ID as the parent for the burst table.

### Primary Output Artifacts

Primary burst results are `burst_table` artifacts.

Preferred payload order:

1. If a per-file `.bur` path exists, register that file path with
   `kind="burst_table"` and `data_format="bur"` inferred from the suffix.
2. If no `.bur` path exists but result data rows exist, convert the per-file rows
   to a pandas DataFrame and call `register_result(kind="burst_table", data=df)`,
   which stores a typed msgpack `BurstTable`.

Use:

```python
register_result(
    kind="burst_table",
    data=bur_path_or_dataframe,
    sample_id=request.mfdb.sample_id,
    parent_artifact_id=input_artifact_id,
    operation_type="burst_selection",
    parameters=extract_burst_parameters(request),
    metadata=build_burst_metadata(request, result, input_path),
    db=db,
)
```

Metadata must include:

- `plugin`: `"burst_selection"`
- `contract_version`: the value from `api.contract.CONTRACT_VERSION`
- `input_file`: normalized input path
- `output_role`: `"burst_table"`
- `n_bursts`, `n_selected`, `n_photons` when available
- `macro_time_resolution` when available
- JSON-safe `photon_filter`, `burst_detection`, and `gmm` settings
- `windows`, `detectors`, `selected_setup`

Parameters must include only scalar values appropriate for `mfdb_parameter`,
for example:

- `min_photons`
- `photon_window`
- `time_window`
- `filter_active`
- `count_rate_n_ph_max`
- `count_rate_time_window`
- `delta_macro_time_min`
- `delta_macro_time_max`
- `gmm_max_components`

Do not store full nested settings as parameter rows; nested settings belong in
metadata.

### Optional Selection Payload

If the analysis code exposes start/stop indices or a selection mask, register an
additional typed `burst_selection` payload:

```python
register_result(
    kind="burst_selection",
    data=BurstSelection(
        source_artifact_id=burst_table_artifact_id,
        start_indices=start_indices,
        stop_indices=stop_indices,
        criteria=criteria_dict,
    ),
    sample_id=request.mfdb.sample_id,
    parent_artifact_id=burst_table_artifact_id,
    operation_type="burst_selection",
    metadata={...},
    db=db,
)
```

This is optional for PRD-04 unless the current API already exposes the mask or
start/stop arrays without re-reading raw TTTR data. Do not invent a mask from
summary-only `.bur` rows.

### Sidecar Artifacts

Sidecar outputs are not primary scientific burst tables. Register them only when
they are actually created:

| Role | Artifact kind | Data | Parent |
| --- | --- | --- | --- |
| `hdf5` | `processed_data` | HDF5 path | first burst table artifact |
| `zip` | `processed_data` | zip path | first burst table artifact |
| `output_folder` | `external_reference` or `processed_data` | folder path or zip path | first burst table artifact |
| `mti_dir` | `external_reference` | folder path | matching burst table artifact |

Use metadata role labels so mfdb-admin can distinguish primary burst tables from
packaged convenience outputs.

## Operation Boundaries

The primary operation type is always:

```text
burst_selection
```

`register_result()` creates one operation per registered artifact. That is
acceptable for PRD-04. Do not bypass PRD-03 just to force all outputs into one
operation row.

If a later PRD needs a single multi-output operation, extend `result_registry`
with a batch API instead of hand-writing operation rows here.

## Error Handling

MFDB registration must be best effort for user-facing analysis flows:

- Missing DB: return empty artifact IDs and warnings.
- Invalid `sample_id`: do not crash analysis; return a warning.
- Invalid source artifact ID: do not crash analysis; return a warning.
- Failed primary burst-table registration: continue with other files.
- Failed sidecar registration: keep primary burst-table artifacts.

Tests that call `BurstMFDBPipeline` directly should verify that failure paths are
reported and do not leave partial rows for the failed artifact. PRD-03 already
covers transaction rollback inside `register_result()`.

## Tasks

### Task 1: Replace The Legacy Core Pipeline Boundary

File: `chisurf/core/mfdb/pipeline.py`

- Remove direct manual calls using stale arguments such as `artifact_type`,
  `storage_mode="local"`, and `status="success"`.
- Remove direct execution of `analyze_files_handler()` from core.
- Delete the core pipeline module and remove `BurstPipeline` from
  `chisurf.core.mfdb` exports.
- Verify no remaining stale values:

```bash
rg 'artifact_type|storage_mode="local"|status="success"|status=.*success' \
  chisurf/plugins/burst/burst_selection
```

Expected: no matches, unless a deprecation message quotes them in prose.

### Task 2: Add MFDB Request And Result Contract Fields

Files:

- `chisurf/plugins/burst/burst_selection/api/models.py`
- `chisurf/plugins/burst/burst_selection/api/contract.py`
- `chisurf/plugins/burst/burst_selection/api/serialization.py` if needed

Implement:

- `MFDBContext`
- `AnalysisRequest.mfdb`
- `AnalysisResult.output_paths_by_file`
- `AnalysisResult.mfdb_artifacts`
- `AnalysisResult.warnings`
- payload normalization for nested `mfdb`
- contract descriptor documentation for the new fields

Contract requirements:

- Existing payloads without `mfdb` still work.
- `analysis_request_to_payload()` round-trips `mfdb` as JSON-safe dictionaries.

### Task 3: Preserve Per-File Output Paths

File: `chisurf/plugins/burst/burst_selection/api/selection.py`

- `analyze_file()` must populate `output_paths_by_file[str(path)]`.
- `analyze_request()` must merge `output_paths_by_file` without overwriting
  per-file `"bur"` paths.
- Keep existing `output_paths` for compatibility.

This is required before MFDB registration can reliably map each `.bur` file back
to its source TTTR file.

### Task 4: Implement The Plugin-Owned MFDB Pipeline

File to create: `chisurf/plugins/burst/burst_selection/api/mfdb.py`

Implement:

- `BurstRegistrationResult`
- `BurstMFDBPipeline`
- `extract_burst_parameters(request)`
- `build_burst_metadata(request, result, input_path)`
- `registration_result_to_payload(result)` or rely on dataclass serialization

Use the PRD-03 result registry. Do not manually write artifact, operation, or
object-store rows unless adding non-primary sidecar links that PRD-03 cannot
express.

### Task 5: Register From The Service Adapter

File: `chisurf/plugins/burst/burst_selection/backend/services.py`

After:

```python
result = analyze_request(request)
```

call the MFDB pipeline when `request.mfdb.enabled` is true:

```python
registration = BurstMFDBPipeline().register_run(request, result)
result.mfdb_artifacts = dataclass_to_dict(registration)
result.warnings.extend(registration.warnings)
```

Do not call the pipeline from the pure `analyze_request()` function unless adding
a separate explicit helper named `analyze_and_register_request()`.

### Task 6: Pass MFDB Context From GUI And Client

Files:

- `chisurf/plugins/burst/burst_selection/gui/client.py`
- `chisurf/plugins/burst/burst_selection/gui/tool.py`
- any GUI adapter that constructs analysis payloads

Add optional `mfdb` parameter to `BurstSelectionClient.analyze_files()`.

GUI behavior:

- If a sample picker exists, pass its selected `sample_id`.
- If no sample is selected, pass no sample and still allow analysis.
- Do not create samples in PRD-04; sample creation belongs to PRD-02/PRD-02b.
- Show/log MFDB warnings from the result without failing the analysis run.

### Task 7: Add Focused Tests

Create: `test/fio/test_burst_pipeline_mfdb.py`

Required tests:

1. Registers one raw input artifact per input file when no source artifact is
   supplied.
2. Reuses `source_artifact_ids` and does not duplicate a raw input artifact.
3. Registers a `burst_table` artifact for a per-file `.bur` output.
4. Registers a typed msgpack `burst_table` artifact from DataFrame rows when no
   `.bur` path exists.
5. Creates `derived_from` provenance from each burst table to its matching raw
   input artifact.
6. Links registered artifacts to `sample_id` when provided.
7. Records scalar burst parameters in `mfdb_parameter`.
8. Does not crash and reports warnings when MFDB is unavailable.
9. Handles two input files without overwriting per-file output paths.
10. Leaves zero artifact/object rows for failed invalid-sample registration.

Use the real `MFDatabase`, `create_sample()`, and `read_result()` helpers. Use
`db.conn`, not `db.con`.

Also update plugin-local tests under:

```text
chisurf/plugins/burst/burst_selection/tests/
```

Required plugin tests:

- `analysis_request_from_payload()` accepts nested `mfdb` context.
- `analysis_result_to_payload()` includes `mfdb_artifacts` and warnings.
- service handler returns `mfdb_artifacts` on a successful run when registration
  is enabled and a DB is available.

## Verification Commands

Run the narrow MFDB/payload tests:

```bash
PYTHONPATH="modules/chinet:modules/imp-tricks/src:." \
  /Users/tpeulen/mambaforge/envs/arm64/bin/python3 \
  -m pytest -p no:cov -o addopts='' \
  test/fio/test_payload_codec.py \
  test/fio/test_result_registry.py \
  test/fio/test_burst_pipeline_mfdb.py
```

Run the Burst Selection plugin tests:

```bash
PYTHONPATH="modules/chinet:modules/imp-tricks/src:." \
  /Users/tpeulen/mambaforge/envs/arm64/bin/python3 \
  -m pytest -p no:cov -o addopts='' \
  chisurf/plugins/burst/burst_selection/tests
```

Run a stale-vocabulary search:

```bash
rg 'artifact_type|storage_mode="local"|status="success"|parameter_name|parameter_value|db\.con' \
  chisurf/plugins/burst/burst_selection \
  test/fio/test_burst_pipeline_mfdb.py
```

Expected: no implementation/test matches except intentional prose in
documentation.

## Definition Of Done

- [ ] Prerequisite: each named setup resolves to an `mfdb_setup` row via
      `setup_id_for_name()`; its detector and PIE/window definitions are stored
      as structured `mfdb_setup_detector_channel` / `mfdb_setup_pie_window` rows
      (the reading/processing base that PRD-08 later extends), not as opaque
      blobs.
- [ ] Prerequisite: `mfdb_flr_ext.dic` dictates the schema — every new item
      declares the `_chisurf_schema` bridge; the `CREATE TABLE` DDL for the
      setup/detector/window tables is **generated from the dictionary** (Task
      P1a), not hand-written in `schema.py`; the total-coverage
      `validate_mapping()` gate passes with no unmapped items and no curated
      allow-list; no field vocabulary is hardcoded a second time in
      `schema.py` / `entity_registry.py`; and the setup with its detector/window
      definitions displays in mfdb-admin with labels/enums sourced from the
      dictionary.
- [ ] Prerequisite: the `burst_selection` operation populates
      `mfdb_operation.setup_id`, so a `burst_table` artifact traces back to the
      setup that produced it.
- [ ] Core no longer owns or executes Burst Selection analysis.
- [ ] `chisurf/core/mfdb/pipeline.py` is removed and no core MFDB export points
      to plugin-owned burst analysis or registration.
- [ ] Burst Selection request contract includes `MFDBContext`.
- [ ] Burst Selection result contract preserves per-file output paths and can
      return MFDB artifact IDs/warnings.
- [ ] Plugin-owned `BurstMFDBPipeline` registers inputs and burst-table outputs
      through PRD-03 result registry.
- [ ] Per-file `burst_table` artifacts derive from the matching raw TTTR
      artifact.
- [ ] Existing source artifact IDs are reused instead of duplicating raw inputs.
- [ ] Sample IDs, when provided, link registered artifacts to the sample.
- [ ] MFDB registration failures do not fail GUI/CLI/RPC analysis.
- [ ] mfdb-admin can list registered `burst_table` artifacts and their
      provenance edges.
- [ ] Focused MFDB tests and Burst Selection plugin tests pass.
