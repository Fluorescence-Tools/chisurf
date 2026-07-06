---
type: PRD
prd: "04"
title: "PRD-04: Stable Burst Pipeline MFDB Integration"
description: Register burst-selection results in MFDB with stable, queryable provenance across all callers
status: in-progress
phase: "0"
resource: chisurf/plugins/burst/burst_selection
tags: [prd, mfdb, fret]
timestamp: '2026-07-05T00:00:00Z'
---

# Summary
When the Burst Selection plugin produces burst tables, those results are
registered in MFDB with stable, queryable provenance (raw TTTR artifacts →
burst_selection operation → burst_table artifacts → optional sidecar outputs),
identically across GUI, CLI, RPC, and direct API callers. The scientific
analysis API stays the source of truth for computing burst results; MFDB
registration is a narrow archival side effect layered on top. It also makes the
acquisition setup — detector definitions plus PIE/microtime window (channel)
definitions and the TTTR reading routine — a first-class MFDB record so a burst
run can be traced back to what setup produced the photons.

# Status
In progress. Burst Selection is the reference implementation for
workflow-ready plugins with MFDB archival; it must not reimplement storage,
operation/parameter recording, or payload serialization owned by the registry
and codec layers. The main body is implemented: `chisurf/core/mfdb/pipeline.py`
is removed, `chisurf/plugins/burst/burst_selection/api/mfdb.py` holds
`BurstMFDBPipeline`, and `register_result()` / `register_raw_measurement()`
accept and forward `setup_id`. The setup/channel prerequisite and its four
sidequests (per-user ownership, FCS channel definitions, g-factor calibration
provenance, time-versioned calibration) are implemented and reviewed.

# Goal
Register burst tables with stable, queryable provenance:

```text
raw TTTR artifact(s)
  -> operation: burst_selection
  -> burst_table artifact(s)
  -> optional packaged sidecar/output artifacts
```

The pipeline must be stable across GUI, CLI, RPC, and direct API callers.
`analyze_request(request)` remains the source of truth for computing results;
MFDB registration is layered on top of the analysis result. PRD-04 must not
reimplement object storage, operation recording, parameter recording, or payload
serialization — those belong to the result-registry and codec layers.

# Design principles
- **Keep analysis pure.** `analyze_request(request)` computes and writes requested
  files; it must not require MFDB and stays directly testable without a database.
- **Register after success.** MFDB registration runs only after analysis succeeds.
- **Use the result registry for writes** (`register_raw_measurement()` /
  `register_result()`); do not hand-write artifact/operation/object rows.
- **Prefer per-file provenance.** Per input TTTR file, register one raw input
  artifact and one primary burst-table artifact when output rows exist.
- **Preserve optionality.** MFDB failure must not break burst analysis in any
  flow; produce warnings and empty artifact IDs.
- **Avoid core→plugin dependencies.** Burst orchestration lives under
  `chisurf/plugins/burst/burst_selection/`; core MFDB modules expose only generic
  helpers.
- **Keep the external contract explicit and canonical.** MFDB context is nested
  under `mfdb`; legacy top-level provenance fields and RPC aliases are not part of
  the reference implementation.

# Prerequisite: setup and channel definitions in MFDB
Full provenance tracing is only possible if the *setup* that produced the photons
is itself a first-class, queryable MFDB record. In chisurf the setup is defined by
its detector definitions and its PIE/microtime window (channel) definitions plus
the TTTR reading routine — the same objects the Detector Wizard edits and that the
burst request carries as `selected_setup`, `windows`, and `detectors`.

**Scope.** The setup record owned here is only about *reading and processing* the
data — which detector channels exist, the PIE/microtime window definitions, and
the TTTR reading routine. The optical/spectroscopic extension (light sources,
filters, dichroics, objectives, detector hardware) is owned by the
optical-configuration PRD, which *extends* this same setup; PRD-04 does not define
it. The calibration PRD attaches calibration values to the same setup.

## Current gap
- Detector setups persist to `mfdb_setup` as **opaque JSON blobs**
  (`configuration_json` / `detectors_json` / `timing_resolution_json` /
  `burst_defaults_json`): not queryable, not validated.
- `mfdb_flr_ext.dic` has **no save block** for `mfdb_setup` or the detector /
  channel/window definitions, so admin cannot resolve field descriptions or
  enumerate valid values.
- mfdb-admin's "Setups" entity shows only setup-level rows; the definitions inside
  the blobs are not structured records.
- `selected_setup` is a free-text label; nothing guarantees a matching row exists
  and `mfdb_operation.setup_id` is not populated, so the chain has no setup node.

## Required outcome
1. **Setup persisted as a resolvable record.** Each named setup is an `mfdb_setup`
   row addressed by the deterministic id from `setup_id_for_name()`
   (`tttr_detector_setup:<slug>`). The pipeline resolves `request.mfdb.setup_id`
   (or derives it from `selected_setup`) to a real row; a missing row warns,
   never raises.
2. **Channel/detector definitions are structured, not opaque**, following a
   dictionary-described schema. This reading/processing record is the base the
   optical PRD later extends.
3. **flrCIF / `.dic` alignment — the dictionary is the single source of truth.**
   `mfdb_flr_ext.dic` is authoritative for item names, types (`_item_type.code`),
   units (`_item_units.code`), enumerations (`_item_enumeration.value`), and
   descriptions. SQL schema and admin display **derive from / validate against**
   the dictionary — no second hand-maintained copy.
4. **Correct mfdb-admin display** of the setup and its detector/window definitions
   as structured rows; selecting a `burst_table` navigates to its setup.
5. **Operation links to setup.** `mfdb_operation.setup_id` is populated for the
   `burst_selection` operation.

## Source of truth: the `.dic` dictates the schema
For the tables this prerequisite adds there is **no hand-written SQL**. Flow:
1. Author the field vocabulary once in `mfdb_flr_ext.dic` — one item block per
   stored field (type, units where physical, enumerations for closed sets,
   mandatory flag, category key, parent links for FKs, description).
2. Bind every item to its column with the mandatory `_chisurf_schema.table_name`
   / `_chisurf_schema.column_name` bridge — the new tables (`mfdb_setup`,
   `mfdb_setup_detector_channel`, `mfdb_setup_pie_window`) are not `flr_`-prefixed,
   so `DictionarySchemaMap._mapped_categories()` only discovers them via the
   bridge.
3. **Generate the DDL from the dictionary** (Task P1a) — never hand-write
   `CREATE TABLE` for these tables. `DictionarySchemaMap.validate_mapping()` then
   cannot diverge because schema and dictionary share one source.
4. Admin pulls labels/descriptions/enumerations from the dictionary
   (`MmcifDictionary` / `DictionarySchemaMap`), not literals in `EntitySpec`.

Existing hand-written tables in `schema.py` are grandfathered; the generator is
wired only for the new categories in this PRD.

## Prerequisite tasks
- **P1 — reading/processing setup schema.** Decision (locked): **Option A, child
  tables** (pinned-JSON Option B rejected: cannot satisfy admin-listable
  structured rows + SQL-queryability without a view layer, and the optical PRD must
  be able to extend the detector channel with columns/links). Tables:
  `mfdb_setup_detector_channel` (one row per channel: `setup_id` FK ON DELETE
  CASCADE, channel name, one typed column per detector field `channels`,
  `micro_time_ranges`, `g_factor`, `l1`, `l2`, `g_factor_channels` — JSON `text`
  only for genuinely list-valued fields, scalars typed); `mfdb_setup_pie_window`
  (per window: `setup_id` FK, name, `start` int, `end` int); `tttr_reading`
  scalars (`macro_time_resolution`, `micro_time_resolution`, `micro_time_binning`)
  as **real typed columns** on `mfdb_setup` — the blob fallback is not acceptable.
  Indexes on `setup_id`. Bump `SCHEMA_VERSION`, add a migration that creates the
  generated tables and backfills existing `tttr_detector_setup:*` rows from their
  blobs without data loss.
- **P1a — generate table DDL from the dictionary.** Add
  `chisurf/core/mfdb/schema_from_dictionary.py`: for each bridged category emit
  `CREATE TABLE`/`ALTER TABLE ADD COLUMN` from `DictCategory`/`DictItem` metadata —
  column ← `_chisurf_schema.column_name`; SQL type ← `_item_type.code` via a single
  `TYPE_CODE_SQL_MAP` (`int/uint/integer`→INTEGER, `float/double/num`→REAL, else
  TEXT); `NOT NULL` ← mandatory; PK ← `DictCategory.key_item`; FK ← item parent
  link / `_chisurf_schema.foreign_key`; DEFAULT ← `_item_default.value`; audit
  columns `created_at`/`updated_at`/`deleted_at` by fixed convention. Emit
  deterministic SQL; wire into both the fresh-DB build and the migration; unit-test
  the generated DDL and round-trip against `introspect_sqlite_schema`.
- **P2 — persist through the setup repository, not loose JSON.** Extend
  `save_setup()` / `get_setup()` / `list_setups()` to read/write the structured
  rows, keeping `setup_id_for_name()` → `tttr_detector_setup:<slug>`. The wizard
  stops round-tripping opaque `setup_data` blobs once structured storage exists;
  legacy JSON import still works for first-run migration.
- **P3 — flrCIF `.dic` save blocks** authored **first** (P1a generates from them):
  `save_` category/item blocks for the setup, detector-channel, and PIE-window
  fields, each with the `_chisurf_schema` bridge, type codes, units where physical,
  enumerations where closed; column names equal to the schema columns.
- **P4 — mfdb-admin display** of the setup and its channel definitions as
  structured columns (not raw JSON); labels/descriptions/enumerations sourced from
  the dictionary via `DictionarySchemaMap`/`MmcifDictionary`.
- **P5 — resolve and link the setup in the burst operation.** `MFDBContext.setup_id`
  / `setup_version`, `register_*` forwarding of `setup_id`, and
  `setup_id_for_name(selected_setup)` derivation already exist. Remaining: before
  forwarding, look up the `mfdb_setup` row; if absent, warn and forward an empty
  id so `mfdb_operation.setup_id` stays NULL (never a dangling FK). Record
  `setup_version` in `build_burst_metadata()`.
- **P6 — prerequisite tests**, including a **total-coverage dictionary gate**:
  iterate **every** item in the new categories discovered from the dictionary and
  assert `validate_mapping()` is True — no hand-maintained allow-list.

# Setup sidequests
Four related sidequests extend the setup work; all follow the same governing rule
(`.dic` dictates the schema, dictionary-declared columns, total-coverage gate,
best-effort MFDB).

- **Sidequest (per-user setups + public sharing).** Detector setups are a
  scientist's own settings but live in a machine-wide `detector_setups.json`
  migrated globally with no owner. Add `created_by_user_id` (FK
  `flr_sample_users(user_id)`) and `is_public` columns to `mfdb_setup`
  (owner column, **not** ACL — locked decision). Make setup ids user-scoped
  (`tttr_detector_setup:<user_slug>:<name_slug>`) so same-named setups don't
  collide. Per-user idempotent auto-migration stamps the active (or default) user
  and imports only if that user has no setups yet; after a verified import the
  legacy JSON is deleted (MFDB is authoritative). Reads return own + public +
  builtin; only the owner may edit or toggle public; an owner-only "Public
  (visible to all users)" checkbox lives in the Detector Wizard (new setups default
  private). JSON stays a lossless, owner-agnostic export/import format (portability,
  `.csp` files); ownership is applied on import.
- **Sidequest B (FCS channel definitions — same treatment).** FCS channel-pair
  setups are the FCS analogue and get the identical pattern: stored as `mfdb_setup`
  rows (`setup_type="fcs_channel_setup"`, user-namespaced id) with a structured
  child table `mfdb_setup_fcs_pair` (per pair: `setup_id` FK, `name`, `channel_a`,
  `channel_b`, `kind`). **Correlator settings are per pair** (locked):
  `n_bins`/`n_casc`/`make_fine` are dictionary-declared typed columns on
  `mfdb_setup_fcs_pair` — not a JSON blob; the setup-level columns become defaults
  seeding a new pair. mfdb-admin shows the FCS setup and pair rows structured;
  ownership/public reuse the same columns and shared load/save scoping helpers (no
  forked mechanism). Per-user idempotent migration of `fcs_channel_setups.json`;
  the FCS preset UI drops the global correlator group box, makes per-row
  Bins/Cascades/Fine editable with a per-row delete button, and gets a
  construction smoke test. Default Save goes to MFDB only (no user-folder JSON;
  the confirmation names MFDB); legacy JSON is migrated-and-removed; JSON becomes
  export-only via an explicit path. No-MFDB Save degrades **softly** — it still
  succeeds via a JSON fallback but warns that MFDB was unavailable and the setup is
  not stored / not assigned to a user.
- **Sidequest C (g-factor calibration provenance).** The reference decay (VV/VH
  polarization-resolved "Jordi" file) is the calibration evidence and must live in
  MFDB. Register it as a queryable artifact (`raw_measurement` / typed
  `tcspc_decay`), archive the computed g-factor via `register_calibration()`
  (`calibration_type="g_factor"`, parented to the reference decay) with scalar
  parameters (`g_factor`, `g_factor_stddev`, `g_factor_uncorrected/corrected`,
  `r_inf`, region, `decay_shift`, `flip`, `use_bg`, `bg_vv/vh`, `l1`, `l2`), and
  add a dictionary-declared `g_factor_calibration_id` column to
  `mfdb_setup_detector_channel` linking a channel to its calibration. Fix the
  broken g-factor plugin GUI/MFDB path and the FCS read-only-construction
  regression. Reuse the result-registry calibration wrapper; the pure math already
  passes.

  ```text
  reference decay (VV/VH file)       [raw_measurement / tcspc_decay artifact]
    -> operation: calibration (calibration_type = "g_factor")
    -> calibration_data result       [g_factor (+stddev), l1, l2, region, bg, r_inf]
    -> consumed by mfdb_setup_detector_channel  [g_factor/l1/l2 + calibration link]
  ```
- **Sidequest D (time-versioned setup calibration).** A setup drifts (detectors
  age, alignment shifts), so calibration factors are date-dependent and must be
  versioned rather than overwritten. Split the stable **structural definition**
  (routing, PIE windows, TTTR reading — stays on `mfdb_setup` / child tables) from
  the time-varying **calibration** (`g_factor`, `l1`, `l2`, room for `gamma`,
  crosstalk) which moves to a new append-only dated-snapshot table
  `mfdb_setup_calibration` (`calibration_snapshot_id` PK, `setup_id` FK,
  `channel_name`, factors, `g_factor_calibration_id` link to the Sidequest C
  evidence, `calibrated_at` full ISO date+time, `method`, `notes`, owner/audit
  columns, index on `(setup_id, channel_name, calibrated_at)`). New factors INSERT
  a snapshot; `mfdb_setup_detector_channel` factors become a cache of the latest.
  The Detector menu gains a date/calibration combobox next to the setup combobox
  (distinct `calibrated_at`, most-recent first, `Latest` on top); repository API
  `add_setup_calibration` / `list_setup_calibration_dates` /
  `get_setup_calibration(setup, date|"latest")`. Migration backfills one snapshot
  per existing channel. Analyses record both `setup_id` and the calibration
  snapshot/date used, so a re-run reproduces the exact calibration.

# Stable pipeline contract
## Ownership
The stable implementation lives in
`chisurf/plugins/burst/burst_selection/api/mfdb.py` (plugin-owned; may import
Burst Selection API models and result-registry primitives).
`chisurf/core/mfdb/pipeline.py` is **removed**; core MFDB exports no
`BurstPipeline`.

## Request context
Add a nested MFDB context to `AnalysisRequest` (so future provenance attributes
don't churn the signature):

```python
@dataclass
class MFDBContext:
    enabled: bool = True
    sample_id: str = ""
    source_artifact_ids: dict[str, str] = field(default_factory=dict)
    register_missing_inputs: bool = True
```

Semantics: `enabled=False` skips registration; `sample_id` links all artifacts;
`source_artifact_ids` maps a normalized input path to an existing
`raw_measurement` artifact (reuse instead of re-registering);
`register_missing_inputs=True` registers inputs not already mapped.
`analysis_request_from_payload()` / `_to_payload()` / `contract_descriptor()`
accept `{"mfdb": {"sample_id": …, "source_artifact_ids": {…}}}`; top-level
provenance fields are intentionally unsupported.

## Result contract
Extend `AnalysisResult` with `output_paths_by_file: dict[str, dict[str, str]]`
(the stable per-file output map — `output_paths` stays a convenience last-path
role map), `mfdb_artifacts: dict` (filled by the registration layer), and
`warnings: list[str]` (non-fatal MFDB failures).

## Pipeline API
`api/mfdb.py` defines `BurstRegistrationResult` (`input_artifacts`,
`burst_table_artifacts`, `sidecar_artifacts`, `warnings`; keys are normalized
input paths for the first two, stable roles like `"hdf5"`/`"zip"`/`"output_folder"`
for sidecars) and `BurstMFDBPipeline(db=None).register_run(request, result)` which:
returns empty immediately when `mfdb.enabled` is false; normalizes paths with
`Path(path).resolve()`; reuses `source_artifact_ids[path]` when provided; registers
missing raw inputs; registers one `burst_table` per input file that produced rows
or a `.bur`; registers sidecars only after primary tables; never raises into the
UI flow (appends warnings instead).

# Artifact and provenance model
- **Input artifacts** — each input TTTR file is a `raw_measurement` via
  `register_raw_measurement(file_path, sample_id, metadata={plugin, role:
  "raw_tttr", filetype, selected_setup}, db)`. A supplied `source_artifact_ids` id
  is used as the burst-table parent without re-registering.
- **Primary output** — `burst_table` artifacts. Preferred payload order: (1) if a
  per-file `.bur` path exists, register that path with `data_format="bur"`;
  (2) else, if result rows exist, convert per-file rows to a DataFrame and
  `register_result(kind="burst_table", data=df)` (stores a typed msgpack
  `BurstTable`). `operation_type="burst_selection"`,
  `parent_artifact_id=input_artifact_id`. Metadata must include `plugin`,
  `contract_version`, `input_file`, `output_role`, `n_bursts`/`n_selected`/
  `n_photons` when available, `macro_time_resolution` when available, JSON-safe
  `photon_filter`/`burst_detection`/`gmm` settings, and `windows`/`detectors`/
  `selected_setup`. Parameters are scalar-only (`min_photons`, `photon_window`,
  `time_window`, `filter_active`, `count_rate_n_ph_max`, `count_rate_time_window`,
  `delta_macro_time_min/max`, `gmm_max_components`); nested settings go in metadata.
- **Optional selection payload** — if the analysis exposes start/stop indices or a
  mask, register an additional typed `burst_selection` payload
  (`source_artifact_id`, `start_indices`, `stop_indices`, `criteria`), parented to
  the burst table. Optional for PRD-04 unless the API already exposes the mask
  without re-reading raw TTTR; do not invent a mask from summary-only `.bur` rows.
- **Sidecar artifacts** — registered only when actually created: `hdf5` →
  `processed_data`; `zip` → `processed_data`; `output_folder` →
  `external_reference`/`processed_data`; `mti_dir` → `external_reference`; each
  parented to the (first) burst table with metadata role labels.

## Operation boundaries
Primary operation type is always `burst_selection`. `register_result()` creates
one operation per registered artifact — acceptable for PRD-04. A future
single-multi-output operation should extend the registry with a batch API, not
hand-write operation rows here.

## Error handling
MFDB registration is best-effort: missing DB, invalid `sample_id`, invalid source
artifact id → warnings, not crashes; a failed primary registration continues with
other files; a failed sidecar keeps the primary tables. Direct
`BurstMFDBPipeline` tests verify failure paths report and leave no partial rows.

# Tasks (main body)
1. **Replace the legacy core pipeline boundary** — delete
   `chisurf/core/mfdb/pipeline.py`, remove `BurstPipeline` from exports and the
   stale vocabulary (`artifact_type`, `storage_mode="local"`, `status="success"`),
   and stop executing burst analysis from core.
2. **Add MFDB request/result contract fields** (`MFDBContext`, `AnalysisRequest.mfdb`,
   `output_paths_by_file`, `mfdb_artifacts`, `warnings`, nested-`mfdb` payload
   normalization, contract descriptor docs). Existing payloads without `mfdb` still
   work; `_to_payload()` round-trips `mfdb` JSON-safely.
3. **Preserve per-file output paths** — `analyze_file()` populates
   `output_paths_by_file[str(path)]`; `analyze_request()` merges without
   overwriting per-file `.bur` paths.
4. **Implement the plugin-owned MFDB pipeline** (`BurstRegistrationResult`,
   `BurstMFDBPipeline`, `extract_burst_parameters`, `build_burst_metadata`) using
   the result registry.
5. **Register from the service adapter** — after `analyze_request(request)`, when
   `request.mfdb.enabled`, call `BurstMFDBPipeline().register_run(...)`, set
   `result.mfdb_artifacts`, extend warnings. Do not register from pure
   `analyze_request()` unless via an explicit `analyze_and_register_request()`.
6. **Pass MFDB context from GUI/client** — optional `mfdb` on
   `BurstSelectionClient.analyze_files()`; pass a picked `sample_id` when present,
   allow analysis without one, do not create samples here, surface warnings without
   failing the run.
7. **Focused tests** (`test/fio/test_burst_pipeline_mfdb.py`): one raw artifact per
   input when no source supplied; reuse of `source_artifact_ids`; `.bur` →
   `burst_table`; DataFrame → typed msgpack `burst_table`; `derived_from` from each
   table to its raw input; sample linkage; scalar parameters recorded; no-crash +
   warnings when MFDB unavailable; two-file per-file path handling; zero rows left
   for a failed invalid-sample registration. Plugin-local tests: nested `mfdb`
   payload acceptance, `mfdb_artifacts`/warnings in the result payload, service
   handler returns `mfdb_artifacts` on a successful enabled run. Use the real
   `MFDatabase` and `db.conn` (not `db.con`).

# Definition of Done
- Prerequisite: each named setup resolves to an `mfdb_setup` row; detector/window
  definitions are structured `mfdb_setup_detector_channel` / `mfdb_setup_pie_window`
  rows, not blobs.
- Prerequisite: `mfdb_flr_ext.dic` dictates the schema — every new item declares
  the `_chisurf_schema` bridge; the DDL is generated from the dictionary (Task
  P1a), not hand-written; the total-coverage `validate_mapping()` gate passes with
  no allow-list; no vocabulary is hardcoded a second time; and the setup displays
  in mfdb-admin with dictionary-sourced labels/enums.
- Prerequisite: the `burst_selection` operation populates `mfdb_operation.setup_id`.
- Core no longer owns or executes Burst Selection analysis;
  `chisurf/core/mfdb/pipeline.py` is removed.
- Request contract includes `MFDBContext`; result contract preserves per-file
  output paths and returns MFDB artifact IDs/warnings.
- Plugin-owned `BurstMFDBPipeline` registers inputs and burst-table outputs through
  the result registry; per-file `burst_table` artifacts derive from the matching
  raw TTTR artifact; existing source artifact IDs are reused; sample IDs link when
  provided.
- MFDB registration failures never fail GUI/CLI/RPC analysis; mfdb-admin lists
  registered `burst_table` artifacts and their provenance edges.
- Focused MFDB tests and Burst Selection plugin tests pass.

# Review outcomes
Durable decisions and defects surfaced by the code reviews and punch-lists (the
back-and-forth is condensed; all listed items were resolved unless noted):

- **Prerequisite (round 1 → round 2).** Round 1 rejected the first cut: the SQL
  was hand-written **and duplicated** across `CREATE_TABLES_SQL`, the migration,
  and the `.dic` (three drift sites); the `.dic` declared `tttr_reading` columns
  that did not exist; the gate test hid the divergence with a curated allow-list;
  the structured tables were write-only while the wizard still read the blob; and
  residual hardcodes remained (widget special-cases by column name, a stale
  `validate_setup_config` vocabulary, a legacy `SCHEMAS` fallback). Round 2
  confirmed **all blockers resolved**: `schema_from_dictionary.py` generates the
  DDL, the timing columns exist by construction, the FK is expressed via
  `_chisurf_schema.foreign_key`, the gate iterates every dictionary item with no
  allow-list, the wizard reads the child tables (blob is fallback-only), and the
  hardcodes are gone. Remaining minor/optional: index SQL is still a small
  hardcode; a missing dictionary category currently degrades to a no-op comment
  (prefer fail-loud); enumerations are enforced at the app layer, not via SQL
  `CHECK`; column order is alphabetical.
- **Per-user sidequest.** Implemented to the locked decisions (owner column, not
  ACL; user-namespaced ids; `is_public` + owner-only checkbox; per-user idempotent
  migration; own+public scoping; gate extended). Fixed a privacy inversion where
  `is_public` defaulted **public** at the storage layer but private in the GUI —
  now private at every layer (column `DEFAULT 0`, `save_setup` coercion `1 if
  is_public is True else 0`), so a non-GUI caller can't silently leak setups. The
  "ownerless == builtin == shared" invariant is documented inline.
- **Sidequest B (FCS).** Architecture sound (shared `tttr_setup_utils`, structured
  `mfdb_setup_fcs_pair`, dict-declared correlator columns, gate extended), but the
  shared-module extraction regressed the already-green detector path: JSON save
  broke on `str` paths (fixed: coerce to `Path`); an FCS dialog wrote to a DB
  missing `is_public` because `ALTER TABLE` errors were swallowed by
  `except OperationalError: pass` (fixed: `_ensure_column()` checks
  `PRAGMA table_info` and adds only if absent — never silently advance the version
  past a failed column add; construction is read-only via `skip_migration`/DI); and
  a monkeypatch seam moved (fixed via constructor dependency injection). Lesson:
  run the detector **and** FCS suites together — passing only the new tests hid the
  regressions.
- **Sidequest B addendum 2 (MFDB-default save).** Fixed the confirmation message
  hard-coding a JSON path and the silent user-folder JSON write. Locked policy: no-
  MFDB save is a **soft fallback with warning** (still succeeds via JSON, warns the
  setup isn't in the DB), not a hard fail; and the legacy JSON is deleted only when
  the migration actually imported it that call (gate the unlink on a verified
  import).
- **Sidequest C (g-factor).** Solid — the provenance chain is built end to end
  (reference decay registered, g-factor archived as a parented calibration, channel
  carries a dictionary-declared `g_factor_calibration_id`, FCS read-only regression
  fixed). Follow-ups: assert the calibration→reference-decay edge in tests; guard
  the `notes` f-string against a `None` g-factor so a missing value fails loudly
  rather than being swallowed; optionally add an FK on `g_factor_calibration_id`
  and a C3 round-trip test.
- **Sidequest D (time-versioned calibration).** Table, generated DDL, gate,
  append-only API, latest-per-channel resolution, migration backfill, and the date
  combobox are all present and tested. Two defects to fix: `save_setup` appended a
  snapshot on **every** save with no change detection (structural re-saves polluted
  the calibration history) — dedup-on-change against the latest snapshot, or route
  calibration only through `add_setup_calibration`; and the date combobox resolved
  the setup id **without the active user**, so it was empty for per-user setups —
  pass `_resolve_active_user_id()` to `setup_id_for_name` at both call sites. The
  D6 reproducibility metadata (burst pipeline records `calibrated_at`) is
  implemented but inherits the same user-scoping concern.

# Relationships
- Builds on [PRD-030](prd-030.md) codecs and the [PRD-03](prd-03.md) registry; adds setup/channel records extended later by an optical-configuration PRD.
- Reference archival path over the [MFDB (current)](/architecture/mfdb.md) store for a [plugin](/architecture/plugin-system.md); relates to [Plugins target](/specs/plugins.md).
