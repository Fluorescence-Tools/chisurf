---
type: PRD
prd: "02b"
title: "PRD-02b: MFDB Admin Overhaul — Manual Inspection & Editing"
description: Make the mfdb-admin plugin inspect, add, and edit every record the sample data model produces
status: done
phase: "foundation"
resource: chisurf/plugins/core/mfdb_admin/
tags: [prd, mfdb, gui]
timestamp: '2026-07-05T00:00:00Z'
---

# Summary
The mfdb-admin plugin must be able to inspect, add, and edit every record the
sample data model produces — entities, multi-probe positions with flrCIF fields,
FRET pairs, optical properties and spectra, and per-sample key-value metadata —
so a user can manually verify the database is correct before downstream
workflows rely on it. The existing admin GUI predates the sample rewrite and can
only show raw table columns, so it cannot display or edit the structured data,
create samples through the definition API, or surface the full-description and
export-validation views. This PRD is the manual verification gate between the
data model and trusting it enough to wire into result-registry and pipeline
work.

# Status
Done. Serves as the human verification gate for the sample-tracking data model.

# Goal
The mfdb-admin plugin must **inspect, add, and edit** every record the
[PRD-02](prd-02.md) data model produces. Before progressing to downstream PRDs
(result registry, burst pipeline, plugin integration), the user needs to
manually verify that the sample/probe/entity/FRET data is correct in the
database. PRD-02 added `EntityDefinition`, multi-probe `ProbeDefinition` with
flrCIF position fields, `FretPairDefinition`, default-spectra auto-population,
and vocabulary validation — none of which is visible or editable in the current
admin GUI, whose Sample tab shows only raw `flr_sample` fields. Without this, the
only way to verify PRD-02 correctness is test code or raw SQL, which is
unacceptable for a scientific application where the user must see and confirm
what the database contains. PRD-02b is the manual verification gate (Phase 2c',
the GUI counterpart of PRD-02) between implementing the data model and trusting
it in downstream workflows.

# Background — what exists
- `gui/tool.py` (~4569 lines) — `MFDBWidget` with 21 tabs; Sample/Condition/
  Entities/Probes/Positions tabs show only raw table columns.
- `gui/client.py` (~752 lines) — `MFDBClient` RPC wrapper.
- `backend/services.py` (~1330 lines) — RPC handlers; `save_sample_handler`
  uses raw SQL (not `create_sample()`); missing full-description,
  export-validation, and probe-edit handlers.
- `backend/measurement_services.py` (~1850 lines) — raw data, processing runs,
  provenance edges; works but not sample-aware in the PRD-02 sense.
- The mfdb package provides `create_sample()`, `get_sample_full_description()`,
  `validate_sample_for_export()`, `set_sample_metadata()`, `suggest_pdbx_keys()`,
  `EntityDefinition`/`ProbeDefinition`/`FretPairDefinition`/`SampleDefinition`,
  `DEFAULT_FLUOROPHORE_SPECTRA`, `compute_forster_radius()`, and the
  `SampleCreateRequest`/`SampleUpdateRequest` canonical inputs.

# Tasks

## Task 1: Wire `sample_manager` into the admin backend
The current `save_sample_handler` bypasses `create_sample()` (so optical
properties, spectra, FRET pairs, new position fields, and vocabulary validation
are ignored on save) and `get_sample_handler` uses `db.get_sample_full()`
(flr_sample + key_values only) rather than `get_sample_full_description()`. Add
14 RPC handlers (each `(params, auth=None)` → open MFDatabase → `_require_auth`
→ call sample_manager → return dict):
`samples.full_description`, `samples.validate_export`, `samples.create_structured`,
`entities.list`/`save`/`delete`, `probes.save`, `probes.optical_properties.save`,
`probes.positions.list`, `fret_pairs.list`/`save`/`delete`, `pdbx.suggest_keys`,
`pdbx.validate_value`. Register them in `register_services()` (plus the
`sample_database.*` back-compat alias loop) and `manifest.json`. Change
`save_sample_handler` to detect structured data (a `probes` list with position
fields) and delegate to `create_sample()`, keeping the raw-SQL path for legacy
flat dicts. Add 16 matching client methods to `client.py`.

## Task 2: Overhaul the Sample tab
Replace the flat `QFormLayout` with a `QSplitter(Vertical)`: top = sample header
form; bottom = a `QTabWidget` with sub-tabs Entities, Probes & Positions, FRET
Pairs, Condition, and Full Description; an action-buttons bar between. Entities
sub-panel: editable table (id, name, type combo from `ENTITY_TYPES`, sequence,
details) fed by `client.list_entities(sample_id)`. Probes & Positions: editable
probe table (name combo, entity, seq_id, comp_id, asym_id, atom_id, mutation/
modification flags, abs/em/QY read-only) auto-filling from
`DEFAULT_FLUOROPHORE_SPECTRA`, scoped to the sample via
`list_probe_positions`. FRET Pairs: donor/acceptor combos, R₀, κ², n, overlap,
with optional "Recompute R₀" via `compute_forster_radius()`. Embed the Condition
form. Full Description: read-only JSON view with Refresh/Copy/Validate.
`load_sample()` uses `get_sample_full_description()`; `collect_sample()` emits a
structured dict with `entities`, `probes`, `fret_pairs`, `condition`,
`key_values`.

## Task 3: Overhaul the Entities tab (standalone)
Add Refresh / New / Edit / Delete (with usage-warning confirmation) / Show-probes
buttons; fix the `fill_entities()` dead code so the sequence column is populated
(depends on the R15-1 fix so `add_entity()` actually stores sequences); make the
table editable; show an entity count in the tab title.

## Task 4: Overhaul the Probes tab (standalone)
Expand columns to include link_type, reactive, center_atom, ext_coeff; add
Refresh / New (dialog with vocabulary combos + auto-fill) / Edit / Delete /
Optical-properties / Spectra buttons; scope probes to the current sample with a
"Show all / Show sample" toggle; add probe-name autocomplete from
`DEFAULT_FLUOROPHORE_SPECTRA`.

## Task 5: New FRET Pairs tab
Create `fret_pairs_tab()` over `flr_fret_forster_radius` (columns id, sample,
donor, acceptor, R₀, κ², n, overlap_integral, details) with a sample filter,
CRUD buttons, optional Recompute-R₀, wired into `setup_ui()` and
`_all_items_sources`. Requires the R15-3 schema fix (`sample_id` was `INTEGER
NOT NULL` while `mfdb_sample.sample_id` is `TEXT PRIMARY KEY`).

## Task 6: Overhaul the Label Positions tab
Expand columns to show the flrCIF fields added by PRD-02 (atom_id,
mutation_flag, modification_flag, auth_name), update `fill_positions()` and the
`list_probe_positions_handler` join to return them, add CRUD buttons, and make
the table editable.

## Task 7: Full-description preview panel
Wrap the All Items table in a horizontal splitter with a read-only JSON preview
(Copy / Validate); wire `currentItemChanged` so selecting a sample shows
`get_sample_full_description(id)` and other items show their raw dict. Optionally
a standalone Full Description tab (overlaps Task 2.6 — one is primary).

## Task 8: Metadata tab enhancements
PDBx key autocomplete from `suggest_pdbx_keys(prefix)` (fires at 3+ chars,
completes full PDBx paths); value validation on save via
`validate_pdbx_value(key, value)` (warnings, not errors); group key-values by
prefix (flr.* / pdbx.* / chisurf.* / custom).

## Task 9: Import/Export tab enhancements
Show an import summary with created-record counts; run
`validate_sample_export` before export with an "export anyway?" dialog; optional
"Preview flrCIF" button.

## Task 10: Standardize all tables
Give every mfdb-admin table the same baseline interaction model via a
`_setup_standard_table(...)` helper: a dedicated checkbox column (column 0,
replacing "first column is checkable"), full-row selection, a standard
right-click context menu (Open details, Copy checked IDs / selected row / cell,
check-selected/all-visible/invert, select-all/clear, Delete checked…), bulk
actions on checked rows, and clear confirmation before destructive actions.
Deletion must show counts + first-10 IDs, run usage checks (per item kind:
sample, experiment, user, device, setup, entity, probe, probe position, raw
data, processing run, processed data, object, analysis, project/version) with a
mandatory-acknowledgement second dialog when dependencies exist, and report via
the status bar. Audit and migrate every existing table; do not fake deletion in
the GUI — add backend/client delete only where repository semantics are clear,
defaulting to **soft delete** (`deleted_at`, preserved audit log, never physically
delete user files).

## Task 11: Workflow-oriented docks
Make mfdb-admin a provenance browser, not just a row editor. Add a
`Measurements` dock aggregating raw data / processing runs / processed products
(columns include sample, sample QA red/yellow/green, project, experiment, setup/
device, status, location; filter by kind / QA / project / experiment). Make
`Projects` first-class (project/version identity, owner, visibility, linked
experiment, data count; selecting a project filters the other workflow docks).
Add a `Project Data` dock over project-linked artifacts/objects/samples with
validation state. Make provenance first-class (edge table, upstream/downstream
lists, graph view) traceable from any Projects/Measurements/Project-Data/Objects/
Analyses selection, with background fetch + main-thread apply. Add an `Overview`
dock summarizing database health (path, schema version, counts, quality
warnings, recent activity).

## Task 12: Reorganize navigation and polish action names/tooltips
Target dock order: Overview, Projects, Measurements, Project Data, Samples,
Experiments, Processing, Objects, Provenance, Provenance graph, Import/Export,
Admin. Rename "Reset database from source…" to `Reset` (accurate
backup-first tooltip — fix the implementation or the tooltip, never ship a
misleading one). Add tooltips to every toolbar/menu/table action; describe object
type + effect on buttons. Route all long/destructive operations through the
status bar / status label / preview pane, using `QThread` (worker fetches/mutates,
GUI thread updates widgets — never mutate Qt widgets from the worker thread).

## Task 13: General UX polish
Unblock GUI startup: run `refresh()` data fetching in a background
`_MFDBBackgroundTask` (or sequential `QTimer` calls) with a progress
dialog/bar. Wrap table + detail form in `QSplitter(Vertical)` across the listed
tabs. Unlock detail editing (remove `setReadOnly(True)` on descriptive/name/JSON
fields, add a "Save Changes" button per tab committing via `client.update_*`,
keep primary IDs read-only). Show a disabled "No data available" placeholder row
for empty tables.

# Implementation notes
Handlers should call `sample_manager` functions (e.g.
`create_structured_sample_handler` builds a `SampleCreateRequest` →
`create_sample(db, request.to_sample_definition())` → returns
`get_sample_full_description`), not raw SQL. The GUI uses `QTableWidget`,
`QFormLayout`, `DockArea`, `_install_table_context_menu`, and status-label
updates — follow these patterns. `MFDBClient` supports in-process mode
(`inprocess=True`, a direct `ServiceDispatcher`, the primary local mode); new
handlers must be registered in `register_services()` to work in both modes.

**Prerequisite R15 fixes (before starting):** R15-1 `add_entity()` accepts but
never stores `sequence`; R15-2 `asym_id="A"` default makes `not self.asym_id`
always False, breaking legacy chain_id / default-spectra lookups; R15-3
`flr_fret_forster_radius.sample_id` INTEGER vs `mfdb_sample.sample_id` TEXT FK
mismatch; R15-6 `add_fret_forster_radius()` ignores `forster_radius_id`; R15-7
`__init__.py` missing exports for new PRD-02 symbols.

# Definition of Done — MVP
- [ ] Task 1: 14 RPC handlers registered and working; `save_sample_handler`
      delegates to `create_sample()` for structured data
- [ ] Task 2.8/2.9: `load_sample()` uses `get_sample_full_description()`;
      `collect_sample()` emits structured entities/probes/fret_pairs
- [ ] Task 3: Entities tab view/add/edit/delete with sequence column populated
- [ ] Task 4.1-4.3: Probes tab expanded, sample-scoped, add/edit/delete
- [ ] Task 5: FRET pairs tab view/add/delete
- [ ] Task 6: Label positions — all flrCIF fields visible and editable
- [ ] Task 7: Full-description preview for any selected sample
- [ ] Task 8.1: Metadata tab PDBx key autocomplete
- [ ] Task 10.1-10.4: dedicated checkbox column, standard context menu, bulk
      delete with confirmation/usage checks/status, all docks audited/migrated
- [ ] Task 11.1-11.4: Measurements / Projects / Project Data docks; provenance
      traceable from project/measurement/data/object/analysis selections
- [ ] Task 12.2-12.4: Reset renamed with accurate tooltip; actions have
      tooltips/clear labels; long operations report and don't block the GUI
- [ ] All new RPC handlers have tests

Nice-to-have (post-MVP): spectra plots per probe; auto-compute R₀ from spectral
overlap; probe-name auto-fill from DEFAULT_FLUOROPHORE_SPECTRA; flrCIF export
preview; import summary counts; embedded Condition sub-tab; metadata grouping and
value validation; keyboard shortcuts; grouped `Admin` dock; persisted column/
filter preferences; workflow timeline; safe provenance-edge delete.

# Tests
Backend handler tests (`test/test_admin_handlers.py`) cover full_description,
validate_export (complete/incomplete), create_structured (+ auto-filled spectra),
entity list/save-with-sequence/delete, probe save-all-fields and
optical-properties, positions list (flrCIF fields), FRET-pair CRUD, PDBx
suggest/validate, and legacy-vs-structured `save_sample_handler`. GUI tests cover
the standard checkbox column and context menu, checked-vs-selected bulk delete,
usage-second-confirmation, no-delete-when-backend-missing, the Measurements/
Projects/Project-Data docks and project-selection filtering, provenance tracing
from a measurement, and the Reset label / tooltip / non-blocking status
reporting.

# Relationships
- Depends on [PRD-02](prd-02.md); gates downstream [PRD-03](prd-03.md) and [PRD-04](prd-04.md).
- A [plugin](/architecture/plugin-system.md) surfacing the [MFDB (current)](/architecture/mfdb.md) store; aligns with the [Plugins target](/specs/plugins.md).
