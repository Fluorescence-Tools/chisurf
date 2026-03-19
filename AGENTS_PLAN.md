# AGENTS_PLAN.md - Full-Fidelity Project Save/Restore (Including 1:1 UI)

## Objective

Implement a **deterministic, high-performance, full-fidelity** project persistence system that restores:

- all datasets and metadata
- all experiments/setups
- all fit groups and local fits
- all models and model-specific state
- all parameters (values, bounds, fixed, output, etc.)
- all parameter links (intra-fit and cross-fit/global)
- all fit ranges/masks
- **entire UI state 1:1** (layout, windows, selection, active tabs, control values)

No backward compatibility is required.

This plan intentionally avoids “serialize the entire live object graph” (pickling) and instead uses:

- stable persistent identities (UUIDs)
- deterministic reconstruction
- explicit state contracts for models and UI components

---

## Non-Negotiable Requirements

1. **No index-based identity for persistence**
   - Fit index/order cannot be authoritative.
   - Use stable UUIDs (via `unique_identifier`) for all persistent entities.

2. **No generic object pickling for core state**
   - Persist explicit JSON-safe state contracts.
   - Qt objects are restored by declarative UI snapshots, not pickled Python objects.

3. **Strict schema versioning**
   - Introduce a new mandatory project format version.
   - Refuse loading older formats with clear error text.

4. **Transactional save + deterministic load order**
   - Save to temp + atomic replace.
   - Load by dependency graph order (data -> fits -> models -> params -> links -> UI).

---

## Decisions (Answered)

- Plots: no full plot restoration required (do not persist view ranges/cursors/overlays).
- Fit results/residual caches: recompute on load (do not persist derived results unless required for UI).
- Plugins: do not restore plugin windows; only restore main window + core fit windows/layout.
- Raw files: do not embed large raw acquisition files (store derived arrays + metadata; keep original paths only for reference).

---

## Execution Tracking (Cross-Referenced)

Use this section as the quick status board. Detailed behavior changes are logged in `CHANGELOG.md`.

Legend: `[DONE]` complete, `[PARTIAL]` implemented in part, `[PENDING]` not started.

### Phase Status Snapshot

- **Phase 1 - Identity + Registry Foundations:** `[PARTIAL]`
  - Registry module implemented at `chisurf/project/registry.py` ✅
  - O(1) lookup for datasets, fits, parameters, windows ✅
  - Lifecycle hooks for automatic registration ✅
  - Tests added (9 passing) ✅
  - Integration with runtime not yet wired

- **Phase 2 - Runtime Linking Correctness Rewrite:** `[PARTIAL]`
  - Group-link master orientation stabilized to first local fit ✅
  - Per-parameter unlink emits structured history ✅
  - Cross-reference: `CHANGELOG.md` -> Unreleased -> Fixed -> "Fit-group parameter linking now uses a stable master fit" and "Per-parameter unlink from main checkbox branch was not recorded in structured history".

- **Phase 3 - Project v3 Save Pipeline:** `[IMPLEMENTED]`
  - Project format v3 implemented with deterministic JSON ordering ✅
  - UID-based entity references throughout ✅
  - Structured sections: meta, datasets, experiments, fits, links, ui, artifacts ✅
  - Helper methods: get_dataset(), get_fit(), list_dataset_uids(), list_fit_uids() ✅
  - Tests added (6 new tests) ✅
  - **NO backward compatibility** - old formats not supported ✅

- **Phase 4 - Project v3 Load Pipeline:** `[IMPLEMENTED]`
  - Project.load() supports v3 format only ✅
  - Raises error for older formats ✅
  - Loads meta, datasets, experiments, fits, links, ui_state, artifacts ✅

- **Phase 5 - UI 1:1 Restoration Hardening:** `[PARTIAL]`
  - UI state module created at chisurf/project/ui_state.py ✅
  - Functions for capturing/restoring main window geometry, dock state, MDI area ✅
  - Dataset/fit selector state helpers ✅
  - Active tab tracking ✅
  - Integrated into project load/save via core_fit.py ✅

- **Phase 6 - Performance + Reliability:** `[PARTIAL]`
  - Windows save path fixes implemented ✅
  - Save-time plotting stability improved ✅
  - Performance instrumentation added to project save (timing logs) ✅

- **Phase 7 - Operation History and Full State Tracing:** `[PARTIAL]`
  - History system with checkpoints implemented ✅
  - Structured events for fit-run, fit-range, unlink actions ✅
  - No full state tracing or baseline snapshot persistence
  - Cross-reference: `CHANGELOG.md` -> Unreleased -> Fixed entries

- **Phase 8 - Undo/Redo History Navigation:** `[PARTIAL]**
  - Checkpoint-based replay implemented ✅
  - Cursor navigation and parameter/fit-range reconstruction ✅
  - Entity lifecycle replay handlers implemented ✅ (Phase 10)
  - No complete scientific state reconstruction
  - UI synchronization incomplete
  - Cross-reference: History replay fixes in `CHANGELOG.md`

- **Phase 9 - MVC Action Core:** `[COMPLETE]`
  - Full action registry and dispatcher system ✅
  - Controller architecture with services layer complete ✅
  - All 43+ action types registered and tested ✅
  - MCP/LLM integration via Chato working ✅
  - All tests passing (33/33)
  - GUI handlers migrated to use controller actions ✅
  - Added missing actions: fit.load, fit.save, fit.save_all, fit.close_all, parameter.value, fit.update, fit.range_set ✅
  - Added model actions: model.add_component, model.remove_component, model.normalize_amplitudes, model.absolute_amplitudes, model.change_irf, model.unload_irf, model.update, model.set_correction, model.set_linearization, model.unload_lintable, model.unload_background_curve, model.remove_local_fit, model.clear_local_fits, model.append_global_parameter, model.append_fit ✅
  - Migrated model widgets: lifetime.py (6/6), pda/widgets.py (15/15), anisotropy.py (2/2), convolve.py (4/4), corrections.py (7/7), discrete_distance.py (2/2), gaussian.py (3/3), generic.py (2/2), global_model/widget.py (4/4) chisurf.run() calls ✅
  - Migrated plot handlers: residual_image.py, lineplot.py (2/2 chisurf.run() calls) ✅
  - Fixed fit index issues: Ensured correct fit indexing in all migrated widgets ✅
  - Added comprehensive fit index tests: 5 new tests for multi-fit scenarios, edge cases, and detection logic ✅
  - Added correction-specific tests: 1 new test for correction operations ✅
  - Added background curve tests: 1 new test for background curve operations ✅
  - Added global model tests: 1 new test for global model operations ✅
  - Added new services: parameter_service.py, model_service.py ✅
  - Cross-reference: `CHANGELOG.md` -> Unreleased -> Changed -> central action registry/dispatcher entries.

- **Phase 10 - Complete State Reconstruction for Undo/Redo:** `[COMPLETE]`
  - Entity lifecycle reconstruction implemented ✅
  - Replay handlers for fit_add/fit_close/dataset_add/dataset_remove ✅
  - Integration into main cursor navigation ✅
  - Tests added and passing (5 new tests)
  - All remaining model widgets migrated (PCH, parameter scan) ✅
  - New parameter_scan action added with comprehensive tests ✅
  - All migrations completed: 55/55 calls (100% complete) ✅

### Done Items That Are Fully Closed

- **Phase 9 - MVC Action Core:** `[COMPLETE]`
  - Full action registry/dispatcher system implemented and tested
  - Controller architecture with services layer complete
  - MCP/LLM integration working via Chato
  - All action controller tests passing (17/17)

- **History System with Checkpoints:** `[COMPLETE]`
  - Operation history with periodic state snapshots
  - Checkpoint-based replay for faster undo/redo navigation
  - All history tests passing (8/8)

- **Phase 10 - Entity Lifecycle Replay:** `[PARTIAL - INFRASTRUCTURE COMPLETE]`
  - `reconstruct_entity_lifecycle()` function implemented
  - `replay_fit_add()`, `replay_fit_close()`, `replay_dataset_add()`, `replay_dataset_remove()` handlers implemented
  - `apply_entity_lifecycle()` integration in main cursor handler
  - 5 new tests added and passing
  - Runtime validation needed for actual entity creation/destruction during undo/redo

- **Phase 1 - Registry System:** `[IMPLEMENTED]`
  - Created `chisurf/project/registry.py` with O(1) lookup
  - Thread-safe registry with lifecycle hooks
  - 9 tests added and passing
  - Wired into fit/dataset lifecycle (fits and datasets now auto-register)
  - Registry sync on project load ✅

- **Phase 5 - UI State Persistence:** `[IMPLEMENTED]`
  - Created `chisurf/project/ui_state.py` with UI state capture/restore functions
  - Main window geometry, dock state, MDI area support
  - Dataset/fit selector state helpers
  - Active tab tracking

- **Phase 6 - Performance Instrumentation:** `[IMPLEMENTED]`
  - Added timing instrumentation to project save
  - Logs show save duration, dataset count, fit count

- **Project Roundtrip Test:** `[FIXED]`
  - Fixed `test_project_json_roundtrip.py` to use correct list API

- **Closed bug-fix items linked to the plan are tracked in** `BUGS_FIXED.md` (latest entries include grouped dataset restore and anisotropy calibration load hardening).
- **Primary changelog source remains** `CHANGELOG.md` (Unreleased section). This plan file now serves as the phase-level rollup.

---

## Architecture Overview

### 1) Persistent Identity Layer

**Current Status: PARTIAL**

- Stable IDs exist for some entities via `chisurf.base.Base.unique_identifier`
- No centralized registry system implemented
- UID resolution happens ad-hoc in various components
- Missing: `chisurf/project/registry.py` module

**Implementation detail:**

- `chisurf.base.Base.unique_identifier` used as canonical UID source
- Some parameter classes lack proper UID support
- No O(1) lookup registries implemented

### 2) Project Format v3 (Hard Cutover)

**Current Status: NOT STARTED - Using v2 format**

Current implementation uses project format v2 with basic structure:
- `project.json` manifest
- Simple dict-based datasets, experiments, fits
- No UID-based referencing system
- No deterministic ordering guarantees
- No separate array storage

**Planned v3 structure:**

Top-level sections:

- `meta`
- `datasets`
- `experiments`
- `fits`
- `links`
- `ui`
- `artifacts`

Large numeric arrays stored separately (`arrays.npz` or chunked binary files) and referenced by UID.

#### On-disk layout (directory)

- `project.json` (manifest; JSON only; deterministic ordering)
- `arrays.npz` (numpy arrays keyed by string; compressed)
- `blobs/` (optional; binary payloads not suited for npz)
  - e.g. images, thumbnails, plugin artifacts

#### Deterministic serialization rules

- Sort keys for stable diffs (JSON `sort_keys=True`).
- Store lists in deterministic order:
  - datasets: by `dataset_uid`
  - fit groups: by creation order + recorded `created_at` (but referenced by UID)
  - parameters: by canonical parameter path (see below)

### 3) Declarative Model State Contracts

Each model class must implement:

- `get_constructor_state() -> dict` (minimal deterministic init args)
- `get_state() -> dict` (all mutable model internals)
- `set_state(state: dict) -> None`

If a model lacks this contract, save fails with explicit model path and reason.

#### Model state contract definition

Every model class used in fits must support:

- `get_constructor_state() -> dict`
  - only what’s needed to rebuild the same model structure deterministically
  - must be JSON-safe
  - must not include live Qt objects

- `get_state() -> dict`
  - includes all mutable internal state beyond scalar parameters
  - examples: parse equations, component counts, LUT settings, IRF selection

- `set_state(state: dict) -> None`
  - must be idempotent
  - must tolerate missing optional keys
  - must not assume any UI exists

Additionally, all models must provide a way to enumerate parameters with stable `param_path` (directly or via a helper).

#### What is stored per local fit

- `fit_uid`, `dataset_uid`, `model_class` (module + class)
- `model_constructor_state`
- `model_state`
- `parameters`: map `{param_uid: {param_path, scalar_state...}}`
- `fit_range` and `mask` (if present)

### 4) Explicit Link Graph

- Store links as endpoint UIDs, never index paths.
- Types:
  - local parameter link: `source_param_uid -> target_param_uid`
  - global/formula links: explicit source UID + formula + dependency UIDs
- Restore links only after all parameters exist.

#### Link records (runtime + persistence)

- Local link (most common):
  - `{type: "param_link", source_param_uid, target_param_uid}`

- Fit-group “same-name link across locals” becomes explicit endpoint links (no implicit selected-fit semantics).

- Global links:
  - avoid `origin_fit_index` completely
  - store explicit source param uid(s) and explicit target param uid or global param uid
  - if formula-driven linking remains, store:
    - formula string
    - dependency param uids
    - and validate at load that evaluation resolves to the expected target uid

In all cases, link restoration is a separate phase after model/parameter instances exist.

### 5) UI Snapshot System (1:1 Restore)

Define UI snapshot contracts for (main window only):

- main window geometry/state
- MDI area state + z-order + active subwindow
- open fit windows and their associated `fit_group_uid`
- per-window geometry, docking state, maximized/minimized flags
- selected fit group, selected local fit, selected dataset
- active tabs/pages in all major panels
- ribbon/menu expand/collapse/pin states
- parameter panel expanded/collapsed groups
- plot view state: not required (see Decisions)
- plugin windows: not restored (see Decisions)

UI plugin contract (opt-in):

- `get_ui_state() -> dict`
- `set_ui_state(state: dict) -> None`

Plugins are not part of project UI restore.

#### UI fidelity definition

“1:1” means that after load, the user sees the same application session state as before save:

- same open windows (fit windows + plugin windows)
- same docking/MDI layout and geometry
- same active window, selected fit, selected dataset
- same active tabs and control values
- same plot view state (zoom/pan, scaling, toggles) where feasible

We treat UI restoration as a declarative snapshot applied after the object graph is reconstructed.

Standard Qt approach to include:

- `QWidget.saveGeometry()` / `restoreGeometry()` for top-level window geometry.
- `QMainWindow.saveState(version)` / `restoreState(state, version)` for dock widgets and toolbars.

Important Qt constraint: `QMainWindow.saveState()` identifies `QDockWidget` and `QToolBar` instances by their `objectName`, which must be unique and stable across runs.

References:

- Qt `QMainWindow.saveState/restoreState` docs: https://doc.qt.io/qt-5/qmainwindow.html#saveState
- Qt `QWidget.saveGeometry` docs: https://doc.qt.io/qt-5/qwidget.html#saveGeometry
- Qt `QSettings` docs (typical storage mechanism): https://doc.qt.io/qt-5/qsettings.html

#### UI snapshot boundaries

- Persist Qt framework state using:
  - `QMainWindow.saveGeometry()` and `QMainWindow.saveState()` (hex-encoded)
  - MDI area state where available

- Persist per-window semantic state (required for correctness):
  - which `fit_group_uid` a fit window corresponds to
  - which local fit is selected inside a fit group
  - plot configuration values that are not captured by Qt saveState
  - ribbon pin/collapse state, currently selected tool pages

Because `saveState()` is sensitive to widget existence and objectNames, restore is two-pass:

1. Recreate all required dock widgets/subwindows (skeleton)
2. Apply Qt `restoreState` / `restoreGeometry`
3. Apply per-widget semantic state via `set_ui_state` hooks

Also note: several common widgets have their own state blobs and should be persisted where used:

- `QSplitter.saveState()` / `restoreState()` for splitter positions.
- `QHeaderView.saveState()` / `restoreState()` for table/tree column order/width/hidden.

References:

- Qt `QSplitter.saveState/restoreState` docs: https://doc.qt.io/qt-5/qsplitter.html#saveState
- Qt `QHeaderView.saveState/restoreState` docs: https://doc.qt.io/qt-5/qheaderview.html#saveState

---

## Implementation Phases

## Phase 1 - Identity + Registry Foundations

- Add/verify unique identifiers for all persistent entities.
- Introduce centralized registry manager with lifecycle hooks:
  - on fit creation/removal
  - on model rebuild
  - on parameter regeneration
- Remove hot-path dependency on fit-index scanning for parameter ownership.

Deliverables:

- A `Registry` component (new module) that is the single source of truth for:
  - fit groups, local fits
  - datasets
  - parameters
  - windows

- Lifecycle hooks (signal-like):
  - `on_fit_group_created(fg)`
  - `on_fit_group_removed(fg_uid)`
  - `on_model_rebuilt(fit_uid)`
  - `on_parameters_regenerated(model_uid)`

- Stable parameter identity mapping:
  - assign `param_uid` to regenerated parameters by matching `param_path`
  - enforce uniqueness of `param_path` within a model

Acceptance:

- Any parameter resolves owner fit and model in O(1).
- No ambiguous ownership in grouped fits (FCS/TCSPC).

## Phase 2 - Runtime Linking Correctness Rewrite

- Replace widget/macro operations that write via `chisurf.fits[idx]...` strings.
- Use direct object mutations or UID-resolved targets.
- Link menu callbacks capture source/target parameter UIDs directly.
- Refactor fit-group linking to explicit source parameter and target set.

Concrete changes:

- Eliminate all writes that look like `chisurf.fits[...].model.parameters_all_dict[...]` in GUI handlers.
- Link menu callbacks must capture:
  - `source_param_uid`
  - `target_param_uid`
- When applying a link from UI:
  - resolve both UIDs via registry
  - set `source.link = target` (direct object link)
  - update affected controllers via `finalize()`

Guard rails:

- Before linking, validate:
  - not self-link
  - no recursion (cycle detection) using UID graph
  - both endpoints exist and belong to expected fits (optional)

Acceptance:

- No wrong-fit updates when selected fit changes.
- FCS grouped linking and TCSPC VV/VH linking are deterministic.

## Phase 3 - Project v3 Save Pipeline

- Implement project writer with:
  - schema validation
  - deterministic ordering
  - array artifact writer
  - UID reference integrity checks
- Save model constructor state + model state + parameter state by UID.
- Save full link graph by UID.
- Save UI snapshot tree by window/panel/plugin.

Concrete project.json schema (high level)

- `meta`:
  - `project_format_version`
  - `created_at`
  - `chisurf_version`
  - `platform` (optional)

- `datasets`: `{dataset_uid: {name, filename, arrays: {x_key, y_key, ex_key, ey_key}, metadata...}}`
  - arrays refer to keys in `arrays.npz`

- `experiments`: `{experiment_uid: {class, state, setup_state...}}`
  - only include what is required to rebuild models the same way

- `fits`: `{fit_group_uid: {name, local_fits: [fit_uid...], global_model_state...}}`
  - `local_fit[fit_uid]`: dataset uid, model class, model states, param states, fit range/mask

- `links`: `{links: [ ...link_records... ]}`
  - endpoint UIDs only

- `ui`:
  - `main_window`: geometry/state hex
  - `mdi`: layout + subwindow list + active window uid
  - `dock_widgets`: open/closed state, objectName mapping
  - `fit_windows`: list of window records (each with `window_uid`, `fit_group_uid`, geometry, semantic state)
  - `plugins`: list of plugin window records (module/id + state)

Save-time validation checklist:

- Every referenced UID exists.
- Every parameter record has a valid `param_path`.
- No duplicated `param_path` within any model.
- All link endpoints exist.
- All models claim the required state contract.
- Every dock widget / toolbar participating in Qt `saveState` has a stable, unique `objectName`.

Acceptance:

- Save fails fast on non-serializable model/plugin with actionable message.
- Saved payload is deterministic for identical session states.

## Phase 4 - Project v3 Load Pipeline

Load order:

1. validate schema/version
2. load datasets/experiments
3. instantiate fit groups/local fits
4. instantiate models from constructor state
5. apply parameter scalar state
6. apply links
7. finalize/update models
8. reconstruct windows
9. apply UI state (selection, tabs, geometry, plot view)

Load-time reconstruction rules:

- Never assume fit ordering from the file.
- Rebuild fit groups and local fits by UID.
- Rebuild models from recorded class path; apply constructor state; then `set_state`.
- After model creates its parameter structure:
  - match parameters by `param_path`
  - assign `param_uid` from persisted mapping
  - apply scalar parameter state

Link restoration:

- Apply link records only after all params are registered.
- If any link endpoint is missing, fail load (strict mode).

UI restoration:

- Create all windows declared in `ui` first.
- Apply Qt `restoreState` only after all named dock widgets exist.
- Apply semantic UI state last (selected fit/dataset, active tabs).
- Use a deferred queue (e.g. `QTimer.singleShot(0, ...)`) to apply UI state that depends on the event loop.

Acceptance:

- Loaded session reproduces pre-save scientific and UI state 1:1.

## Phase 5 - UI 1:1 Restoration Hardening

- Add UI state adapters for core windows/widgets.
- Add plugin UI snapshot manager and registration hooks.
- Add deferred-restore queue for widgets that initialize asynchronously.
- Enforce idempotent `set_ui_state`.

Core UI components that need explicit adapters (expected):

- dataset selector widget (selection + expanded state)
- fit selector widget (selected fit group + selected local fit)
- plot option panels (checkboxes, axis modes)
- ribbon state (pinned/collapsed + active page)
- parameter table widgets (expanded groups + currently focused editor)

MDI subwindows:

- Each subwindow must have:
  - a stable `window_uid`
  - an `objectName` that is deterministic across runs
  - a record linking it to the underlying logical entity (`fit_group_uid` or plugin id)

Plugin windows:

- Plugin manager must support:
  - enumerating open plugin windows
  - recreating them by plugin id/module
  - applying their `ui_state` if provided

Acceptance:

- Repeated save->load cycles do not drift layout or control states.

## Phase 6 - Performance + Reliability

- Batch updates and block expensive redraws during restore.
- Single final UI refresh after full apply.
- Add timing instrumentation for save/load phases.
- Add integrity checks and concise diagnostics report.

Targets:

- O(1) UID lookups in runtime paths
- No warning flood in successful runs
- Stable load times for large projects

Performance budgets (initial targets):

- Save: < 2s for small projects; < 10s for large projects (100+ datasets)
- Load: < 3s for small projects; < 15s for large projects

Reliability requirements:

- Transactional save (temp dir + atomic rename)
- Load must either succeed fully or fail with no partial session corruption
- Any failure should include:
  - which UID failed
  - which component failed (dataset/model/link/ui)
  - next action suggestion

## Phase 7 - Operation History and Full State Tracing

- Introduce a first-class history subsystem for all state-changing user actions.
- Do not rely exclusively on `chisurf.run(...)` text replay for traceability.
- Record structured history entries for:
  - dataset load/remove/group
  - fit creation/removal
  - model mutations (add/remove components)
  - parameter changes (value/fixed/bounds)
  - links/unlinks (source/target param UID)
  - fit run/stop/results update
  - project save/load

History entry schema (minimum):

- `event_id` (UUID)
- `timestamp`
- `action_type`
- `source_uid` (entity initiating change)
- `target_uid` (optional)
- `payload` (JSON-safe delta)
- `summary` (human-readable line for history browser)

Implementation notes:

- Add a central recorder, e.g. `chisurf.history.record(...)`, callable from GUI, macros, and model code.
- Emit both:
  - structured event (for replay/reconstruction)
  - human-readable log line (for immediate user browsing)
- For direct object mutations in widgets, call history recorder explicitly at action boundaries.
- Keep recorder low overhead; append-only with optional batching to disk.

UI browsing requirements:

- Add history browser panel for chronological event list.
- Support filtering by action type and entity UID.
- Selecting an event shows details (`payload`, source/target names, fit/model context).
- Replace legacy `plainTextEditHistory` in `chisurf/gui/gui.ui` with a dedicated history browser widget host.

Restore/replay requirements:

- Persist history stream inside project bundle (`history.jsonl` or equivalent).
- Support deterministic replay from initial snapshot + events.
- Validate replay reaches same final UID graph and parameter/link state.

## Phase 8 - Undo/Redo History Navigation (Time Travel UX)

- Add standard actions:
  - `Undo` mapped to `Ctrl+Z`
  - `Redo` mapped to `Ctrl+Y`
- Maintain a history cursor (`current_event_index`) independent of raw append-only history.
- History browser highlights current cursor row.
- Clicking a history row sets cursor to that event and triggers jump-to-state reconstruction.

State jump semantics:

- Reconstruct from baseline snapshot + replay up to cursor index.
- Add optional checkpoint snapshots every N events (e.g. 50) for faster jumps.
- Exclude non-deterministic/non-state events from replay cursor semantics.

UX/interaction rules:

- If text-edit widgets own focus, preserve native text undo/redo behavior.
- Otherwise, `Ctrl+Z/Ctrl+Y` control history cursor navigation.
- Disable/queue undo-redo while a fit is actively running.

Acceptance:

- `Ctrl+Z/Ctrl+Y` navigates session states deterministically.
- Clicking history entry jumps to that state.
- Current state is visually obvious in history browser.
- Replay result matches expected fit/data/link state at cursor positions.

Implementation status note:

- Implemented: history browser cursor, highlight, and keyboard navigation hooks (`Ctrl+Z`/`Ctrl+Y`) for history browsing.
- Implemented: deterministic replay of navigation context (dataset/fit focus) and parameter subset restoration (value/fixed/bounds/link) based on events up to cursor.
- **CRITICAL LIMITATION**: Current replay only applies parameter/range/setup changes to existing fits and datasets. It does NOT create/destroy fits or load/remove datasets.
- Pending: full baseline snapshot persistence (saving checkpoints in project bundles for instant restore).

Phase 8 sub-phases (current focus):

- **8.1 Logging/Diagnostics (INFO level):** add high-signal navigation/replay logs for cursor moves, replay window sizes, applied parameter operations, unresolved keys, and fit-window activation results. ✅
- **8.2 Update/Finalize Consolidation:** revisit parameter update logic so replay/apply paths trigger consistent model+GUI refresh (not only via scattered finalize calls). ✅
- **8.3 Active Window Synchronization:** history navigation must activate the corresponding MDI fit window and keep `cs.current_fit`/`current_fit_idx` aligned. ✅
- **8.4 Replay Coverage Expansion:** incrementally expand replay from navigation+parameter subset toward full scientific state reconstruction. ❌
- **8.5 Fit/Dataset Lifecycle Replay:** Add replay handlers for `fit_add`/`fit_close` and `dataset_add`/`dataset_remove` to actually create/destroy entities during undo/redo. ❌

## Phase 10 - Complete State Reconstruction for Undo/Redo

**NEW PHASE ADDED**: Address the critical gap where undo/redo doesn't work for fit creation, data loading, and other entity lifecycle operations.

### Problem Analysis

Current history replay only handles:
- Navigation state (dataset/fit selection)
- Parameter state (values, bounds, fixed, links)
- Fit range state
- Setup state

**Missing**: Actual creation/destruction of fits and datasets during replay.

When user does:
1. Load dataset → `dataset_add` event recorded
2. Create fit → `fit_add` event recorded  
3. Undo (Ctrl+Z) → Only navigation/parameter changes are reverted, but dataset and fit still exist!

### Solution Design

Add replay handlers that can:
1. **Create fits** from `fit_add` events during forward replay
2. **Destroy fits** from `fit_close` events during backward replay  
3. **Load datasets** from `dataset_add` events during forward replay
4. **Remove datasets** from `dataset_remove` events during backward replay

### Implementation Plan

#### 10.1 Add Replay Handlers for Entity Lifecycle

- `replay_fit_add(event)`: Create fit group with same dataset, model, parameters
- `replay_fit_close(event)`: Remove fit group by UID/name
- `replay_dataset_add(event)`: Load dataset from original path or recreate from snapshot
- `replay_dataset_remove(event)`: Remove dataset by UID/name

#### 10.2 Extend Replay State Format

Add new sections to replay state:
```python
{
    # ... existing sections ...
    "entities": {
        "fits_to_create": [{"uid": "...", "dataset_uid": "...", "model": "...", "params": {...}}],
        "fits_to_destroy": ["fit_uid_1", "fit_uid_2"],
        "datasets_to_load": [{"uid": "...", "path": "...", "checksum": "..."}],
        "datasets_to_remove": ["dataset_uid_1", "dataset_uid_2"]
    }
}
```

#### 10.3 Integrate with Cursor Navigation

Modify `_on_history_cursor_changed` to:
1. Apply entity lifecycle changes BEFORE parameter/navigation changes
2. Handle forward/backward direction correctly
3. Preserve UID consistency across undo/redo cycles

#### 10.4 Add Safety Checks

- Prevent undo of "initial load" operations that would leave empty session
- Warn before undoing operations that affect many dependent entities
- Validate entity existence before applying parameter changes

### Acceptance Criteria

- `Ctrl+Z` after fit creation removes the fit and all its parameters
- `Ctrl+Y` after undoing fit creation restores the fit with original parameters
- `Ctrl+Z` after dataset loading removes the dataset
- `Ctrl+Y` after undoing dataset loading restores the dataset
- Multiple undo/redo cycles maintain consistent state
- History browser shows accurate state at each cursor position

### Risk Mitigation

1. **Incremental implementation**: Start with fit lifecycle, then datasets
2. **Backup snapshots**: Create full domain snapshots before major undo operations
3. **Validation tests**: Add regression tests for each entity type
4. **User confirmation**: Add warnings for destructive undo operations

## Future - Generic Action Infrastructure (MCP/LLM Ready)

- Add a first-class action registry for all user/system operations.
- Provide stable action IDs and payload contracts (schema-like validation).
- Execute actions through a single dispatcher with structured history emission.
- Keep existing macros as compatibility wrappers during migration.

Core goals:

- decouple UI events from execution internals
- make operations machine-addressable for MCP/LLM control
- standardize audit/replay metadata across GUI, CLI, scripts, and automation

Proposed architecture:

- `chisurf.actions.ActionRegistry`
  - `register(name, handler, metadata)`
  - `execute(name, payload, context)`
  - `list_actions()`
- history integration:
  - `action_execute_start`
  - `action_execute_finish`
  - `action_execute_error`
- optional reversibility metadata for future command-level undo

Initial action set (pilot):

- `dataset.add`, `dataset.remove`, `dataset.group`
- `fit.add`, `fit.close`, `fit.run`
- `project.save`, `project.load`, `project.close`

Migration strategy:

1. introduce registry and pilot actions
2. route selected GUI handlers through actions
3. keep macros as wrappers calling actions
4. migrate remaining paths incrementally
5. expose action catalog for MCP/LLM adapters

Acceptance criteria (future phase):

- action catalog is discoverable at runtime
- pilot actions fully operational and history-traceable
- replay can consume action stream deterministically
- external controller can invoke actions without GUI-specific coupling

## Phase 9 - MVC Action Core (Detailed Plan)

Objective:

- Move all state-changing operations to a GUI-independent action core.
- Make GUI, scripts, and future MCP controllers use the same controller path.
- Eliminate duplicate/ambiguous history events and stabilize undo/redo.

### 9.1 Architecture Split (Strict MVC)

Model (no Qt imports):

- `chisurf/runtime/actions.py`
  - `ActionSpec`: action metadata (name, schema, replayable, dedupe policy)
  - `ActionRegistry`: register/list/discover actions
  - `ActionDispatcher`: validate -> dedupe -> execute -> history emit
- `chisurf/runtime/history.py`
  - append-only canonical event stream
  - event cursor and replay window APIs
  - deterministic replay helpers
- `chisurf/runtime/state.py`
  - baseline snapshot representation
  - incremental checkpoint support for replay performance

Controller:

- `chisurf/controllers/action_controller.py`
  - `execute(name, payload, context)`
  - `undo_step()` / `redo_step()`
  - `jump_to_event(event_id)`
  - orchestrates model replay and applies domain changes
- existing macros become adapters calling controller methods
- GUI handlers call controller (never mutate model directly)
- script API calls controller directly

View:

- history browser renders model history/cursor state only
- fit/dataset widgets render current domain state only
- view emits intent signals (undo/redo/click-row), controller performs state changes

Acceptance:

- Core modules import without Qt/GUI dependencies.
- GUI becomes a pure controller+view consumer of core actions.

### 9.2 Canonical Action Schema

Each action record must include:

- `event_id`, `timestamp`, `action_type`
- `payload` (validated/normalized)
- `source_uid`, `target_uid` (optional)
- `replayable` (bool)
- `dedupe_key` and `dedupe_policy`
- `status` (`start|finish|error` when applicable)

Action metadata in registry:

- `schema` (required payload keys and types)
- `replayable` flag
- `dedupe_policy`: `none|drop_duplicates|coalesce_latest`
- `side_effect_class`: `state|io|ui|diagnostic`

Acceptance:

- All emitted events match schema and are replay-classified.

### 9.3 Dedupe and Coalescing (Central)

Implement in dispatcher (not GUI):

- fingerprint: `action_type + normalized_payload + source_uid`
- default dedupe window: 200 ms (configurable)
- policy by action:
  - `none` for fit run lifecycle and project save/load
  - `drop_duplicates` for repeated identical toggles
  - `coalesce_latest` for rapid spinner/slider-like edits

Acceptance:

- duplicate action bursts no longer pollute history.
- undo/redo lands on meaningful state changes.

### 9.4 Action Catalog (Initial Migration Set)

Dataset actions:

- `dataset.add`
- `dataset.remove`
- `dataset.group`

Fit actions:

- `fit.add`
- `fit.close`
- `fit.run.start`
- `fit.run.finish`
- `fit.run.abort`

Parameter actions:

- `parameter.value.set`
- `parameter.fixed.set`
- `parameter.bounds.set`
- `parameter.bounds_on.set`
- `parameter.link.set`
- `parameter.link.clear`

Project actions:

- `project.save`
- `project.load`
- `project.close`
- `app.reinitialize.start`
- `app.reinitialize.finish`

Acceptance:

- Existing GUI/macro paths for these operations route through the dispatcher.

### 9.5 Undo/Redo and Cursor Semantics

Rules:

- cursor moves over replayable `state` actions only
- `Ctrl+Z` -> previous replayable state event
- `Ctrl+Y` -> next replayable state event
- clicking history row sets cursor to that event

Reconstruction:

- baseline snapshot + replay to cursor
- add checkpoints every N events (default 50)
- invalidate/rebuild checkpoints when history is loaded/replaced

Acceptance:

- fit selection, active window, and parameter/link GUI state visibly change on undo/redo.

### 9.6 Active Window / current_fit Synchronization

On replay apply:

- resolve target fit group from replay state
- set `cs.current_fit` and `current_fit_idx`
- activate matching MDI subwindow
- refresh fit/dataset selectors and parameter panel

Acceptance:

- history navigation updates both model state and active fit window consistently.

### 9.7 Migration Strategy (Low Risk)

Step 1:

- create dispatcher/registry/history core modules
- no behavior change yet

Step 2:

- route one vertical slice end-to-end:
  - `parameter.value.set`
  - `parameter.link.set/clear`
  - undo/redo cursor for those actions

Step 3:

- migrate fit actions (`fit.add/close/run`) and dataset actions

Step 4:

- migrate project lifecycle actions

Step 5:

- deprecate scattered `_record_history` helpers in GUI/macros
- keep compatibility wrappers temporarily

Step 6:

- expose action catalog for MCP adapter

Acceptance:

- no direct history writes remain in view classes.

### 9.8 Testing and Verification

Unit tests (headless):

- schema validation per action
- dedupe/coalescing behavior
- replay determinism for parameter, fit, dataset, and project actions

Integration tests:

- GUI undo/redo changes active window + parameter/link visuals
- project save/close/load with history preserved and replayable

Performance tests:

- replay latency with 1k+ events
- checkpoint speedup effectiveness

Logging requirements (INFO level):

- cursor move summary
- replay range size and apply counts
- unresolved keys and activation failures

Definition of done for Phase 9:

- action dispatcher is the single state-change entry path.
- history stream is deduplicated, replayable, and GUI-independent.
- undo/redo and click-jump visibly update model + active window + parameter GUI.
- controller API is ready for script and MCP execution.

---

## Testing Plan

### A. Scientific State Roundtrip

- TCSPC single + grouped (VV/VH)
- FCS grouped with cross-links
- Mixed model sessions
- Verify numerically equivalent fit outputs after restore

### B. Link Integrity

- intra-fit links
- cross-fit/global links
- link/unlink/fixed/bounds/value operations after restore

### C. UI Fidelity (1:1)

- golden-state tests for:
  - active fit/dataset selections
  - open windows and geometry
  - tab/ribbon states
  - splitter/header states (where used)

### D. Stress/Scale

- large dataset counts
- many fit groups and links
- repeated save/load cycles

### E. Failure Injection

- missing artifact files
- unknown model class
- broken plugin state
- malformed UID references

### F. History/Replay

- verify every state-changing operation emits a history event
- verify history browser renders and filters events correctly
- verify save/load preserves history stream
- verify replay from initial snapshot reproduces final state (links, params, fits)

Expected behavior: precise error messages + no partial silent corruption.

Add “UI 1:1” verification strategy:

- Snapshot-driven tests that compare:
  - window count and types
  - window geometry
  - active selection (fit/dataset/local fit)
  - critical control values
  - link graph equivalence (by UID)

---

## Suggested File Touchpoints

- `chisurf/project/project.py`
- `chisurf/project/fit_state.py`
- `chisurf/macros/core_fit.py`
- `chisurf/history.py`
- `chisurf/fitting/fit.py`
- `chisurf/fitting/parameter.py`
- `chisurf/parameter.py`
- `chisurf/gui/widgets/fitting/parameter_widgets.py`
- UI snapshot helpers (new module): `chisurf/project/ui_state.py`
- Registry manager (new module): `chisurf/project/registry.py`
- tests under `test/` for persistence, linking, and UI fidelity

---

## Definition of Done

1. Save/load reproduces full scientific state and **UI state 1:1**.
2. No fit-index-based persistence paths remain.
3. Linking correctness issues in grouped fits are eliminated.
4. Project format v3 is strict and validated.
5. Full test suite for persistence/linking/UI passes in CI.
6. Structured operation history exists, is browseable, and supports full state tracing/replay checks.
7. **Undo/redo works for all operations including fit creation/destruction and dataset loading/removal** (Phase 10 complete).

---

## Notes

- Qt also supports OS-level session management via `QSessionManager`, but it is not a replacement for a domain-specific project format. Qt’s docs explicitly recommend using `QSettings` to save/restore application settings (geometry/state, recently used files, etc.).
  - Reference: https://doc.qt.io/qt-5/qsessionmanager.html
