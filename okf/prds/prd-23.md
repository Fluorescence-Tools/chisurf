---
type: PRD
prd: "23"
title: "PRD-23: Thin Widgets / View–API Separation"
description: Makes GUI widgets pure view — no data processing, no database or acquisition-library calls, no side effects on construction — with mandatory construction smoke tests and a shared dockable-tool base.
status: in-progress
phase: "cross-cutting"
resource: chisurf/gui/widgets/tools/
tags: [prd, gui]
timestamp: '2026-07-05T00:00:00Z'
---

# Summary
PRD-23 makes GUI widgets pure view: they call only the api/RPC client, hold no database or photon-library logic, and perform no side effects (especially no database writes) on construction. It mandates a construction smoke test per tool to catch missing-import and side-effect-on-init regressions, and introduces a shared dockable-tool base implementing Load/Save, drag-drop, dock management, and connectivity-aware MFDB behaviour once instead of per plugin. This removes the class of bugs where logic hidden inside a widget ships broken, and enforces the GUI half of the transformer contract.

# Status
In-progress (cross-cutting, STATUS TABLE authoritative). The shared `ChisurfDockTool` base, residual-logic extraction, construction smoke tests, and a repo-wide static guard against database writes in widget `__init__` have landed for the reference transformers plus a third tool; remaining dockable tools migrate opportunistically.

# Goal
GUI widgets become pure **view**: no data processing, no DB/`tttrlib` calls, no side effects on construction. All state and I/O live behind the api/RPC layer. Mandatory construction smoke tests and a shared dockable-tool base remove the class of bugs where logic hidden in widgets ships broken.

# Evidence (why)
- A missing `QComboBox` import crashed `DetectorWizardPage` at construction — never caught because the widget-build path had no test.
- The FCS dialog wrote to MFDB **on construction** (migrating/registering as a side effect), causing the "auth required" / write-on-open issues.
- Each tool re-implements Load/Save, drag-drop, docks differently (shifter vs others), so fixes don't propagate.

# Design
- **View-only widgets.** Widgets call the api/RPC client only; no `tttrlib`, no `MFDatabase`, no registration logic in `gui/`. Construction is read-only and cannot perform DB writes (the [PRD-16](prd-16.md) split, enforced).
- **Mandatory construction smoke test** per tool (mirror `test/gui/test_detector_wizard_page.py`): build the widget offscreen, assert it constructs — catches missing-import / side-effect-on-init regressions.
- **Shared dockable-tool base** (`ChisurfDockTool` or similar): standard Load/Save tool-menu, drag-drop target, dock management, MFDB-connection status, and the "From MFDB / Save to MFDB vs file" connectivity-aware behaviour — implemented once, reused by every transformer tool.
- Lean on the existing MVC controller work (MVC complete) so state lives in the model/controller, not the widget.

# Tasks
1. Define `ChisurfDockTool` base (tool-menu, docks, drag-drop, MFDB status, connectivity-aware load/save) and migrate the Microtime Shifter + FCS dialog + detector wizard onto it.
2. Move any residual `tttrlib`/DB logic out of `gui/` into `api/` + RPC.
3. Add construction smoke tests for every tool; make it a checklist item for new transformers (PRD-16 conformance).
4. Forbid DB writes during widget `__init__` (read-only construction); migration/registration only on explicit user action.

# Definition of Done
- [ ] Widgets are view-only (no DB/`tttrlib`); construction is read-only.
- [ ] A shared dockable-tool base is used by the transformer tools; Load/Save/docks are not re-implemented per plugin.
- [ ] Every tool has a construction smoke test; new transformers must add one.

# Definition of Clean
Layer purity (view ↔ api/RPC); no side effects on construction; GUI smoke tests mandatory; reuse the base, don't fork.

# Implementation status

**Task 2 (move residual DB logic out of `gui/`) — landed for both reference transformers (the non-GUI half).**

- **Burst Selection.** The raw-input registration/lookup logic that lived in `gui/tool.py` (importing `register_result`/`MFDatabase`/`resolve_database_path` into the view) moved to `api/mfdb.py`: `file_md5`, `sample_id_for_raw_path`, `raw_artifact_id_for_path`, `raw_file_data_format`, `register_raw_input_for_sample`, and a new `acquire_mfdb_connection()` (global-or-default connection acquisition). `gui/tool.py` now imports these from `..api.mfdb` (re-exported under the old private names for callers/tests) and its `_db()` is a one-liner over `acquire_mfdb_connection()`. The view no longer imports `MFDatabase`/`register_result`/`resolve_database_path`.
- **Microtime Shifter.** `api/mfdb.py` gained `active_mfdb_connection()`; the GUI `_db()` and the `MicrotimeShiftMFDBPipeline` default both use it, so the view no longer imports `_get_global_db` directly.
- **Import side-effects (overlaps Task 4).** Both plugin `__init__.py` files no longer import the Qt GUI tool eagerly — `BurstSelectionTool`/`MicrotimeShifterTool` resolve lazily via module `__getattr__` (PEP 562). The `api`/`cli` layers are now importable **headlessly** (no Qt binding required), which the new `tests/test_api_mfdb.py` exercises (and which previously made the api/cli tests uncollectable when Qt was absent).

**Task 1 (shared dockable-tool base) — landed.** `chisurf/gui/widgets/tools/` provides `ChisurfDockTool(QMainWindow)` and `PathDropListWidget`. The base factors the boilerplate both transformer tools had copied: window-level path drag-drop (dispatched to an overridable `on_paths_dropped` hook → `_add_paths` by convention), the byte-identical drop-list widget, window-geometry persistence helpers, and lazy MFDB-connectivity accessors (`acquire_mfdb_connection` hook, `mfdb_connection`, `mfdb_connected`) that do **no** work on construction. Burst Selection and Microtime Shifter now subclass it: each deleted its private `DropListWidget` (now `PathDropListWidget`) and its duplicated window `dragEnterEvent`/`dropEvent`, and routes connection acquisition through the `acquire_mfdb_connection` hook.

**Task 3 (construction smoke tests) — landed for both reference transformers.** `test/gui/test_chisurf_dock_tool.py` covers the base (read-only construction, drop dispatch, MFDB hooks, geometry). Each tool has an offscreen construction smoke test that builds the real widget (not `__new__`) and asserts it constructs, reuses `ChisurfDockTool`, and opens no MFDB connection on init: `burst_selection/tests/test_construction_smoke.py` (new) and the existing `tttr_microtime_shifter/tests/test_gui.py` (augmented).

**Task 4 (read-only construction) — enforced repo-wide.** Beyond the two tools (whose smoke tests assert `_mfdb_db is None` after construction), the rule is now an automated static guard: `test/test_no_db_writes_in_widget_init.py` AST-parses every GUI module and fails if any class `__init__` *directly* calls an MFDB write/open entrypoint (`register_*`, `MFDatabase(...)`, `save_setup`/`add_*`, `set_object_sample_id`, `reconcile_schema`, …); calls inside nested callbacks defined in `__init__` are deferred and not flagged (meta-tested both ways). The guard is AST-only (no Qt — runs in any CI). The one pre-existing violation it surfaced — `lightpath_simulator/gui/easy_mode.py` opening `MFDatabase` in a dye-table widget's `__init__` (which can trigger schema-reconcile writes) — was fixed by deferring the adapter open to first tooltip render (`_ensure_db_adapter`). The allowlist is empty.

**Third tool migrated — TTTR Time-Window (`tttr_time_windows`).** It had the same copied `DropListWidget` + window drag-drop, but its list filtered by supported TTTR extension. Rather than fork, `PathDropListWidget` gained an optional `path_filter` predicate (None = accept all, preserving the burst/shifter behavior; a predicate restricts drag-accept and drop), so the tool now reuses the shared widget (`PathDropListWidget(..., path_filter=_is_supported_path)`), subclasses `ChisurfDockTool`, drops its window `dragEnter`/`dropEvent`, and lazy-loads its GUI in `__init__.py`. Covered by `tttr_time_windows/tests/test_construction_smoke.py` and an added base test for the filter variant. This proves the base generalises beyond the two reference transformers, including the extension-filtered drop case.

**Still to do:** migrate the remaining `QMainWindow` dockable tools onto `ChisurfDockTool` opportunistically (e.g. `tttr_image_browser`, `trace_browser`, `tttr_lut_tools`, `pch`, FCS dialog, detector wizard — though the detector wizard is a `QWizardPage`, not a `QMainWindow`, so it does not fit this base). The FCS write-on-construction bug is already fixed (its `register_result` is in an explicit action handler, not `__init__`; the guard confirms). The GUI base/smoke work was written without a Qt-capable environment; run the smoke/base tests under `QT_QPA_PLATFORM=offscreen` with Qt bindings to confirm (the static guard needs no Qt).

# Relationships
- Enforces the GUI half of [PRD-16](prd-16.md) (transformer contract); adds construction smoke tests as a conformance checklist item.
- Generalizes fixes made to the reference-transformer tools.
- Leans on the completed MVC controller separation so state lives in the model/controller.
- Targets the [Plugins target](/specs/plugins.md); widgets talk only through the [RPC target](/specs/rpc.md).
