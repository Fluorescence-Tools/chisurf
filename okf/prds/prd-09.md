---
type: PRD
prd: "09"
title: "PRD-09: Microtime Shifter — Workflow Plugin + MFDB Provenance"
description: Convert the microtime-shifter tool into a layered api/backend/cli/gui workflow plugin with RPC and full MFDB provenance.
status: done
phase: "0"
resource: chisurf/plugins/tttr/tttr_microtime_shifter
tags: [prd, plugins]
timestamp: '2026-07-05T00:00:00Z'
---

# Summary
The microtime-shifter tool was a single-file Qt window with direct file read/write, no layering, and no provenance. This PRD restructures it into a workflow-ready plugin (pure `api` shift logic, `backend` RPC service, `cli`, `gui`) modeled on the burst-selection template, communicating over ZMQ JSON-RPC. Dropped TTTR files are identified and deduplicated in the MFDB object store, shifted files are written and registered as derived artifacts, and the per-channel shift values are persisted as dictionary-declared MFDB data so the operation is fully traceable. Save is connectivity-aware: with MFDB it registers into the managed object store; without it, it falls back to a file dialog and warns. Ten non-negotiable "Definition of Clean" standards govern the work (layer purity, no blobs, generated DDL, behavior-asserting tests, GUI smoke, DI over monkeypatching, idempotence).

# Status
Done. Split into api/backend/cli/gui with a versioned contract, dictionary-declared shift values, object-store dedup, and connectivity-aware save.

# Goal
Convert `chisurf/plugins/tttr/tttr_microtime_shifter` from a single-file
`QMainWindow` into a **workflow-ready plugin** modelled on Burst Selection: a
`core`/`api`/`cli`/`gui` split that communicates via RPC, with full MFDB
provenance — dropped TTTR files are identified/deduplicated in the object store,
shifted files are written and registered as derived artifacts, and the
per-channel shift values are persisted as **dictionary-declared** MFDB data so the
operation is fully traceable.

The plugin being replaced was `wizard.py` — a `MicroTimeShifter(QMainWindow)` with
inline Load…/Save… buttons and a drag-drop `FileLineEdit`,
`self.shifts = {channel: shift}` + `self.global_shift`, and direct `tttrlib`
read/write of shifted files. No split, no RPC, no MFDB. Burst Selection is the
canonical workflow-plugin template (its `api/contract.py`, `api/models.py` +
`MFDBContext`, `api/selection.py`, `api/mfdb.py`, `backend/services.py`,
`cli/main.py`, `gui/client.py` + `gui/tool.py`).

# Governing rules
- **MFDB is the store; the `.dic` dictates the schema.** Anything structured the
  plugin persists (the shift values especially) is declared in
  `chisurf/core/mfdb/data/mfdb_flr_ext.dic` with `_chisurf_schema` bridges and is
  covered by the total-coverage `validate_mapping()` gate. No opaque JSON blobs
  for the shift values.
- MFDB registration is best-effort for the GUI flow — failures warn, never crash
  the shifter.
- Reuse the result registry (`register_raw_measurement` / `register_result`) and
  the object store; do not hand-roll object/artifact writes.

# Definition of Clean (non-negotiable engineering standards)
These encode lessons from the burst-pipeline work; a review rejects the work if
any are violated.

1. **Layer purity.** `api/shift.py` has no Qt and no DB imports — pure functions,
   directly unit-testable. The GUI calls the API **only via the RPC client**; no
   `tttrlib` read/write and no MFDB calls in the widget. `backend/services.py` is
   the only place wiring pure-API → MFDB.
2. **Single source of truth / no blobs.** Shift values are dictionary-declared
   structured columns. No correlator-style JSON blob, no second copy of field
   names/labels in code. `validate_mapping()` passes for every new item, checked
   by a total-coverage test deriving its item list from the dictionary.
3. **No hardcoded SQL.** New tables/columns are generated from the `.dic` via the
   dictionary-driven DDL generator; migrations add columns with the verified
   `_ensure_column` helper — never `try/except OperationalError: pass`.
4. **Best-effort MFDB, but fail loud on real bugs.** MFDB-unavailable warns and the
   shift still works; genuine errors (bad migration, missing column, failed
   registration) must surface, not be swallowed.
5. **Tests assert behavior, not existence.** MFDB tests assert the `derived_from`
   edge (shifted→raw), object-store **dedup** (same content not re-stored), and
   that stored shifts read back equal to what was applied. Pure-shift tests assert
   the actual micro-time values.
6. **GUI build coverage.** A construction smoke test for the dockable tool (mirror
   `test/gui/test_detector_wizard_page.py`); every new Qt symbol must be imported.
7. **Dependency injection over monkeypatching.** Tests inject DB/RPC seams
   (explicit `db_path` / in-process client), not `monkeypatch.setattr` of internals.
8. **No dangling legacy.** `wizard.py` is removed, not left as dead/duplicate code.
9. **Determinism + idempotence.** Re-running a shift with the same inputs is
   idempotent (no duplicate artifacts/objects); generated DDL is stable.
10. **Run the whole relevant suite together** (plugin tests + the dict gate) so a
    new category can't silently break the gate or regress sibling plugins.

# Tasks
1. **API layer (`api/`).** `api/models.py`: `ShiftRequest` (`files`,
   `global_shift`, `channel_shifts: dict[int,int]`, `filetype`, `output_dir`,
   `mfdb: MFDBContext`) and `ShiftResult` (`output_paths_by_file`,
   `applied_shifts_by_file`, `mfdb_artifacts`, `warnings`). `api/shift.py`: **pure**
   shift logic — read a TTTR file, apply global + per-channel micro-time shifts,
   write the shifted file, return the output path and exact shifts applied (the
   refactor of the widget's `load_file`/save logic). `api/contract.py`:
   `CONTRACT_VERSION`, `shift_request_from_payload`/`*_to_payload`,
   `shift_result_to_payload`, `contract_descriptor`, service-success envelope.
   `api/mfdb.py`: registration (Task 4).
2. **Backend service + RPC (`backend/services.py`).** RPC handlers on the ChiSurf
   ServiceDispatcher: `microtime_shift.apply`, `microtime_shift.load_metadata`
   (used routing channels + n micro-time channels for a file),
   `microtime_shift.identify` (object-store lookup, Task 4). Handlers call the pure
   API then `api/mfdb.py`.
3. **CLI (`cli/main.py`).** Headless entry: input file(s), shifts, output dir; run
   the pure API; optionally register in MFDB.
4. **MFDB treatment — connectivity-aware save (two modes).**
   - **Connected to MFDB → register, don't write a loose file.** Save writes the
     shifted bytes into the **managed object store** and records full provenance
     (raw → `microtime_shift` operation with the shift values → shifted artifact,
     `derived_from` edge). The GUI confirms "shifted data registered in MFDB".
   - **Not connected → old behaviour.** A Save dialog writes the shifted TTTR to a
     user-chosen path; the user is warned it wasn't registered (warn, don't fail —
     work is never lost). "Connected" means a usable MFDB resolves
     (`get_db()`/object store available). An explicit "Save to file…" stays
     available even when connected, for export/portability.

     ```text
     dropped TTTR file  [raw_measurement, deduplicated by content hash]
       -> operation: microtime_shift  [global + per-channel shifts, .dic-declared]
       -> shifted TTTR  [processed_data/"shifted_tttr", in object store,
                         derived_from the raw input]
     ```

     On load/drop, compute the file content hash and look it up in `mfdb_object`
     (by `content_md5`); reuse the existing artifact/object if present, else
     register via `register_raw_measurement`; surface "already in MFDB" vs "newly
     registered". Save the shifted file then `register_result(kind=
     "processed_data", data=<shifted path>, data_format inferred,
     parent_artifact_id=<raw>, operation_type="microtime_shift", parameters=
     <shifts>)`. Persist the applied shifts with the operation so re-loading a
     shifted file restores the values that produced it.
5. **Shift values need a `.dic` entry (structured, dictionary-declared).**
   Preferred: a structured child table `mfdb_microtime_shift` — one row per
   (operation/artifact, routing_channel) with `shift` (micro-time channels), plus a
   `global_shift` at the operation/artifact level; declared in `mfdb_flr_ext.dic`
   with `_chisurf_schema` bridges, generated via the dictionary-driven generator,
   added to the gate's coverage set, `SCHEMA_VERSION` bump. Or (only if the table
   is rejected in review, with the reason recorded): per-channel shifts as
   `mfdb_parameter` rows named `shift_ch<idx>` plus `global_shift`, with matching
   `.dic` item descriptions. Either way the shift values must be queryable and
   dictionary-described — no undocumented blob; units = micro-time channels.
6. **GUI (`gui/`) — client + dockable tool.** `gui/client.py`: RPC client wrapping
   `microtime_shift.*` (in-process + ZMQ). `gui/tool.py`: the `QMainWindow`
   restructured — move Load/Save (button + line edit) into a tool menu/toolbar out
   of the central layout (drag-drop still identifies the file in the object store);
   put the shift controls (global + per-channel spinboxes) in one `QDockWidget` and
   the plot (micro-time histogram / shifted overlay) in another; the GUI calls the
   API only via RPC; connectivity-aware Save reflects the current mode in the UI
   and keeps an explicit "Save to file…"; show MFDB status without blocking.
7. **Plugin packaging.** Add `manifest.json` (display name, gui/cli entrypoints),
   keep the icon; `__init__.py` loads the manifest and re-exports the GUI tool;
   move legacy `wizard.py` content into the split or delete it.
8. **Tests.** Pure `api/shift.py` (known shift → expected micro times;
   read→shift→write→read round-trip); contract round-trip; MFDB (dropped file
   identified + deduped, shifted file registered as `processed_data` derived from
   raw, shifts stored and readable, dict gate covers the new items);
   connectivity-aware save (connected → registers, writes no loose file; injected
   null DB → still produces the shifted file + not-registered warning); GUI
   construction smoke test.

# Definition of Done
- `tttr_microtime_shifter` is split into `api`/`backend`/`cli`/`gui` with a
  versioned contract; the GUI talks to the API only via RPC.
- Load/Save live in a tool menu; shift controls and the plot are separate custom
  docks; drag-drop still works.
- Dropped TTTR files are identified in the MFDB object store (deduped by content
  hash); raw inputs registered once.
- Save is connectivity-aware: connected → shifted data registered as
  `processed_data` derived from the raw input with the shift values recorded; not
  connected → Save dialog writes the file with a not-registered warning. Work is
  never lost.
- Applied global + per-channel shifts are persisted and restored.
- Shift values are **dictionary-declared** (structured, gate-covered), not a blob.
- MFDB failures warn but never crash the shifter.
- API/contract/MFDB tests and a GUI smoke test pass; all ten Definition of Clean
  standards are met.

# Relationships
- First consumer and acceptance case for [PRD-10](prd-10.md) (dataset browser "From MFDB…").
- Its per-channel shift storage is generalized/retired into [PRD-11](prd-11.md)'s role-indexed operation parameters.
- Registration path exercised the silent-failure bug documented for [PRD-10](prd-10.md).
- Builds on the [plugin system](/architecture/plugin-system.md); target in [Plugins target](/specs/plugins.md).
