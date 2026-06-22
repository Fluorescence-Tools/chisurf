# PRD-09: Microtime Shifter — Workflow Plugin + MFDB Treatment

## Goal

Convert `chisurf/plugins/tttr/tttr_microtime_shifter` from a single-file
`QMainWindow` into a **workflow-ready plugin** modelled on Burst Selection
(`chisurf/plugins/burst/burst_selection`): a `core` / `api` / `cli` / `gui`
split that communicates via RPC, with full MFDB provenance — dropped TTTR files
are identified/deduplicated in the object store, shifted files are written and
registered as derived artifacts, and the per-channel shift values are persisted
as **dictionary-declared** MFDB data so the operation is fully traceable.

## Reference implementation (read first)

Burst Selection is the canonical workflow-plugin template:

- `chisurf/plugins/burst/burst_selection/api/contract.py` — `CONTRACT_VERSION`,
  `*_from_payload` / `*_to_payload`, `contract_descriptor`, service envelope.
- `chisurf/plugins/burst/burst_selection/api/models.py` — `AnalysisRequest` /
  `AnalysisResult` + `MFDBContext`.
- `chisurf/plugins/burst/burst_selection/api/selection.py` — pure analysis.
- `chisurf/plugins/burst/burst_selection/api/mfdb.py` — MFDB registration
  (`register_raw_measurement` / `register_result`, parent/derived edges).
- `chisurf/plugins/burst/burst_selection/backend/services.py` — RPC handlers +
  state; `cli/main.py`; `gui/client.py` + `gui/tool.py`.

Current shifter (to replace): `tttr_microtime_shifter/wizard.py` —
`MicroTimeShifter(QMainWindow)` with inline Load…/Save… buttons + a drag-drop
`FileLineEdit`, `self.shifts = {channel: shift}` + `self.global_shift`, and
direct `tttrlib` read/write of shifted files. No split, no RPC, no MFDB.

## Governing rules (carried from PRD-04)

- **MFDB is the store; the `.dic` dictates the schema.** Anything structured the
  plugin persists (the shift values especially) is declared in
  `chisurf/core/mfdb/data/mfdb_flr_ext.dic` with `_chisurf_schema` bridges and is
  covered by the total-coverage `validate_mapping()` gate. No opaque JSON blobs
  for the shift values.
- MFDB registration is best-effort for the GUI flow — failures warn, never crash
  the shifter.
- Reuse PRD-03 (`register_raw_measurement` / `register_result`) and the object
  store; do not hand-roll object/artifact writes.

## Definition of Clean (non-negotiable engineering standards)

These encode lessons already learned in PRD-04. A review will reject the work if
any are violated.

1. **Layer purity.** `api/shift.py` has **no** Qt and **no** DB imports — pure
   functions over data, directly unit-testable. The GUI calls the API **only via
   the RPC client**; no `tttrlib` read/write and no MFDB calls inside the widget.
   `backend/services.py` is the only place that wires pure-API → MFDB.
2. **Single source of truth / no blobs.** Shift values are dictionary-declared
   structured columns (Task 5). No correlator-style JSON blob, no second copy of
   field names/labels in code. `DictionarySchemaMap.validate_mapping()` must pass
   for every new item, checked by a **total-coverage** test that derives its item
   list from the dictionary (no hand-maintained allow-list).
3. **No hardcoded SQL.** New tables/columns are generated from the `.dic` via the
   PRD-04 Task-P1a generator; migrations add columns with the verified
   `_ensure_column` helper — never `try/except OperationalError: pass`.
4. **Best-effort MFDB, but fail loud on real bugs.** MFDB-unavailable warns and
   the shift still works; but genuine errors (bad migration, missing column,
   failed registration) must surface, not be swallowed.
5. **Tests assert behavior, not just existence.** MFDB tests assert the
   `derived_from` edge (shifted→raw), object-store **dedup** (same content not
   re-stored), and that the stored shifts read back equal to what was applied —
   not merely "a row exists." Pure-shift tests assert the actual micro-time
   values.
6. **GUI build coverage.** Add a construction smoke test for the dockable tool
   (mirror `test/gui/test_detector_wizard_page.py`). Every new Qt symbol used must
   be imported — a missing import that only fails at construction is a defect the
   smoke test must catch.
7. **Dependency injection over monkeypatching.** Tests inject the DB / RPC seams
   (e.g. an explicit `db_path` / in-process client), not `monkeypatch.setattr` of
   internals, wherever the seam can be a parameter.
8. **No dangling legacy.** `wizard.py` is removed (its logic moved into the
   split), not left as dead/duplicate code. No import to a deleted module.
9. **Determinism + idempotence.** Re-running a shift with the same inputs is
   idempotent (no duplicate artifacts/objects); generated DDL is stable.
10. **Run the whole relevant suite together** (plugin tests + the dict gate
    `test/fio/test_setup_prerequisites.py`) so a new category can't silently break
    the gate or regress sibling plugins.

## Tasks

### Task 1: API layer (`api/`)

- `api/models.py`:
  - `ShiftRequest`: `files: list[str]`, `global_shift: int`,
    `channel_shifts: dict[int, int]`, `filetype: str | None`,
    `output_dir: str | None`, `mfdb: MFDBContext` (reuse the Burst pattern).
  - `ShiftResult`: `output_paths_by_file: dict[str, str]`,
    `applied_shifts_by_file: dict[str, dict]`, `mfdb_artifacts: dict`,
    `warnings: list[str]`.
- `api/shift.py` — **pure** shift logic (no GUI, no DB): read a TTTR file, apply
  the global + per-channel micro-time shifts, write the shifted file; return the
  output path and the exact shifts applied. Directly unit-testable. This is the
  refactor of the current `load_file` / save logic out of the widget.
- `api/contract.py` — `CONTRACT_VERSION`, `shift_request_from_payload` /
  `*_to_payload`, `shift_result_to_payload`, `contract_descriptor`, and the
  service-success envelope, mirroring Burst's contract.
- `api/mfdb.py` — registration (see Task 4).

### Task 2: Backend service + RPC (`backend/`)

- `backend/services.py` — register RPC handlers on the ChiSurf
  ServiceDispatcher (ZMQ JSON-RPC), e.g. `microtime_shift.apply`,
  `microtime_shift.load_metadata` (return used routing channels + n micro-time
  channels for a file), `microtime_shift.identify` (object-store lookup, Task 3).
  Handlers call the pure `api/shift.py` then `api/mfdb.py`. Mirror
  `burst_selection/backend/services.py`.

### Task 3: CLI (`cli/`)

- `cli/main.py` — headless entry: take input file(s), shifts, output dir; run the
  pure API; optionally register in MFDB. Mirror `burst_selection/cli/main.py`.

### Task 4: MFDB treatment

**Save is connectivity-aware (two modes).** The Save action adapts to whether the
plugin is connected to MFDB (same pattern as PRD-04 setup persistence):

- **Connected to MFDB → register, don't write a loose file.** Save writes the
  shifted bytes into the **managed object store** and records full provenance
  (raw → `microtime_shift` operation with the shift values → shifted artifact,
  `derived_from` edge). No user-folder file is required. The GUI confirms "shifted
  data registered in MFDB" and (optionally) which artifact id.
- **Not connected to MFDB → old behaviour (save to file).** Save falls back to the
  current flow: a Save dialog writes the shifted TTTR to a user-chosen path. The
  user is informed it was saved to a file and not registered in MFDB (warn, do not
  fail — work is never lost). This is the existing `open_save_dialog` / `save_file`
  path, preserved.
- "Connected" means a usable MFDB resolves (`get_db()` / object store available).
  Both modes share the same pure shift output; only the persistence target
  differs. A "Save to file…" action also remains available explicitly even when
  connected, for export/portability (mirrors the JSON-export rule for setups).

Provenance chain (connected mode):

```text
dropped TTTR file  [raw_measurement, deduplicated in object store by content hash]
  -> operation: microtime_shift  [global + per-channel shifts, .dic-declared]
  -> shifted TTTR  [processed_data/"shifted_tttr", registered in object store,
                    derived_from the raw input]
```

- **Identify dropped files in the object store.** On load/drop, compute the file
  content hash and look it up in `mfdb_object` (by `content_md5`). If the raw
  TTTR is already stored, reuse that artifact/object instead of re-registering;
  otherwise register it with `register_raw_measurement`. Surface "already in
  MFDB" vs "newly registered" to the GUI.
- **Save and register the shifted file in the object store.** Write the shifted
  TTTR, then `register_result(kind="processed_data", data=<shifted file path>,
  data_format="ptu"/inferred, parent_artifact_id=<raw>,
  operation_type="microtime_shift", parameters=<shifts>, ...)` so the shifted
  bytes live in the managed object store, derived from the raw input.
- **Keep the shifts.** The applied global + per-channel shifts are persisted with
  the operation so re-loading a previously shifted file restores/show the shift
  values that produced it.

### Task 5: Shift values need a `.dic` entry (structured, dictionary-declared)

The shift values are not free parameters in a blob — they are dictionary-declared
MFDB data. Choose and record:

- **Preferred:** a structured child table `mfdb_microtime_shift` — one row per
  (operation/artifact, routing_channel): `shift` (micro-time channels),
  plus a `global_shift` on the operation/artifact level. Declared in
  `mfdb_flr_ext.dic` with `_chisurf_schema` bridges, generated via the PRD-04
  Task-P1a generator, and added to `_SETUP_CATEGORIES`/the gate's coverage set so
  `validate_mapping()` checks it. `SCHEMA_VERSION` bump.
- **Or** record per-channel shifts as `mfdb_parameter` rows with names like
  `shift_ch<idx>` plus `global_shift`, **and** add the corresponding `.dic`
  item descriptions. Only acceptable if the structured table is rejected in
  review; record the reason.

Either way: the shift values must be queryable and dictionary-described — no
undocumented blob. Units = micro-time channels.

### Task 6: GUI (`gui/`) — client + dockable tool

- `gui/client.py` — RPC client wrapping the `microtime_shift.*` methods (mirror
  `burst_selection/gui/client.py`; in-process client + ZMQ).
- `gui/tool.py` — the `QMainWindow`, restructured:
  - **Move Load/Save (button + line edit) into a tool menu / toolbar**, out of
    the inline central layout. Drag-drop still works (a drop target identifies
    the file in the object store, Task 3).
  - **Custom docks:** put the **microtime-shift controls** (global + per-channel
    shift spinboxes) in one `QDockWidget` and the **plot** (micro-time
    histogram / shifted overlay) in another, so they are detachable/rearrangeable.
  - The GUI calls the API only via the RPC client — no direct `tttrlib` read/write
    or DB writes in the widget.
  - **Connectivity-aware Save** (Task 4): when MFDB is connected, Save registers
    the shifted data in MFDB (no loose file needed); when not connected, Save
    falls back to the file Save dialog and warns that the result was not
    registered. Reflect the current mode in the UI (e.g. the Save action label /
    a status indicator) and keep an explicit "Save to file…" available regardless.
  - Show MFDB status (identified vs newly registered; shifted data registered;
    warnings) without blocking the shift.

### Task 7: Plugin packaging

- Add `manifest.json` (display name, entrypoints: gui/cli) like the modern
  plugins; keep the icon. `__init__.py` loads the manifest and re-exports the GUI
  tool. Move legacy `wizard.py` content into the split or delete it.

### Task 8: Tests

- `api/shift.py` pure tests: applying a known global/per-channel shift produces
  the expected micro times; round-trip read→shift→write→read.
- Contract round-trip: `shift_request_from_payload` / `*_to_payload`.
- MFDB tests: dropped file identified in the object store (dedup — second
  registration of the same content does not duplicate the object); shifted file
  registered as `processed_data` derived from the raw input; shift values stored
  and readable; the dict gate covers the new shift items.
- Connectivity-aware save: with MFDB available, save registers the shifted
  artifact (and records the shifts) and writes no loose file; with MFDB
  unavailable (inject a null/absent DB), save still produces the shifted file at
  the requested path and reports the not-registered warning. Both preserve the
  shifted output.
- A GUI construction smoke test for the dockable tool (mirror
  `test/gui/test_detector_wizard_page.py`).

## Definition of Done

- [ ] `tttr_microtime_shifter` is split into `api` / `backend` / `cli` / `gui`
      with a versioned contract; the GUI talks to the API only via RPC.
- [ ] Load/Save (button + line edit) live in a tool menu; microtime-shift
      controls and the plot are separate custom docks; drag-drop still works.
- [ ] Dropped TTTR files are identified in the MFDB object store (deduplicated by
      content hash); raw inputs are registered once.
- [ ] Save is connectivity-aware: connected to MFDB → shifted data registered in
      the object store as `processed_data` derived from the raw input, with the
      shift values recorded as provenance; not connected → old behaviour (Save
      dialog writes the shifted file to a user-chosen path) with a warning that it
      was not registered. Work is never lost in either mode.
- [ ] The applied global + per-channel shifts are persisted and restored.
- [ ] Shift values are **dictionary-declared** in `mfdb_flr_ext.dic` (structured,
      gate-covered) — not stored in an opaque blob.
- [ ] MFDB failures warn but never crash the shifter.
- [ ] API/contract/MFDB tests and a GUI smoke test pass.
- [ ] All ten "Definition of Clean" standards are met (layer purity, no blobs,
      generated DDL, behavior-asserting tests, GUI smoke test, DI over
      monkeypatching, no dangling legacy, idempotence, gate stays green).
