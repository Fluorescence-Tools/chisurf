# PRD-28: ndXplorer ↔ MFDB Burst-Selection Integration (manual-test round trip)

## Goal

Close the loop so a **burst selection** can move between the Burst Selection tool,
MFDB, and **ndXplorer** through the existing sample/measurement selection widget —
giving a hands-on end-to-end test of the Phase-2 operation/transformer spine:

```
raw measurement ──Burst Selection──▶ burst_table (registered in MFDB, operation
                                     microtime/burst params) ──┐
                                                               ▼
                              MfdbDatasetPickerDialog (sample/measurement select)
                                                               │
                                                               ▼
                                                          ndXplorer (visualize)
```

Two directions:
1. **Burst Selection → ndXplorer:** a "Send to ndXplorer" action that opens the
   just-produced burst selection in ndXplorer.
2. **ndXplorer ← MFDB:** an "Open burst selection from MFDB" action in/around
   ndXplorer that lets the user pick a registered burst selection via the
   sample/measurement selection widget and loads it.

## Why

Phase 2 made Burst Selection register its results as a conformant transformer
(`operation_type="burst_selection"`, typed `.dic` parameters, validated). There is
no GUI path yet to *open* a registered burst selection back into an analysis tool, so
the round trip can't be exercised by hand. ndXplorer is chisurf's burst/MFD explorer
and already ingests burst data; wiring it to the MFDB dataset picker makes the whole
pipeline manually testable.

## Existing pieces to reuse (do not reinvent)

- **Selection widget:** `chisurf/gui/widgets/mfdb/dataset_browser.py` →
  `MfdbDatasetPickerDialog.pick_dataset(kinds=[...], formats=[...])` (PRD-10). The
  Microtime Shifter already uses it (`tttr_microtime_shifter/gui/tool.py:325`). This
  *is* the "sample measurement selection widget" — filter it to burst kinds.
- **Open/resolve path:** the `datasets.open` RPC / dataset-open handler returns a local
  readable path for an artifact (`MfdbDatasetPickerDialog` returns the selected
  artifact; `mfdb.datasets.open` resolves its object to a path).
- **Burst registration:** `burst_selection/api/mfdb.py` already registers a
  `burst_table` artifact (`kind="burst_table"`, `operation_type="burst_selection"`).
- **ndXplorer ingest:** `NDXplorer.open_files(file_type="burst_dir",
  file_handles=<path>, append=False)` and
  `ndxplorer.__main__.open_path_like_drop(ndx, path)` (`modules/ndxplorer`).

## Design

### A. ndXplorer ← MFDB (open a registered burst selection)

Add an **"Open from MFDB…"** entry point that:
1. Opens `MfdbDatasetPickerDialog.pick_dataset(kinds=["burst_table"], parent=…)`
   (optionally also `processed_data` with a burst format) — the sample/measurement
   selection widget, scoped by the canonical user (PRD-17) so "Mine"/"All" work.
2. Resolves the chosen artifact to a local path via the dataset-open path (the
   object store file, or the burst directory the burst table references).
3. Launches/*reuses* an `NDXplorer` instance and calls
   `open_files(file_type="burst_dir", file_handles=path)` (or `open_path_like_drop`).

Placement: a small chisurf-side launcher (so MFDB/Qt-picker code stays out of the
external `ndxplorer` module) — e.g. in the `chisurf/plugins/ndxplorer` wrapper: a menu
action "ndXplorer: Open burst selection from MFDB". Keep `modules/ndxplorer`
dependency-free of chisurf.

### B. Burst Selection → ndXplorer (send the current result)

Add a **"Send to ndXplorer"** action to the Burst Selection tool that, after a run,
takes the produced burst selection's path (the `.bur`/burst directory it just wrote,
or the registered `burst_table` artifact resolved to a path) and opens it in ndXplorer
via the same `open_files`/`open_path_like_drop` call.

### C. Thin, contract-respecting wiring

- GUI actions only orchestrate: pick → resolve path → hand to ndXplorer. No analysis
  logic in the widgets (PRD-23).
- The selection widget and `datasets.open` are reused as-is; the only new code is the
  two launcher actions + path resolution.

## Tasks

1. Path resolution helper: given a selected MFDB `burst_table` artifact, return the
   local burst-data path ndXplorer can open (object-store file or referenced burst
   dir). Reuse the dataset-open handler.
2. ndXplorer launcher action "Open burst selection from MFDB" (picker → resolve →
   `NDXplorer.open_files`). Reuse/instantiate `NDXplorer`; do not modify
   `modules/ndxplorer`.
3. Burst Selection "Send to ndXplorer" action (resolve current result path → open).
4. Construction smoke tests for both actions; a path-resolution unit test. Manual
   acceptance: process a measurement → it registers → pick it in the widget → it opens
   in ndXplorer.

## Definition of Done

- [ ] From ndXplorer, a user can pick a registered burst selection via the
      sample/measurement selection widget and open it.
- [ ] From Burst Selection, a user can send the current result to ndXplorer.
- [ ] No analysis logic in the GUI actions; `modules/ndxplorer` stays chisurf-free;
      the picker + `datasets.open` are reused, not reimplemented.
- [ ] Smoke + path-resolution tests pass; the round trip works by hand.

## Definition of Clean

Reuse `MfdbDatasetPickerDialog` + `datasets.open` (no new browser); thin GUI
orchestration only; `modules/ndxplorer` untouched; identity via the PRD-17 resolver so
the picker scopes correctly.

## Implementation status

Direction A is implemented:
- `chisurf/plugins/ndxplorer/mfdb_launcher.py` — `open_burst_selection_from_mfdb()`
  (pick via `MfdbDatasetPickerDialog` → resolve path via `mfdb.datasets.open` →
  `open_path_like_drop`), plus `resolve_dataset_path` (tested against the real
  in-process client) and `send_path_to_ndxplorer` (direction B helper).
- `chisurf/plugins/ndxplorer_open_burst/` — the menu entry
  **"Tools:Open Burst in ndXplorer"** invoking the launcher.
- Manual test now: run that menu action (or, from the Code Editor,
  `from chisurf.plugins.ndxplorer.mfdb_launcher import open_burst_selection_from_mfdb;
  open_burst_selection_from_mfdb()`), pick a registered burst selection, and it opens
  in ndXplorer.

Remaining: a "Send to ndXplorer" button on the Burst Selection tool (direction B,
`send_path_to_ndxplorer`); confirm ndXplorer ingests the resolved artifact path
(burst_table object vs. the burst-dir `external_reference`) during interactive use.

## Relationship

Manual-test enabler for **PRD-04** (burst pipeline) and the **Phase-2** spine
(PRD-11/16). Reuses **PRD-10** (dataset browser / picker) and **PRD-17** (identity).
Apply **PRD-23** (thin widgets). Mirror the pattern the Microtime Shifter already uses
for the picker.
