# Burst Selection Plugin — Current Status

_Last updated: 2026-06-10_

## Overview

The Burst Selection plugin has been refactored into a clean API/CLI/GUI/server
architecture. All modules are implemented, tested with real BH SPC132 data, and
passing. **The codebase is in good shape** — the main outstanding objective is
HDF5/ZIP API writer parity (now implemented), plus final GUI equivalence
verification in a desktop environment.

## Module-by-module status

| Module | Lines | Coverage | Status |
|--------|-------|----------|--------|
| `api/` | ~670 | API+CLI+server tests | ✅ Complete |
| `cli/` | ~155 | CLI integration tests | ✅ Complete |
| `gui/tool.py` | ~1140 | New GUI default tests | ✅ Implemented |
| `gui/adapter.py` | ~200 | Used by tests | ✅ Complete |
| `gui/legacy/` | ~962 | Preserved | ✅ Preserved |
| `server/` | ~240 | Server+ZMQ tests | ✅ Complete |
| `tests/` | 19 files | 29 passing | ✅ Complete |

## What's implemented

- **API** (`api/`): Models, TTTR loading, photon filtering, burst detection,
  burst summary (DataFrame), feature extraction, GMM fitting, `.bur` read/write,
  HDF5 write (legacy-compatible), ZIP packaging, serialization, statistics.
- **CLI** (`cli/`): `analyze`, `inspect`, `fit-gmm`, `serve` — all functional.
- **Migrated GUI** (`gui/tool.py`): Full PyQt GUI with drag/drop, batch
  processing, filter controls, histogram/burst/dT/MCS/decay plots, GMM fitting,
  output format selection (CSV/bur, MFD-HDF, ZIP, Remove Folder).
- **Legacy GUI** (`gui/legacy/`): Preserved, selectable via `USE_LEGACY_GUI`.
- **Server** (`server/`): `analyze_files`, `inspect_bur`, `fit_gmm_from_bur`
  RPC methods; ZMQ JSON-RPC client; `ServiceDispatcher` registration.
- **Tests** (`tests/`): 29 tests covering API (real data), CLI (real data),
  server/ZMQ, services, GUI defaults — all passing, zero skips/xfail.

## What's still needed

### 1. Desktop GUI verification
The migrated GUI has never been opened and exercised in a real desktop
environment with Qt rendering available. The headless test environment cannot
test widget instantiation at the `QApplication` level.

**Action:** Someone needs to run the plugin on a Mac with a display:
```bash
cd /Users/tpeulen/dev/chisurf
python -m chisurf
# Navigate: Spectroscopy > Single-Molecule > Burst-Selection
```
Then compare against legacy (set `USE_LEGACY_GUI = True` in `__init__.py`).

### 2. HDF5/ZIP API writer
✅ **DONE as of 2026-06-10.** `api/io.py` now contains:
- `write_hdf5(dataframes, path)` — legacy-compatible HDF5 with category_map
- `zip_output_folder(output_folder, zip_path)` — ZIP compression
- `get_unique_folder_path(base_path)` — unique directory resolver

These are exported from `api/__init__.py`. Tests are still needed (see below).

### 3. GUI parity gaps (known)
- `show_channel_selection` / `show_clear_button` kwargs now work (fixed 2026-06-10)
- `show_decay_button` / `show_filter_button` have no direct equivalents in
  migrated GUI (legacy had dedicated tool buttons; migrated GUI has dock tabs)
- `appendPlainText` → `append` bug (fixed 2026-06-10)

### 4. HDF5/ZIP writer tests
Tests for `write_hdf5` and `zip_output_folder` are not yet written. They should
verify that:
- HDF5 output can be read back and matches the original DataFrame
- Category map round-trips correctly
- ZIP archive preserves directory structure

### 5. tttrlib.BurstFilter integration
Not started. The current API wraps existing ChiSurf mask-based behavior
(`burst.py`, `count_rate.py`). `tttrlib.BurstFilter` can replace these
internally once parity tests exist.

### 6. Electron/WebUI
Not started. Planned as post-stabilization milestone.

## Test results

```
$ /Users/tpeulen/mambaforge/envs/arm64/bin/python3 -m pytest \
    chisurf/plugins/burst/burst_selection/tests -q --no-cov
29 passed in 6.98s
```

All CLI, server, ZMQ, services, and GUI-default tests pass. No skips or xfail.

## Key files

| File | Purpose |
|------|---------|
| `__init__.py` | Plugin metadata, `USE_LEGACY_GUI` switch, CLI entrypoint |
| `wizard.py` | Plugin entry point, instantiates `BurstSelectionTool` |
| `api/io.py` | TTTR loading, `.bur` R/W, HDF5 write, ZIP, unique path |
| `api/selection.py` | Photon filtering, burst detection, burst summary, `analyze_file` |
| `api/features.py` | Feature extraction, GMM fitting |
| `api/models.py` | Dataclasses for all settings/request/result types |
| `api/serialization.py` | JSON-compatible serialization helpers |
| `api/stats.py` | DataFrame summary statistics |
| `cli/main.py` | Click CLI: `analyze`, `inspect`, `fit-gmm`, `serve` |
| `gui/tool.py` | Migrated PyQt GUI with legacy-style controls and plots |
| `gui/adapter.py` | GUI ↔ API bridge helpers |
| `gui/legacy/burst_selector.py` | Preserved legacy GUI |
| `server/methods.py` | RPC handler functions |
| `server/services.py` | `ServiceDispatcher` registration |
| `server/client.py` | ZMQ JSON-RPC client wrapper |
| `docs/PRD.md` | Product requirements & implementation plan |
| `docs/NEW_GUI_MIGRATION.md` | PyQt → WebUI migration phases |
| `tests/` | 29 tests with real BH SPC132 data |
