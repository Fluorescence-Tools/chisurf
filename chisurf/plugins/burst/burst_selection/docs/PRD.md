# PRD: Burst Selection API, CLI, GUI, and ZMQ Server

## Goal

Refactor the Burst Selection plugin into separated API, CLI, GUI, and ZMQ communication layers while preserving the current user experience and producing the same analysis results. The Burst Selection plugin is the starting point for a future single-molecule data analysis server.

## Decisions

1. Burst-selection ZMQ methods must be registered in the existing ChiSurf `ServiceDispatcher` instead of only running as a standalone server.
2. `tttrlib.BurstFilter` may become the canonical burst engine later, but the first refactor must preserve the current ChiSurf mask-based behavior and must not break the UI.
3. Tests and implementation modules for this work live under `chisurf/plugins/burst/burst_selection/`.
4. Plugin-local tests must be included in the default pytest run.

## Current baseline

- Current GUI entry point: `chisurf/plugins/burst/burst_selection/wizard.py`.
- Current GUI implementation: `chisurf/plugins/burst/burst_selection/burst_selector.py`.
- Current plugin metadata: `chisurf/plugins/burst/burst_selection/__init__.py`.
- Current GUI uses `WizardTTTRPhotonFilter.save_selection` to create `.bur` files, then reads those files back for display.
- Existing core helpers:
  - Burst mask detection: `chisurf/core/fluorescence/burst/burst.py`.
  - Burst DataFrame generation: `chisurf/core/fio/fluorescence/burst.py`.
  - `.bur` read/write: `chisurf/core/fio/fluorescence/burst.py`.

## Current implementation status

- Accomplished:
  - PRD and implementation plan added and kept as a living status document.
  - API modules added under `chisurf/plugins/burst/burst_selection/api/`.
  - CLI moved into `chisurf/plugins/burst/burst_selection/cli/` with a top-level compatibility entrypoint at `cli.py`.
  - GUI adapter moved into `chisurf/plugins/burst/burst_selection/gui/adapter.py` with a top-level compatibility shim at `gui.py`.
  - Legacy GUI implementation moved into `chisurf/plugins/burst/burst_selection/gui/legacy/burst_selector.py`; new GUI entrypoint is controlled by `USE_LEGACY_GUI`.
  - Plugin root cleaned up with `README.md`, `docs/`, compatibility shims, and package metadata only.
  - ZMQ server/client/services moved into `chisurf/plugins/burst/burst_selection/server/`.
  - `BurstSelectionTool` imports the GUI adapter through the `gui` submodule and keeps the existing GUI output path.
  - Output-format toggle handlers now synchronize CSV/MFD-HDF, ZIP, and Remove Folder state instead of being no-ops.
  - Click CLI added with `analyze`, `inspect`, `fit-gmm`, and `serve` commands.
  - Burst Selection RPC methods registered in `ServiceDispatcher`.
  - ZMQ client/server integration tested with `ZmqServer`.
  - Plugin-local tests added and included in the default pytest run.
  - Core CLI plugin discovery/forwarding fixed so `python -m chisurf.core.cli burst-selection ...` forwards subcommands.
  - FRET plugin CLI metadata fixed to avoid plugin discovery warnings.
  - Server proxy integration tests fixed by patching `chisurf.core.api` instead of non-existent `cs.core.api`.
  - Server startup stderr collection fixed to drain available stderr before terminating a live child process.
  - HDF5/ZIP API writers added to `api/io.py` (`write_hdf5`, `zip_output_folder`, `get_unique_folder_path`) — parity with legacy format.
  - Action bar buttons connected: Add TTTR files, Batch, Process, Save .bur, Clear.
  - Setup selection comboBox added, populated from `detector_setups.json`, updates channel/microtime/burst parameters on change.
  - `appendPlainText` → `append` bug fixed.
  - Constructor visibility kwargs (`show_channel_selection`, `show_clear_button`) now actually control UI visibility.
  - Detailed .md instruction files written under `docs/` for GUI verification, HDF5/ZIP tests, tttrlib.BurstFilter integration, and WebUI planning.
- Verified:
  - API burst summary output matches `chisurf.core.fio.fluorescence.burst.generate_burst_dataframe`.
  - `.bur` read/write compatibility works for the API path.
  - HDF5 roundtrip preserves all columns and category_map; ZIP archives preserve directory structure.
  - ZMQ client/server round trip works for `burst_selection.inspect_bur`.
  - Burst Selection services appear in `ServiceDispatcher`.
  - Targeted Burst Selection tests pass: `39 passed` (up from 29).
  - Ruff passes on all new/modified files.
  - Bundled real BH SPC SPC132 single-molecule DNA example data is included under `chisurf/plugins/burst/burst_selection/tests/data/bh_spc132_sm_dna/`.
  - I/O tests cover HDF5 roundtrip, category_map structure, empty input, ZIP packaging, unique folder path resolution.
- Still needed:
  - Integrate `tttrlib.BurstFilter` behind the API while preserving current mask-based parity (see `docs/INSTRUCTION_TTTRLIB_BURSTFILTER.md`).
  - Desktop GUI verification in a real Qt environment (see `docs/INSTRUCTION_GUI_VERIFICATION.md`).
  - Electron/WebUI planning and prototype (see `docs/INSTRUCTION_WEBUI.md`).

## Target architecture

```text
chisurf/plugins/burst/burst_selection/
  __init__.py
  README.md
  wizard.py
  cli.py                         # compatibility entrypoint
  gui.py                         # compatibility adapter shim
  docs/
    PRD.md
    NEW_GUI_MIGRATION.md
  api/
    __init__.py
    models.py
    selection.py
    features.py
    io.py
    stats.py
    serialization.py
  cli/
    __init__.py
    main.py
  gui/
    __init__.py
    adapter.py
    tool.py
    assets/
    legacy/
      burst_selector.py
  server/
    __init__.py
    client.py
    methods.py
    services.py
  tests/
    data/bh_spc132_sm_dna/
```

The GUI should become an adapter around the API. The CLI and ZMQ server must call the same API as the GUI.

## API requirements

The API must provide deterministic, non-Qt functions for:

- TTTR loading.
- Photon filtering.
- Burst start/stop detection.
- Burst summary DataFrame generation.
- Feature computation, including proximity ratio.
- GMM fitting.
- `.bur` read/write compatibility.
- Result serialization.

Initial functions:

```python
load_tttr(path, filetype=None) -> tttrlib.TTTR
apply_photon_filters(tttr, settings) -> np.ndarray
find_bursts(selected_mask) -> np.ndarray
summarize_bursts(start_stop, filename, tttr, windows, detectors) -> pd.DataFrame
compute_features(df) -> pd.DataFrame
fit_gmm(values, settings) -> dict
read_bur(path) -> pd.DataFrame
write_bur(df, path) -> None
```

The first implementation should wrap existing ChiSurf behavior to keep outputs identical.

## CLI requirements

Add a Click CLI registered through plugin metadata:

```python
cli_entrypoint = "burst-selection=chisurf.plugins.burst.burst_selection.cli:cli"
```

Initial commands:

```bash
csc burst-selection analyze FILE [FILE ...] --setup setup.json --setup-name NAME
csc burst-selection inspect BUR_FILE --feature "Proximity Ratio"
csc burst-selection fit-gmm BUR_FILE --feature "Proximity Ratio"
csc burst-selection serve
```

The CLI must not import Qt at startup.

## ZMQ server requirements

Use the existing ChiSurf ZMQ transport and JSON-RPC 2.0 style.

Initial RPC methods:

```text
burst_selection.analyze_files
burst_selection.inspect_bur
burst_selection.fit_gmm_from_bur
```

Initial event topics:

```text
burst_selection.job.created
burst_selection.job.progress
burst_selection.job.completed
burst_selection.job.failed
burst_selection.job.cancelled
```

The server must use:

- `chisurf.server.transport.zmq.ZmqServer`.
- `chisurf.server.dispatcher.ServiceDispatcher`.
- `chisurf.server.jobs.JobManager`.
- Existing `ServiceResult` error conventions.

## GUI requirements

The GUI must keep the same menu name and user workflow:

- Drag/drop TTTR files.
- Batch processing dialog.
- Channel setup.
- GMM settings.
- Histogram and GMM plotting.
- DataFrame editor.
- `.bur`, HDF5, and ZIP output options.

Current refactor status: the GUI has a non-Qt adapter layer under `gui/adapter.py` for settings conversion, saved `.bur` loading, UI DataFrame construction, and API analysis wrappers. The existing `WizardTTTRPhotonFilter.save_selection` path is still used for `.bur`, HDF5, and ZIP output to preserve the existing workflow. The new API currently writes `.bur` output directly; HDF5 and ZIP output remain GUI-only until a parity-safe API writer is added.

The GUI should call API functions for computation and display results from API result objects.

## Test requirements

Add tests under:

```text
chisurf/plugins/burst/burst_selection/tests/
```

Minimum coverage:

- API selection and feature functions using real TTTR data only.
- `.bur` read/write compatibility.
- GMM fitting.
- CLI commands.
- ZMQ server/client round trip.
- GUI adapter helper tests.
- GUI widget instantiation without changing UI behavior, once a headless Qt platform is available.

Update `pyproject.toml` so these tests run by default.

## Implementation steps

1. [x] Add PRD and implementation plan.
2. [x] Add baseline API modules without changing GUI behavior.
3. [x] Add tests that compare API results against current behavior where possible.
4. [x] Refactor the GUI to call the API while preserving the same results.
4b. [x] First migrated PyQt GUI implemented in `gui/tool.py` with API-backed analysis.
5. [x] Add Click CLI.
6. [x] Add ZMQ server/client.
7. [x] Register burst-selection methods with `ServiceDispatcher`.
8. [x] Reorganize plugin into clean submodules: `api/`, `cli/`, `gui/`, `server/`, and `tests/`.
9. [x] Run targeted and default tests for changed files.
10. [x] GUI button connections and setup selection from `detector_setups.json`.
11. [x] HDF5/ZIP API writer parity (`api/io.py`).
12. [x] Real-data tests for I/O (39 passing).
13. [x] Detailed .md instruction files for remaining work under `docs/`.

## Acceptance criteria

- Existing Burst Selection GUI still opens and behaves the same.
- Existing `.bur` output remains compatible.
- Plugin structure is organized into `api/`, `cli/`, `gui/`, `server/`, and `tests/` submodules with compatibility shims where needed.
- API, CLI, GUI adapter, and server share the same analysis path where the API is used.

  - Real BH SPC SPC132 example data is available under `chisurf/plugins/burst/burst_selection/tests/data/bh_spc132_sm_dna/` and is used by plugin-local tests.
  - ZMQ server can inspect a `.bur` request and is structured to analyze burst-selection requests through `burst_selection.analyze_files`.

- Burst-selection methods appear in `meta.methods`.
- Plugin-local tests are included in the default pytest run.
- No existing tests regress.
