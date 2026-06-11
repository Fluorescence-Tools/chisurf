# New Burst Selection GUI Migration Plan

## Goal

Replace the legacy `BurstSelectionTool` implementation with a new GUI layer that uses the Burst Selection API for analysis while preserving the existing user workflow and `.bur` compatibility.

## Long-term direction

The new GUI should become a local WebUI, for example Electron or a similar framework. Local WebUI processes should communicate with ChiSurf through the existing ZMQ JSON-RPC transport.

For now, keep the PyQt path as the temporary implementation target so the migration can proceed without changing the current ChiSurf GUI runtime. The PyQt GUI may reuse existing ChiSurf modules where helpful to minimize duplication. The future Electron/WebUI process should depend only on ChiSurf core and ZMQ JSON-RPC, not the PyQt GUI layer.

## Control switch

The plugin exposes:

```python
USE_LEGACY_GUI = True
```

Current behavior:

- `USE_LEGACY_GUI = True` imports `chisurf.plugins.burst.burst_selection.gui.legacy.burst_selector.BurstSelectionTool`.
- `USE_LEGACY_GUI = False` imports `chisurf.plugins.burst.burst_selection.gui.tool.BurstSelectionTool`, which currently raises `NotImplementedError` until the new GUI is ready.

This switch allows stepwise migration without breaking the current plugin.

## Migration phases

### Phase 1: Stabilize legacy path

- Keep `gui/legacy/burst_selector.py` unchanged except for submodule imports and asset path fixes.
- Keep output behavior identical:
  - CSV/`.bur`
  - MFD-HDF
  - ZIP output
  - Remove Folder behavior
- Preserve `WizardTTTRPhotonFilter.save_selection` as the authoritative output path.
- Keep real-data tests pinned to the bundled BH SPC132 example.

### Phase 2: Add temporary PyQt shell, then WebUI shell

Create a new GUI implementation under:

```text
gui/tool.py
```

Temporary implementation target:

- Keep PyQt as the short-term path so existing tests and workflows continue.
- Do not introduce Electron/WebUI until the API/ZMQ contract is stable.

Long-term implementation target:

- Replace the PyQt shell with a local WebUI, for example Electron or a similar framework.
- The WebUI should communicate with ChiSurf through ZMQ JSON-RPC.
- Keep the same service contracts used by the CLI/server:
  - `burst_selection.analyze_files`
  - `burst_selection.inspect_bur`
  - `burst_selection.fit_gmm_from_bur`

Initial responsibilities:

- Load the existing `.ui` layout or replace it with a new Qt layout.
- Reuse existing menu/workflow concepts:
  - drag/drop TTTR files
  - batch processing dialog
  - channel setup
  - GMM settings
  - histogram and GMM plotting
  - DataFrame editor
  - `.bur`, HDF5, and ZIP output options
- Do not migrate computation yet. The shell can still call the legacy tool for UI behavior while the API path is prepared.

Acceptance:

- `USE_LEGACY_GUI = False` opens a GUI shell or clearly reports the incomplete migration state.
- No legacy computation is called from the new shell yet.

### Phase 3: Wire analysis through API

Replace legacy computation calls with API calls:

- Use `api.selection.analyze_file` / `analyze_request`.
- Use `gui.adapter.analysis_settings_from_wizard` or a new settings mapper for the migrated GUI.
- Use `gui.adapter.make_ui_dataframe` for the table view.
- Use `api.features.extract_features` for histogram feature data.
- Keep `.bur` output compatible with `api.io.write_bur`.

Acceptance:

- New GUI analysis output matches legacy `.bur` output for the bundled BH SPC132 file.
- Real-data parity tests compare new GUI/API output against `generate_burst_dataframe`.

### Phase 4: Migrate output writers

Move output behavior out of `WizardTTTRPhotonFilter.save_selection`:

- `.bur` writer: already exists in `api.io.write_bur`.
- HDF5 writer: add parity-safe API writer.
- ZIP writer: add API-level packaging helper.
- Remove-folder behavior: keep as an explicit post-process option.

Acceptance:

- `.bur`, HDF5, and ZIP outputs from the new GUI match the legacy workflow.
- ZIP/Remove Folder state is synchronized by GUI controls, not hidden side effects.

### Phase 5: Replace histogram GMM behavior

Important constraint from user: do not fit GMMs to histograms by default.

Migration target:

- Histogram display remains available.
- GMM fitting is explicit user action only.
- Prefer feature-table GMM fitting through `api.features.fit_gmm`.
- Avoid automatic `GaussianMixture.fit(filtered_data.reshape(-1, 1))` from histogram update paths.

Acceptance:

- Opening or refreshing the GUI does not run GMM fitting automatically.
- GMM fitting is triggered only by an explicit command/action.
- Tests verify no default histogram GMM fitting path.

### Phase 6: Flip the plugin switch

When the new GUI passes real-data and UI-equivalence tests:

- Set `USE_LEGACY_GUI = False`.
- Keep `gui/legacy/` available for rollback.
- Add a short deprecation note in `README.md`.

Acceptance:

- Default plugin import uses the new GUI.
- Legacy import remains available through `gui/legacy.burst_selector`.

## Test strategy

Use the bundled real BH SPC132 data:

```text
tests/data/bh_spc132_sm_dna/m000.spc
```

Required real-data tests:

- API analysis parity against `generate_burst_dataframe`.
- `.bur` read/write compatibility.
- CLI `analyze`, `inspect`, and explicit `fit-gmm`.
- ZMQ `analyze_files`, `inspect_bur`, and `fit_gmm_from_bur`.
- GUI adapter settings mapping.
- No fake TTTR objects.

GUI tests should run only when a working headless Qt platform is available.

## Current status

- Legacy path: preserved under `gui/legacy/` and controlled by `USE_LEGACY_GUI`.
- First migrated PyQt GUI: implemented in `gui/tool.py`; it uses the API for analysis and does not run GMM fitting by default.
- Plugin default: `USE_LEGACY_GUI = False` for the first migrated GUI iteration.
- API/CLI/server: implemented and tested with real data.
- Long-term WebUI/Electron target: pending after the PyQt migration stabilizes.
- HDF5/ZIP API writers: pending.
- Default histogram GMM fitting: must not be migrated as default behavior.
