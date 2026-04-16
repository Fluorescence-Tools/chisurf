# CHANGELOG

## [Unreleased]

### Fixed

- **macOS App Bundle Build now generates a `.dmg` artifact**:
  - Updated `build_tools/osx/build-osx-app.sh` to create a disk image (`.dmg`) using `hdiutil`.
  - The DMG contains the `ChiSurf.app` bundle and a convenient link to `/Applications`.
  - Updated `.github/workflows/pixi-ci.yml` to propagate `CHISURF_VERSION` to the DMG build step via `$GITHUB_ENV`.
  - Fixed a regression in `build-osx-app.sh` where `PYTHONPATH: unbound variable` caused CI failures by quoting the launcher heredoc delimiter.
  - Hardened the `ChiSurf` launcher to use absolute paths for `PYTHONPATH` and `QT_PLUGIN_PATH`, ensuring the bundle remains self-contained regardless of the current working directory.

- **Optimized NDXplorer data loading with background computation and caching**:
  - Offloaded expensive data processing and column computation to background threads to prevent UI blocking.
  - Implemented `FileMetadataCache` in `modules/ndxplorer` to accelerate subsequent loads of previously processed datasets.
  - Resolved UI hang during large folder ingestion by ensuring background tasks do not block the main event loop.

- **Improved NDXplorer plot responsiveness and axis selection**:
  - Fixed root cause of empty axis selectors after folder-drop loads by hardening parameter metadata collection.
  - Ensured UI correctly re-enables after all data loading and append paths.
  - Added automatic hiding of image-specific controls for non-image datasets.

- **Kristine FCS file loading now correctly restores count rate, acquisition time, and masks**:
  - Re-synchronized `read_kristine` indexing in `chisurf/fio/fluorescence/fcs/kristine.py` to match the non-transposed `(n_points, n_columns)` data layout.
  - Added full support for an optional 5th 'mask' column in Kristine FCS files and generic CSV datasets.
  - Corrected duration and count rate extraction from column 2 (indices `[0, 2]` and `[1, 2]`) and uncertainty/weight extraction from column 3.
  - Resolved a bug in `read_fcs` (`chisurf/fio/fluorescence/fcs/__init__.py`) where CSV-loaded datasets were not correctly populated into the internal dataset list.
  - This resolves a regression where count rates were read as near-zero (lag-time values), causing small weights and suppressed weighted residuals in FCS fits.
  - Extended `DataCurve` and `DataGroup` (`chisurf/data.py`) to properly store, serialize, and load mask data, fixing crashes when loading 5-column ASCII datasets.

- **Resolved Windows access violation crash during MCP server startup**:
  - Refactored GUI synchronization to ensure Qt objects (`_GuiExecutor`, `_GuiSyncInvoker`) are initialized on the main GUI thread during startup, avoiding `killTimer` and cross-thread destruction warnings.
  - Implemented explicit `asyncio` event loop policy for the MCP server thread on Windows, utilizing `ProactorEventLoopPolicy` for improved stability with HTTP transports.
  - Pre-initialized GUI executors during the `startup_interface` stage to eliminate race conditions and thread-safety issues during lazy initialization in background threads.
  - Added pre-import of `chisurf.mcp.server` in the main thread to ensure proper module and Qt class registration before the background MCP thread starts.

- **Fixed ValueError: too many values to unpack (expected 4) and associated "huge residuals" during Curve/DataCurve slicing**:
  - Resolved a fundamental regression in `Curve.__getitem__` (in `chisurf/curve.py`) where incorrectly using `d.flatten()` on 2xN arrays caused slices to return x-axis values (time/lags) as y-axis values (correlation/amplitudes).
  - Updated `calculate_weighted_residuals` in `chisurf/fitting/__init__.py` and `DataCurve.save` in `chisurf/data.py` to handle the 5-tuple returned by `DataCurve` slicing.
  - Refactored `DataCurve.__getitem__` in `chisurf/data.py` to directly access data properties, bypassing the unstable legacy `super()` implementation.
  - Corrected `DataCurve.__getitem__` type hints for better developer visibility and tool consistency.
  - This fix restores accuracy to FCS fits, weighted residual plots, and chi-squared calculations which were broken by the cross-axis slicing error.

- **Resolved persistent SIGBUS (EXC_ARM_DA_ALIGN) startup crash on macOS ARM64 (Apple Silicon)** by sanitizing UI elements (Emojis, special symbols) that triggered incompatible rendering paths in CoreText.
- **Hardened `build-osx-app.sh` by removing `DYLD_LIBRARY_PATH`** to prevent library conflicts with system frameworks and ensured the ribbon correctly renders in-window by changing `RibbonBar` inheritance from `QMenuBar` to `QWidget`.
- **Fixed `ModuleNotFoundError` for `chisurf.plugins._dev` in bundled application** by implementing a global development-plugin proxy system in `chisurf/plugins/__init__.py`.
- **Restored `ChiSurf.icns` icon association and launcher environment fixes** for the macOS application bundle.

### Changed

- **Updated `quest` dependencies for NumPy 2.0 compatibility**:
  - Updated `modules/quest/pyproject.toml` to require `numpy >= 2.0`.
  - Removed obsolete `pymol-open-source` dependency from build environment.

- **ChiSurf distribution now managed exclusively by pixi with rattler-build**:
  - Removed conda-build recipe (`conda-recipe/` directory deleted) — superseded by `rattler-recipe/`
  - Removed `.condaignore` file (conda-build-specific)
  - Removed `micromamba` from runtime dependencies; moved to `[feature.build.dependencies]` in `pixi.toml`
  - All build and distribution tasks now use `pixi run` with the `build` feature
  
- **Implemented dynamic entry points discovery for build recipes and installers**:
  - Created `rattler-recipe/collect_entry_points.py` to dynamically discover all entry points at build time from two sources:
    1. Static entry points from `pyproject.toml` (`[project.scripts]` and `[project.gui-scripts]`)
    2. Dynamic plugin entry points via AST-parsing of `cli_entrypoint` variables in `chisurf/plugins/*/` `__init__.py` files
  - Added `collect-entry-points` pixi task (required before build) that generates `rattler-recipe/entry_points.json`
  - Updated `rattler-recipe/recipe.yaml` to load entry points from generated JSON instead of hardcoding — supports all 26 entry points (18 static + 8 dynamic from plugins)
  - Updated `build_tools/win/create_installer_script.py` to read from `entry_points.json`, enabling plugin CLI entry points to appear in Inno Setup installer menus
  - Updated `build_tools/win/build-setup.bat` to call entry points collection before package builds
  
- **Updated CI workflow for pixi-based distribution and GitHub Releases**:
  - Renamed `.github/workflows/conda-release.yml` → `build-release.yml` (more accurately reflects pixi-based build)
  - Removed Anaconda.org upload step (no longer conda-centric)
  - Added GitHub Releases upload for tagged releases using `softprops/action-gh-release@v1`
  - Artifacts (`.conda` and `.tar.bz2` packages) are now uploaded to GitHub Releases on tag creation
  
- **Windows installer (`setup.exe`) remains on build path**:
  - Inno Setup flow preserved; uses pixi-managed environment instead of raw micromamba
  - `build-setup.bat` still creates distributable environment via `micromamba create` (now a build-only dependency)

### Added

- **Jordi G-factor plugin now supports slow-dye (FP) anisotropy-mixing calibration in estimate mode**: after fast-dye tail matching for `g`, users can load a separate FP Jordi file to estimate linked `l1/l2` from steady-state anisotropy (`rho` default `16 ns`, lifetime estimated from decay first moment).
  - Added FP controls/outputs and estimate warnings in `chisurf/plugins/jordi_g_factor/__init__.py` (`load_fp_jordi_file`, `calculate_fp_mixing_estimate`, linked `l1=l2` solve, `dt` input, editable `tau`/target `rS` with manual override toggles, single `l1/l2` estimate output).
  - Moved Jordi g-factor plugin controls into a Designer-editable UI file `chisurf/plugins/jordi_g_factor/wizard.ui` and simplified linked output display to a single `l1/l2 est` field.
  - Updated Jordi plotting in `chisurf/plugins/jordi_g_factor/__init__.py`: tail-matched decays (`VV` and `VH*G`) are now overlaid in the main decay plot, and the secondary plot now shows time-resolved anisotropy `r(t)` with uncorrected/corrected traces.
  - Added direct manual override inputs in `chisurf/plugins/jordi_g_factor/__init__.py` / `chisurf/plugins/jordi_g_factor/wizard.ui`: editable `g` value (accepted on user entry) and editable linked `l1/l2` value (accepted directly on user entry).
  - Clamped anisotropy display range to `[-0.5, 1.5]` in the `r(t)` plot for both fast and slow dye traces.
  - Switched Jordi plugin entry points to dynamic loading (`importlib`) in `chisurf/gui/widgets/experiments/tcspc/csv_tcspc_widget.py` and `chisurf/gui/widgets/wizard/tttr_channeldefinition/tttr_channel_definition_tttr_io.py`.
  - Updated the TCSPC CSV reference button flow in `chisurf/gui/widgets/experiments/tcspc/csv_tcspc_widget.py` to propagate plugin-derived `l1/l2` only when FP/slow-dye calibration was actually processed (`fp_estimate_available`), in addition to `g` and `vh_shift`.
  - Updated TCSPC CSV reference launch to initialize Jordi plugin `dt [ns/ch]` from current reader settings (`dt` + optional rebin scaling) in `chisurf/gui/widgets/experiments/tcspc/csv_tcspc_widget.py`.
  - Migrated batch decay processing into `chisurf/plugins/jordi_g_factor/__init__.py` (new `Batch...` action): now opens as a modal batch window and applies full corrected anisotropy processing (`g`, `l1`, `l2`, shift, optional background correction) while exporting Jordi-anisotropy-compatible batch outputs (`filename`, `r_inf`, `region_min`, `region_max`, `bg_vv`, `bg_vh`, `g_factor`).
  - Simplified FP override UX in `chisurf/plugins/jordi_g_factor/__init__.py` / `chisurf/plugins/jordi_g_factor/wizard.ui`: removed manual tau/rS checkboxes; entering values in `tau` or target `rS` now directly enables manual override and supersedes computed values.

### Deprecated

- **Deprecated embedded browser plugin and moved notebook opening to system browser**.
  - Marked `chisurf/plugins/misc/browser/__init__.py` (`Tools:Miscellaneous:Browser`) as hidden/deprecated (`menu_hidden = True`, `deprecated = True`) so it no longer appears in normal plugin menus.
  - Updated notebook menu launch in `chisurf/gui/__init__.py` (`populate_notebooks` / `add_notebook`) to open notebook URLs via `webbrowser.open_new_tab(...)` instead of routing through the embedded browser plugin.

- **Deprecated plugin**: `chisurf/plugins/jordi_anisotropy/__init__.py` (`Jordi Anisotropy Decay`) is now marked obsolete.
  - Added plugin metadata flags (`menu_hidden = True`, `deprecated = True`) so it no longer appears in normal plugin menus.
  - Added explicit runtime deprecation warning and in-window notice to direct users to the Jordi G-Factor + reader anisotropy workflow.
  - Improved `r(t)` computation in `chisurf/plugins/jordi_g_factor/__init__.py` to use separate `l1` and `l2` terms in the anisotropy denominator; corrected traces now use active corrected parameters (`g`, `l1`, `l2`) while raw traces remain uncorrected (`l1=l2=0`).
  - Extended detector wizard integration in `chisurf/gui/widgets/wizard/tttr_channeldefinition/tttr_channel_definition_tttr_io.py` (`on_calc_g_factor`) to prefill FP `dt` from effective microtime resolution and propagate plugin-provided `l1/l2` estimates to detector setup fields alongside `g_factor`.
  - Added focused calibration math coverage in `test/test_jordi_g_factor_fp_calibration.py`.

- **Completed Phase 11 - Advanced Undo/Redo Features**: Enhanced the undo/redo system to support complex model operations and improved history navigation UX.
  - Added model actions (`model_add_component`, `model_remove_component`, `model_normalize_amplitudes`, `model_absolute_amplitudes`, `model_change_irf`, `model_unload_irf`, `model_update`, `model_set_correction`, `model_set_linearization`, `model_unload_lintable`, `model_unload_background_curve`, `model_remove_local_fit`, `model_clear_local_fits`, `model_append_global_parameter`, `model_append_fit`) to state-bearing actions for undo/redo navigation in `chisurf/gui/widgets/history_browser.py` (`_is_state_action`).
  - Extended domain snapshot capture in `chisurf/history_replay.py` (`capture_domain_snapshot`) to include comprehensive model state (components, configuration, IRF settings, corrections, etc.).
  - Implemented model state reconstruction in `chisurf/history_replay.py` (`reconstruct_model_state`) to rebuild model configuration and component state from history events.
  - Added model state application in `chisurf/gui/main.py` (`_apply_model_state`) to apply reconstructed model state during history navigation, including component add/remove operations, IRF changes, correction settings, and other model configurations.
  - Updated snapshot-to-replay conversion in `chisurf/history_replay.py` (`snapshot_to_replay_state`) to handle model state.
  - Added comprehensive tests in `test/test_history_replay_state.py` (`test_reconstruct_model_state`, `test_reconstruct_model_state_multiple_fits`) for model undo/redo functionality.
  - Cross-reference: `AGENTS_PLAN.md` Phase 11, `handover.md` Phase 11 completion.

### Added

- **NDXplorer now supports drag-and-drop for `.er4` sampling files**: 
  - Dropping `.er4` files onto the preview area now automatically triggers the sampling import dialog.
  - Implemented in `modules/ndxplorer/ndxplorer/utils/working_path_helpers.py`.

- **Added dataset masking support in NDXplorer**:
  - Implemented `MaskDrawingWidget`, `MaskOverlayWidget`, and `DrawingOverlayWidget` in `modules/ndxplorer` for interactive ROI-based masking of 2D datasets.
  - Integrated drawing overlay layer into surface plots for immediate visual mask feedback.

- **Added test fixtures for NDXplorer masking**: 
  - Included sample fluorescence data and unit test fixtures for dataset masking validation in `modules/ndxplorer/ndxplorer/tests/fixtures/`.

- **Dev Mode Source Jumping**: Developer-focused feature for navigating from UI elements to their source code locations.
  - Added `is_dev_mode()` and `dev_mode_settings()` helpers in `chisurf/settings/__init__.py` to gate dev-mode features (mirrors `enable_experimental` setting).
  - Added `chisurf/gui/devtools/source_jump.py` with source resolution functions (`resolve_widget_source`, `resolve_object_source`, `resolve_focused_widget_source`, `resolve_fit_window_source`, `open_in_editor`) for mapping widgets/objects to source file locations.
  - Added `chisurf/gui/widgets/code_badge.py` with `CodeBadgeButton` (tiny floating `</>` overlay) and `install_code_badge()` for adding jump-to-source buttons on widgets.
  - Upgraded `chisurf/plugins/misc/code_editor/text_editor.py` (`CodeEditor`) with `open_file(path, line)` and `goto_line(line)` methods for line-aware source navigation.
  - Modified `chisurf/gui/decorators.py` (`init_with_ui`) to store `_chisurf_ui_path` on widgets for `.ui` file resolution.
  - Added code badges on main dock panels (`Read data`, `Datasets`, `Analysis`, `Plot settings`, `History`, `Macro`) and MDI fit windows when dev mode is enabled.
  - Added Developer menu with "Open Source for Focus" (`Ctrl+Alt+J`), "Refresh Code Badges", and "Dev Mode Settings..." actions.
  - Added `gui.dev_mode` settings section in `chisurf/settings/settings_chisurf.yaml` for badge visibility and location configuration.
- **History checkpoint system for faster undo/redo navigation**: large history streams now use periodic state snapshots to accelerate cursor jumps, avoiding full event replay from the start.
  - Added checkpoint infrastructure in `chisurf/history.py` (`OperationHistory.__init__`, `set_checkpoint_capture`, `create_checkpoint`, `get_checkpoint_before`, `get_events_from_checkpoint`, `clear_checkpoints`, `checkpoint_count`, `_maybe_create_checkpoint`).
  - Added domain state capture function in `chisurf/history_replay.py` (`capture_domain_snapshot`) to serialize current datasets, fits, parameters, links, fit ranges, and setup state.
  - Added snapshot-to-replay-state converter in `chisurf/history_replay.py` (`snapshot_to_replay_state`) for applying checkpoint state during cursor navigation.
  - Updated history cursor handler in `chisurf/gui/main.py` (`_on_history_cursor_changed`) to use checkpoints when available, falling back to full event replay if no checkpoint exists.
  - Wired checkpoint capture into history browser setup in `chisurf/gui/main.py` (history browser placeholder replacement).
  - Default checkpoint interval is 50 events; checkpoints are automatically invalidated when history is loaded/replaced.

### Fixed

- **macOS app bundle (`ChiSurf.app`) did not work — three bugs fixed in `build_tools/osx/build-osx-app.sh`**:
  1. **`PYTHONPATH: unbound variable` crash at build time**: heredoc delimiter was unquoted (`<< LAUNCHER`), so bash expanded `$PYTHONPATH` at write-time; `-u` flag in `set -euo pipefail` aborted the build. Fixed: `<< 'LAUNCHER'` (quoted delimiter).
  2. **"No such file or directory" for python at launch time**: `SCRIPT_DIR` in the launcher resolved to `ChiSurf.app/Contents` instead of `ChiSurf.app`, causing doubled paths (`Contents/Contents/bin/python`). Fixed: `cd "$(dirname "$0")/../.."` to go two levels up correctly.
  3. **Missing dylibs (`libquadmath`, Qt, HDF5, etc.) caused `numpy` ImportError**: only 4 dylibs were hardcoded for copying; all transitive dependencies were absent. Fixed: `cp -a "$APP_PATH/lib/"*.dylib` copies every dylib from the conda env.
  - Also hardened the launcher: `PYTHONNOUSERSITE=1` prevents user site-packages interference; `PYTHONPATH` is set exclusively (not appended) so the dev tree can't shadow bundle packages; `cd "$HOME"` at startup avoids the CWD being picked up as a package root; `QT_PLUGIN_PATH` points into the bundle.

- **TCSPC convolve widget FWHM display now updates even when synthetic IRF is active**.
  - Added `ConvolveWidget._refresh_fwhm_display` in `chisurf/models/tcspc/widgets/convolve.py` and wired it into initialization/state/IRF-change/unload paths so the FWHM field is populated from the active IRF model even without a loaded external IRF curve.

- **TTTR Audifier now opens as its own top-level window instead of a dock widget**.
  - Updated `chisurf/plugins/tttr/audifier/__init__.py` (`_bootstrap_plugin`) to stop docking into the main window and always show a standalone widget window.

- **Plugin visibility now defaults to hiding broken and CLI-only plugins unless dev/experimental mode is enabled**.
  - Updated plugin filtering in `chisurf/gui/__init__.py` (`populate_plugins`), `chisurf/gui/widgets/ribbon/ribbon_plugins.py` (`_create_plugins_category`), and `chisurf/gui/widgets/ribbon/ribbon_categories.py` (`_add_setup_plugins_to_main`, `_add_help_plugins_to_main`).
  - In normal mode, broken and CLI-only plugins are hidden entirely; in dev/experimental mode they remain visible for diagnostics.

- **Global-fit dataset could be removed accidentally**: dataset removal guard checked only `"Global Dataset"`, so reserved `"Global-fit"` entries could disappear.
  - Added canonical global-dataset guard in `chisurf/macros/core_data.py` (`_is_global_fit_dataset`) and applied it to dataset removal.
  - Added explicit restore operation `restore_global_fit_dataset` in `chisurf/macros/core_data.py` and exposed action `dataset.restore_global_fit` in `chisurf/actions/dataset_actions.py`.

- **MCP fit feedback ergonomics for LLM agents**: fit quality had to be inferred indirectly, making autonomous retry logic brittle.
  - Added per-fit `chi2` in `describe_state` output in `chisurf/mcp/server.py`.
  - Added `get_fit_quality` MCP tool in `chisurf/mcp/server.py` to report fit name/dataset/chi2 directly.
  - Added `run_fit_with_quality` MCP tool in `chisurf/mcp/server.py` to run fits with optional retries and return before/after chi2 plus improvement flags.

- **TCSPC CSV reader `dt` propagation bug in setup widget**: when "scale with rebin" was off, `dt` was incorrectly rewritten as `1.0 * rebin` instead of using the spinbox value.
  - Fixed `CsvTCSPCWidget.onParametersChanged` in `chisurf/gui/widgets/experiments/tcspc/csv_tcspc_widget.py` to preserve the user-entered `dt` when scaling is disabled.

- **IRF actions incomplete in action-controller path**: IRF unload/change behavior was partially broken due missing macro support and action->macro argument mismatch.
  - Added robust IRF resolution + unload implementation in `chisurf/macros/model.py` (`change_irf`, `unload_irf`).
  - Updated `chisurf/actions/model_actions.py` IRF actions to pass resolved fit objects (`fit=...`) instead of invalid `fit_idx` kwargs.

- **GUI IRF change regression for non-first fit windows**: IRF selection from a fit widget could target the wrong fit and the selected IRF was immediately cleared.
  - Updated `ConvolveWidget.change_irf` in `chisurf/models/tcspc/widgets/convolve.py` to resolve and pass the active fit-group index and to stop auto-dispatching `model.unload_irf` right after `model.change_irf`.
  - Updated `ConvolveWidget.onUnloadIRF` to unload IRF for the correct active fit group.
  - Hardened IRF actions in `chisurf/actions/model_actions.py` to default to `cs.current_fit` when `fit_index` is omitted.

- **TCSPC fit creation via MCP/GUI failed silently due missing widget handlers**: `fit.add` emitted history events but created no fits because per-dataset creation raised UI callback `AttributeError`s that were swallowed by the batch loop.
  - Restored missing IRF unload callback in `chisurf/models/tcspc/widgets/convolve.py` (`ConvolveWidget.onUnloadIRF`).
  - Restored missing lifetime UI button callbacks in `chisurf/models/tcspc/widgets/lifetime.py` (`LifetimeWidget.onAddLifetime`, `LifetimeWidget.onRemoveLifetime`, `LifetimeWidget.onAbsoluteAmplitudes`).
  - Added handler contract coverage in `test/test_convolve_widget_contract.py` to guard required callback presence.

- **MCP debug/control usability for live ChiSurf sessions**: MCP connection mode and diagnostics were difficult to debug from OpenCode when targeting a running GUI process.
  - Added MCP transport CLI options and streamable-HTTP runtime support in `chisurf/mcp/server.py` (`_parse_args`, `resolve_transport_kwargs`, `main`) so ChiSurf can expose a stable remote MCP endpoint.
  - Added minimal debug-oriented MCP tools in `chisurf/mcp/server.py` (`ping`, `list_runtime_vars`, `get_runtime_var`, `set_runtime_var`, `show_message`) for quick runtime inspection and on-screen PyQt message feedback.
  - Updated GUI MCP autostart invocation in `chisurf/gui/__init__.py` (`setup_gui` stage `start_mcp`) to launch MCP as streamable HTTP on `127.0.0.1:8765/mcp`.
  - Updated OpenCode MCP config in `opencode.json` to connect as remote MCP (`type: remote`, `url: http://127.0.0.1:8765/mcp`) instead of spawning a separate local MCP process.
  - Added focused coverage for transport kwargs in `test/test_mcp_server.py` (`test_resolve_transport_kwargs_streamable_http`, `test_resolve_transport_kwargs_stdio`).

- **Ribbon auto-fold not working after unpinning**: the ribbon would stay visible even when unpinned because the auto-fold timer was never started when toggling auto-fold on.
  - Fixed `toggle_auto_fold(enabled=True)` in `chisurf/gui/widgets/ribbon/ribbon_auto_fold.py` to restart the timer when enabling auto-fold.
  - Fixed `_setup_pin_button()` to stop the timer if ribbon starts pinned, preventing inconsistent state.
  - Removed redundant timer start from `__init__` since `_setup_auto_fold()` handles it.
  - Added `pinned` setting to `chisurf/settings/settings_chisurf.yaml` for visibility.
- **DataGroup classes now properly initialize unique_identifier**: `ExperimentDataCurveGroup` and `ExperimentDataGroup` were not passing `*args` and `**kwargs` to the `Base` class during initialization, causing `unique_identifier` to be missing from `meta_data` and raising `KeyError` when grouping datasets.
  - Fixed in `chisurf/data.py` by changing inheritance order to `class DataGroup(chisurf.base.Base, list)` so `super().__init__()` properly calls both base initializers.
- **DataGroup classes now have a filename property**: grouped datasets were missing the `filename` attribute that the dataset selector expects, causing `AttributeError` when updating the UI after grouping.
  - Added `filename` property to `DataGroup` in `chisurf/data.py` that returns the filename of the first member.
  - Fixed in `chisurf/data.py` (`DataGroup.__init__`) to forward `*args` and `**kwargs` to the base class.
- **Chato direct FCS command now supports folder-based global fits with linked diffusion/shape parameters**: commands targeting a directory (e.g. `...\\test\\data\\fcs\\kristine`) previously failed with `File not found` because only single-file paths were supported.
  - Updated `chisurf/plugins/_dev/chato/frontend/dock.py` (`_extract_first_path_like`, `_resolve_fcs_curve_paths`, `_linkable_global_fcs_parameter_names`, `_handle_direct_fcs_fit_command`) to load multiple FCS curves from a folder, group datasets, create one global fit, link `td*` and `s/shape` parameters across the fit group, and run optimization.
  - Added focused regression coverage in `test/test_chato_direct_fcs_paths.py`.

- Prevented silent history-event loss for unregistered action types by adding dispatcher-error fallback in `chisurf/runtime/actions.py` (`record_action`): if routed execution fails (e.g., unknown action name), the event is now still recorded via direct history fallback instead of being dropped.
- **Plugin widget UI loading now fails fast with clear path errors after plugin moves/restructure**: missing or stale `.ui` paths previously surfaced as less actionable Qt load failures.
  - Added cached UI path resolution and explicit existence checks in `chisurf/gui/decorators.py` (`_resolve_ui_path_cached`, `resolve_ui_path`, `init_with_ui.load_ui`).
- **Grouped dataset fits now default-link non-nuisance parameters across local fits**: grouped fits previously started largely unlinked, requiring repetitive manual linking for lifetime/rotational/distance parameters.
  - Added automatic non-nuisance linking in `chisurf/macros/core_fit.py` (`_collect_group_nuisance_parameter_names`, `_auto_link_non_nuisance_group_parameters`, `add_fit`), excluding nuisance-style groups (`generic`, `corrections`, `convolve`, and `*nuisance*`).
  - Added a structured history event `fit_group_auto_link` recording how many master/follower links were created.
- **Adding multiple fits from GUI is faster for multi-selection workflows**: creating many fits in one action previously repeated top-level UI repaint/unfreeze cycles and full `cs.update()` calls for each fit.
  - Updated `chisurf/macros/core_fit.py` (`add_fit`) to batch multi-dataset fit creation with deferred UI refresh (`_defer_cs_update`) and shared UI-freeze scope (`_ui_updates_frozen`).
- **Data table no longer stalls UI during frequent fit refreshes**: table updates used expensive per-refresh geometry autosizing and copied curve data on every update.
  - Optimized `chisurf/plots/table_plot.py` (`_FitTableModel.set_arrays`, `FitTablePlot.__init__`, `FitTablePlot._get_arrays`, `FitTablePlot._refresh_arrays_into_model`, `FitTablePlot.showEvent`) to avoid `ResizeToContents`, avoid unnecessary model resets, avoid deep curve copies, and defer updates while hidden.
- **Grouped-parameter unlink now clears stale underline styling on follower widgets**: after removing fit-group links, dependent parameter rows could remain visually underlined until a later UI refresh, even though links were already removed.
  - Added explicit post-link/unlink visual refresh in `chisurf/gui/widgets/fitting/parameter_widgets.py` (`FittingParameterWidget._refresh_group_link_visuals`, `FittingParameterWidget.onLinkFitGroup`) so all same-named parameter controllers in the active fit group are finalized immediately.
  - Added regression contract coverage in `test/test_parameter_widget_link_visuals.py` (`test_group_unlink_refreshes_visual_state_for_related_widgets_contract`).
- **Project load no longer passes non-finite anisotropy calibration values into lifetime model construction**: loading saved projects with `NaN`/`inf` reader calibration (`g_factor/l1/l2`) could crash on Windows during anisotropy parameter initialization.
  - Hardened calibration parsing/application in `chisurf/macros/core_fit.py` (`_coerce_finite_float`, `_resolve_dataset_anisotropy_calibration`, `_apply_anisotropy_calibration_to_fit`, `add_fit`) to ignore non-finite values before model creation.
  - Removed a stray `fit_close` history emit from the end of `load_project` in `chisurf/macros/core_fit.py` that referenced undefined close-context variables.
  - Added regression coverage in `test/test_core_fit_anisotropy_calibration.py` (`test_resolve_dataset_anisotropy_calibration_ignores_non_finite_values`, `test_apply_anisotropy_calibration_ignores_non_finite_values`).
- **Project save/load now preserves grouped datasets and grouped-fit membership**: grouped datasets were reloaded as flat datasets and local fits were reattached to separate fit groups.
  - Added dataset tree snapshot persistence (`ui_state.dataset_layout`) in `chisurf/macros/core_fit.py` (`save_project`) and restored grouped dataset objects during `load_project`.
  - Updated fit reconstruction in `chisurf/macros/core_fit.py` (`load_project`) to map grouped local-fit dataset IDs back to a single grouped dataset index (deduplicated), preserving original grouped fit composition.
- **Grouped fit filenames now keep per-fit identity labels (VV/VH, etc.)**: indexed-only names made grouped exports hard to map back to channels/conditions.
  - Updated grouped save naming in `chisurf/fitting/fit.py` (`FitGroup.save`) to derive a per-fit suffix from each local dataset name (common prefix removed), with safe fallback to index when needed.
- **Grouped fit export no longer overwrites to a single output set**: `FitGroup.save(...)` reused the same filename for every local fit, so only one fit's files survived on disk.
  - Fixed grouped filename generation in `chisurf/fitting/fit.py` (`FitGroup.save`) to write each local fit to a unique indexed base (`_00`, `_01`, ...), preserving all grouped outputs.
- **Fit-group parameter linking now uses a stable master fit**: middle-checkbox linking from a non-first local fit could make that selected fit the master, leading to inconsistent master/follower orientation across sessions.
  - Updated fit-group linking in `chisurf/macros/core_fit.py` (`link_fit_group`) so group links always target the parameter of the first local fit; non-master flags are reset before relinking.
  - Added regression coverage in `test/test_fit_group_link_master.py` (`test_link_fit_group_always_uses_first_fit_as_master`).
- **Line-plot metrics overlay no longer uses rich-text HTML updates during fit refresh**: `pyqtgraph.TextItem.setHtml(...)` remained a crash hot path in some Windows save/switch workflows.
  - Switched line-plot metrics rendering to plain text and added a save-time overlay suspension guard in `chisurf/plots/lineplot/lineplot.py` (`LinePlot._build_metrics_overlay_text`, `LinePlot.update`) and `chisurf/macros/core_fit.py` (`save_fit`).
- **`Ctrl+S` now reliably triggers "Save Current Fit"**: save-fit keyboard shortcut behavior was inconsistent depending on focus/widget context.
  - Bound `Ctrl+S` as an application shortcut for `actionSaveCurrentFit` in `chisurf/gui/main.py` (`Main.define_actions`).
- **Fit-export filenames now drop source extensions from dataset names**: exports based on names like `...dat VV` previously produced awkward filenames such as `...dat VV_info.txt` and `...dat VV_weighted residuals.csv`.
  - Updated save-name stem handling in `chisurf/macros/core_fit.py` (`save_fit`, `save_fits`) to remove trailing source extensions before writing CSV/TXT/DOCX/fit.json outputs.
- **Windows save could fail with `FileNotFoundError` for deep fit-export paths**: long VV/VH export filenames (e.g. `..._weighted residuals.csv`) could exceed classic Windows path limits and fail mid-save even when the folder exists.
  - Added long-path normalization in `chisurf/fio/ascii.py` (`_windows_extended_path`, `Csv.save`) to automatically use Windows extended-length path prefixes for deep output paths.
- **Save-fit crash in grouped VV/VH lifetime workflows**: saving grouped fits could trigger an access-violation in `pyqtgraph.TextItem.setHtml(...)` while fit selection changes refreshed line plots.
  - Hardened metrics-overlay updates in `chisurf/plots/lineplot/lineplot.py` (`LinePlot._metrics_text_alive`, `LinePlot.update`) to verify Qt object lifetime (`sip.isdeleted`) before calling `updateTextPos()`/`setHtml()`.
- **TCSPC CSV `Reference` picker now uses the central last-used-folder flow**: `pushButton_inspect` now opens via the shared `get_filename(...)` helper and honors `chisurf.working_path` consistently.
  - Updated file selection in `chisurf/gui/widgets/experiments/tcspc/csv_tcspc_widget.py` (`CsvTCSPCWidget.openJordiGFactorPlugin`) and hardened cancel handling in `chisurf/gui/widgets/general.py` (`get_filename`) so cancel no longer resets `chisurf.working_path`.
- **Anisotropy fit setup now respects TCSPC reader `g_factor/l1/l2` values**: fit creation previously applied only `g_factor`, so `l1/l2` from CSV reader settings were ignored when initializing anisotropy model parameters.
  - Updated fit-calibration resolution/application in `chisurf/macros/core_fit.py` (`_resolve_dataset_anisotropy_calibration`, `_apply_anisotropy_calibration_to_fit`, `add_fit`) to propagate `g_factor`, `l1`, and `l2` from reader/metadata into model kwargs and anisotropy parameters.
- **TCSPC CSV reader UI now exposes `l1/l2` calibration controls near `rep.rate` and `g-factor`**: users can edit anisotropy leakage factors directly in the reader panel and optionally link `l1`/`l2` with a dedicated checkbox (default off).
  - Updated UI and parameter propagation in `chisurf/gui/widgets/experiments/tcspc/tcspc_csv.ui` (`doubleSpinBox_l1`, `doubleSpinBox_l2`, `checkBox_link_l1_l2`, `actionL1L2Changed`) and `chisurf/gui/widgets/experiments/tcspc/csv_tcspc_widget.py` (`CsvTCSPCWidget.updateUI`, `CsvTCSPCWidget.onParametersChanged`, `_sync_l2_from_l1`, `_on_link_l1_l2_toggled`).
- **TCSPC reader anisotropy calibration now carries `g_factor`, `l1`, and `l2` consistently**: reader calibration was effectively g-only and left `l1/l2` scattered across downstream components.
  - Updated calibration defaults/annotation in `chisurf/experiments/tcspc/reader.py` (`TCSPCReader._default_anisotropy_calibration`, `TCSPCReader._annotate_anisotropy_calibration`, `TCSPCReader.read`) and extended CSV reader calibration propagation in `chisurf/fio/fluorescence/tcspc.py` (`read_tcspc_csv`, `_annotate_anisotropy_meta`).
- **Grouped line-plot autocorrelation traces were hard to compare in "display group" mode**: autocorrelation curves from multiple local fits overlapped on the same baseline.
  - Added automatic per-fit vertical offsets for grouped autocorrelation traces in `chisurf/plots/lineplot/lineplot.py` (`LinePlot._plot_group_curves`, `LinePlot._compute_group_curve_offsets`) so grouped `a.corr.` curves are stacked for readability while keeping the active fit centered.
- **Grouped line-plot weighted residuals were hard to compare in "display group" mode**: weighted residual traces from multiple local fits overlapped on the same baseline, reducing readability.
  - Added automatic per-fit vertical offsets for grouped weighted residuals in `chisurf/plots/lineplot/lineplot.py` (`LinePlot._plot_group_curves`, `LinePlot._compute_group_curve_offsets`) so traces are stacked like pyqtgraph multi-curve displays while keeping the active fit centered.
- **Plugin check dangerous-call detection no longer flags `ast.literal_eval`**: replaced brittle substring scanning for `eval`/`exec` with AST-based call analysis so safe calls like `ast.literal_eval(...)` pass while true dangerous calls are still blocked.
  - Updated plugin safety checks in `chisurf/macros/plugin_check.py` (`_contains_dangerous_operations`, `_resolve_call_name`, `PluginTestRunner._test_plugin`, `PluginTestRunner._test_plugin_chisurf_way`)
- **FCS CPM now uses total detector count rate by default**: corrected CPM display to use total count rate per molecule instead of per-detector mean when total-rate metadata is available.
  - Updated total-rate resolution in `chisurf/models/fcs/fcs.py` (`ParseFCSWidget._resolve_total_mean_count_rate`, `ParseFCSWidget.update_model`) and applied it to CPM/CPM_all and signal countrate `S`
  - Updated TTTR correlator `.cor` export in `chisurf/gui/widgets/wizard/tttr_correlator/tttr_correlator.py` (`WizardTTTRCorrelator.save_correlations`) to write total detector count rate `(ca + cb) / duration / 1000`
  - Updated Zeiss Confocor3 import in `chisurf/fio/fluorescence/fcs/confocor3.py` (`read_zeiss_fcs`) to store explicit `mean_count_rate_total` metadata for two-channel curves
- **TTTR microtime histogram plugin launch crash**: fixed missing UI file resolution that raised `FileNotFoundError` when loading the plugin wizard.
  - Corrected UI path in `chisurf/plugins/tttr/microtime_histogram/wizard.py` (`MicrotimeHistogram.__init__` decorator)
- **Dataset removal now confirms and closes dependent fits**: removing datasets/groups now displays a single confirmation dialog, logs how many dependent fits would be closed, and automatically closes those fits before removing the data.
  - Added dataset expansion/fit dependency helpers in `chisurf/macros/core_data.py` and fire the confirmation dialog only once.
- **TCSPC fits honor reader g_factor**: anisotropy models now derive the initial `g` value from the TCSPC reader metadata so the g parameter matches the detector calibration used during data import.
  - Added dataset-level g_factor resolution/parsing in `chisurf/macros/core_fit.py` (`_resolve_dataset_g_factor`, `_apply_g_factor_to_fit`, `add_fit`).
  - Hardened g-factor assignment in `chisurf/macros/core_fit.py` so anisotropy objects with scalar `_g` state no longer crash fit creation (`'float' object has no attribute 'value'`).
- **TCSPC anisotropy widget `r_ss` showed `NaN` persistently**: fixed missing model/fit wiring so the `r_ss` output parameter can resolve lifetime-model values instead of falling back to invalid context.
  - Passed `fit`/`model` when constructing `AnisotropyWidget` in `chisurf/models/tcspc/widgets/lifetime.py` and `chisurf/models/tcspc/widgets/gaussian.py`
  - Added context inference fallback in `chisurf/models/tcspc/widgets/anisotropy.py` (`AnisotropyWidget.__init__`)
- **TCSPC anisotropy widget `rS,I` mismatch**: corrected intensity-domain steady-state anisotropy computation to match ChiSurf anisotropy channel conventions and objective-mixing model.
  - `rS,I` now unmixed VV/VH channel integrals using `l1/l2` and computes anisotropy with ChiSurf convention `r = (VV - VH) / (g*VV + 2*VH)` in `chisurf/models/tcspc/widgets/anisotropy.py` (`_compute_steady_state_anisotropy_intensity`).
  - Fixed grouped dataset handling: for `VV/VH` fits the widget now uses both curves from `ExperimentDataCurveGroup` (instead of only `current_dataset`) before unmixing/integration.
  - Fixed g-factor insensitivity in local `VV`/`VH` views by always using dual-channel computation whenever both channels are available from stacked data or fit-group pairing.
  - Corrected channel extraction for stacked/local VH paths to avoid mixing up 1D/2D channel arrays; this removes large `rS,I` bias (often ~2x) caused by using wrong channel slices in dual-group contexts.

- **Fitting progress dialog wraps long fit names**: prevents the fit-run progress window from becoming extremely wide when a fit has a very long name.
  - Updated `FittingControllerWidget.onRunFit` in `chisurf/gui/widgets/fitting/fit_controller.py` to wrap/break the fit name for the progress label.
  - Updated `EnhancedProgressDialog` in `chisurf/gui/widgets/progress.py` to use a word-wrapping label; added `wrap_text` helper.

- **Line plot defaults to group display for multi-fit FitGroups**: when plotting a fit group with multiple local fits, the "display group" checkbox is auto-enabled so all fits are rendered by default.
  - Updated `LinePlot._auto_enable_display_group_if_grouped` and call sites in `chisurf/plots/lineplot/lineplot.py`.

- **Line plot metrics overlay is more compact for multi-fit groups**: reduces on-plot clutter by using a smaller font and rendering chi2r/DW values as a lightweight table with the active fit highlighted.
  - Updated metrics overlay rendering in `chisurf/plots/lineplot/lineplot.py` (`LinePlot._build_metrics_overlay_html`).

- **Output parameters are visually distinct in the fit parameter editor**: output/result parameters now render with a subtle text highlight (bold + color) by default.
  - Updated `FittingParameterWidget._update_role_visuals` in `chisurf/gui/widgets/fitting/parameter_widgets.py`.

- **TCSPC generic widget output labels renamed for compactness**: `nPh(Fl)`/`nPh(Bg)` are now shown as `#PhF`/`#PhB`.
  - Updated output parameter labels in `chisurf/models/tcspc/widgets/generic.py`.
- **Add/remove action buttons now have semantic colors in TCSPC editors**: add buttons are styled green and remove buttons red for quicker visual affordance.
  - Updated button styling in `chisurf/models/tcspc/widgets/anisotropy.py` (`AnisotropyWidget.__init__`), `chisurf/models/tcspc/widgets/gaussian.py` (`GaussianWidget.__init__`), and `chisurf/models/tcspc/widgets/lifetime.py` (`LifetimeWidget.__init__`).
- **Anisotropy add/remove controls are explicit normal push buttons**: prevented any toggle/radio-button behavior by forcing non-checkable push-button semantics for the anisotropy component actions.
  - Updated `chisurf/models/tcspc/widgets/anisotropy.py` (`AnisotropyWidget.__init__`).
- **Anisotropy add/del controls moved next to diagnostics actions**: relocated `add`/`del` to the left of `VV/VH diag` and inserted a horizontal spacer between component controls and diagnostics tools for clearer grouping.
  - Updated control-row layout in `chisurf/models/tcspc/widgets/anisotropy.py` (`AnisotropyWidget.__init__`).

- **Simplified ChiSurf version naming scheme**: Changed from date-style `YY.YYYYMMDD` to simplified `YY.XX.YY` for releases and `YY.devZZZ` for dev builds. Updated version computation in `chisurf/info.py` (`_compute_version`), build-time baking in `build_tools/_build_backend.py`, conda fallback in `conda-recipe/meta.yaml`, Windows build bootstrap in `build_tools/win/build-setup.bat`, and updater version parsing/sorting in `chisurf/plugins/chisurf/updater/updater.py`. Updated documentation in `docs/VERSIONING.md`, `docs/RELEASES.md`, and `build_tools/BUILD_INSTRUCTIONS.md`.

- **Resolved persistent SIGBUS (EXC_ARM_DA_ALIGN) startup crash on macOS ARM64 (Apple Silicon)** by sanitizing UI elements (Emojis, special symbols) that triggered incompatible rendering paths in CoreText.
- **Hardened `build-osx-app.sh` by removing `DYLD_LIBRARY_PATH`** to prevent library conflicts with system frameworks and ensured the ribbon correctly renders in-window by changing `RibbonBar` inheritance from `QMenuBar` to `QWidget`.
- **Fixed `ModuleNotFoundError` for `chisurf.plugins._dev` in bundled application** by implementing a global development-plugin proxy system in `chisurf/plugins/__init__.py`.
- **Restored `ChiSurf.icns` icon association and launcher environment fixes** for the macOS application bundle.

### Changed

- **Updated `quest` dependencies for NumPy 2.0 compatibility**:
  - Updated `modules/quest/pyproject.toml` to require `numpy >= 2.0`.
  - Removed obsolete `pymol-open-source` dependency from build environment.

- **ChiSurf distribution now managed exclusively by pixi with rattler-build**:
  - Removed conda-build recipe (`conda-recipe/` directory deleted) — superseded by `rattler-recipe/`
  - Removed `.condaignore` file (conda-build-specific)
  - Removed `micromamba` from runtime dependencies; moved to `[feature.build.dependencies]` in `pixi.toml`
  - All build and distribution tasks now use `pixi run` with the `build` feature
  
- **Implemented dynamic entry points discovery for build recipes and installers**:
  - Created `rattler-recipe/collect_entry_points.py` to dynamically discover all entry points at build time from two sources:
    1. Static entry points from `pyproject.toml` (`[project.scripts]` and `[project.gui-scripts]`)
    2. Dynamic plugin entry points via AST-parsing of `cli_entrypoint` variables in `chisurf/plugins/*/` `__init__.py` files
  - Added `collect-entry-points` pixi task (required before build) that generates `rattler-recipe/entry_points.json`
  - Updated `rattler-recipe/recipe.yaml` to load entry points from generated JSON instead of hardcoding — supports all 26 entry points (18 static + 8 dynamic from plugins)
  - Updated `build_tools/win/create_installer_script.py` to read from `entry_points.json`, enabling plugin CLI entry points to appear in Inno Setup installer menus
  - Updated `build_tools/win/build-setup.bat` to call entry points collection before package builds
  
- **Updated CI workflow for pixi-based distribution and GitHub Releases**:
  - Renamed `.github/workflows/conda-release.yml` → `build-release.yml` (more accurately reflects pixi-based build)
  - Removed Anaconda.org upload step (no longer conda-centric)
  - Added GitHub Releases upload for tagged releases using `softprops/action-gh-release@v1`
  - Artifacts (`.conda` and `.tar.bz2` packages) are now uploaded to GitHub Releases on tag creation
  
- **Windows installer (`setup.exe`) remains on build path**:
  - Inno Setup flow preserved; uses pixi-managed environment instead of raw micromamba
  - `build-setup.bat` still creates distributable environment via `micromamba create` (now a build-only dependency)

### Added

- **Jordi G-factor plugin now supports slow-dye (FP) anisotropy-mixing calibration in estimate mode**: after fast-dye tail matching for `g`, users can load a separate FP Jordi file to estimate linked `l1/l2` from steady-state anisotropy (`rho` default `16 ns`, lifetime estimated from decay first moment).
  - Added FP controls/outputs and estimate warnings in `chisurf/plugins/jordi_g_factor/__init__.py` (`load_fp_jordi_file`, `calculate_fp_mixing_estimate`, linked `l1=l2` solve, `dt` input, editable `tau`/target `rS` with manual override toggles, single `l1/l2` estimate output).
  - Moved Jordi g-factor plugin controls into a Designer-editable UI file `chisurf/plugins/jordi_g_factor/wizard.ui` and simplified linked output display to a single `l1/l2 est` field.
  - Updated Jordi plotting in `chisurf/plugins/jordi_g_factor/__init__.py`: tail-matched decays (`VV` and `VH*G`) are now overlaid in the main decay plot, and the secondary plot now shows time-resolved anisotropy `r(t)` with uncorrected/corrected traces.
  - Added direct manual override inputs in `chisurf/plugins/jordi_g_factor/__init__.py` / `chisurf/plugins/jordi_g_factor/wizard.ui`: editable `g` value (accepted on user entry) and editable linked `l1/l2` value (accepted directly on user entry).
  - Clamped anisotropy display range to `[-0.5, 1.5]` in the `r(t)` plot for both fast and slow dye traces.
  - Switched Jordi plugin entry points to dynamic loading (`importlib`) in `chisurf/gui/widgets/experiments/tcspc/csv_tcspc_widget.py` and `chisurf/gui/widgets/wizard/tttr_channeldefinition/tttr_channel_definition_tttr_io.py`.
  - Updated the TCSPC CSV reference button flow in `chisurf/gui/widgets/experiments/tcspc/csv_tcspc_widget.py` to propagate plugin-derived `l1/l2` only when FP/slow-dye calibration was actually processed (`fp_estimate_available`), in addition to `g` and `vh_shift`.
  - Updated TCSPC CSV reference launch to initialize Jordi plugin `dt [ns/ch]` from current reader settings (`dt` + optional rebin scaling) in `chisurf/gui/widgets/experiments/tcspc/csv_tcspc_widget.py`.
  - Migrated batch decay processing into `chisurf/plugins/jordi_g_factor/__init__.py` (new `Batch...` action): now opens as a modal batch window and applies full corrected anisotropy processing (`g`, `l1`, `l2`, shift, optional background correction) while exporting Jordi-anisotropy-compatible batch outputs (`filename`, `r_inf`, `region_min`, `region_max`, `bg_vv`, `bg_vh`, `g_factor`).
  - Simplified FP override UX in `chisurf/plugins/jordi_g_factor/__init__.py` / `chisurf/plugins/jordi_g_factor/wizard.ui`: removed manual tau/rS checkboxes; entering values in `tau` or target `rS` now directly enables manual override and supersedes computed values.

### Deprecated

- **Deprecated embedded browser plugin and moved notebook opening to system browser**.
  - Marked `chisurf/plugins/misc/browser/__init__.py` (`Tools:Miscellaneous:Browser`) as hidden/deprecated (`menu_hidden = True`, `deprecated = True`) so it no longer appears in normal plugin menus.
  - Updated notebook menu launch in `chisurf/gui/__init__.py` (`populate_notebooks` / `add_notebook`) to open notebook URLs via `webbrowser.open_new_tab(...)` instead of routing through the embedded browser plugin.

- **Deprecated plugin**: `chisurf/plugins/jordi_anisotropy/__init__.py` (`Jordi Anisotropy Decay`) is now marked obsolete.
  - Added plugin metadata flags (`menu_hidden = True`, `deprecated = True`) so it no longer appears in normal plugin menus.
  - Added explicit runtime deprecation warning and in-window notice to direct users to the Jordi G-Factor + reader anisotropy workflow.
  - Improved `r(t)` computation in `chisurf/plugins/jordi_g_factor/__init__.py` to use separate `l1` and `l2` terms in the anisotropy denominator; corrected traces now use active corrected parameters (`g`, `l1`, `l2`) while raw traces remain uncorrected (`l1=l2=0`).
  - Extended detector wizard integration in `chisurf/gui/widgets/wizard/tttr_channeldefinition/tttr_channel_definition_tttr_io.py` (`on_calc_g_factor`) to prefill FP `dt` from effective microtime resolution and propagate plugin-provided `l1/l2` estimates to detector setup fields alongside `g_factor`.
  - Added focused calibration math coverage in `test/test_jordi_g_factor_fp_calibration.py`.

- **Completed Phase 11 - Advanced Undo/Redo Features**: Enhanced the undo/redo system to support complex model operations and improved history navigation UX.
  - Added model actions (`model_add_component`, `model_remove_component`, `model_normalize_amplitudes`, `model_absolute_amplitudes`, `model_change_irf`, `model_unload_irf`, `model_update`, `model_set_correction`, `model_set_linearization`, `model_unload_lintable`, `model_unload_background_curve`, `model_remove_local_fit`, `model_clear_local_fits`, `model_append_global_parameter`, `model_append_fit`) to state-bearing actions for undo/redo navigation in `chisurf/gui/widgets/history_browser.py` (`_is_state_action`).
  - Extended domain snapshot capture in `chisurf/history_replay.py` (`capture_domain_snapshot`) to include comprehensive model state (components, configuration, IRF settings, corrections, etc.).
  - Implemented model state reconstruction in `chisurf/history_replay.py` (`reconstruct_model_state`) to rebuild model configuration and component state from history events.
  - Added model state application in `chisurf/gui/main.py` (`_apply_model_state`) to apply reconstructed model state during history navigation, including component add/remove operations, IRF changes, correction settings, and other model configurations.
  - Updated snapshot-to-replay conversion in `chisurf/history_replay.py` (`snapshot_to_replay_state`) to handle model state.
  - Added comprehensive tests in `test/test_history_replay_state.py` (`test_reconstruct_model_state`, `test_reconstruct_model_state_multiple_fits`) for model undo/redo functionality.
  - Cross-reference: `AGENTS_PLAN.md` Phase 11, `handover.md` Phase 11 completion.

### Added

- **NDXplorer now supports drag-and-drop for `.er4` sampling files**: 
  - Dropping `.er4` files onto the preview area now automatically triggers the sampling import dialog.
  - Implemented in `modules/ndxplorer/ndxplorer/utils/working_path_helpers.py`.

- **Added dataset masking support in NDXplorer**:
  - Implemented `MaskDrawingWidget`, `MaskOverlayWidget`, and `DrawingOverlayWidget` in `modules/ndxplorer` for interactive ROI-based masking of 2D datasets.
  - Integrated drawing overlay layer into surface plots for immediate visual mask feedback.

- **Added test fixtures for NDXplorer masking**: 
  - Included sample fluorescence data and unit test fixtures for dataset masking validation in `modules/ndxplorer/ndxplorer/tests/fixtures/`.

- **Dev Mode Source Jumping**: Developer-focused feature for navigating from UI elements to their source code locations.
  - Added `is_dev_mode()` and `dev_mode_settings()` helpers in `chisurf/settings/__init__.py` to gate dev-mode features (mirrors `enable_experimental` setting).
  - Added `chisurf/gui/devtools/source_jump.py` with source resolution functions (`resolve_widget_source`, `resolve_object_source`, `resolve_focused_widget_source`, `resolve_fit_window_source`, `open_in_editor`) for mapping widgets/objects to source file locations.
  - Added `chisurf/gui/widgets/code_badge.py` with `CodeBadgeButton` (tiny floating `</>` overlay) and `install_code_badge()` for adding jump-to-source buttons on widgets.
  - Upgraded `chisurf/plugins/misc/code_editor/text_editor.py` (`CodeEditor`) with `open_file(path, line)` and `goto_line(line)` methods for line-aware source navigation.
  - Modified `chisurf/gui/decorators.py` (`init_with_ui`) to store `_chisurf_ui_path` on widgets for `.ui` file resolution.
  - Added code badges on main dock panels (`Read data`, `Datasets`, `Analysis`, `Plot settings`, `History`, `Macro`) and MDI fit windows when dev mode is enabled.
  - Added Developer menu with "Open Source for Focus" (`Ctrl+Alt+J`), "Refresh Code Badges", and "Dev Mode Settings..." actions.
  - Added `gui.dev_mode` settings section in `chisurf/settings/settings_chisurf.yaml` for badge visibility and location configuration.
- **History checkpoint system for faster undo/redo navigation**: large history streams now use periodic state snapshots to accelerate cursor jumps, avoiding full event replay from the start.
  - Added checkpoint infrastructure in `chisurf/history.py` (`OperationHistory.__init__`, `set_checkpoint_capture`, `create_checkpoint`, `get_checkpoint_before`, `get_events_from_checkpoint`, `clear_checkpoints`, `checkpoint_count`, `_maybe_create_checkpoint`).
  - Added domain state capture function in `chisurf/history_replay.py` (`capture_domain_snapshot`) to serialize current datasets, fits, parameters, links, fit ranges, and setup state.
  - Added snapshot-to-replay-state converter in `chisurf/history_replay.py` (`snapshot_to_replay_state`) for applying checkpoint state during cursor navigation.
  - Updated history cursor handler in `chisurf/gui/main.py` (`_on_history_cursor_changed`) to use checkpoints when available, falling back to full event replay if no checkpoint exists.
  - Wired checkpoint capture into history browser setup in `chisurf/gui/main.py` (history browser placeholder replacement).
  - Default checkpoint interval is 50 events; checkpoints are automatically invalidated when history is loaded/replaced.

### Fixed

- **macOS app bundle (`ChiSurf.app`) did not work — three bugs fixed in `build_tools/osx/build-osx-app.sh`**:
  1. **`PYTHONPATH: unbound variable` crash at build time**: heredoc delimiter was unquoted (`<< LAUNCHER`), so bash expanded `$PYTHONPATH` at write-time; `-u` flag in `set -euo pipefail` aborted the build. Fixed: `<< 'LAUNCHER'` (quoted delimiter).
  2. **"No such file or directory" for python at launch time**: `SCRIPT_DIR` in the launcher resolved to `ChiSurf.app/Contents` instead of `ChiSurf.app`, causing doubled paths (`Contents/Contents/bin/python`). Fixed: `cd "$(dirname "$0")/../.."` to go two levels up correctly.
  3. **Missing dylibs (`libquadmath`, Qt, HDF5, etc.) caused `numpy` ImportError**: only 4 dylibs were hardcoded for copying; all transitive dependencies were absent. Fixed: `cp -a "$APP_PATH/lib/"*.dylib` copies every dylib from the conda env.
  - Also hardened the launcher: `PYTHONNOUSERSITE=1` prevents user site-packages interference; `PYTHONPATH` is set exclusively (not appended) so the dev tree can't shadow bundle packages; `cd "$HOME"` at startup avoids the CWD being picked up as a package root; `QT_PLUGIN_PATH` points into the bundle.

- **TCSPC convolve widget FWHM display now updates even when synthetic IRF is active**.
  - Added `ConvolveWidget._refresh_fwhm_display` in `chisurf/models/tcspc/widgets/convolve.py` and wired it into initialization/state/IRF-change/unload paths so the FWHM field is populated from the active IRF model even without a loaded external IRF curve.

- **TTTR Audifier now opens as its own top-level window instead of a dock widget**.
  - Updated `chisurf/plugins/tttr/audifier/__init__.py` (`_bootstrap_plugin`) to stop docking into the main window and always show a standalone widget window.

- **Plugin visibility now defaults to hiding broken and CLI-only plugins unless dev/experimental mode is enabled**.
  - Updated plugin filtering in `chisurf/gui/__init__.py` (`populate_plugins`), `chisurf/gui/widgets/ribbon/ribbon_plugins.py` (`_create_plugins_category`), and `chisurf/gui/widgets/ribbon/ribbon_categories.py` (`_add_setup_plugins_to_main`, `_add_help_plugins_to_main`).
  - In normal mode, broken and CLI-only plugins are hidden entirely; in dev/experimental mode they remain visible for diagnostics.

- **Global-fit dataset could be removed accidentally**: dataset removal guard checked only `"Global Dataset"`, so reserved `"Global-fit"` entries could disappear.
  - Added canonical global-dataset guard in `chisurf/macros/core_data.py` (`_is_global_fit_dataset`) and applied it to dataset removal.
  - Added explicit restore operation `restore_global_fit_dataset` in `chisurf/macros/core_data.py` and exposed action `dataset.restore_global_fit` in `chisurf/actions/dataset_actions.py`.

- **MCP fit feedback ergonomics for LLM agents**: fit quality had to be inferred indirectly, making autonomous retry logic brittle.
  - Added per-fit `chi2` in `describe_state` output in `chisurf/mcp/server.py`.
  - Added `get_fit_quality` MCP tool in `chisurf/mcp/server.py` to report fit name/dataset/chi2 directly.
  - Added `run_fit_with_quality` MCP tool in `chisurf/mcp/server.py` to run fits with optional retries and return before/after chi2 plus improvement flags.

- **TCSPC CSV reader `dt` propagation bug in setup widget**: when "scale with rebin" was off, `dt` was incorrectly rewritten as `1.0 * rebin` instead of using the spinbox value.
  - Fixed `CsvTCSPCWidget.onParametersChanged` in `chisurf/gui/widgets/experiments/tcspc/csv_tcspc_widget.py` to preserve the user-entered `dt` when scaling is disabled.

- **IRF actions incomplete in action-controller path**: IRF unload/change behavior was partially broken due missing macro support and action->macro argument mismatch.
  - Added robust IRF resolution + unload implementation in `chisurf/macros/model.py` (`change_irf`, `unload_irf`).
  - Updated `chisurf/actions/model_actions.py` IRF actions to pass resolved fit objects (`fit=...`) instead of invalid `fit_idx` kwargs.

- **GUI IRF change regression for non-first fit windows**: IRF selection from a fit widget could target the wrong fit and the selected IRF was immediately cleared.
  - Updated `ConvolveWidget.change_irf` in `chisurf/models/tcspc/widgets/convolve.py` to resolve and pass the active fit-group index and to stop auto-dispatching `model.unload_irf` right after `model.change_irf`.
  - Updated `ConvolveWidget.onUnloadIRF` to unload IRF for the correct active fit group.
  - Hardened IRF actions in `chisurf/actions/model_actions.py` to default to `cs.current_fit` when `fit_index` is omitted.

- **TCSPC fit creation via MCP/GUI failed silently due missing widget handlers**: `fit.add` emitted history events but created no fits because per-dataset creation raised UI callback `AttributeError`s that were swallowed by the batch loop.
  - Restored missing IRF unload callback in `chisurf/models/tcspc/widgets/convolve.py` (`ConvolveWidget.onUnloadIRF`).
  - Restored missing lifetime UI button callbacks in `chisurf/models/tcspc/widgets/lifetime.py` (`LifetimeWidget.onAddLifetime`, `LifetimeWidget.onRemoveLifetime`, `LifetimeWidget.onAbsoluteAmplitudes`).
  - Added handler contract coverage in `test/test_convolve_widget_contract.py` to guard required callback presence.

- **MCP debug/control usability for live ChiSurf sessions**: MCP connection mode and diagnostics were difficult to debug from OpenCode when targeting a running GUI process.
  - Added MCP transport CLI options and streamable-HTTP runtime support in `chisurf/mcp/server.py` (`_parse_args`, `resolve_transport_kwargs`, `main`) so ChiSurf can expose a stable remote MCP endpoint.
  - Added minimal debug-oriented MCP tools in `chisurf/mcp/server.py` (`ping`, `list_runtime_vars`, `get_runtime_var`, `set_runtime_var`, `show_message`) for quick runtime inspection and on-screen PyQt message feedback.
  - Updated GUI MCP autostart invocation in `chisurf/gui/__init__.py` (`setup_gui` stage `start_mcp`) to launch MCP as streamable HTTP on `127.0.0.1:8765/mcp`.
  - Updated OpenCode MCP config in `opencode.json` to connect as remote MCP (`type: remote`, `url: http://127.0.0.1:8765/mcp`) instead of spawning a separate local MCP process.
  - Added focused coverage for transport kwargs in `test/test_mcp_server.py` (`test_resolve_transport_kwargs_streamable_http`, `test_resolve_transport_kwargs_stdio`).

- **Ribbon auto-fold not working after unpinning**: the ribbon would stay visible even when unpinned because the auto-fold timer was never started when toggling auto-fold on.
  - Fixed `toggle_auto_fold(enabled=True)` in `chisurf/gui/widgets/ribbon/ribbon_auto_fold.py` to restart the timer when enabling auto-fold.
  - Fixed `_setup_pin_button()` to stop the timer if ribbon starts pinned, preventing inconsistent state.
  - Removed redundant timer start from `__init__` since `_setup_auto_fold()` handles it.
  - Added `pinned` setting to `chisurf/settings/settings_chisurf.yaml` for visibility.
- **DataGroup classes now properly initialize unique_identifier**: `ExperimentDataCurveGroup` and `ExperimentDataGroup` were not passing `*args` and `**kwargs` to the `Base` class during initialization, causing `unique_identifier` to be missing from `meta_data` and raising `KeyError` when grouping datasets.
  - Fixed in `chisurf/data.py` by changing inheritance order to `class DataGroup(chisurf.base.Base, list)` so `super().__init__()` properly calls both base initializers.
- **DataGroup classes now have a filename property**: grouped datasets were missing the `filename` attribute that the dataset selector expects, causing `AttributeError` when updating the UI after grouping.
  - Added `filename` property to `DataGroup` in `chisurf/data.py` that returns the filename of the first member.
  - Fixed in `chisurf/data.py` (`DataGroup.__init__`) to forward `*args` and `**kwargs` to the base class.
- **Chato direct FCS command now supports folder-based global fits with linked diffusion/shape parameters**: commands targeting a directory (e.g. `...\\test\\data\\fcs\\kristine`) previously failed with `File not found` because only single-file paths were supported.
  - Updated `chisurf/plugins/_dev/chato/frontend/dock.py` (`_extract_first_path_like`, `_resolve_fcs_curve_paths`, `_linkable_global_fcs_parameter_names`, `_handle_direct_fcs_fit_command`) to load multiple FCS curves from a folder, group datasets, create one global fit, link `td*` and `s/shape` parameters across the fit group, and run optimization.
  - Added focused regression coverage in `test/test_chato_direct_fcs_paths.py`.

- Prevented silent history-event loss for unregistered action types by adding dispatcher-error fallback in `chisurf/runtime/actions.py` (`record_action`): if routed execution fails (e.g., unknown action name), the event is now still recorded via direct history fallback instead of being dropped.
- **Plugin widget UI loading now fails fast with clear path errors after plugin moves/restructure**: missing or stale `.ui` paths previously surfaced as less actionable Qt load failures.
  - Added cached UI path resolution and explicit existence checks in `chisurf/gui/decorators.py` (`_resolve_ui_path_cached`, `resolve_ui_path`, `init_with_ui.load_ui`).
- **Grouped dataset fits now default-link non-nuisance parameters across local fits**: grouped fits previously started largely unlinked, requiring repetitive manual linking for lifetime/rotational/distance parameters.
  - Added automatic non-nuisance linking in `chisurf/macros/core_fit.py` (`_collect_group_nuisance_parameter_names`, `_auto_link_non_nuisance_group_parameters`, `add_fit`), excluding nuisance-style groups (`generic`, `corrections`, `convolve`, and `*nuisance*`).
  - Added a structured history event `fit_group_auto_link` recording how many master/follower links were created.
- **Adding multiple fits from GUI is faster for multi-selection workflows**: creating many fits in one action previously repeated top-level UI repaint/unfreeze cycles and full `cs.update()` calls for each fit.
  - Updated `chisurf/macros/core_fit.py` (`add_fit`) to batch multi-dataset fit creation with deferred UI refresh (`_defer_cs_update`) and shared UI-freeze scope (`_ui_updates_frozen`).
- **Data table no longer stalls UI during frequent fit refreshes**: table updates used expensive per-refresh geometry autosizing and copied curve data on every update.
  - Optimized `chisurf/plots/table_plot.py` (`_FitTableModel.set_arrays`, `FitTablePlot.__init__`, `FitTablePlot._get_arrays`, `FitTablePlot._refresh_arrays_into_model`, `FitTablePlot.showEvent`) to avoid `ResizeToContents`, avoid unnecessary model resets, avoid deep curve copies, and defer updates while hidden.
- **Grouped-parameter unlink now clears stale underline styling on follower widgets**: after removing fit-group links, dependent parameter rows could remain visually underlined until a later UI refresh, even though links were already removed.
  - Added explicit post-link/unlink visual refresh in `chisurf/gui/widgets/fitting/parameter_widgets.py` (`FittingParameterWidget._refresh_group_link_visuals`, `FittingParameterWidget.onLinkFitGroup`) so all same-named parameter controllers in the active fit group are finalized immediately.
  - Added regression contract coverage in `test/test_parameter_widget_link_visuals.py` (`test_group_unlink_refreshes_visual_state_for_related_widgets_contract`).
- **Project load no longer passes non-finite anisotropy calibration values into lifetime model construction**: loading saved projects with `NaN`/`inf` reader calibration (`g_factor/l1/l2`) could crash on Windows during anisotropy parameter initialization.
  - Hardened calibration parsing/application in `chisurf/macros/core_fit.py` (`_coerce_finite_float`, `_resolve_dataset_anisotropy_calibration`, `_apply_anisotropy_calibration_to_fit`, `add_fit`) to ignore non-finite values before model creation.
  - Removed a stray `fit_close` history emit from the end of `load_project` in `chisurf/macros/core_fit.py` that referenced undefined close-context variables.
  - Added regression coverage in `test/test_core_fit_anisotropy_calibration.py` (`test_resolve_dataset_anisotropy_calibration_ignores_non_finite_values`, `test_apply_anisotropy_calibration_ignores_non_finite_values`).
- **Project save/load now preserves grouped datasets and grouped-fit membership**: grouped datasets were reloaded as flat datasets and local fits were reattached to separate fit groups.
  - Added dataset tree snapshot persistence (`ui_state.dataset_layout`) in `chisurf/macros/core_fit.py` (`save_project`) and restored grouped dataset objects during `load_project`.
  - Updated fit reconstruction in `chisurf/macros/core_fit.py` (`load_project`) to map grouped local-fit dataset IDs back to a single grouped dataset index (deduplicated), preserving original grouped fit composition.
- **Grouped fit filenames now keep per-fit identity labels (VV/VH, etc.)**: indexed-only names made grouped exports hard to map back to channels/conditions.
  - Updated grouped save naming in `chisurf/fitting/fit.py` (`FitGroup.save`) to derive a per-fit suffix from each local dataset name (common prefix removed), with safe fallback to index when needed.
- **Grouped fit export no longer overwrites to a single output set**: `FitGroup.save(...)` reused the same filename for every local fit, so only one fit's files survived on disk.
  - Fixed grouped filename generation in `chisurf/fitting/fit.py` (`FitGroup.save`) to write each local fit to a unique indexed base (`_00`, `_01`, ...), preserving all grouped outputs.
- **Fit-group parameter linking now uses a stable master fit**: middle-checkbox linking from a non-first local fit could make that selected fit the master, leading to inconsistent master/follower orientation across sessions.
  - Updated fit-group linking in `chisurf/macros/core_fit.py` (`link_fit_group`) so group links always target the parameter of the first local fit; non-master flags are reset before relinking.
  - Added regression coverage in `test/test_fit_group_link_master.py` (`test_link_fit_group_always_uses_first_fit_as_master`).
- **Line-plot metrics overlay no longer uses rich-text HTML updates during fit refresh**: `pyqtgraph.TextItem.setHtml(...)` remained a crash hot path in some Windows save/switch workflows.
  - Switched line-plot metrics rendering to plain text and added a save-time overlay suspension guard in `chisurf/plots/lineplot/lineplot.py` (`LinePlot._build_metrics_overlay_text`, `LinePlot.update`) and `chisurf/macros/core_fit.py` (`save_fit`).
- **`Ctrl+S` now reliably triggers "Save Current Fit"**: save-fit keyboard shortcut behavior was inconsistent depending on focus/widget context.
  - Bound `Ctrl+S` as an application shortcut for `actionSaveCurrentFit` in `chisurf/gui/main.py` (`Main.define_actions`).
- **Fit-export filenames now drop source extensions from dataset names**: exports based on names like `...dat VV` previously produced awkward filenames such as `...dat VV_info.txt` and `...dat VV_weighted residuals.csv`.
  - Updated save-name stem handling in `chisurf/macros/core_fit.py` (`save_fit`, `save_fits`) to remove trailing source extensions before writing CSV/TXT/DOCX/fit.json outputs.
- **Windows save could fail with `FileNotFoundError` for deep fit-export paths**: long VV/VH export filenames (e.g. `..._weighted residuals.csv`) could exceed classic Windows path limits and fail mid-save even when the folder exists.
  - Added long-path normalization in `chisurf/fio/ascii.py` (`_windows_extended_path`, `Csv.save`) to automatically use Windows extended-length path prefixes for deep output paths.
- **Save-fit crash in grouped VV/VH lifetime workflows**: saving grouped fits could trigger an access-violation in `pyqtgraph.TextItem.setHtml(...)` while fit selection changes refreshed line plots.
  - Hardened metrics-overlay updates in `chisurf/plots/lineplot/lineplot.py` (`LinePlot._metrics_text_alive`, `LinePlot.update`) to verify Qt object lifetime (`sip.isdeleted`) before calling `updateTextPos()`/`setHtml()`.
- **TCSPC CSV `Reference` picker now uses the central last-used-folder flow**: `pushButton_inspect` now opens via the shared `get_filename(...)` helper and honors `chisurf.working_path` consistently.
  - Updated file selection in `chisurf/gui/widgets/experiments/tcspc/csv_tcspc_widget.py` (`CsvTCSPCWidget.openJordiGFactorPlugin`) and hardened cancel handling in `chisurf/gui/widgets/general.py` (`get_filename`) so cancel no longer resets `chisurf.working_path`.
- **Anisotropy fit setup now respects TCSPC reader `g_factor/l1/l2` values**: fit creation previously applied only `g_factor`, so `l1/l2` from CSV reader settings were ignored when initializing anisotropy model parameters.
  - Updated fit-calibration resolution/application in `chisurf/macros/core_fit.py` (`_resolve_dataset_anisotropy_calibration`, `_apply_anisotropy_calibration_to_fit`, `add_fit`) to propagate `g_factor`, `l1`, and `l2` from reader/metadata into model kwargs and anisotropy parameters.
- **TCSPC CSV reader UI now exposes `l1/l2` calibration controls near `rep.rate` and `g-factor`**: users can edit anisotropy leakage factors directly in the reader panel and optionally link `l1`/`l2` with a dedicated checkbox (default off).
  - Updated UI and parameter propagation in `chisurf/gui/widgets/experiments/tcspc/tcspc_csv.ui` (`doubleSpinBox_l1`, `doubleSpinBox_l2`, `checkBox_link_l1_l2`, `actionL1L2Changed`) and `chisurf/gui/widgets/experiments/tcspc/csv_tcspc_widget.py` (`CsvTCSPCWidget.updateUI`, `CsvTCSPCWidget.onParametersChanged`, `_sync_l2_from_l1`, `_on_link_l1_l2_toggled`).
- **TCSPC reader anisotropy calibration now carries `g_factor`, `l1`, and `l2` consistently**: reader calibration was effectively g-only and left `l1/l2` scattered across downstream components.
  - Updated calibration defaults/annotation in `chisurf/experiments/tcspc/reader.py` (`TCSPCReader._default_anisotropy_calibration`, `TCSPCReader._annotate_anisotropy_calibration`, `TCSPCReader.read`) and extended CSV reader calibration propagation in `chisurf/fio/fluorescence/tcspc.py` (`read_tcspc_csv`, `_annotate_anisotropy_meta`).
- **Grouped line-plot autocorrelation traces were hard to compare in "display group" mode**: autocorrelation curves from multiple local fits overlapped on the same baseline.
  - Added automatic per-fit vertical offsets for grouped autocorrelation traces in `chisurf/plots/lineplot/lineplot.py` (`LinePlot._plot_group_curves`, `LinePlot._compute_group_curve_offsets`) so grouped `a.corr.` curves are stacked for readability while keeping the active fit centered.
- **Grouped line-plot weighted residuals were hard to compare in "display group" mode**: weighted residual traces from multiple local fits overlapped on the same baseline, reducing readability.
  - Added automatic per-fit vertical offsets for grouped weighted residuals in `chisurf/plots/lineplot/lineplot.py` (`LinePlot._plot_group_curves`, `LinePlot._compute_group_curve_offsets`) so traces are stacked like pyqtgraph multi-curve displays while keeping the active fit centered.
- **Plugin check dangerous-call detection no longer flags `ast.literal_eval`**: replaced brittle substring scanning for `eval`/`exec` with AST-based call analysis so safe calls like `ast.literal_eval(...)` pass while true dangerous calls are still blocked.
  - Updated plugin safety checks in `chisurf/macros/plugin_check.py` (`_contains_dangerous_operations`, `_resolve_call_name`, `PluginTestRunner._test_plugin`, `PluginTestRunner._test_plugin_chisurf_way`)
- **FCS CPM now uses total detector count rate by default**: corrected CPM display to use total count rate per molecule instead of per-detector mean when total-rate metadata is available.
  - Updated total-rate resolution in `chisurf/models/fcs/fcs.py` (`ParseFCSWidget._resolve_total_mean_count_rate`, `ParseFCSWidget.update_model`) and applied it to CPM/CPM_all and signal countrate `S`
  - Updated TTTR correlator `.cor` export in `chisurf/gui/widgets/wizard/tttr_correlator/tttr_correlator.py` (`WizardTTTRCorrelator.save_correlations`) to write total detector count rate `(ca + cb) / duration / 1000`
  - Updated Zeiss Confocor3 import in `chisurf/fio/fluorescence/fcs/confocor3.py` (`read_zeiss_fcs`) to store explicit `mean_count_rate_total` metadata for two-channel curves
- **TTTR microtime histogram plugin launch crash**: fixed missing UI file resolution that raised `FileNotFoundError` when loading the plugin wizard.
  - Corrected UI path in `chisurf/plugins/tttr/microtime_histogram/wizard.py` (`MicrotimeHistogram.__init__` decorator)
- **Dataset removal now confirms and closes dependent fits**: removing datasets/groups now displays a single confirmation dialog, logs how many dependent fits would be closed, and automatically closes those fits before removing the data.
  - Added dataset expansion/fit dependency helpers in `chisurf/macros/core_data.py` and fire the confirmation dialog only once.
- **TCSPC fits honor reader g_factor**: anisotropy models now derive the initial `g` value from the TCSPC reader metadata so the g parameter matches the detector calibration used during data import.
  - Added dataset-level g_factor resolution/parsing in `chisurf/macros/core_fit.py` (`_resolve_dataset_g_factor`, `_apply_g_factor_to_fit`, `add_fit`).
  - Hardened g-factor assignment in `chisurf/macros/core_fit.py` so anisotropy objects with scalar `_g` state no longer crash fit creation (`'float' object has no attribute 'value'`).
- **TCSPC anisotropy widget `r_ss` showed `NaN` persistently**: fixed missing model/fit wiring so the `r_ss` output parameter can resolve lifetime-model values instead of falling back to invalid context.
  - Passed `fit`/`model` when constructing `AnisotropyWidget` in `chisurf/models/tcspc/widgets/lifetime.py` and `chisurf/models/tcspc/widgets/gaussian.py`
  - Added context inference fallback in `chisurf/models/tcspc/widgets/anisotropy.py` (`AnisotropyWidget.__init__`)
- **TCSPC anisotropy widget `rS,I` mismatch**: corrected intensity-domain steady-state anisotropy computation to match ChiSurf anisotropy channel conventions and objective-mixing model.
  - `rS,I` now unmixed VV/VH channel integrals using `l1/l2` and computes anisotropy with ChiSurf convention `r = (VV - VH) / (g*VV + 2*VH)` in `chisurf/models/tcspc/widgets/anisotropy.py` (`_compute_steady_state_anisotropy_intensity`).
  - Fixed grouped dataset handling: for `VV/VH` fits the widget now uses both curves from `ExperimentDataCurveGroup` (instead of only `current_dataset`) before unmixing/integration.
  - Fixed g-factor insensitivity in local `VV`/`VH` views by always using dual-channel computation whenever both channels are available from stacked data or fit-group pairing.
  - Corrected channel extraction for stacked/local VH paths to avoid mixing up 1D/2D channel arrays; this removes large `rS,I` bias (often ~2x) caused by using wrong channel slices in dual-group contexts.

- **Resolved persistent SIGBUS (EXC_ARM_DA_ALIGN) startup crash on macOS ARM64 (Apple Silicon)** by sanitizing UI elements (Emojis, special symbols) that triggered incompatible rendering paths in CoreText.
- **Hardened `build-osx-app.sh` by removing `DYLD_LIBRARY_PATH`** to prevent library conflicts with system frameworks and ensured the ribbon correctly renders in-window by changing `RibbonBar` inheritance from `QMenuBar` to `QWidget`.
- **Fixed `ModuleNotFoundError` for `chisurf.plugins._dev` in bundled application** by implementing a global development-plugin proxy system in `chisurf/plugins/__init__.py`.
- **Restored `ChiSurf.icns` icon association and launcher environment fixes** for the macOS application bundle.

## Previous Releases

(Add previous release notes here if available)
