# CHANGELOG

## [Unreleased]

### Fixed

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
- **NDXplorer UI Blocking Bug**: Fixed issue where NDXplorer UI remained blocked/disabled after data loading completed. The UI now properly re-enables itself after both normal data loading and append operations.
  - Added `update_ui_enabled_state()` call in normal data loading path in `modules/ndxplorer/ndxplorer/io/file_operations.py`
  - Added `update_ui_enabled_state()` call in append data loading path in `modules/ndxplorer/ndxplorer/io/file_operations.py`
  - This ensures UI controls are re-enabled consistently after all data loading operations
- **NDXplorer Folder-Drop Disabled UI**: Fixed follow-up issue where dropping a folder path could still leave the UI disabled after successful load.
  - Updated `update_ui_enabled_state()` in `modules/ndxplorer/ndxplorer/core/plot_main.py` to detect loaded data from both `data_source` and fallback `_data_source`
  - This aligns UI enable/disable logic with both data-manager and direct `_data_source` load paths
- **NDXplorer Empty Axis Selectors After Folder Drop**: Fixed follow-up issue where axes remained `(-1, '')`, histograms were skipped, and comboboxes appeared empty after successful folder-drop load.
  - Updated `SurfacePlotWidget.update()` in `modules/ndxplorer/ndxplorer/plotting/plot_control.py` to fall back to `data_source.parameter_names` when `data_source.data.columns` is empty
  - This preserves axis selector population for loader paths that provide parameter metadata before a full dataframe column map is available
- **NDXplorer Folder-Drop Parameter Population Regression**: Fixed root cause where folder-drop loads populated 0 axis parameters even though computed parameters existed.
  - Updated parameter collection in `modules/ndxplorer/ndxplorer/io/file_operations.py` to read names from active `data_source` (data-manager path) instead of only `_data_source`
  - Added fallback to `_data_source` only when the active accessor is unavailable
- **NDXplorer SM Axis Bootstrap**: Fixed follow-up issue where single-molecule folder loads could still leave axis selectors at `(-1, '')` and prevent histogram rendering.
  - Updated combobox population in `modules/ndxplorer/ndxplorer/io/file_operations.py` to always include raw `data_source.parameter_names` for all load types (not only image data)
  - Added explicit default index selection (`0`) when comboboxes contain items but have no active index
- **NDXplorer Non-Image UI Cleanup**: Hide image/frame controls when loaded data is not image-based.
  - Updated `check_and_set_image_axes()` in `modules/ndxplorer/ndxplorer/utils/axis_helpers.py` to call `hide_frame_selection()` when no `X pixel`/`Y pixel` axes are found
  - Updated `setup_frame_selection()` and `hide_frame_selection()` in `modules/ndxplorer/ndxplorer/plotting/plot_control.py` to show/hide the whole `groupBoxImage` (including pause/playback controls) for image vs non-image datasets
- **NDXplorer SM Plot Visibility Regression**: Fixed case where histograms were computed and sent to the renderer, but the UI still showed placeholder/empty content.
  - Updated data-presence visibility gate in `modules/ndxplorer/ndxplorer/plotting/plot_update_helpers.py` to use `data_source` (data-manager aware) instead of legacy `_data_source` only
  - Prevents `_show_empty_plots()`/background fallback from overriding valid SM histogram rendering
- **Burst Reader Column Truncation**: Fixed `.bur`/extra table loading that could drop a valid last column (e.g., `Last File`) and reduce available combobox parameters.
  - Updated `_process_burst_analysis_dir()` in `modules/ndxplorer/ndxplorer/io/reader.py` to only apply `drop_last_column` when the last header is legacy-empty (`""`/`"Unnamed..."`), not unconditionally
  - Prevents silent column loss during burst folder ingestion
- **NDXplorer Data Button False "No Data" Message**: Fixed DataFrame editor check to use active `data_source` instead of legacy `_data_source`.
  - Updated `show_dataframe_editor()` in `modules/ndxplorer/ndxplorer/core/plot_main.py` to validate and open data from `self.data_source`
- **Undo/redo fit-run replay did not restore fit-generated parameter state**: fit lifecycle events were tracked, but pre/post fit parameter values were not snapshotted for replay.
  - Added before/after parameter snapshots in `chisurf/gui/widgets/fitting/fit_controller.py` (`FittingControllerWidget._collect_parameter_snapshot`, `FittingControllerWidget.onRunFit`)
  - Updated replay in `chisurf/history_replay.py` (`reconstruct_parameter_state`) to consume `parameter_snapshot_before` / `parameter_snapshot_after`
  - Added regression test in `test/test_history_replay_state.py` (`TestHistoryReplayState.test_fit_run_snapshots_restore_parameter_state`)
- **Undo/redo replay did not restore fit-range state across cursor jumps**: parameter state could change while range and plots stayed stale.
  - Added structured `fit_range_set` event recording in:
    - `chisurf/gui/widgets/fitting/fit_controller.py` (`FittingControllerWidget.onFitRangeChanged`, `FittingControllerWidget.onAutoFitRange`)
    - `chisurf/plots/lineplot/lineplot.py` (line region handler)
    - `chisurf/plots/residual_image.py` (ROI handler)
  - Added before/after fit-range snapshots in `chisurf/gui/widgets/fitting/fit_controller.py` (`FittingControllerWidget._collect_fit_range_snapshot`, `FittingControllerWidget.onRunFit`)
  - Added range reconstruction in `chisurf/history_replay.py` (`reconstruct_fit_range_state`)
  - Added application of reconstructed range in `chisurf/gui/main.py` (`Main._apply_fit_range_state`, `Main._on_history_cursor_changed`)
  - Marked `fit_range_set` as state-bearing in `chisurf/gui/widgets/history_browser.py` (`HistoryBrowserWidget._is_state_action`)
  - Added regression test in `test/test_history_replay_state.py` (`TestHistoryReplayState.test_reconstruct_fit_range_state`)
- **Per-parameter unlink from main checkbox branch was not recorded in structured history**.
  - Added explicit `parameter_unlink` trace in `chisurf/gui/widgets/fitting/parameter_widgets.py` (`FittingParameterWidget.onLinkFitGroup`)
- **`fit_close` history action emitted at wrong lifecycle point** (project load tail instead of actual close action).
  - Moved `fit_close` history recording to `chisurf/macros/core_fit.py` (`close_fit`)
  - Removed misplaced `fit_close` emit from project-load tail in `chisurf/macros/core_fit.py`

- **Fit-run progress window now autocloses reliably**: the fitting progress dialog could remain open after fitting finished; it now closes immediately by default after completion.
  - Updated `EnhancedProgressDialog.finish` in `chisurf/gui/widgets/progress.py` to support synchronous finalization when `close_delay_ms=0`.
  - Updated `FittingControllerWidget.onRunFit` in `chisurf/gui/widgets/fitting/fit_controller.py` to default `fit_progress_close_delay_ms` to `0`.

- **Output parameters refresh when switching fits**: changing the selected local fit could update parameters/plots but leave dependent output/result parameters stale.
  - Updated `change_selected_fit_of_group` in `chisurf/macros/core_fit.py` to `finalize()` the selected model and refresh parameter controllers for the associated fit after selection changes.

- **Output parameters refresh after parameter edits**: editing a parameter (e.g. TCSPC anisotropy `g`) could update plots but leave dependent output parameters (e.g. `rS,I`) stale until a manual refresh.
  - Updated `FittingParameterWidget._trigger_model_update` in `chisurf/gui/widgets/fitting/parameter_widgets.py` to recompute model outputs (`model.finalize()`), refresh output parameter controllers for the associated fit, and additionally refresh visible output widgets in the active fit window as a fallback.

- **Computed IRF width sign no longer changes shape behavior**: negative IRF width (`iw`) could compensate skew sign and produce ambiguous computed IRF behavior versus `ik`.
  - Updated computed-IRF generation in `chisurf/models/tcspc/nusiance.py` (`Convolve._process_irf`) to always use positive width magnitude (`abs(iw)` with epsilon floor), so only `ik` sign controls asymmetry direction.
### Changed

- **Completed Phase 10 - Complete State Reconstruction for Undo/Redo**: Finalized the MVC action core migration by completing all remaining model widget migrations and adding comprehensive parameter scan support, achieving 100% migration coverage (55/55 calls) for full undo/redo functionality.
  - Migrated remaining PCH model widget operations in `chisurf/models/pch/widgets.py`: replaced 2 `chisurf.run()` calls with action controller calls for component add/remove operations using existing `model.add_component` and `model.remove_component` actions.
  - Added new `parameter.scan` action with full service implementation in `chisurf/controllers/services/parameter_service.py` (`scan_parameter`) and controller handler in `chisurf/controllers/action_controller.py` (`_handle_parameter_scan`).
  - Registered `parameter_scan` action spec in `chisurf/runtime/actions.py` with proper schema validation (`parameter_name`, `fit_index`, `scan_range`, `n_steps`).
  - Migrated parameter scan plot interaction in `chisurf/plots/parameter_scan/parameter_scan.py` from direct `chisurf.run()` execution to action controller with fallback compatibility.
  - Added comprehensive test coverage for parameter scan action in `test/test_action_controller.py` (`test_parameter_scan_routes_to_registered_handler`).
  - Updated `AGENTS_PLAN.md` to mark Phase 10 as complete with full migration statistics.
  - Updated `TODO.md` to reflect completion of all model widget migrations.
  - Cross-reference: `AGENTS_PLAN.md` Phase 10, `TODO.md` Architecture section.

- **Completed Phase 9 - MVC Action Core migration**: All GUI state-changing operations now route through the central action controller, completing the MVC architecture foundation for structured history, deterministic replay, undo/redo, and MCP/LLM automation.
  - Added missing actions: `fit.load`, `fit.save`, `fit.save_all`, `fit.close_all`, `parameter.value`, `fit.update`, `fit.range_set` with corresponding service methods in `chisurf/controllers/services/fit_service.py`, `chisurf/controllers/services/parameter_service.py`, and `chisurf/controllers/services/model_service.py`.
  - Added model management actions: `model.add_component`, `model.remove_component`, `model.normalize_amplitudes`, `model.absolute_amplitudes`, `model.change_irf`, `model.unload_irf`, `model.update`, `model.set_correction`, `model.set_linearization`, `model.unload_lintable`, `model.unload_background_curve`, `model.remove_local_fit`, `model.clear_local_fits`, `model.append_global_parameter`, `model.append_fit` with service methods in `chisurf/controllers/services/model_service.py`.
  - Migrated remaining GUI handlers to use controller actions: `onSetupChanged` (setup.select), `onCloseAllFits` (fit.close_all), `onLoadFit` (fit.load), `onSaveFit` (fit.save), `onSaveFits` (fit.save_all) in `chisurf/gui/main.py`.
  - Migrated model widget operations in `chisurf/models/tcspc/widgets/lifetime.py`: replaced 6 `chisurf.run()` calls with action controller calls for parameter updates, fit updates, and model component operations.
  - Migrated PDA model widget operations in `chisurf/models/pda/widgets.py`: replaced 15 `chisurf.run()` calls with action controller calls for parameter updates, fit updates, and model component operations. Added `_get_fit_index_for_model()` helper function to ensure correct fit indexing.
  - Migrated TCSPC model widget operations in `chisurf/models/tcspc/widgets/anisotropy.py` and `chisurf/models/tcspc/widgets/convolve.py`: replaced 6 `chisurf.run()` calls with action controller calls for model component and IRF operations.
  - Migrated TCSPC corrections widget in `chisurf/models/tcspc/widgets/corrections.py`: replaced 7 `chisurf.run()` calls with action controller calls for correction operations (pile-up, reverse, DNL, window function, linearization).
  - Migrated TCSPC discrete distance widget in `chisurf/models/tcspc/widgets/discrete_distance.py`: replaced 2 `chisurf.run()` calls with action controller calls for FRET rate component operations.
  - Migrated TCSPC Gaussian widget in `chisurf/models/tcspc/widgets/gaussian.py`: replaced 3 `chisurf.run()` calls with action controller calls for Gaussian component operations and model updates.
  - Migrated TCSPC generic widget in `chisurf/models/tcspc/widgets/generic.py`: replaced 2 `chisurf.run()` calls with action controller calls for background curve operations.
  - Migrated global model widget in `chisurf/models/global_model/widget.py`: replaced 4 `chisurf.run()` calls with action controller calls for local fit management, global parameter operations, and fit appending.
  - Migrated plot interaction handlers in `chisurf/plots/residual_image.py` and `chisurf/plots/lineplot/lineplot.py`: replaced 2 `chisurf.run()` calls with `fit.range_set` action controller calls and fixed fit indexing to use `getattr(self.fit, "selected_fit_index", 0)`.
  - Fixed critical fit indexing issues: Corrected hardcoded `fit_index: 0` to use proper fit detection in all migrated widgets, ensuring actions operate on the correct fit rather than always using `fits[0]`. This was a critical bug that would have caused actions to modify the wrong fit in multi-fit scenarios.
  - Added comprehensive controller tests in `test/test_action_controller.py` (33 tests total, +17 new tests including 5 specifically for fit index handling, 1 for correction operations, 1 for background curve operations, and 1 for global model operations):
    - `test_fit_index_handling_with_multiple_fits()` - Verifies different fit indices are handled correctly
    - `test_fit_index_default_handling()` - Ensures proper default behavior
    - `test_fit_index_validation()` - Tests edge cases and error handling
    - `test_fit_index_detection_logic()` - Validates model-to-fit mapping logic
    - `test_fit_index_edge_cases()` - Tests type conversion and boundary conditions
    - `test_model_correction_actions()` - Tests correction-specific operations
    - `test_model_background_curve_actions()` - Tests background curve operations
    - `test_model_global_model_actions()` - Tests global model operations
  - Updated `AGENTS_PLAN.md` to mark Phase 9 as complete with expanded action coverage.
  - Cross-reference: `AGENTS_PLAN.md` Phase 9, `TODO.md` Architecture section.

- Added general Chato MCP control endpoints for capability/state introspection and action execution so automation is no longer limited to TCSPC-specific routes.
  - Extended `chisurf/plugins/_dev/chato/frontend/dock.py` (`ChatoDock._dispatch_action_request`) with `/core/action_catalog`, `/core/describe_state`, `/core/execute_action`, and `/core/execute_plan` handlers.
  - Added core MCP tools in `chisurf/plugins/_dev/chato/mcp/core.py` (`discover_capabilities`, `describe_state`, `execute_action`, `execute_plan`) with in-process action-server proxying and local fallback behavior.
- Extended the Chato agentic backend with general controller-driven tools and switched the agent entrypoint from TCSPC-only naming to a ChiSurf-general path.
  - Added `core_action_catalog`, `core_describe_state`, `core_execute_action`, and `core_execute_plan` to `chisurf/plugins/_dev/chato/backend/agentic.py` (`_make_tcspc_tools`, `_tool_help_text`) for generic introspection and action execution.
  - Added `run_chisurf_agent(...)` in `chisurf/plugins/_dev/chato/backend/agentic.py` and kept `run_tcspc_agent(...)` as a compatibility wrapper.
  - Updated `chisurf/plugins/_dev/chato/backend/langchain.py` to call `run_chisurf_agent(...)` when agentic mode is selected.
  - Added focused regression coverage in `test/test_chato_agentic_tools.py`.
- Reduced TCSPC-only wizard capture so general fit/analysis requests can flow to generic agentic execution.
  - Updated `chisurf/plugins/_dev/chato/backend/langchain.py` (`_tcspc_wizard_intent`) to trigger the wizard only for explicit TCSPC context instead of generic `/fit` requests.
  - Expanded default agentic keyword triggers in `chisurf/plugins/_dev/chato/core/config.py` (`CHATO_AGENTIC_KEYWORDS`) for non-TCSPC workflows (`fit`, `analysis`, `fcs`, `pda`, `rics`, `pch`, etc.).
  - Added intent behavior tests in `test/test_chato_agentic_tools.py` (`test_tcspc_wizard_intent_is_not_generic_fit_trigger`, `test_tcspc_wizard_intent_for_explicit_tcspc_context`).
- Added Chato UI provider/API-key settings with two provider modes (`Local LLM`, `Mistral API`), with API keys persisted in the ChiSurf user settings folder instead of `QSettings`.
  - Extended settings UI in `chisurf/plugins/_dev/chato/frontend/widgets.py` (`SettingsDialog`) with a two-option provider selector and masked API-key field.
  - Added a provider connectivity test control in `chisurf/plugins/_dev/chato/frontend/widgets.py` (`SettingsDialog.test_provider_btn`) that validates endpoint/API key by querying provider models.
  - Added authenticated model listing support in `chisurf/plugins/_dev/chato/frontend/workers.py` (`ModelsWorker`) and authenticated chat/model requests in `chisurf/plugins/_dev/chato/backend/client.py` (`LlamaCppClient`).
  - Added key-store helpers in `chisurf/plugins/_dev/chato/frontend/dock.py` (`_api_keys_file_path`, `_load_api_key_store`, `_save_api_key_store`, `_get_stored_api_key`, `_set_stored_api_key`) using `<chisurf_user_settings>/chato_api_keys.json`.
  - Updated settings persistence flow in `chisurf/plugins/_dev/chato/frontend/dock.py` (`_load_settings`, `_save_settings`, `_apply_settings`) so API keys are loaded from and saved to the user-folder key store, not `QSettings`.
  - Added provider/model guard in `chisurf/plugins/_dev/chato/frontend/dock.py` (`_fetch_provider_models`, `_pick_supported_chat_model`, `_apply_settings`) to auto-switch `chat_model` to a supported model when the selected model is not available on the current endpoint; for `mistral_api`, preferred fallback order is `devstral-medium-latest`, `devstral-small-latest`, `codestral-latest`, `mistral-large-latest`.
  - Added a direct-chat fallback path in `chisurf/plugins/_dev/chato/frontend/workers.py` (`ChatWorker.run`) when LangChain request shaping fails (including provider-specific 4xx/422 cases), so Chato can still chat through the configured endpoint.
  - Updated `chisurf/plugins/_dev/chato/backend/langchain.py` and `chisurf/plugins/_dev/chato/backend/agentic.py` to propagate provider/API-key context and avoid provider-incompatible request parameters for Mistral (`top_p` omitted on Mistral paths).
- Fixed Mistral LangChain request incompatibility in `chisurf/plugins/_dev/chato/backend/langchain.py` by omitting `max_tokens` on Mistral paths, preventing `max_completion_tokens`-style 422 payload rejections.
- Moved the local runtime GPU/CPU indicator from the main Chato dock toolbar into Settings and made it local-provider specific.
  - Removed toolbar badge usage in `chisurf/plugins/_dev/chato/frontend/dock.py` and switched to a computed `local_runtime` setting field.
  - Added `Local runtime` status row in `chisurf/plugins/_dev/chato/frontend/widgets.py` (`SettingsDialog.local_runtime_label`) that is shown only when provider is `Local LLM`.
- Fixed Mistral chat instability from LangChain provider payload incompatibilities by bypassing LangChain for `mistral_api` provider mode in `chisurf/plugins/_dev/chato/frontend/workers.py` (`ChatWorker.run`) and using direct authenticated chat client requests.
- Fixed RAG embedding auth handling for remote providers and reduced repeated auth-error noise.
  - Added bearer-auth header support in `chisurf/plugins/_dev/chato/rag/rag.py` (`_embed_texts_openai`) using configured API key envs.
  - Added throttled warning behavior in `chisurf/plugins/_dev/chato/rag/rag.py` (`_rag_retrieve_hits`) for repeated 401/Unauthorized embedding failures.
  - Updated `chisurf/plugins/_dev/chato/frontend/dock.py` (`_apply_settings`) and `chisurf/plugins/_dev/chato/frontend/widgets.py` (`_on_provider_changed`) to default Mistral embedding config to `openai:mistral-embed`.
- Simplified and cleaned the Chato initialization instruction set to reduce contradictory/duplicated rules while preserving strict grounding behavior.
  - Rewrote `chisurf/plugins/_dev/chato/instructions/chato_init_prompt.md`.
- Restored operational agent behavior for Mistral provider mode so action requests (e.g. fitting commands) run through LangChain/agentic orchestration instead of always short-circuiting to plain chat fallback.
  - Updated `chisurf/plugins/_dev/chato/frontend/workers.py` (`ChatWorker.run`) to use LangChain path first for Mistral as well, with direct client fallback only on failure.
  - Reinforced tool-first behavior in prompt files: `chisurf/plugins/_dev/chato/instructions/chato_default_system_prompt.md` and `chisurf/plugins/_dev/chato/instructions/chato_init_prompt.md`.
- Added direct operational fallback for explicit FCS fit commands with local file paths so Chato executes load+fit actions instead of returning non-operational guidance.
  - Added `chisurf/plugins/_dev/chato/frontend/dock.py` handlers (`_extract_first_path_like`, `_guess_fcs_setup_name`, `_handle_direct_fcs_fit_command`) and wired them into `on_send` before chat-model execution.
  - Hardened direct FCS execution selection in `chisurf/plugins/_dev/chato/frontend/dock.py` (`_handle_direct_fcs_fit_command`, `_resolve_available_setup_names`) to use GUI-valid experiment/setup names and fallbacks (`FCS` then available setup list), fixing combo-box name mismatches.
  - Coupled direct FCS execution with active GUI selection state in `chisurf/plugins/_dev/chato/frontend/dock.py` (`_couple_dataset_to_ui`, `_handle_direct_fcs_fit_command`) by explicitly selecting the imported dataset before fit creation and issuing `fit.set_dataset` after fit creation.
- Hardened fit creation fallback for GUI variants without `current_model_name`.
  - Updated `chisurf/macros/core_fit.py` (`add_fit`) to resolve model name from multiple sources (`current_model_name`, `current_model_class.name`, first experiment model) instead of assuming one GUI attribute is always present.
- Added explicit operational command example to Chato instructions and continuation handoff artifact.
  - Updated `chisurf/plugins/_dev/chato/instructions/chato_default_system_prompt.md` and `chisurf/plugins/_dev/chato/instructions/chato_init_prompt.md` with the concrete FCS fit command example.
  - Added `chisurf/plugins/_dev/chato/catchup.md` and linked it from `chisurf/plugins/_dev/chato/README.md`.
- Added plugin UI-path contract coverage in `test/test_plugin_ui_paths_contract.py` (`test_all_plugin_init_with_ui_paths_exist`, `test_plugin_direct_loadui_string_targets_exist`) to assert plugin `.ui` references resolve after plugin moves.
- Added grouped-linking default contract coverage in `test/test_grouped_default_linking_contract.py` (`test_grouped_fits_auto_link_non_nuisance_parameters_contract`).
- Added performance regression contracts in `test/test_performance_contracts.py` for batched fit creation and data-table refresh behavior.
- Added a phase-level execution tracker with explicit `[DONE]/[PARTIAL]/[PENDING]` status and changelog cross-references in `AGENTS_PLAN.md` (`Execution Tracking (Cross-Referenced)`), so implementation progress can be traced from plan milestones to concrete release notes.
- Expanded history-navigation diagnostics and replay synchronization in `chisurf/gui/main.py` (`Main._apply_parameter_state`, `Main._apply_fit_range_state`, `Main._select_fit_by_name`) with high-signal INFO logs for unresolved replay keys/fit-groups and explicit `chisurf.current_fit` synchronization during cursor-driven fit activation.
- Introduced an MCP-ready central action core in `chisurf/runtime/actions.py` (`ActionSpec`, `ActionRegistry`, `ActionDispatcher`) with payload schema validation and dedupe-policy metadata (`none`, `drop_duplicates`, `coalesce_latest`) and wired lazy access via `chisurf/__init__.py` (`action_dispatcher`, `action_registry`).
- Migrated first command/history vertical slice to the central dispatcher:
  - parameter operation tracing in `chisurf/gui/widgets/fitting/parameter_widgets.py` (`FittingParameterWidget._trace_operation`) now executes through `chisurf.action_dispatcher` before fallback to raw history recording,
  - console command history in `chisurf/gui/main.py` (`Main.init_console`) now emits `run_command` through the same dispatcher.
- Reduced maintenance overhead by consolidating duplicated history-emission logic into `chisurf/runtime/actions.py` (`record_action`) and reusing it across `chisurf/macros/core_data.py`, `chisurf/macros/core_fit.py`, `chisurf/gui/widgets/fitting/fit_controller.py`, `chisurf/plots/lineplot/lineplot.py`, `chisurf/gui/main.py`, and `chisurf/gui/widgets/fitting/parameter_widgets.py`.
- Added MCP-facing action catalog discovery in `chisurf/runtime/actions.py` (`ActionSpec.to_dict`, `ActionRegistry.catalog`, `get_action_catalog`) and lazy runtime accessor `chisurf.action_catalog` in `chisurf/__init__.py`, exposing canonical action metadata plus dot-style `mcp_name` aliases.
- Extended project persistence metadata in `chisurf/macros/core_fit.py` by embedding action-catalog snapshots in `proj.extra["action_catalog"]` during project/fit saves and by recording `history_loaded` flags on `project_load`/`fit_load` history events for better replay provenance.
- Added focused coverage for the action core in `test/test_action_dispatcher.py` (`TestActionDispatcher`) validating registry defaults, payload schema checks, and duplicate suppression behavior.
- Consolidated ad-hoc root test scripts into a dedicated `unittests/` area and removed empty placeholder root tests.
  - Added focused pytest-style coverage in `unittests/test_jordi_rebin_fix.py` (`test_vv_vh_rebin_reshape_groups_are_stable`) and `unittests/test_jordi_io_and_anisotropy.py` (`test_jordi_roundtrip_split_channels`, `test_vv_vh_spectrum_equals_concatenated_components`).
  - Added contract-style pytest checks in `unittests/test_lineplot_group_display_contract.py`, `unittests/test_tcspc_reader_contract.py`, and `unittests/test_irf_normalization_contract.py` to preserve key fixes previously only covered by manual scripts.
  - Removed obsolete/empty/manual root scripts including `test_fcs2d_fitting.py`, `test_menu_switch.py`, `test_plugin_check*.py`, `test_active_fit_only.py`, `test_irf_normalization.py`, and related ad-hoc `test_*.py` diagnostics now replaced by `unittests/` coverage.
- Updated `modules/ndxplorer/ndxplorer/io/file_operations.py` to ensure UI state is properly restored after data loading
- Added lightweight `fit_mask_set` observability event in `chisurf/plots/table_plot.py` (`DataTablePlot._set_mask`) for mask edits.
- Completed routing of remaining plot-side history emits through the central action router by switching `chisurf/plots/table_plot.py` (`DataTablePlot._set_mask`) and `chisurf/plots/residual_image.py` (ROI fit-range sync path) to `record_action(...)`; direct `history.record(...)` usage now remains only inside `chisurf/runtime/actions.py`.
- Added MCP-style action invocation compatibility in `chisurf/runtime/actions.py` by resolving dotted aliases (e.g. `project.save`) to canonical action IDs (e.g. `project_save`) in `ActionDispatcher.execute`, and exposed `chisurf.action_execute` in `chisurf/__init__.py` for external controller adapters.
- Added missing lifecycle action specs (`fit_save`, `fit_load`, `fit_group_link`, `fit_group_unlink`) in `chisurf/runtime/actions.py` to keep routed execution aligned with existing project/fit lifecycle events.
- Improved MVC maintainability by extracting controller-domain logic into focused services under `chisurf/controllers/services/` (`project_service.py`, `dataset_service.py`, `fit_service.py`, `setup_service.py`) and simplifying `chisurf/controllers/action_controller.py` to a thin routing layer.
- Added architecture contract documentation in `docs/architecture_mvc_actions.md` and linked it from `docs/index.rst`.
- Tightened lifecycle action schemas in `chisurf/runtime/actions.py` for stronger payload validation (`project_save/load`, `fit_add/run/set_dataset`, `dataset_remove/group/ungroup`).
- Added integration-style controller coverage for setup parameter orchestration in `test/test_action_controller.py` (`test_setup_params_set_applies_nested_keys`) to verify dotted nested key updates (e.g. `noise_model.weight_type`) and history emission.
- Routed fit-dataset switching in `chisurf/gui/widgets/fitting/fit_controller.py` (`FittingControllerWidget.change_dataset`) through controller action `fit.set_dataset` when selected dataset exists in imported datasets, reducing direct fit mutation in GUI path.
- Routed experiment/setup loading configuration in key wizard flows through controller actions (`experiment.set`, `setup.select`, `setup.params.set`) in `chisurf/gui/main.py` (`onExperimentChanged`), `chisurf/plugins/tttr/microtime_histogram/wizard.py`, `chisurf/plugins/fluorescence_decay/irf_estimator/__init__.py`, and `chisurf/gui/widgets/wizard/fcs_merger/fcs_merger.py`.
- Expanded Phase 8 replay coverage for setup state by adding `reconstruct_setup_state(...)` in `chisurf/history_replay.py` and applying setup replay in `chisurf/gui/main.py` during history cursor navigation (`_apply_setup_state`), including replay support for `experiment_set`, `setup_select`, and `setup_params_set` events.
- Added focused replay test coverage in `test/test_history_replay_state.py` (`test_reconstruct_setup_state`).
- Improved Phase 8 dataset navigation replay for ungroup operations by extending `dataset_ungroup` payload with `expanded_names` in `chisurf/macros/core_data.py` and consuming it in `chisurf/history_replay.py` (`reconstruct_navigation_state`).
- Added replay regression coverage for ungroup navigation in `test/test_history_replay_state.py` (`test_reconstruct_navigation_state_dataset_ungroup`).
- Added dataset-UID-aware replay metadata for dataset lifecycle events in `chisurf/macros/core_data.py` (`loaded_uids`, `removed_uids`, `group_uid`, `group_uids`, `expanded_uids`) and extended `chisurf/history_replay.py` navigation reconstruction to track/select `selected_dataset_uid`.
- Updated history cursor apply path in `chisurf/gui/main.py` to prefer dataset UID selection during replay (`_select_dataset_by_identity`) with name fallback.
- Added replay coverage for UID-based dataset ambiguity handling in `test/test_history_replay_state.py` (`test_reconstruct_navigation_state_prefers_dataset_uid`).
- Added UID-carrying parameter link/unlink payload fields in `chisurf/gui/widgets/fitting/parameter_widgets.py` and extended replay reconstruction in `chisurf/history_replay.py` to preserve source/link UID metadata (`source_*_uid`, `link_uid`) for more deterministic link replay resolution.
- Updated parameter replay apply in `chisurf/gui/main.py` (`_apply_parameter_state`) to prefer parameter UID matching (source and link target) before name-based fallback.
- Added replay regression coverage for parameter link UID metadata in `test/test_history_replay_state.py` (`test_reconstruct_parameter_state_link_uid_metadata`).
- Expanded parameter-value/fixed/bounds event payloads in `chisurf/gui/widgets/fitting/parameter_widgets.py` to include source UID context (`fit_uid`, `local_fit_uid`, `parameter_uid`) across both main-row and detail-popup editors, improving replay determinism and link resolution consistency.
- Expanded Phase 8 fit navigation replay with fit UID tracking in `chisurf/history_replay.py` (`fit_uids`, `selected_fit_uid`) and updated cursor apply in `chisurf/gui/main.py` to prefer UID-based fit selection (`_select_fit_by_identity`) with name fallback.
- Added replay coverage for fit UID navigation state in `test/test_history_replay_state.py` (`test_reconstruct_navigation_state_tracks_fit_uid`) and updated base navigation coverage to assert `selected_fit_uid`.
- Added architecture note `docs/history_project_mcp.md` and linked it from `docs/index.rst` to document the current history-routing, project metadata, and MCP action invocation model.
- Added initial controller layer `chisurf/controllers/action_controller.py` (`ActionController`) and routed project lifecycle GUI entrypoints through controller execution (`project.save`, `project.load`, `project.close`) in `chisurf/gui/main.py` and `chisurf/gui/project_helpers.py`, keeping macros as adapters during migration.
- Extended controller migration into fit lifecycle callsites by routing `fit.add` and `fit.close` intents through `ActionController` in `chisurf/gui/fit_helpers.py`, `chisurf/gui/widgets/fitting/fit_list.py`, `chisurf/gui/widgets/fitting/fit_subwindow.py`, and dependent-fit closure in `chisurf/macros/core_data.py`.
- Added structured `fit_close` history emission in `chisurf/macros/core_fit.py` (`close_fit`) so controller-routed fit closure remains explicitly traceable in operation history.
- Extended controller coverage for dataset lifecycle by adding `dataset.add`, `dataset.remove`, and `dataset.group` handlers in `chisurf/controllers/action_controller.py` and routing key GUI entrypoints in `chisurf/gui/main.py` and `chisurf/gui/widgets/experiments/widgets.py` through controller execution.
- Routed fit-run lifecycle event invocation through `ActionController` (`fit.run.start`, `fit.run.finish`, `fit.run.abort`) from `chisurf/gui/widgets/fitting/fit_controller.py` while preserving existing payload/history semantics.
- Added controller-dispatched fit execution (`fit.run.execute`) by moving `FittingControllerWidget.onRunFit` to call `ActionController` and retaining execution logic in `FittingControllerWidget._run_fit_impl` for behavior parity.
- Added macro-level dataset delegation to controller actions in `chisurf/macros/core_data.py` (`add_dataset`, `remove_datasets`, `group_datasets`) with `_from_controller` guards to avoid recursion while keeping direct macro/script entrypoints aligned with controller routing.
- Routed remaining dataset/fit lifecycle entrypoints in key experiment/plugin flows through `ActionController` (`dataset.add`, `dataset.remove`, `dataset.group`, `fit.add`) across `chisurf/gui/main.py`, `chisurf/gui/widgets/experiments/{widgets.py, rics.py, pch.py, tcspc/tcspc_tttr_reader_control_widget.py}`, `chisurf/plugins/chisurf/batch_analysis/wizard.py`, `chisurf/plugins/tttr/microtime_histogram/wizard.py`, `chisurf/plugins/fluorescence_decay/irf_estimator/__init__.py`, `chisurf/gui/widgets/wizard/fcs_merger/fcs_merger.py`, and `chisurf/plugins/fluorescence_decay/tr_anisotropy/wizard.py`.
- Added routed dataset ungroup support via `dataset.ungroup` in `chisurf/controllers/action_controller.py` and `chisurf/macros/core_data.py` (`ungroup_datasets`), with structured `dataset_ungroup` history and updated dataset-selector ungroup UI path in `chisurf/gui/widgets/experiments/widgets.py`.
- KISS simplification pass: removed redundant local fallback wrappers around controller dispatch in `chisurf/macros/core_data.py` (dataset delegation gates), `chisurf/gui/widgets/fitting/fit_controller.py` (`onRunFit`), and `chisurf/plugins/chisurf/batch_analysis/wizard.py` (`dummy_run_fit`) to keep a single explicit controller execution path.
- Added controller actions `fit.set_dataset` and `fit.run` in `chisurf/controllers/action_controller.py` and action-registry specs in `chisurf/runtime/actions.py`; `fit.run` now emits structured run lifecycle events and `fit.set_dataset` emits structured `fit_data_set` assignment history.
- Migrated batch-analysis execution flow in `chisurf/plugins/chisurf/batch_analysis/wizard.py` (`dummy_run_fit`) from string-command fit assignment/run to controller actions (`dataset.add`, `fit.set_dataset`, `fit.run`).
- Added command-driven action-catalog export in `chisurf/macros/core_fit.py` (`export_action_catalog`) with JSON/YAML output support, and registered `action_catalog_export` in `chisurf/runtime/actions.py` for structured tracking of export operations.
- Completed batch-analysis loaded-dataset execution routing in `chisurf/plugins/chisurf/batch_analysis/wizard.py` so both file-based and preloaded-dataset branches use controller actions (`fit.set_dataset`, `fit.run`) instead of direct fit assignment/run calls.
- Added controller/MCP execution support for catalog export by handling `action.catalog.export` in `chisurf/controllers/action_controller.py` (`_handle_action_catalog_export`), routing it to `core_fit.export_action_catalog`.
- Added MCP-ready experiment/loading control actions in `chisurf/controllers/action_controller.py` and `chisurf/runtime/actions.py`: `experiment.set`, `setup.select`, and `setup.params.set`, allowing controller-driven setup parameter updates (including nested dotted keys) for flows such as PDA time-window (`minimum_time_window_length`) and FCS noise-model configuration before dataset load.
- Added focused CPM regression coverage in `test/test_fcs_cpm_total_rate.py` to validate total-vs-per-detector count-rate metadata resolution.
- Updated package versioning to produce PEP 440-compatible, reproducible versions derived from git tags/metadata (dev snapshots use `.post0.devN`); Windows installer marks dev builds via name and `_dev` suffix.
- Added read-only steady-state anisotropy output (`r_ss`) to the TCSPC anisotropy widget, with lifetime-model path first and decay/data-based fallback when no lifetime spectrum is available (`chisurf/models/tcspc/widgets/anisotropy.py`, `AnisotropyWidget._compute_steady_state_anisotropy`).
- Updated direct `r_ss` fallback computation to subtract channel/background offsets before integration (uses per-channel metadata `bg_vv/bg_vh` when available, otherwise model/global background).
- Split anisotropy steady-state outputs into `r_SS,L` (lifetime-derived) and `r_SS,I` (intensity-derived) in the TCSPC anisotropy widget; added parameter-registry tooltip metadata for both (`chisurf/models/tcspc/widgets/anisotropy.py`, `chisurf/settings/constants/fitting_parameters.json`).
- Clarified anisotropy tooltip text for `r_SS,I` and `r_SS,L`, explicitly documenting that `r_SS,I` includes background correction (per-channel metadata or global fallback) while `r_SS,L` is lifetime-domain.
- Compact anisotropy widget layout: `r_SS,L` and `r_SS,I` now share a single row to reduce vertical space usage.
- Reworked anisotropy parameter/output row layout to grid-based placement so regular and output parameter widgets align with consistent column widths in the TCSPC anisotropy panel.
- Added central anisotropy integral utilities (`chisurf/fluorescence/anisotropy/integrals.py`) including Eq. 2.4-22 style `r_e` calculation and Perrin-corrected `G` estimation from total channel intensities (`tau`, `rho`, `r0`, `l1`, `l2`).
- Added anisotropy diagnostic outputs `VV_bg-corr` and `VH_bg-corr` (background-corrected integrated channel intensities) to the TCSPC anisotropy widget and parameter tooltip registry.
- Expanded `rS,I` / `rS,L` tooltip descriptions with explicit domain/formula details and correction pipeline notes.
- Switched intensity-domain anisotropy diagnostics from trapezoidal integration to direct channel summation (`sum(VV)`, `sum(VH)`) for `rS,I`, `VV_bg-corr`, and `VH_bg-corr` to match TTTR/count-based workflow.
- For dual-fit anisotropy groups (VV/VH stacks), `g`, `l1`, and `l2` are now linked by default from the first (top/VV) fit to downstream fits, matching the usual middle-click linking workflow (`chisurf/models/tcspc/anisotropy.py`, `Anisotropy.set_polarization_by_group_position`).
- Updated TCSPC `Generic` widget so `nPh(Bg)` and `nPh(Fl)` are read-only output parameters (fitting-parameter widgets) instead of plain line edits; added tooltip registry entries (`nPh_bg`, `nPh_fl`) for consistent lookup and reuse.
- Added anisotropy diagnostics UI refinements: compact `Diag` status line (10% green, 10-20% yellow, >20% red), hidden-by-default VV/VH diagnostic outputs with a short toggle button (`VV/VH diag`), and a new `show r(t)` button to plot corrected vs uncorrected time-resolved anisotropy decays.
- Updated anisotropy decay diagnostics to honor per-fit IRF timeshifts: in dual VV/VH fit groups, channel traces are automatically shifted to a common IRF reference (top VV fit) before corrected/uncorrected `r(t)` computation.
- Expanded anisotropy decay diagnostics with live controls for `g`, `l1`, `l2`, `BgVV`, `BgVH`, and VV/VH timeshifts (prefilled from active fit/group defaults), plus Reset and CSV export in the same dialog.
- Added fixed anisotropy diagnostic plot scaling (`r(t)` y-range 0.0..0.5) and an optional `link l1/l2` checkbox in the diagnostics dialog.
- Simplified anisotropy shift controls to a single relative channel shift (`dVH-VV`) instead of two independent shifts.
- Extended `show r(t)` diagnostics to plot four curves: data raw/corrected and model raw/corrected anisotropy decays, with CSV export containing both data and model traces.
- Diagnostic defaults now prefill `BgVV`/`BgVH` from fit/curve context; relative VV/VH shift is intentionally not auto-prefilled from fit timeshifts (user-controlled), and `link l1/l2` is enabled by default.
- Expanded focused history tests; verified passing:
  - `test/test_history.py`
  - `test/test_history_replay_fitlink.py`
  - `test/test_history_replay_state.py`

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

## Previous Releases

(Add previous release notes here if available)
