# ChiSurf TODO List

## High Priority

### AI Settings Migration
- [ ] Migrate ALL Chato API settings to centralized AI settings plugin
  - [x] Provider: Mistral only
  - [x] Base URL: https://api.mistral.ai/v1
  - [x] API Key
  - [x] Chat Model: mistral-small/medium/large
  - [x] Text Embedding: mistral-embed
  - [x] Code Embedding: codestral-embed
  - [x] Temperature, top_p, max_tokens
  - [ ] RAG Settings: Keep in Chato (not needed for now)
  - [ ] Remove all hardcoded AI settings from Chato config
  - [ ] Chato must import ALL AI settings from ai_settings module
  - [ ] Single source of truth: chisurf/settings/ai_settings.py
  - Location: chisurf/settings/ai_settings.py, chisurf/plugins/ai_settings/

### Phase 9 - MVC Action Core Completion ✅
- [COMPLETED] All GUI handlers migrated to use action controller
- [COMPLETED] Added missing actions: fit.load, fit.save, fit.save_all, fit.close_all
- [COMPLETED] Added comprehensive tests (21 total, +4 new)
- [COMPLETED] Updated documentation and status tracking

### Plugin Check System
- [BROKEN] Dangerous pattern detection in plugin check is too simplistic
  - Current pattern matching flags safe operations like `ast.literal_eval`
  - Need more sophisticated parsing to distinguish safe vs dangerous eval usage
  - Location: `chisurf/macros/plugin_check.py` lines 344-349 and 521-530
  - Status: Temporary fix applied (added space before eval) but needs proper solution

## Medium Priority

### Plots / UX
- Auto-enable group display for line plots of FitGroups
  - When a `LinePlot` is created for a `FitGroup` with more than one local fit, automatically check the "display group" checkbox and render all fits in the plot.
  - Touchpoints: `chisurf/plots/lineplot/lineplot.py` (`LinePlot.update`, `LinePlotControl.display_group`)

- Lineplot: reduce legend/metrics footprint for multi-fit groups
  - Use smaller fonts for legend and for the chi2 overlay to save space.
  - When multiple fits are displayed, show chi2 and Durbin-Watson in a lightweight table; highlight the current fit row in bold.
  - Touchpoints: `chisurf/plots/lineplot/lineplot.py` (legend + `DraggableTextItem` overlay rendering)

### Parameters / UX
- Output parameters should be visually distinct
  - Highlight output (read-only/result) parameters so they stand out from editable inputs.
  - Touchpoints: `chisurf/gui/widgets/fitting/parameter_widgets.py` (`FittingParameterWidget._update_role_visuals`), `chisurf/settings/settings_chisurf.yaml` (`parameter.role_color_output`).

### Architecture (MVC / GUI Decoupling)
- Audit all UI elements/controllers for MVC compliance
  - Identify direct model mutations in widgets/views and move them into controller/service layers
  - Define and document allowed UI-to-model interaction patterns
  - Add a phased refactor plan so core workflows can run without GUI dependencies
  - Goal: ChiSurf core should be operable headless (CLI/API-first), with GUI as an optional frontend
- [COMPLETED] Complete migration of remaining GUI handlers to use action controller
  - Migrated: onSetupChanged, onCloseAllFits, onLoadFit, onSaveFit, onSaveFits
  - Added missing actions: fit.load, fit.save, fit.save_all, fit.close_all, parameter.value, fit.update
  - Added model actions: model.add_component, model.remove_component, model.normalize_amplitudes, model.absolute_amplitudes
  - Added comprehensive tests (25 total, +8 new)
- [COMPLETED] Fix remaining chisurf.run() calls in model widgets
  - Fixed: lifetime.py (6/6 chisurf.run() calls migrated)
  - Added parameter, fit update, and model component actions
- [COMPLETED] Migrate plot interaction handlers to action controller
  - Fixed: residual_image.py (1/1 chisurf.run() calls migrated)
  - Fixed: lineplot.py (1/1 chisurf.run() calls migrated)
  - Added: fit.range_set action with service method
  - Total tests: 26 (+1 new for fit range)
- [COMPLETED] Continue migrating other model widgets (TCSPC, PDA, FCS)
  - Fixed: PDA widgets (15/15 chisurf.run() calls migrated)
  - Fixed: TCSPC widgets - anisotropy.py (2/2), convolve.py (4/4)
  - Migrated: parameter updates, fit updates, model component operations, IRF operations
  - Remaining: Other TCSPC widgets, FCS widgets, PCH widgets, parameter_scan.py
- [COMPLETED] Fix fit index issues in migrated widgets
  - Fixed: residual_image.py, lifetime.py, pda/widgets.py
  - Added helper function _get_fit_index_for_model() in pda/widgets.py
  - Ensured all action controller calls use correct fit index instead of hardcoded 0
  - Verified all tests still pass (26/26)
- [COMPLETED] Add comprehensive fit index tests
  - Added test_fit_index_handling_with_multiple_fits() - Tests different fit indices
  - Added test_fit_index_default_handling() - Tests default behavior
  - Added test_fit_index_validation() - Tests edge cases
  - Added test_fit_index_detection_logic() - Tests model-to-fit mapping
  - Added test_fit_index_edge_cases() - Tests type handling
  - Total tests: 31 (+5 new comprehensive fit index tests)
- [COMPLETED] Migrate TCSPC corrections widget
  - Fixed: corrections.py (7/7 chisurf.run() calls migrated)
  - Added new actions: model.set_correction, model.set_linearization, model.unload_lintable
  - Added comprehensive tests for correction operations
  - Total tests: 32 (+1 new for corrections)
- [COMPLETED] Migrate additional TCSPC widgets
  - Fixed: discrete_distance.py (2/2), gaussian.py (3/3), generic.py (2/2)
  - Added new action: model.unload_background_curve
  - Added comprehensive test for background curve operations
  - Total tests: 33 (+1 new for background curve)
- [COMPLETED] Migrate global model widget
  - Fixed: global_model/widget.py (4/4 chisurf.run() calls migrated)
  - Added new actions: model.remove_local_fit, model.clear_local_fits, model.append_global_parameter, model.append_fit
  - Added comprehensive tests for global model operations
  - Total tests: 33 (+1 new for global model)
- [COMPLETED] Migrate remaining TCSPC, FCS, PCH, and other model widgets
  - Completed: pch/widgets.py (2), parameter_scan.py (1)
  - All migrations complete: 55/55 calls (100% complete)

### History Replay (Phase 8)
- Full baseline + full-domain replay is still incomplete
  - Current replay covers navigation, parameter state, fit ranges, and setup state.
  - Remaining coverage should include broader scientific/domain object reconstruction from baseline + events.
  - Touchpoints: `chisurf/history_replay.py`, `chisurf/gui/main.py` (`_on_history_cursor_changed`).

- Dataset replay currently relies on dataset names for selection
  - UID metadata and UID-first selection are now implemented for current lifecycle events.
  - Remaining work: continue removing residual name-only fallbacks in replay apply paths where practical.
  - Touchpoints: `chisurf/history_replay.py` (`reconstruct_navigation_state`), history payload schemas.

- Parameter link replay still resolves by fit/local-fit/parameter names
  - UID-aware link payload fields and UID-first replay resolve are now implemented for current events.
  - Remaining work: continue reducing residual name-based fallback paths in replay resolution.
  - Touchpoints: `chisurf/history_replay.py` (`reconstruct_parameter_state`), `chisurf/gui/main.py` (`_apply_parameter_state`), link/unlink payload emitters in `parameter_widgets.py`.

- Fit replay still has name fallback for older events without UIDs
  - Fit UID tracking/selection is implemented for current events.
  - Remaining risk: old streams without fit UID fields may still select by name.
  - Touchpoints: `chisurf/history_replay.py` (`reconstruct_navigation_state`), `chisurf/gui/main.py` (`_select_fit_by_identity`).

### Unified UX (Single-Molecule)
- Unify selection UX for single-molecule setups and plugin settings
  - Standardize setup selection flow across relevant plugins
  - Align naming, defaults, and state persistence behavior between setup widgets and plugin dialogs
  - Reduce duplicated configuration controls by centralizing shared setup/settings components

### TCSPC / Anisotropy UI
- Add steady-state anisotropy output (`r_ss`) to the anisotropy widget (similar to CPM in FCS)
  - For models with lifetime spectrum, use model-provided steady-state anisotropy
  - If no lifetime spectrum is available, compute via decay-based fallback from anisotropy + data channels
  - Keep it read-only and auto-updating with parameter/model changes

- Human QA task: stabilize anisotropy and verify consistency
  - Perform careful manual validation of anisotropy behavior and outputs across representative datasets/workflows.
  - Ensure results are consistent/reproducible across repeated runs and UI interactions.
  - This is a human verification task (not an AI-automation task).

### Release Process / Versioning
- Define and document the release/maintenance workflow (max 1 major/year)
  - Use per-major maintenance branches (`25`, `26`, `27`, ...) plus `development` for next-major work
  - Define backport policy for old majors (cherry-pick minimal fixes)
  - Define version format (date-style, PEP 440 compatible) and tag naming
  - Add a release checklist (freeze, changelog, tag, build artifacts, publish)
  - Define dev build channeling (pip pre-releases / conda dev label) and how to flag dev installers

### Performance
- Speed up ChiSurf startup time without sacrificing early responsiveness
  - Profile import/initialization hotspots (module imports, plugin discovery, GUI construction)
  - Defer heavy initialization (lazy-load plugins/resources) and show UI early
  - Add a startup timing log and a simple performance regression check

### Documentation
- Update plugin development documentation with security guidelines
- Document plugin check system behavior and skip detection logic

### Licensing
- Determine and document ChiSurf licensing baseline
  - Research all third-party software/libraries used by ChiSurf and collect their licenses
  - Keep ChiSurf on the most permissive license allowed by dependency/license constraints
  - Create and maintain a central license inventory document with a software/license table
  - Track the minimal (most restrictive) effective license state from dependencies over time

### Testing
- Add comprehensive tests for plugin check dangerous pattern detection
- Test edge cases for safe eval patterns vs dangerous eval patterns

## Low Priority

### Code Cleanup
- Refactor plugin check to use AST parsing instead of string matching
- Consolidate duplicate dangerous pattern detection logic

---

## Notes
- Items marked as [BROKEN] need immediate attention
- Items marked as [FIXME] are known issues that need fixing
- Items marked as [TODO] are planned improvements
