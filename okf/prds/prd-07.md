---
type: PRD
prd: "07"
title: "PRD-07: Plugin MFDB Integration"
description: Have high-priority plugins register their results in MFDB via a single registration call at each output point.
status: superseded
phase: "unassigned"
resource: chisurf/plugins
tags: [prd, plugins]
timestamp: '2026-07-05T00:00:00Z'
---

# Summary
This PRD proposed adding a single `register_result()` call at the output point of each high-priority plugin (lifetime fitting, maximum entropy, anisotropy, the companion photon-data exploration tool, microtime histogram, FCS correlation, etc.) so their results land in MFDB with sample-id propagation and provenance edges, wrapped in try/except so plugins still work without MFDB. It defined priority tiers and a per-plugin recipe (find output point, register, propagate sample id, test).

# Status
Superseded by [PRD-16](prd-16.md), the strict, uniform transformer contract that replaces these ad-hoc per-plugin registration calls. Retained here as the record of the original per-plugin approach and its plugin inventory.

# Goal
High-priority plugins register their results in MFDB via the result registry
(PRD-03). Not a rewrite — just add one `register_result()` call at the output
point of each plugin.

# Principle
Every plugin integration follows the same pattern:
1. Find where the plugin produces its output (file, array, plot data).
2. Add `register_result()` after that point.
3. Wrap in try/except so the plugin works without MFDB.
4. Add `sample_id` propagation if the plugin has a GUI.

No other changes to the plugin.

# Priority tiers
- **Tier 1 — smFRET workflow (do first):** on the critical path for smFRET data
  processing.
- **Tier 2 — supporting analysis:** results worth archiving but off the critical
  path.
- **Tier 3 — future:** would benefit from MFDB but low priority.

## Tier 1 plugins
1. **burst_selection** — skip; covered in PRD-04.
2. **lltf** (Lazy Lifetime Fitting, `chisurf/plugins/fluorescence_decay/lltf/`) —
   produces lifetime fit results (amplitudes, lifetimes, chi-squared). At the fit
   output, call `register_fit_result(...)` with `chi2_reduced`, per-component
   lifetimes/amplitudes, and `parent_artifact_id` /`sample_id` from
   `fit.data.meta_data`.
3. **maxent_decay** (Maximum Entropy,
   `chisurf/plugins/fluorescence_decay/maxent_decay/`) — lifetime/FRET-distance
   distributions. `register_result(kind="fit_result", method="maximum_entropy",
   operation_type="local_fit", ...)`.
4. **tr_anisotropy** (`chisurf/plugins/fluorescence_decay/tr_anisotropy/`) —
   rotation correlation times, anisotropy params, g-factor.
   `register_result(kind="fit_result", operation_type="local_fit", ...)`; if
   g-factor is determined, also register it as a calibration ([PRD-05](prd-05.md)).
5. **ndxplorer** (MFD analysis, `chisurf/plugins/ndxplorer/`) — multi-parameter
   histograms, FRET-efficiency distributions, population selections.
   `register_result(kind="processed_data",
   operation_type="population_selection", ...)`.
6. **microtime_histogram** (`chisurf/plugins/tttr/microtime_histogram/`) — TCSPC
   decay histograms from TTTR files. `register_result(kind="processed_data",
   operation_type="histogram_construction", ...)`.
7. **fcs_correlator** (`chisurf/plugins/fcs/fcs_correlator/`) — correlation
   functions. `register_result(kind="correlation_data",
   operation_type="correlation", ...)`.
8. **jordi_g_factor** (`chisurf/plugins/jordi_g_factor/`) — covered in
   [PRD-05](prd-05.md); register g-factor as calibration.

## Tier 2 plugins
9. **pch** (Photon Counting Histogram, `chisurf/plugins/pch/`) — brightness, N;
   `register_fit_result(...)`.
10. **fcs_calculator** (`chisurf/plugins/fcs/fcs_calculator/`) — diffusion
    coefficient, hydrodynamic radius, effective volume;
    `register_result(kind="calibration_data",
    metadata={"calibration_type": "confocal_volume"}, ...)`.
11. **burst_bva** (Burst Variance Analysis, `chisurf/plugins/burst/burst_bva/`) —
    `E_mean`, `sigma_E`; `register_result(kind="processed_data",
    metadata={"analysis_type": "BVA"}, ...)`.
12. **intensity_trace** (`chisurf/plugins/tttr/intensity_trace/`) — HMM state
    sequences, dwell times; `register_result(kind="processed_data",
    metadata={"method": "HMM"}, ...)`.
13. **clsm** (`chisurf/plugins/microscopy/clsm/`) — image/ROI analysis;
    `register_result(kind="image_data", operation_type="image_analysis", ...)`.

## Tier 3 plugins (future)
Add `register_result()` when someone works on them: `fret_calculator`,
`kappa2_dist`, `fret` (modelling — AV distance distributions), `fcs_2d`,
`burst_mle_analysis`, `burst_fcs_correlator`, `psf_determination`,
`quenching_estimator`.

# Implementation guide (per plugin)
1. **Read the plugin.** Open the directory; read `__init__.py`; find the main
   calculation/output function.
2. **Find the output point.** A function that returns results, writes a file,
   updates a GUI widget (`setText`, `setData`), or emits a signal
   (`self.result_ready.emit`).
3. **Add the registration call**, right after the output point:
   `register_result(kind=..., data=..., sample_id=..., parent_artifact_id=...,
   operation_type=..., parameters={...}, metadata={...})` — valid kinds/types per
   the architecture docs — wrapped in try/except that logs at debug on failure.
4. **Propagate `sample_id`.** GUI: add a `SamplePicker` (PRD-02) and pass the
   selected id. Headless: add a `--sample-id` argument. Downstream plugin: read
   `sample_id` from the input data's metadata.
5. **Test.** Create a temporary MFDB, run the plugin's main function on test data,
   assert an artifact was registered and the provenance edge exists (if
   `parent_artifact_id` was set).

# Definition of Done
- [ ] All Tier 1 plugins (7) register results in MFDB
- [ ] All Tier 1 plugins have `sample_id` propagation
- [ ] All Tier 1 plugins have tests for MFDB registration
- [ ] At least 3 Tier 2 plugins register results in MFDB
- [ ] mfdb-admin shows results from all integrated plugins

# Relationships
- Superseded by [PRD-16](prd-16.md) (uniform transformer contract).
- Its registration model is formalized by [PRD-11](prd-11.md) (operation-node abstraction in MFDB).
- Touches the [plugin system](/architecture/plugin-system.md) and [Plugins target](/specs/plugins.md).
