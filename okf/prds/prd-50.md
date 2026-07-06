---
type: PRD
prd: "50"
title: "PRD-50: Photon Distribution Analysis (PDA) Family"
description: Wrap the existing PDA histogram engine in ChiSurf models and AutoForm view specs, covering static distance-distribution PDA, dynamic/N-state kinetic PDA, error surfaces, three-color PDA, and a kinetic consistency check.
status: draft
phase: "unassigned"
resource: chisurf/core/models/pda/
tags: [prd, fret]
timestamp: '2026-07-05T00:00:00Z'
---

# Summary
Photon Distribution Analysis fits the shot-noise-broadened FRET-efficiency histogram of single-molecule bursts to recover inter-dye distance distributions and, in its dynamic form, kinetic exchange between conformational states. The `tttrlib.Pda` C++ engine already computes the histograms, but no ChiSurf model, `view.json`, or plugin wraps it. This PRD adds the model + schema + AutoForm UI + fit integration in staged scope: static (single/multi-Gaussian, Lorentzian) PDA, dynamic/N-state kinetic PDA, Support-Plane/MCMC error surfaces, three-color PDA, and a kinetic consistency check. Dual-color models are already ported to the PRD-38 model/view-spec split with headless coverage; time-binned dynamic PDA, a 3-state variant, and PDA-specific error surfaces remain.

# Status
Draft / unassigned (STATUS TABLE authoritative). Dual-color static plus a dynamic two-state model done under AutoForm; several follow-ups (GUI light-path hook, time-binned dynamic PDA, 2D residual plot, PDA error surfaces) open.

Parent: [PRD-49](prd-49.md) (Phase 1, first target). Related: PRD-38
(model/view-spec split), PRD-40 (declarative editors), PRD-04 (burst pipeline),
PRD-28 (ndxplorer).

# Motivation

PDA quantitatively fits the shot-noise-broadened FRET-efficiency histogram of
single-molecule bursts to recover **inter-dye distance distributions** and, in its
dynamic form, **kinetic exchange between conformational states** — resolving whether
histogram width is shot noise, static heterogeneity, or dynamics. The incumbent
suite's PDA and three-color PDA apps are mature; ChiSurf has the compute engine but
no analysis surface.

**Current state.** `tttrlib.Pda` (C++) already computes PDA 1D/2D histograms given
per-channel background, an amplitude/probability spectrum `pF`, and `hist2d_nmin/nmax`
bounds — but **no ChiSurf model, `view.json`, or plugin wraps it**, so PDA is
unavailable at the application level (PRD-49 status: ENGINE-ONLY). This is the
highest-reuse gap: the math exists; we need the model+UI+fit integration.

# Progress (updated)

**Done (dual-color, AutoForm + view.json):**
- Ported the three existing PDA models to the PRD-38 model/view-spec split — pure
  models + `*.view.json`, registered directly in `experiment_configs.yaml`, rendered
  by `build_model_editor` → AutoForm (no hand-built Qt): `PdaSimpleModel`
  (`simple.view.json`), `PdaGaussianDistanceModel` (`pdagauss.view.json`),
  `PdaAnisotropyModel` (`anisotropy.view.json`).
- Added a **dynamic two-state PDA model** (`dynamic.py` +
  `PdaDynamicTwoStateModel` + `dynamic.view.json`): exact two-state occupation-time
  distribution (Bessel form), reduces to static two-Gaussian (slow) and single
  averaged population (fast). Registered in the config.
- Added **γ / α / δ** as read-only computed output parameters on `PdaFretNuisance`
  (`update_correction_factors`), plus a **light-path connection**:
  `PdaFretNuisance.apply_lightpath_matrices(...)` and
  `common.apply_lightpath_to_nuisance(...)` ingest the light-path simulator's
  `crosstalk_matrices` (excitation → ExDG/ExAG, emission → cGD/cGA/cRD/cRA).
- Qt-free distribution plot accessor `common.get_pda_distribution` (string-keyed
  axis) + generalized `resolve_distribution_options` to import dotted/`module:func`
  accessors, so the PDA E-histogram plot is authorable in JSON.
- Headless coverage: `test/gui/test_pda_model_editor.py` (renders + computes for all
  four models; dynamic-limit checks; correction-factor + light-path bridge tests).

**Follow-ups (not yet done):**
- GUI button wiring the live light-path plugin session to a selected PDA model
  (the pure bridge API is done and tested; only the one-click GUI hook remains).
- Full time-binned dynamic PDA (an N-vs-observation-time grid, the number-of-E-bins /
  number-of-time-bins scheme) and a 3-state variant — the current model uses a
  single dimensionless exchange parameter `K_ex`.
- Port the 2D S1S2 residual image plot (`Residual2DPlot`) to a registered view-spec
  plot key (currently only the 1D projection + residual panel are exposed).
- SPA/MCMC error surfaces wired specifically to PDA parameters.

# Scope (staged)

1. **Static distance-distribution PDA.** Model over `tttrlib.Pda`: single- and
   multi-Gaussian (and Lorentzian) P(R) → P(FRET) → shot-noise histogram; fit to a
   1D E (or S_g) histogram from a burst dataset with γ/crosstalk/direct-excitation/
   background corrections. Multi-file global fit.
2. **Dynamic / N-state kinetic PDA.** 2- and 3-state (extensible N-state) kinetic
   networks with exchange rates convolved over the burst-duration / time-window; fit
   rates + state distances/populations.
3. **Error surfaces.** Wire Support-Plane-Analysis, MCMC, and Hessian/covariance
   estimation to PDA parameters via `chisurf/core/fitting/sample.py`.
4. **Three-color tcPDA.** 1D/2D/3D three-color distance distributions, time-binned,
   with Bayesian priors — after (1)–(3) land.
5. **Kinetic consistency check.** Resample burst data from a fitted kinetic scheme and
   compare to the measured histogram (the incumbent's "consistency check"); reuses the
   dynamic-PDA simulator.

# Design

- **Compute core:** wrap `tttrlib.Pda` — do not reimplement histogram math. A thin
  `chisurf/core/fluorescence/pda/` module builds the `pF` spectrum from a P(R) model and
  drives `Pda.hist2d`/1D outputs; the fit objective compares model vs. measured
  histogram (MLE / χ²).
- **Model + schema:** `chisurf/core/models/pda/` model class(es) + `chisurf/core/dataspec/`
  schema, mirroring the structure of `chisurf/core/models/rics/` and `.../fcs/`.
- **UI = AutoForm + `view.json`** (PRD-49 AutoForm mandate): parameters (distances,
  widths, populations, corrections, kinetic rates) via `parameter_table`; histogram/fit
  overlay via an AutoForm plot section. No bespoke Qt. Add a section to
  `chisurf/gui/autoform/sections/` only if none fits (e.g. an overlay-histogram section),
  so other plugins reuse it.
- **Fit wiring:** register with the fitting stack so the model is selectable in the
  add-fit flow (the seam the `test-model-editor` skill exercises).
- **Data source:** consume burst tables / photon selections produced by
  `burst_selection` (E/S, corrections, PIE channels) and MFDB-registered datasets via
  `ChiSurfAPI` — not globals.

# Reuse

- `tttrlib.Pda` — histogram engine (constructor: `hist2d_nmax, hist2d_nmin,
  background_ch1, background_ch2, pF, implementation=PDA_DEFAULT|PDA_OPTIMIZED`).
- `chisurf/core/models/rics/`, `.../fcs/` — model + view.json templates.
- `chisurf/core/fitting/sample.py` — SPA / MCMC / covariance error surfaces.
- `chisurf/gui/autoform/` + `sections/` — declarative UI.
- `burst_selection` / `ndxplorer` — burst tables, corrections, E/S histograms.
- PRD-49 structural-FRET stack — P(R) priors can come from AV models
  (`fps_json_editor`, `fret_line`), a differentiator over the incumbent.

# Acceptance

- **Headless (primary):** a test builds the static PDA model, generates a synthetic
  2-Gaussian E-histogram via `tttrlib.Pda`, fits it, and asserts recovered mean
  distances/widths/populations within tolerance — runnable through the
  `test-model-editor` skill pattern; added under `test/` or `chisurf/core/models/pda/test/`.
- **Dynamic PDA:** synthetic 2-state exchange at known rate is recovered; static-only
  fit is rejected by F-test (`f_test` plugin).
- **Error surface:** SPA/MCMC produces a confidence interval bracketing the true value.
- **UI:** model appears in the add-fit combobox, renders parameter groups and the
  histogram-overlay from `view.json` with no empty groups or crashes (model-editor
  headless checks).
- **tcPDA (later stage):** synthetic three-color dataset recovers the three pairwise
  distances.

# Non-goals

- No new bespoke Qt widgets (PRD-49 AutoForm mandate).
- Not reimplementing PDA histogram math outside `tttrlib`.
- Antibunching/nsFCS and FCCS models are PRD-54, not here.

# Relationships
- Child of [PRD-49](prd-49.md) (Phase 1, first target).
- Wraps the `tttrlib.Pda` engine; consumes burst tables from the burst-selection pipeline.
- Renders via [GUI & AutoForm](/subsystems/gui-autoform.md); reads datasets through [Core target](/specs/core.md) rather than globals.
