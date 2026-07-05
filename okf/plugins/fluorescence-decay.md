---
type: Plugin Group
title: Fluorescence decay
description: TCSPC fluorescence-lifetime plugins for decay analysis, blind IRF estimation, maximum-entropy lifetime/distance distributions and time-resolved anisotropy.
resource: chisurf/plugins/fluorescence_decay/
tags: [plugins, tcspc]
timestamp: '2026-07-05T00:00:00Z'
---

The `fluorescence_decay/` group analyses Time-Correlated Single Photon Counting (TCSPC) microtime histograms: fitting lifetime decays, recovering the instrument response function (IRF), extracting lifetime or FRET-distance distributions, and resolving rotational anisotropy. These tools depend on a correct IRF and instrument corrections, so estimation and calibration steps sit alongside the fitting steps.

| Plugin dir | Display name | What it does |
|---|---|---|
| `lifetime_analysis` | Decay Analysis | Integrated lifetime-analysis window bundling IRF estimation, MaxEnt MEM, Lazy Lifetime Analysis, microtime histograms and Jordi G-factor calibration. |
| `irf_estimator` | IRF Estimation | Blind IRF estimation from decay data via truncated-exponential fitting and Richardson-Lucy deconvolution. |
| `maxent_decay` | MaxEnt MEM | Maximum-entropy analysis of TCSPC decays for lifetime and FRET-distance distributions. |
| `lltf` | Lazy Lifetime Analysis | Fast/low-setup lifetime estimation for TCSPC decay data. |
| `tr_anisotropy` | Anisotropy-Wizard | Guided setup of a linked VV/VH global time-resolved anisotropy fit: loads polarised decays, background-corrects IRFs, sets instrument corrections and defines lifetime/rotation spectra. |
| `jordi_g_factor` | Jordi G-Factor Calculator | Calculates detector G-factors from Jordi-format decay files and can archive calibration provenance through backend services. |
| `jordi_anisotropy` | Jordi Anisotropy Decay | Computes anisotropy decays from Jordi-format VV/VH data, including batch processing. |
| `tttr/microtime_histogram` | Histogram-Microtime | Creates TTTR micro-time histograms for decay inspection before model fitting. |

Each plugin is discovered through its `manifest.json` (`id`, `display_name`, `categories: [Spectroscopy, Fluorescence decay]`) by the infrastructure in `chisurf/core/plugin/`. They obtain decay datasets and register fits through `PluginContext` / `ChiSurfAPI` rather than the legacy globals, and their GUIs (including the `tr_anisotropy` wizard) are rendered from AutoForm view schemes. The integrated `lifetime_analysis` window builds on the shared `NavigationPanelTool` shell and reuses the standalone plugins as panels.

See also: [plugin system](/architecture/plugin-system.md), [Plugins target](/specs/plugins.md), [GUI & AutoForm](/subsystems/gui-autoform.md). Lifetime filters derived from these decays feed the [FCS group](fcs.md) (fFCS, 2D-FLCS) and per-burst lifetime estimation in the [burst group](burst.md).
