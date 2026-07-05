---
type: Plugin Group
title: Burst analysis
description: A suite of single-molecule FRET burst-analysis plugins that select photon bursts from TTTR streams and derive per-burst FRET, variance, lifetime and correlation observables.
resource: chisurf/plugins/burst/
tags: [plugins, smfret]
timestamp: '2026-07-05T00:00:00Z'
---

The `burst/` group covers the confocal single-molecule FRET (smFRET) burst workflow: bursts are selected from Time-Tagged Time-Resolved (TTTR) photon streams, background is estimated, and each burst yields FRET, variance, lifetime and correlation observables that a browser inspects. This pipeline is the subject of [PRD-04](/prds/prd-04.md) (burst pipeline).

| Plugin dir | Display name | What it does |
|---|---|---|
| `burst_analysis` | Burst Analysis | Integrated burst workflow bundling selection, BVA, MLE, browser and background estimation in one navigation-shell window. |
| `burst_selection` | Burst Selection | Selects photon bursts from TTTR data and computes per-burst FRET indicators. |
| `burst_background` | Burst Background Estimation | Estimates detector background rates from TTTR burst data. |
| `burst_bva` | BVA | Burst Variance Analysis — tests whether FRET-efficiency spread exceeds shot-noise (dynamics detection). |
| `burst_mle_analysis` | Burst MLE | Maximum-likelihood fluorescence-lifetime analysis of single-molecule bursts. |
| `burst_fcs_correlator` | Burst-wise FCS | Computes correlation functions per burst from Burst-ID (`.bst`)/BUR files. |
| `burst_browser` | Burst Browser | Inspects burstwise analysis tables and plots. |
| `bid_to_analysis` | BID→Analysis | Converts Seidel-style BID (Burst ID) files into a burstwise analysis folder (BUR/Info/MTI, optional HDF5/SL5) that ChiSurf and companion tools consume. |

Each plugin is discovered through its `manifest.json` (`id`, `display_name`, `categories: [Spectroscopy, Single-Molecule]`) by the plugin infrastructure in `chisurf/core/plugin/`; `bid_to_analysis` is a code-only helper without a manifest. Plugins receive datasets, fits and project state through `PluginContext` / `ChiSurfAPI` rather than the legacy process globals, and their GUIs are rendered from AutoForm view schemes. The integrated windows (`burst_analysis`) build on the shared `NavigationPanelTool` shell.

See also: [plugin system](/architecture/plugin-system.md), [Plugins target](/specs/plugins.md), [GUI & AutoForm](/subsystems/gui-autoform.md). The `.bst`/BUR/SL5 formats interoperate with an established multiparameter-fluorescence suite; a companion photon-data exploration tool can feed and receive burst selections.
