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
| `burst_h2mm` | H2MM | Photon-by-photon Hidden Markov Model — resolves sub-burst FRET-state dynamics on the microsecond scale via a self-contained numba engine, with BIC/ICL state selection and Viterbi dwell/transition analysis. |
| `burst_mle_analysis` | Burst MLE | Maximum-likelihood fluorescence-lifetime analysis of single-molecule bursts. |
| `burst_fcs_correlator` | Burst-wise FCS | Computes correlation functions per burst from Burst-ID (`.bst`)/BUR files. |
| `burst_browser` | Burst Browser | Inspects burstwise analysis tables and plots. |
| `bid_to_analysis` | BID→Analysis | Converts Seidel-style BID (Burst ID) files into a burstwise analysis folder (BUR/Info/MTI, optional HDF5/SL5) that ChiSurf and companion tools consume. |

`burst_h2mm` follows the client-server standard: a Qt-free numba engine (`core/h2mm.py` — cached `A^Δt` and transition-count `ρ` tensors, scaled forward-backward, Baum-Welch EM, Viterbi+ICL) under a database-free RPC service (`burst_h2mm.jobs.compute`) with a thin `H2mmClient` GUI, and it embeds as step 6 of the `burst_analysis` shell, inheriting the burst folder and channel definitions from the shared workflow context. The EM E-step is parallelised over bursts and defers the ρ contraction: rather than forming ξ per gap (`O(N·n⁴)`), it accumulates a per-unique-Δt weight `W[slot,k,m]` in `O(N·n²)` and contracts `ξ = Σ W·ρ` once per slot, so the engine runs faster than the reference `H2MM_C` (≈1.6× at 2 states, ≈3.6× at 4 states) while staying numerically equivalent.

Each plugin is discovered through its `manifest.json` (`id`, `display_name`, `categories: [Spectroscopy, Single-Molecule]`) by the plugin infrastructure in `chisurf/core/plugin/`; `bid_to_analysis` is a code-only helper without a manifest. Plugins receive datasets, fits and project state through `PluginContext` / `ChiSurfAPI` rather than the legacy process globals, and their GUIs are rendered from AutoForm view schemes. The integrated windows (`burst_analysis`) build on the shared `NavigationPanelTool` shell.

Planned work for `burst_background`: replace the current exponential-tail fit with
more robust statistical estimators (e.g. M-estimators) while staying single-threaded
CPU-only and low-latency, backed by broad edge-case unit tests (targeting a ~20%
accuracy improvement on public datasets).

See also: [plugin system](/architecture/plugin-system.md), [Plugins target](/specs/plugins.md), [GUI & AutoForm](/subsystems/gui-autoform.md). The `.bst`/BUR/SL5 formats interoperate with an established multiparameter-fluorescence suite; a companion photon-data exploration tool can feed and receive burst selections.
