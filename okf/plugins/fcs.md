---
type: Plugin Group
title: Correlation (FCS)
description: Fluorescence Correlation Spectroscopy plugins that turn TTTR photon streams into correlation curves and derive diffusion, lifetime-filtered and 2D-lifetime observables.
resource: chisurf/plugins/fcs/
tags: [plugins, fcs]
timestamp: '2026-07-05T00:00:00Z'
---

The `fcs/` group processes raw photon data into correlation curves and the quantities derived from them — diffusion coefficients, concentrations, lifetime-filtered species and exchange dynamics. Fluorescence Correlation Spectroscopy (FCS) studies molecular dynamics, diffusion and interactions from intensity fluctuations in a confocal volume.

| Plugin dir | Display name | What it does |
|---|---|---|
| `fcs_correlator` | FCS Correlator | Wizard that selects/filters TTTR photons, computes correlation functions and merges curves; exposes correlator, filter and merger panels (each an AutoForm `*.view.json`). |
| `fcs_filter_calculator` | FCS Filter Calculator | Computes filtered-FCS (fFCS) lifetime filters from microtime decay patterns. |
| `flc_2d` | 2D-FLCS | Two-dimensional fluorescence lifetime correlation spectroscopy — builds 2D decay-correlation maps to resolve lifetime species and exchange dynamics (see [flc-2d memory](fluorescence-decay.md)). |
| `fcs_merger` | FCS-Merger | Merges/averages multiple FCS correlation curves to improve signal-to-noise. |
| `fcs_calculator` | Diffusion/Volume Calculator | Confocal FCS calculator for tau, D, hydrodynamic radius, effective volume and concentration. |
| `fcs_channel_preset` | FCS Definitions | Defines FCS detector channels per setup (a `Setup` category plugin). |
| `fcs_toolbox` | FCS Tools | Unified FCS toolbox built on the shared `NavigationPanelTool` shell, aggregating the panels above. |
| `fcs_convert` | (CLI) | Click CLI converting FCS correlation files between formats (ALV, ConfoCor3, Kristine, PyCorrFit, `.sin`, CSV, YAML, …). |

Plugins are discovered via `manifest.json` (`id`, `display_name`, `categories: [Spectroscopy, Fluorescence Correlation Spectroscopy]`) by `chisurf/core/plugin/`; `fcs_toolbox` and `fcs_convert` are code/CLI helpers without their own manifest. They read datasets and write fits through `PluginContext` / `ChiSurfAPI`, and their GUIs are AutoForm view schemes. Conversion I/O is shared with `chisurf.core.fio.fluorescence.fcs`.

See also: [plugin system](/architecture/plugin-system.md), [Plugins target](/specs/plugins.md), [GUI & AutoForm](/subsystems/gui-autoform.md). Per-burst FCS lives in the [burst group](burst.md). Formats interoperate with established FCS/multiparameter-fluorescence suites.
