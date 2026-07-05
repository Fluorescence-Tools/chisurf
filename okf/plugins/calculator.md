---
type: Plugin Group
title: Calculator plugins
description: Small physical-quantity estimators — FRET and phasor calculators — gathered under a single calculators hub, plus the κ² orientation-factor calculator.
resource: chisurf/plugins/calculator/
tags: [plugins, calculators]
timestamp: '2026-07-05T00:00:00Z'
---

Calculators are lightweight, mostly-stateless tools that turn a handful of inputs into
a derived fluorescence quantity — no dataset or fit required. They live under
`chisurf/plugins/calculator/`, and a hub plugin groups them (together with the FCS and
FRET-line calculators from other groups) into one two-panel launcher.

| Plugin dir | Display name | What it does |
| --- | --- | --- |
| `calculator/hub` | Main:Tools:Calculators | Hub that groups the FRET-line, FRET/homoFRET, FCS and phasor-plot calculators and embeds the selected one in a two-panel view. |
| `calculator/fret_calculator` | Main:Tools:FRET-Calculator | Combined heteroFRET and homoFRET parameter calculator (R₀, E, distances, anisotropy). |
| `calculator/phasor_calculator` | Main:Tools:Phasor-Calculator | Interactive phasor plot: universal semicircle with reference-lifetime grid, a FRET trajectory and a two-component mixing line; declarative AutoForm view. |
| `kappa2_dist` | Structure:FRET:Kappa2 Distribution | Compute/visualise the κ² orientation-factor distribution (WIC, DWT, isotropic); shared with the modelling group. |

The hub demonstrates the plugin composition pattern: it discovers sibling calculators
via their `manifest.json` and embeds each one's panel rather than reimplementing them.
The phasor calculator is a fully declarative tool — its whole UI is a `view.json`
rendered by [GUI & AutoForm](/subsystems/gui-autoform.md), a good reference for the
"declarative UI where possible" principle in [Plugins target](/specs/plugins.md).
All calculators are discovered and activated the same way as any other plugin
([plugin system](/architecture/plugin-system.md)), receiving a `PluginContext` for the
rare cases they read session state; most are pure functions of their form inputs.
Related domain: the phasor calculator shares math with imaging [Phasor-FLIM](imaging.md),
and the κ² calculator feeds FRET [modelling](modelling.md).
