---
type: Plugin Group
title: Modelling plugins
description: Structural and FRET-modelling tools — accessible-volume editing, restrained rigid-body docking, orientation-factor and hydrodynamics estimation, and FRET-line generation.
resource: chisurf/plugins/modelling/
tags: [plugins, modelling]
timestamp: '2026-07-05T00:00:00Z'
---

The modelling group turns fluorescence observables (FRET efficiencies, anisotropies,
diffusion coefficients) into and out of molecular structure. Most tools live under
`chisurf/plugins/modelling/`; two closely related FRET-modelling plugins
(`fret_line`, `kappa2_dist`) sit at the plugin root but belong to the same domain and
are re-exposed through the `structure_tools` toolbox.

| Plugin dir | Display name | What it does |
| --- | --- | --- |
| `modelling/fret` | Structure:FRET:Docking & Screening | FRET-restrained rigid-body docking, refinement, structure-library screening and error estimation; a thin shim around an external molecular-modeling framework and its accessible-volume/FRET extension. |
| `modelling/fps_json_editor` | Structure:FRET:FPS JSON Editor | Edit `fps.json` files that define FRET accessible-volume dye models; fetch reference structures by PDB ID. |
| `modelling/hydropro` | Structure:Computation:HydroPro | Graphical front-end to an external hydrodynamics tool, computing properties such as the translational diffusion coefficient from atomic or bead-model structures. |
| `modelling/proteinmc` | (no manifest) | Protein Monte-Carlo helpers; a first-class module still lacking a `manifest.json` (see steering note). |
| `modelling/structure_tools` | Structure:Structure Tools | Unified toolbox aggregating FPS JSON Editor, FRET Docking, Kappa2 Distribution, QuEst, HydroPro and Trajectory Tools into one entry point. |
| `fret_line` | Spectroscopy:FRET:FRET Line Generator | Compute static, dynamic, WLC and mixture FRET lines for overlay on smFRET 2D histograms in ndxplorer. |
| `kappa2_dist` | Structure:FRET:Kappa2 Distribution | Compute and visualise the κ² orientation-factor distribution (WIC, DWT, isotropic models). |

Integration follows the standard plugin contract ([plugin system](/architecture/plugin-system.md),
[Plugins target](/specs/plugins.md)): each plugin is discovered by its `manifest.json`,
activated with a `PluginContext` that exposes the API facade for datasets/fits, and
renders its panels declaratively through [GUI & AutoForm](/subsystems/gui-autoform.md).
`fret_docking` persists FPS-style pose vectors and reloads projects to continue a run.
Note `modelling/proteinmc` currently has no manifest — a documented gap against the
one-manifest-identity rule in [Plugins target](/specs/plugins.md).
