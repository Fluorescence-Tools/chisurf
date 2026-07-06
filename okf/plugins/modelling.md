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

## FRET docking engine (durable facts)

The `modelling/fret` engine keeps all computation in importable, GUI-free core
modules (`av.py`, `clash.py`, `distance.py`, `engine.py`, `docking.py`,
`screening.py`, `results.py`, `io.py`), with thin `cli/` (click) and `gui/`
(`wizard.py`) layers on top — the design intent is that the plugin is a strict
**superset of the legacy FPS + OLGA** tools. Load points to keep stable:

- **AV backends are selectable.** `av.py` computes accessible volumes via an
  external labelling library (primary) with a biophysical-modeling-framework
  fallback; `select_backend(name)` and an `--av-backend auto|…` flag pick between
  them, gated by availability flags. The fallback AV path is unavailable on
  Windows and must skip gracefully.
- **`fps.json` is the interchange format.** Never rename its keys (`Positions`,
  `Distances`, `χ²`); multi-body docking carries a `body_id` per position and
  evaluators serialize under an `Evaluators` key. The GUI `LabelStructure` widget
  is owned by `fps_json_editor` and imported by `fret/wizard.py`; they share one
  live Python dict via a `fps_json_payload` getter/setter (no file round-trip).
- **Evaluators** live in `fret/evaluators/` (positions, distance, fret_efficiency,
  chi2, residuals, geometry, av_metrics), all deriving from an `Evaluator` ABC
  with JSON round-trip and DataFrame/CSV export.
- **Informative pair selection** reuses `chisurf/plugins/traj/fret_pair_selection/
  olga_greedy.py` (`select_informative_pairs`) and emits OLGA-compatible reports.

The in-tree `TARGETS.md` and the draft "FRET superset" plan (refine/bootstrap,
sampling fix, OLGA evaluator graph) are captured as [PRD-58](/prds/prd-58.md); the
evaluator subpackage has already landed. The [ChiMOL viewer](/plugins/profiles/chimol.md)
roadmap is [PRD-57](/prds/prd-57.md).

## Idea: structure-aware MFDB sample definer

`fps_json_editor` already implements most of what a flrCIF-grade sample definer
needs — PDB loading, atom-level position picking, accessible-volume simulation,
and distance restraints. Its sub-widgets (`PositionPanel`, `MolView`, `AVWorker`)
are reuse candidates for a future `sample_definer` that produces MFDB
`SampleDefinition`/`ProbeDefinition`/`FretPairDefinition` records (a structure-aware
replacement for the flat sample picker), auto-filling AV parameters and
photophysics from the fluorophore database and computing R0 from spectral overlap.
The cleanest path composes those widgets into a new host rather than modifying
`fps_json_editor`; this feeds the sample-tracking work in
[PRD-02](/prds/prd-02.md), [PRD-06](/prds/prd-06.md) and [PRD-08](/prds/prd-08.md).
