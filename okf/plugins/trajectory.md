---
type: Plugin Group
title: Trajectory tools
description: Molecular-dynamics trajectory utilities for alignment, conversion, joining, energy, FRET, clashes, transforms, and topology export.
resource: chisurf/plugins/traj/
tags: [plugins, structure, trajectory, molecular-dynamics]
timestamp: '2026-07-05T00:00:00Z'
---

The `traj/` group handles molecular-dynamics trajectory files and structure
series. It is part of the structural/modelling side of ChiSurf and complements
the [modelling plugins](/plugins/modelling.md) plus the
[structure model family](/subsystems/models.md).

| Plugin dir | Display name | What it does |
| --- | --- | --- |
| `traj_tools` | Structure:Structure:Traj Tools | Combined dockable workspace that hosts the individual trajectory utilities. |
| `traj_align` | Structure:Trajectory:Align | Aligns trajectories to a reference frame or structure. |
| `traj_convert` | Structure:Trajectory:Convert | Converts trajectory files between supported formats. |
| `traj_join` | Structure:Trajectory:Join | Joins or stacks trajectories. |
| `traj_rotate_translate` | Structure:Trajectory:Rotate/Translate | Applies rigid-body rotation and translation. |
| `traj_save_topology` | Structure:Trajectory:Save Topol | Saves topology or first-frame structure files. |
| `traj_remove_clashes` | Structure:Trajectory:Remove Clashed | Removes frames with steric clashes. |
| `potential_energy` | Structure:Trajectory:Energy Calculator | Computes potential-energy components for structures and trajectories. |
| `traj_energy` | Structure:Trajectory:Trajectory Energy | Calculates and analyzes trajectory energy time series. |
| `fret_trajectory` | Structure:Trajectory:FRET | Calculates FRET observables from molecular-dynamics trajectories. |

Most entries are thin Qt widgets with manifests; `traj_tools` is the
aggregation shell. Keep reusable structure/trajectory math outside widgets when
expanding this group so it can be called from scripts, services, and tests.

See also [modelling plugins](/plugins/modelling.md), [compiled modules](/subsystems/compiled-modules.md),
and [plugin system](/architecture/plugin-system.md).
