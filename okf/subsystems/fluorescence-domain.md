---
type: Subsystem
title: Fluorescence Domain
description: Shared fluorescence algorithms used by fitting models, plugins, readers, and imaging tools.
resource: chisurf/core/fluorescence/
tags: [core, fluorescence, tcspc, fcs, fret, imaging]
timestamp: '2026-07-05T00:00:00Z'
---

# Scope

`chisurf/core/fluorescence/` is the Qt-free scientific kernel for fluorescence
calculations. It sits below the [models](/subsystems/models.md), plugin GUIs,
and [data IO](/subsystems/data-io.md): plugins collect inputs and render
results, while this package implements the numerical operations.

| Subpackage | Role |
| --- | --- |
| `tcspc/` | Decay convolution, pile-up/background/correction helpers, IRF estimation, and phasor calculations. |
| `fcs/` | Correlation, channel setup handling, fFCS filters, normalization, and curve merging. |
| `fret/` | Forster/FRET utilities and FRET-line generation for lifetime/E overlays. |
| `burst/` | Burst statistics, BVA, background, count-rate and change-point helpers. |
| `anisotropy/` | Anisotropy decays/integrals and orientation-factor calculations. |
| `imaging/` | Shared pixel-map helpers for intensity, Number & Brightness, micro-time histograms, and phasor maps. |
| `curation/` | Fluorescence curation helpers, including AI-assisted triage. |

# Reuse pattern

The preferred boundary is: keep numerical code here or in a small plugin
`core/`/`api/` package; keep Qt widgets in plugin `gui/`; expose server-friendly
operations through manifest `services` when long-running or scriptable. This is
the same split used by the [TTTR plugins](/plugins/tttr.md), [imaging plugins](/plugins/imaging.md),
[calculator plugins](/plugins/calculator.md), and [FCS plugins](/plugins/fcs.md).

# Examples

- `fret.fret_line.FRETLineGenerator` sweeps TCSPC FRET model parameters and
  produces FRET-efficiency/lifetime lines reused by the FRET-line plugin.
- `tcspc.phasor` implements phasor coordinates used by calculators and
  per-pixel FLIM tools.
- `imaging.pixel_maps` caches expensive TTTR/CLSM loads and writes standard
  imaging HDF5 tables consumable by downstream analysis.

See also [fitting](/subsystems/fitting.md), [data model](/subsystems/data-model.md),
and [plugin system](/architecture/plugin-system.md).
