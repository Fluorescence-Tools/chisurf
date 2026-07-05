---
type: Subsystem
title: Data model
description: How ChiSurf represents experimental datasets — the Base/Curve/DataCurve hierarchy, data groups, and per-technique data types.
resource: chisurf/core/data.py
tags: [core, data-model, datasets, curves, experiments]
timestamp: '2026-07-05T00:00:00Z'
---

# Object hierarchy

Every domain object descends from `Base` (`chisurf/core/base.py`). `Base`
carries a `name`, a `meta_data` dict, a `unique_identifier` (UUID), and
`to_dict`/`from_dict` (de)serialization. Live instances are tracked in a
process-wide `Base._uuid_index`, so any object can be looked up by its UUID.
`Data(Base)` adds file binding (`filename`, `embed_data`, `read_file_size_limit`).

Numeric data build on the curve stack in `chisurf/core/curve.py`:

| Class | File | Storage / role |
|-------|------|----------------|
| `NCurve` | `curve.py` | one 1-D NumPy array `d` |
| `Curve` | `curve.py` | 2×N array, `d[0]=x`, `d[1]=y`; arithmetic, `normalize`, `<<` shift |
| `CurveGroup` | `curve.py` | plain list of curves |
| `ExperimentalData` | `data.py` | `Data` + `data_reader` + `experiment` links |
| `DataCurve` | `data.py` | `Curve` + `ExperimentalData` + errors/mask |

# DataCurve — the fittable dataset

`DataCurve` (`chisurf/core/data.py`) is the workhorse experimental dataset:
`x`, `y` plus x/y error arrays `ex`, `ey` and a fit `mask`. Its `data` property
stacks these as a 5-row array `(x, y, ex, ey, mask)`. Weights are `1/ey`
(`set_weights`). Flattened N-D data (2-D histograms, images) describe their grid
via `meta_data['grid']` (`ndim`, `shape`, `order`, optional row/col indices) so
GUI range/residual tools can rebuild logical coordinates without technique-specific
knowledge.

# Data groups

Multiple datasets are held in `DataGroup(list, Base)` — a list that also tracks a
`current_dataset`. Specializations: `DataCurveGroup` (proxies `x/y/ex/ey/mask` to
the current curve), `ExperimentDataGroup` (adds `experiment`/`setup`), and
`ExperimentDataCurveGroup` (both). `get_data(...)` filters a dataset list down to
`ExperimentalData`/`ExperimentDataGroup` instances, excluding e.g. `"Global-fit"`.

# Experiments and per-technique types

An `Experiment` (`chisurf/core/experiments/core/experiment.py`) is a registry
binding model classes to `ExperimentReader`s (`.../core/reader.py`). Experiments
are loaded at import from `experiment_configs.yaml` into
`chisurf.core.experiments.types` (`chisurf/core/experiments/__init__.py`). Readers
turn files into the data types above:

- **TCSPC** — `experiments/tcspc/reader.py` yields `DataCurve` decay histograms;
  `tttr_reader.py`/`simulator.py` produce them from TTTR/tttrlib photon streams.
- **FCS** — `experiments/fcs/reader.py` yields an `ExperimentDataCurveGroup` of
  correlation curves.
- **smFRET / PDA / PCH / DEER / RICS** — `experiments/{pda,pch,deer,rics}/`
  readers produce technique-specific `DataCurve`(-group)s consumed by matching
  models.

# Dataset flow

Imported datasets live in the legacy global `chisurf.imported_datasets:
List[DataGroup]` (`chisurf/__init__.py`). New code should reach datasets, fits,
and project state through the [API facade](/architecture/api-facade.md) /
`PluginContext` rather than these [runtime globals](/architecture/runtime-globals.md).
Fitting consumes a `DataCurve` alongside a model — see [fitting](/subsystems/fitting.md),
[models](/subsystems/models.md), and [parameters](/subsystems/parameters.md).
Provenance and metadata about datasets are recorded in the
[MFDB store](/architecture/mfdb.md). See also [Core](/subsystems/core.md) and the
[Core target](/specs/core.md).
