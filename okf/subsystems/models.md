---
type: Subsystem
title: Fitting Models
description: Maps fitting parameters to predicted curves across fluorescence/EPR method families, with each model's editable structure declared as data (view.json) and rendered by AutoForm.
resource: chisurf/core/models/
tags: [core, models, tcspc, fcs, fret]
timestamp: '2026-07-05T00:00:00Z'
---

# The Model abstraction

`Model` (`chisurf/core/models/model.py`) is an abstract subclass of
`FittingParameterGroup` — a model *is* a tree of parameters (see
[parameters](/subsystems/parameters.md)). Subclasses implement
`update_model()` to read experimental data from `self.fit`, evaluate the
functional form, and write the predicted curve into `self.y`. `ModelCurve`
mixes `Model` with `chisurf.core.curve.Curve` so a model exposes `x`/`y`
arrays that the [fitting engine](/subsystems/fitting.md) differences against
the data. `update()` refreshes all nested parameter groups, then calls
`update_model()`; `get_state`/`set_state` snapshot a model for project save.

# Model families

| Subpackage | Fits |
| --- | --- |
| `tcspc/` (`lifetime.py`, `fret.py`, `maxent.py`, `pddem.py`, `mix_model.py`, `av_decay.py`, `fret_structure.py`) | TCSPC decays: multi-exponential lifetimes, FRET (Gaussian/rate/worm-like-chain/single-distance), MaxEnt lifetime & FRET, PDDEM, convolution+anisotropy |
| `fcs/` (`parse.py`, `maxent.py`, `models.yaml`) | FCS correlation curves — equation-preset diffusion/bunching models + MaxEnt |
| `rics/rics.py` | RICS/ICS image-correlation (simple, triplet, immobile, flow, full, 2D Gaussian) |
| `pch/` | Photon-counting-histogram multi-component |
| `pda/` (`simple.py`, `dynamic.py`, `dynamic_mc.py`, `pdagauss.py`, `anisotropy.py`) | Photon-distribution-analysis (static/dynamic state mixtures, Gaussian distance, anisotropy) |
| `deer/deer.py` | DEER/PELDOR dipolar traces → P(r) via Gaussian, Rice, Tikhonov, MaxEnt |
| `structure/` (`proteinmc.py`, `rmf.py`) | Structure/Monte-Carlo modelling |
| `stopped_flow/` | Reaction-kinetics traces |
| `parse/parse.py` | Generic user-typed equation parser (backs the FCS/TCSPC presets) |
| `parameter_transform/`, `global_model/globalfit.py` | Parameter transforms; joint [global fit](/subsystems/fitting.md) over many `Fit`s |

`ParseModel` scans an equation string, auto-creates a `FittingParameter` per
free symbol, and evals it — presets live in `tcspc/tcspc.models.json` and
`fcs/models.yaml`.

# Registration / discovery

Models are not auto-scanned: each experiment lists its model classes by dotted
path in `chisurf/core/settings/experiment_configs.yaml` (per `tcspc`, `fcs`,
`rics`, `pch`, `pda`, `deer`, `structure`, `global`). At startup
`chisurf/gui/main_helper.py` imports each path and calls
`Experiment.add_model_class` (`chisurf/core/experiments/core/experiment.py`);
`Experiment.model_names` then feeds the add-fit UI. A `Fit` instantiates the
chosen class as `model_class(self, **model_kw)`. Users can also drop
`*__override__*.py` files into their settings `models/` dir
(`inject_user_models`) or wrap a callable via `function_to_model_decorator`
(chinet-node backed).

# Data-described editor (PRD-38 / PRD-40)

`Model.view_spec()` returns a UI-agnostic `ModelView` (sections + plot specs,
no Qt). If a class sets `view_spec_file` (e.g. `lifetime.view.json`,
`deer/deer_gauss.view.json`, the `rics/*.view.json` and `pda/*.view.json`), it
loads that hand-editable JSON via `chisurf.core.dataspec`
(`view_spec.py` is a back-compat shim); otherwise it auto-derives one section
per nested parameter group plus a standard plot set. The
[GUI AutoForm renderer](/subsystems/gui-autoform.md) turns the `ModelView`
into the editor, keeping compute (`.py`) and layout (`.view.json`) separate —
see [PRD-38](/prds/prd-38.md) and [PRD-40](/prds/prd-40.md).

See [Core](/subsystems/core.md) and the [Core target](/specs/core.md).
