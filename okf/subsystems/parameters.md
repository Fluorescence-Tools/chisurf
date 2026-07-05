---
type: Subsystem
title: Parameters
description: Scalar Parameter objects, bounds, the link dependency graph, and how free parameters become fitting degrees of freedom.
resource: chisurf/core/parameter.py
tags: [core, parameters, fitting, dependency-graph]
timestamp: '2026-07-05T00:00:00Z'
---

# Parameter

`Parameter` (`chisurf/core/parameter.py`) is a single scalar backed by a
low-level `chinet.Port`. Its `value` can come from three sources, checked in
order: a **link** to another parameter, a Python **callable** (dynamic value), or
the port's stored scalar. Reads clamp to bounds when enabled; writes are ignored
while a callable is the source of truth. `Parameter` supports arithmetic
(`+ - * / ** %`, `__float__`) so expressions read naturally.

| Attribute | Meaning |
|-----------|---------|
| `value` | current scalar (link/callable/port) |
| `bounds`, `bounds_on` | `(lb, ub)` tuple + enforcement flag (on the port) |
| `fixed` | frozen during optimization |
| `link` / `is_linked` | follows another parameter's port |
| `is_link_master` | UI-only hint: this parameter is a link target |
| `controller` | attached GUI widget, if any |

# Links and the dependency graph

Setting `p.link = q` makes `p` a follower whose value tracks `q`. This adds a
directed edge in a shared graph maintained by `chinet` at the `Port` level. The
graph is kept acyclic: `chinet.Port.would_create_cycle` (Kahn's algorithm)
rejects links that would form a cycle, surfaced by the side-effect-free predicate
`Parameter.check_recursive_link(current, target)` and enforced in the `link`
setter (raising `ValueError` on a recursive link). Passing `link = None` unlinks.
Because links live in the port graph, changing a master propagates to all
followers without extra bookkeeping.

# Groups

`ParameterGroup` (`parameter.py`) is a Base-backed collection; attribute writes
matching a contained parameter's name are routed to that parameter's `value`
setter, and reads return the scalar. `FittingParameterGroup`
(`chisurf/core/fitting/parameter.py`) is the group models subclass — it
distinguishes:

- `parameters_all` — every parameter (fixed, linked, free).
- `parameters` — **free** parameters only (`not fixed and not is_linked`).
- `parameter_values` / `parameter_bounds` — vectors over the free set.
- `aggregated_parameters` — nested `FittingParameterGroup`s discovered by
  `find_parameters`, enabling hierarchical models.

# Fitting parameters and degrees of freedom

`FittingParameter(Parameter)` adds fit-specific state: `error_estimate`
(covariance or support-plane), `parameter_scan`/`scan_result` (χ² scans via
`scan` / `adaptive_scan`), and `fit_idx` (which fit uses it, via
`find_fit_idx_of_parameter`). The set of **free** parameters is exactly the
optimizer's degrees of freedom: `Fit.run` (`chisurf/core/fitting/fit.py`) calls
`model.find_parameters(...)` then `leastsqbound(get_wres, model.parameter_values,
bounds=model.parameter_bounds, ...)`. Fixing a parameter or linking it to a
master removes it from `parameters` and therefore from the fit, without deleting
it — the model still reads its value. This is how global fits share one degree of
freedom across datasets.

See [fitting](/subsystems/fitting.md), [models](/subsystems/models.md), the
[data model](/subsystems/data-model.md), and the [action layer](/architecture/action-layer.md)
that mediates parameter edits. Parameters reach GUIs/plugins through the
[API facade](/architecture/api-facade.md), not the
[runtime globals](/architecture/runtime-globals.md).
