---
type: Subsystem
title: Fitting Engine
description: Binds models to datasets over a range, computes weighted residuals and chi², and drives local/global least-squares optimization plus error analysis and sampling.
resource: chisurf/core/fitting/
tags: [core, fitting, optimization, global-analysis]
timestamp: '2026-07-05T00:00:00Z'
---

# What a Fit is

A `Fit` (`chisurf/core/fitting/fit.py`) pairs one `DataCurve` with one
`ModelCurve` and restricts the comparison to an index range `[xmin, xmax]`
(optionally narrowed by an arbitrary 1D `mask`/weight array). See
[data model](/subsystems/data-model.md), [models](/subsystems/models.md),
and [parameters](/subsystems/parameters.md).

| Object | Role |
| --- | --- |
| `Fit` | Single dataset ↔ single model over a range |
| `FitGroup(Fit)` | Many `Fit`s sharing a global model (`GlobalFitModel`) |
| `FittingParameter` | Free/fixed/linked model parameter (`fitting/parameter.py`) |
| `sample_fit` / `sample.py` | MCMC / emcee parameter sampling |
| `support_plane.py` | chi² scans + F-test confidence intervals |

# Fitting flow

1. Evaluate model → `model.update_model()` produces model y-values.
2. Weighted residuals `wres = (data − model) / data_error` over the range
   (`calculate_weighted_residuals` in `fitting/__init__.py`; masked via
   `_apply_fit_mask`).
3. `chi2 = Σ wres²`; `chi2r = chi2 / (n_points − n_free − 1)` (`get_chi2`).
4. `Fit.run()` minimizes `get_wres` with `leastsqbound`
   (`chisurf/core/math/optimization/leastsqbound.py`) over the model's free
   parameters and their bounds, then updates error estimates and pushes model
   state onto a bounded `results` deque.

`leastsqbound` wraps `scipy.optimize` MINPACK `leastsq` (Levenberg–Marquardt),
adding box bounds via an internal↔external parameter transform and a
cancellable `progress_callback` (`OptimizationCancelled`).

# Global analysis / parameter linking

- `FittingParameter.link` ties a parameter to another parameter (across fits);
  linked and `fixed` params are excluded from the free set (`n_free`), so a
  linked param is optimized once but shared everywhere.
- `FitGroup` builds a `GlobalFitModel`
  (`chisurf/core/models/global_model/globalfit.py`) whose
  `weighted_residuals` is the `np.concatenate` of member residuals and whose
  `parameters` is the deduplicated union of member free params — one global
  least-squares problem.
- `FitGroup.run()` optionally fits each member locally first
  (`global_optimize_local_first`) then runs the joint `leastsqbound`.

# Error analysis & sampling

| Method | Entry point | Basis |
| --- | --- | --- |
| Covariance | `covariance_matrix`, `update_error_estimates` | `scipy.linalg.pinvh` of the curvature from `approx_grad` finite differences |
| chi² scan | `Fit.chi2_scan` | brute scan of one parameter (`support_plane.scan_parameter`) |
| Support plane | `Fit.adaptive_chi2_scan` | adaptive scan to the `scipy.stats` F-test threshold → CI crossings |
| MCMC | `sample.walk_mcmc` | Metropolis on `lnprob` (`−0.5·chi2` + uniform-bounds `lnprior`) |
| Ensemble | `sample.sample_emcee`, `sample_fit` | `emcee.EnsembleSampler` over free params; chains saved to disk |

Per-parameter `error_estimate` records whether it came from `cov` or the
support plane (`sp`).

# Exposure

- Active fits live in the `chisurf.fits` list (indexed via `find_fit_idx`) — a
  legacy [runtime global](/architecture/runtime-globals.md).
- The [action layer](/architecture/action-layer.md)
  (`chisurf/core/actions/fit_actions.py`) mediates `run_fit`.
- The [API facade](/architecture/api-facade.md) (`chisurf/core/api/`) exposes
  `run_fit(fit_index/fit_uid)` for GUI, macros, plugins, and the server.

See [Core](/subsystems/core.md) and the [Core target](/specs/core.md).
