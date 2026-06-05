"""

"""
from __future__ import annotations
from typing import Dict

import emcee
import numpy as np

import chisurf
import chisurf.core.fitting


def walk_mcmc(
        fit: chisurf.core.fitting.fit.Fit,
        steps: int,
        step_size: float,
        temp: float = 1.0,
        thin: int = 1,
        chi2max: float = np.inf,
        callback: typing.Callable = None,
        check_cancel: typing.Callable = None
) -> Dict:
    """

    :param fit:
    :param steps:
    :param step_size:
    :param chi2max:
    :param temp:
    :param thin:
    :return:
    """
    dim = fit.model.n_free
    state_initial = fit.model.parameter_values
    n_samples = steps // thin
    # initialize arrays
    lnp = np.empty(n_samples)
    parameter = np.empty((n_samples, dim))
    n_accepted = 0
    state_prev = np.copy(state_initial)
    bounds = fit.model.parameter_bounds

    lnp_prev = np.array(
        chisurf.core.fitting.fit.lnprob(
            parameter_values=state_initial,
            fit=fit,
            chi2max=chi2max,
            bounds=bounds
        )
    )

    while n_accepted < n_samples:

        state_next = state_prev + np.random.normal(0.0, step_size, dim) * state_initial
        lnp_next = chisurf.core.fitting.fit.lnprob(
            parameter_values=state_next,
            fit=fit,
            chi2max=chi2max,
            bounds=bounds
        )

        if not np.isfinite(lnp_next):
            continue

        if (-lnp_next + lnp_prev) / temp > np.log(np.random.rand()):
            # save results
            parameter[n_accepted] = state_next
            lnp[n_accepted] = lnp_next
            # switch previous and next
            np.copyto(state_prev, state_next)
            np.copyto(lnp_prev, lnp_next)
            n_accepted += 1
            if callback:
                callback(n_accepted, n_samples)
        
        if check_cancel and check_cancel():
            break

    chi2 = -2. * lnp / float(fit.model.n_points - fit.model.n_free - 1.0)

    return {
        'chi2r': chi2,
        'parameter_values': parameter,
        'parameter_names': fit.model.parameter_names
    }


def sample_emcee(
        fit: chisurf.core.fitting.fit.Fit,
        steps: int,
        nwalkers: int,
        thin: int = 10,
        std: float = 1e-3,
        chi2max: float = np.inf,
        progress_bar = None,
        substeps: int = None,
        callback: typing.Callable = None,
        check_cancel: typing.Callable = None
) -> Dict:
    """Sample the parameter space by emcee using a number of 'walkers'

    :param fit: the fit to be samples
    :param steps: the number of steps of each walker
    :param thin: an integer (only every ith step is saved)
    :param nwalkers: the number of walkers
    :param chi2max: maximum allowed chi2
    :param std: the standard deviation of the parameters used to randomize the initial set of the walkers
    :return: a list containing the chi2 and the parameter values
    """
    if substeps is None:
        try:
            substeps = int(chisurf.core.settings.cs_settings['optimization']['sampling'].get('substeps', 100))
        except (KeyError, TypeError):
            substeps = 100

    model = fit.model
    ndim = fit.n_free  # Number of free parameters to be sampled (number of dimensions)
    kw = {
        'bounds': fit.model.parameter_bounds,
        'chi2max': chi2max
    }
    sampler = emcee.EnsembleSampler(
        nwalkers=nwalkers,
        ndim=ndim,
        log_prob_fn=chisurf.core.fitting.fit.lnprob,
        args=[fit],
        kwargs=kw
    )
    # Initialize walkers with a robust standard deviation estimate
    p0 = np.array(model.parameter_values)
    bounds = np.array(kw['bounds'])
    std_input = std # input float, e.g., 1e-3
    std_vec = np.zeros(ndim)

    for i in range(ndim):
        lb, ub = bounds[i]
        # 1. Use width of narrow bounds as scale if finite
        if lb is not None and ub is not None and np.isfinite(lb) and np.isfinite(ub):
            # Use 1/1000th of the range as jitter
            std_vec[i] = (ub - lb) * 1e-4
        # 2. Else use relative scale if parameter is non-zero
        elif abs(p0[i]) > 1e-15:
            std_vec[i] = abs(p0[i]) * std_input
        # 3. Last fallback: use absolute input value
        else:
            std_vec[i] = std_input

    if progress_bar is not None and hasattr(progress_bar, 'setMaximum'):
        progress_bar.setMaximum(steps)

    previous_state = []
    for _ in range(nwalkers):
        p = p0 + std_vec * np.random.randn(ndim)
        # Ensure initial state stays within user-provided bounds. 
        # Clip lb if lb > -inf and ub if ub < inf.
        for j in range(ndim):
            lb, ub = bounds[j]
            if lb is not None and np.isfinite(lb):
                p[j] = max(p[j], lb)
            if ub is not None and np.isfinite(ub):
                p[j] = min(p[j], ub)
        previous_state.append(p)

    current_step = 0
    while current_step < steps:
        n_to_run = min(substeps, steps - current_step)
        previous_state = sampler.run_mcmc(
            previous_state,
            nsteps=n_to_run,
            thin_by=thin,
            skip_initial_state_check=True
        )
        current_step += n_to_run
        
        if progress_bar is not None:
            try:
                progress_bar.setValue(current_step)
            except Exception:
                pass

        if callback:
            try:
                callback(current_step, steps, sampler=sampler)
            except Exception:
                pass
        
        if check_cancel and check_cancel():
            break

    chi2 = -2. * sampler.get_log_prob(flat=True) / float(model.n_points - model.n_free - 1.0)
    return {
        'chi2r': chi2,
        'parameter_values': sampler.get_chain(flat=True),
        'parameter_names': fit.model.parameter_names
    }
