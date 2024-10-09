"""

"""
from __future__ import annotations
from typing import Dict

import emcee
import numpy as np

import chisurf
import chisurf.fitting


def walk_mcmc(
        fit: chisurf.fitting.fit.Fit,
        steps: int,
        step_size: float,
        temp: float = 1.0,
        thin: int = 1,
        chi2max: float = np.inf
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
        chisurf.fitting.fit.lnprob(
            parameter_values=state_initial,
            fit=fit,
            chi2max=chi2max,
            bounds=bounds
        )
    )

    while n_accepted < n_samples:

        state_next = state_prev + np.random.normal(0.0, step_size, dim) * state_initial
        lnp_next = chisurf.fitting.fit.lnprob(
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

    chi2 = -2. * lnp / float(fit.model.n_points - fit.model.n_free - 1.0)

    return {
        'chi2r': chi2,
        'parameter_values': parameter,
        'parameter_names': fit.model.parameter_names
    }


def sample_emcee(
        fit: chisurf.fitting.fit.Fit,
        steps: int,
        nwalkers: int,
        thin: int = 10,
        std: float = 1e-3,
        chi2max: float = np.inf,
        progress_bar = None,
        substeps: int = 500
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
    model = fit.model
    ndim = fit.n_free  # Number of free parameters to be sampled (number of dimensions)
    kw = {
        'bounds': fit.model.parameter_bounds,
        'chi2max': chi2max
    }
    sampler = emcee.EnsembleSampler(
        nwalkers=nwalkers,
        ndim=ndim,
        log_prob_fn=chisurf.fitting.fit.lnprob,
        args=[fit],
        kwargs=kw
    )
    std = np.array(fit.model.parameter_values) * std
    if progress_bar is not None:
        from qtpy import QtWidgets
        progress_bar.setMaximum(steps)

    previous_state = [fit.model.parameter_values + std * np.random.randn(ndim) for _ in range(nwalkers)]

    from qtpy import QtWidgets
    large_steps = steps // substeps
    for i in range(large_steps):
        previous_state = sampler.run_mcmc(
            previous_state,
            nsteps=substeps,
            thin_by=thin,
            skip_initial_state_check=True,
            tune=True
        )
        if progress_bar is not None:
            progress_bar.setValue(substeps * i + 1)
            QtWidgets.QApplication.processEvents()

    chi2 = -2. * sampler.flatlnprobability / float(model.n_points - model.n_free - 1.0)
    return {
        'chi2r': chi2,
        'parameter_values': sampler.flatchain,
        'parameter_names': fit.model.parameter_names
    }
