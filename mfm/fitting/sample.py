from __future__ import annotations

import emcee
import numpy as np

import mfm.fitting.fit


def walk_mcmc(
        fit: mfm.fitting.fit.Fit,
        steps: int,
        step_size: float,
        chi2max: float,
        temp: float,
        thin: int
):
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
    parameter = np.zeros((n_samples, dim))
    n_accepted = 0
    state_prev = np.copy(state_initial)
    lnp_prev = np.array(fit.lnprob(state_initial))
    while n_accepted < n_samples:
        state_next = state_prev + np.random.normal(0.0, step_size, dim) * state_initial
        lnp_next = fit.lnprob(state_next, chi2max)
        if not np.isfinite(lnp_next):
            continue
        if (-lnp_next + lnp_prev)/temp > np.log(np.random.rand()):
            # save results
            parameter[n_accepted] = state_next
            lnp[n_accepted] = lnp_next
            # switch previous and next
            np.copyto(state_prev, state_next)
            np.copyto(lnp_prev, lnp_next)
            n_accepted += 1
    return [lnp, parameter]


def sample_emcee(
        fit: mfm.fitting.fit.Fit,
        steps: int,
        nwalkers: int,
        thin: int = 10,
        std: float = 1e-4,
        chi2max: float = np.inf
):
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
    ndim = fit.n_free # Number of free parameters to be sampled (number of dimensions)
    kw = {
        'parameters': fit.model.parameters,
        'bounds': fit.model.parameter_bounds,
        'chi2max': chi2max
    }
    sampler = emcee.EnsembleSampler(
        nwalkers, ndim,
        mfm.fitting.fit.lnprob,
        args=[fit], kwargs=kw
    )
    std = np.array(fit.model.parameter_values) * std
    pos = [fit.model.parameter_values + std * np.random.randn(ndim) for i in range(nwalkers)]
    sampler.run_mcmc(pos, steps, thin=thin)
    chi2 = -2. * sampler.flatlnprobability / float(model.n_points - model.n_free - 1.0)
    return chi2, sampler.flatchain