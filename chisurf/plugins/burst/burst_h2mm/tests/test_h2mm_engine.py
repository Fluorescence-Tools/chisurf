"""Headless correctness tests for the Numba H2MM engine."""

from __future__ import annotations

import numpy as np
import pytest

from chisurf.plugins.burst.burst_h2mm.core import h2mm


def _make_ground_truth() -> h2mm.H2mmModel:
    """Return a well-separated 2-state, 2-stream model."""
    prior = np.array([0.5, 0.5])
    # Moderate switching so bursts contain several transitions.
    trans = np.array([[0.995, 0.005], [0.010, 0.990]])
    # State 0 emits mostly stream 0; state 1 mostly stream 1.
    obs = np.array([[0.85, 0.15], [0.20, 0.80]])
    return h2mm.H2mmModel(prior=prior, trans=trans, obs=obs)


def _simulate(model, n_bursts=400, burst_len=80, rate=0.25, seed=1):
    """Generate bursts with Poisson-spaced (variable-Δt) photon times."""
    rng = np.random.default_rng(seed)
    times = []
    for _ in range(n_bursts):
        gaps = rng.poisson(lam=1.0 / rate, size=burst_len - 1) + 1
        t = np.concatenate([[0], np.cumsum(gaps)]).astype(np.int64)
        times.append(t)
    streams = h2mm.simulate_bursts(model, times, seed=seed + 7)
    return times, streams


def test_prepare_bursts_layout():
    times = [np.array([0, 5, 9]), np.array([0, 3])]
    streams = [np.array([0, 1, 0]), np.array([1, 0])]
    data = h2mm.prepare_bursts(times, streams, n_streams=2)
    assert data.n_bursts == 2
    assert data.n_photons == 5
    # Unique Δt across data: {5,4,3} -> sorted {3,4,5}
    assert list(data.unique_dt) == [3, 4, 5]
    # Last photon of each burst has slot -1.
    assert data.gap_slot[2] == -1
    assert data.gap_slot[4] == -1


def test_loglik_monotonic_increase():
    gt = _make_ground_truth()
    times, streams = _simulate(gt, n_bursts=150, burst_len=60, seed=3)
    data = h2mm.prepare_bursts(times, streams, n_streams=2)

    init = h2mm.factory_model(2, 2, seed=0)
    lls = []
    model = init
    # Run EM one iteration at a time and confirm logL never decreases.
    for _ in range(15):
        model = h2mm.optimize(model, data, max_iter=1, tol=0.0)
        lls.append(model.loglik)
    diffs = np.diff(lls)
    assert np.all(diffs >= -1e-6), f"logL decreased: {lls}"


def test_recovers_ground_truth_two_state():
    gt = _make_ground_truth()
    times, streams = _simulate(gt, n_bursts=600, burst_len=100, seed=11)
    data = h2mm.prepare_bursts(times, streams, n_streams=2)

    fit = h2mm.fit_states(data, n_states=2, n_restarts=2, max_iter=400, seed=0)

    # States may be permuted; align by donor (stream-0) emission probability.
    order = np.argsort(-fit.obs[:, 0])
    obs = fit.obs[order]
    trans = fit.trans[np.ix_(order, order)]

    # Emission probabilities recovered within a loose tolerance.
    assert obs[0, 0] == pytest.approx(0.85, abs=0.08)
    assert obs[1, 1] == pytest.approx(0.80, abs=0.08)
    # Transition rates recovered to the right order of magnitude.
    assert trans[0, 1] == pytest.approx(0.005, abs=0.004)
    assert trans[1, 0] == pytest.approx(0.010, abs=0.006)


def test_bic_prefers_two_states():
    gt = _make_ground_truth()
    times, streams = _simulate(gt, n_bursts=500, burst_len=100, seed=21)
    data = h2mm.prepare_bursts(times, streams, n_streams=2)

    fit1 = h2mm.fit_states(data, n_states=1, n_restarts=1, max_iter=200, seed=0)
    fit2 = h2mm.fit_states(data, n_states=2, n_restarts=2, max_iter=400, seed=0)

    assert fit2.loglik > fit1.loglik
    assert fit2.bic < fit1.bic


def test_viterbi_path_and_icl():
    gt = _make_ground_truth()
    times, streams = _simulate(gt, n_bursts=200, burst_len=80, seed=31)
    data = h2mm.prepare_bursts(times, streams, n_streams=2)
    fit = h2mm.fit_states(data, n_states=2, n_restarts=1, max_iter=300, seed=0)

    path, icl = h2mm.viterbi(fit, data)
    assert path.shape[0] == data.n_photons
    assert path.min() >= 0 and path.max() < 2
    assert np.isfinite(icl)
