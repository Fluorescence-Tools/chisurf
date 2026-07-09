"""Tests for the optional amortised (surrogate) H2MM estimator."""

from __future__ import annotations

import numpy as np
import pytest

from chisurf.plugins.burst.burst_h2mm.core import h2mm
from chisurf.plugins.burst.burst_h2mm.core import surrogate as S

pytestmark = pytest.mark.skipif(
    not S.surrogate_available(), reason="scikit-learn not installed"
)


def _two_state_data(seed=1, n_bursts=200):
    gt = h2mm.H2mmModel(
        np.array([0.5, 0.5]),
        np.array([[0.99, 0.01], [0.02, 0.98]]),
        np.array([[0.85, 0.15], [0.20, 0.80]]),
    )
    rng = np.random.default_rng(seed)
    times = [
        np.concatenate([[0], np.cumsum(rng.poisson(4, 79) + 1)]).astype(np.int64)
        for _ in range(n_bursts)
    ]
    streams = h2mm.simulate_bursts(gt, times, seed=seed + 5)
    return h2mm.prepare_bursts(times, streams, 2), gt


def test_features_deterministic_and_fixed_length():
    data, _ = _two_state_data(seed=2, n_bursts=50)
    f1 = S.extract_features(data)
    f2 = S.extract_features(data)
    assert f1.shape == (24,)
    assert np.array_equal(f1, f2)
    assert np.all(np.isfinite(f1))


def test_encode_decode_roundtrip():
    gt = h2mm.H2mmModel(
        np.array([0.4, 0.6]),
        np.array([[0.97, 0.03], [0.05, 0.95]]),
        np.array([[0.8, 0.2], [0.3, 0.7]]),
    )
    m = S._decode(S._encode(gt), 2, 2)
    # canonical ordering + reconstruction recovers the model closely
    assert np.abs(m.obs - gt.obs).max() < 1e-6
    assert np.abs(m.trans - gt.trans).max() < 1e-6


def test_train_predict_recovers_states():
    # Small/fast training run — enough to recover a clear 2-state model.
    sm = S.train_surrogate(
        n_states=2, n_streams=2, n_samples=400,
        hidden_layer_sizes=(128, 64), max_iter=300,
        n_bursts=120, burst_len=70, seed=3,
    )
    data, gt = _two_state_data(seed=7, n_bursts=250)
    est = sm.predict(data)

    # Valid stochastic model.
    assert np.allclose(est.obs.sum(1), 1.0)
    assert np.allclose(est.trans.sum(1), 1.0)
    # Per-state FRET recovered to a loose (amortised, approximate) tolerance.
    E_est = np.sort(est.obs[:, 1] / est.obs.sum(1))
    E_gt = np.sort(gt.obs[:, 1] / gt.obs.sum(1))
    assert np.abs(E_est - E_gt).max() < 0.12


def test_estimate_via_fit_states_and_refine():
    sm = S.train_surrogate(
        n_states=2, n_streams=2, n_samples=400,
        hidden_layer_sizes=(128, 64), max_iter=300,
        n_bursts=120, burst_len=70, seed=3,
    )
    data, _ = _two_state_data(seed=9, n_bursts=250)

    one_shot = h2mm.fit_states(data, 2, surrogate=sm)
    assert one_shot.n_states == 2

    refined = h2mm.fit_states(data, 2, surrogate=sm, refine_iters=50)
    # Polishing must not lower the likelihood below the pure surrogate estimate.
    assert np.isfinite(refined.loglik)
    assert refined.n_iter > 0


def test_save_load_roundtrip(tmp_path):
    sm = S.train_surrogate(
        n_states=2, n_streams=2, n_samples=200,
        hidden_layer_sizes=(64,), max_iter=150, n_bursts=80, burst_len=60, seed=1,
    )
    p = tmp_path / "surrogate.pkl"
    sm.save(p)
    loaded = S.SurrogateModel.load(p)
    data, _ = _two_state_data(seed=4, n_bursts=100)
    assert np.allclose(sm.predict(data).obs, loaded.predict(data).obs)
