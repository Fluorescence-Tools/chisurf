r"""Amortised (surrogate) neural estimator for H2MM — an **optional** fast path.

Instead of iterating Baum-Welch EM to the maximum-likelihood estimate, this
module trains a small neural network **once** on data drawn from the H2MM
generative model (:func:`~.h2mm.simulate_bursts`) and then estimates the model
parameters of a real dataset in a **single forward pass** — the
simulation-based / amortised-inference idea (see OKF ``PRD-60``).

Why it is a *replacement* for EM, not an initialiser
----------------------------------------------------
Empirically, seeding EM near the optimum does **not** cut its iteration count
much: starting Baum-Welch from the exact generative parameters still takes
almost as many maps as a random start, because EM spends its effort on the
finite-sample "last mile" to *this dataset's* MLE.  So the surrogate earns its
speed by returning an estimate directly (optionally polished by a few EM maps
via ``refine_iters``), not by warm-starting a full EM run.

Accuracy / scope
----------------
The estimate is **approximate** — like fitting on a subsample, it trades a
little statistical precision for a large speed-up, which is safe when the data
over-determine the model (many bursts).  A trained surrogate is specific to a
``(n_states, n_streams)`` and the burst-length / inter-photon-Δt regime it was
trained on; it does not ship pretrained.  Train one with :func:`train_surrogate`
and cache it, or fall back to EM (:func:`~.h2mm.fit_states`).

This module has **no Qt / ChiSurf dependency** and uses only ``numpy`` and
``scikit-learn`` (already a ChiSurf dependency); if scikit-learn is unavailable
:func:`surrogate_available` returns ``False`` and callers fall back to EM.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .h2mm import (
    BurstPhotons,
    H2mmModel,
    _row_normalize,
    njit,
    optimize,
    prepare_bursts,
)

try:
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler

    _HAVE_SKLEARN = True
except Exception:  # pragma: no cover - exercised only without scikit-learn
    _HAVE_SKLEARN = False


# Feature layout version — bump when :func:`extract_features` changes so a stale
# cached model is rejected rather than silently mis-fed.
FEATURES_VERSION = 1
_LOG_FLOOR = 1e-6  # transition probabilities below this are treated as this


def surrogate_available() -> bool:
    """Return whether the optional scikit-learn backend is importable."""
    return _HAVE_SKLEARN


# ---------------------------------------------------------------------------
# Feature extraction (permutation-invariant summary of a dataset)
# ---------------------------------------------------------------------------


@njit(cache=True)
def _windowed_fret(streams, offsets, n_streams, win):
    """Per-photon local acceptor fraction over a ``±win`` photon window."""
    n = streams.shape[0]
    out = np.empty(n)
    scale = 1.0 / max(n_streams - 1, 1)
    for b in range(offsets.shape[0] - 1):
        s = offsets[b]
        e = offsets[b + 1]
        for j in range(s, e):
            a = j - win
            if a < s:
                a = s
            c = j + win + 1
            if c > e:
                c = e
            acc = 0.0
            for k in range(a, c):
                acc += streams[k] * scale
            out[j] = acc / (c - a)
    return out


def extract_features(data: BurstPhotons) -> np.ndarray:
    """Return a fixed-length, permutation-invariant feature vector for ``data``.

    The features summarise the emission structure (windowed local-FRET
    histogram + quantiles), the kinetics (photon-lag autocorrelation of the
    per-photon FRET signal), and the inter-photon timing — everything an
    amortised estimator needs to recover ``(prior, trans, obs)`` without seeing
    the raw sequence.  Independent of burst order and burst count.
    """
    streams = data.streams
    offsets = data.burst_offsets
    p = data.n_streams
    scale = 1.0 / max(p - 1, 1)
    sig = streams.astype(np.float64) * scale  # per-photon FRET-like signal in [0,1]

    feats: list[float] = [float(sig.mean())]

    loc = _windowed_fret(streams, offsets, p, 12)
    hist, _ = np.histogram(loc, bins=10, range=(0.0, 1.0), density=True)
    feats += [float(v) for v in hist]
    feats += [float(v) for v in np.quantile(loc, [0.1, 0.25, 0.5, 0.75, 0.9])]

    # Photon-lag autocorrelation of the FRET signal, per burst then averaged.
    mu = sig.mean()
    for lag in (1, 2, 4, 8, 16, 32):
        num = 0.0
        cnt = 0
        for b in range(data.n_bursts):
            s = int(offsets[b])
            e = int(offsets[b + 1])
            if e - s > lag:
                x = sig[s:e]
                num += float(np.mean((x[:-lag] - mu) * (x[lag:] - mu)))
                cnt += 1
        feats.append(num / cnt if cnt else 0.0)

    # Inter-photon Δt statistics.
    gs = data.gap_slot
    dt = data.unique_dt[gs[gs >= 0]] if data.unique_dt.shape[0] else np.zeros(1)
    feats += [float(dt.mean()), float(dt.std())]

    return np.asarray(feats, dtype=np.float64)


# ---------------------------------------------------------------------------
# Parameter <-> target-vector encoding (canonical state ordering)
# ---------------------------------------------------------------------------


def _canonical_order(model: H2mmModel) -> H2mmModel:
    """Return ``model`` with states sorted by stream-0 emission (label-invariant)."""
    order = np.argsort(-model.obs[:, 0])
    return H2mmModel(
        prior=model.prior[order],
        trans=model.trans[np.ix_(order, order)],
        obs=model.obs[order],
    )


def _encode(model: H2mmModel) -> np.ndarray:
    """Flatten a canonically-ordered model to a regression target vector.

    Emissions and prior are stored as probabilities; the tiny off-diagonal
    transition probabilities are stored in ``log10`` space (their dynamic range
    spans orders of magnitude, so a linear target would ignore them).
    """
    m = _canonical_order(model)
    n = m.n_states
    off = [np.log10(max(m.trans[i, j], _LOG_FLOOR)) for i in range(n) for j in range(n) if i != j]
    return np.concatenate([m.obs.ravel(), np.asarray(off), m.prior.ravel()])


def _decode(vec: np.ndarray, n: int, p: int) -> H2mmModel:
    """Reconstruct a valid :class:`H2mmModel` from a (possibly noisy) target vector."""
    k = 0
    obs = _row_normalize(np.clip(vec[k : k + n * p].reshape(n, p), 1e-6, None))
    k += n * p
    trans = np.eye(n)
    for i in range(n):
        for j in range(n):
            if i != j:
                trans[i, j] = 10.0 ** vec[k]
                k += 1
    for i in range(n):
        trans[i, i] = max(1.0 - (trans[i].sum() - trans[i, i]), 1e-6)
    trans = _row_normalize(trans)
    prior = _row_normalize(np.clip(vec[k : k + n].reshape(1, -1), 1e-9, None)).ravel()
    return _canonical_order(H2mmModel(prior=prior, trans=trans, obs=obs))


# ---------------------------------------------------------------------------
# Trained surrogate container
# ---------------------------------------------------------------------------


@dataclass
class SurrogateModel:
    """A trained amortised estimator plus the scalers and regime metadata."""

    net: object
    x_scaler: object
    y_scaler: object
    n_states: int
    n_streams: int
    features_version: int
    meta: dict

    def predict(self, data: BurstPhotons) -> H2mmModel:
        """Estimate the H2MM model of ``data`` in a single forward pass."""
        if data.n_streams != self.n_streams:
            raise ValueError(
                f"surrogate trained for n_streams={self.n_streams}, got {data.n_streams}"
            )
        x = self.x_scaler.transform(extract_features(data)[None])
        y = self.y_scaler.inverse_transform(self.net.predict(x))[0]
        return _decode(y, self.n_states, self.n_streams)

    def save(self, path: str | Path) -> None:
        """Pickle the surrogate (net + scalers + metadata) to ``path``."""
        with open(path, "wb") as fh:
            pickle.dump(self, fh)

    @staticmethod
    def load(path: str | Path) -> SurrogateModel:
        """Load a surrogate saved by :meth:`save`; validate the feature version."""
        with open(path, "rb") as fh:
            m = pickle.load(fh)
        if not isinstance(m, SurrogateModel):
            raise TypeError(f"{path} is not a SurrogateModel")
        if m.features_version != FEATURES_VERSION:
            raise ValueError(
                f"surrogate feature version {m.features_version} != current {FEATURES_VERSION}; retrain"
            )
        return m


# ---------------------------------------------------------------------------
# Training-data generation + training
# ---------------------------------------------------------------------------


def _random_model(n_states, n_streams, rng) -> H2mmModel:
    """Draw a random, well-ordered H2MM model over a realistic FRET/kinetics range."""
    fret = np.sort(rng.uniform(0.08, 0.92, n_states))
    obs = np.empty((n_states, n_streams))
    if n_streams == 2:
        obs[:, 0] = 1.0 - fret
        obs[:, 1] = fret
    else:
        base = rng.dirichlet(np.ones(n_streams), size=n_states)
        obs = base
    trans = np.eye(n_states)
    for i in range(n_states):
        for j in range(n_states):
            if i != j:
                trans[i, j] = 10.0 ** rng.uniform(-2.8, -1.3)
        trans[i, i] = 1.0 - (trans[i].sum() - trans[i, i])
    prior = np.full(n_states, 1.0 / n_states)
    return H2mmModel(prior=prior, trans=_row_normalize(trans), obs=_row_normalize(obs))


def _fast_simulate(model: H2mmModel, times, rng) -> list[np.ndarray]:
    """Sample photon streams by drawing the hidden state at each photon time.

    Equivalent to :func:`~.h2mm.simulate_bursts` for the *observed* photons —
    both distribute the state at photon ``n`` as ``A^Δt`` propagated from photon
    ``n-1`` — but ``O(photons)`` instead of ``O(clock ticks)`` because the state
    is sampled directly from the interval propagator ``A^Δt`` (cached per unique
    ``Δt``) rather than advanced tick by tick.  This keeps surrogate training
    practical.
    """
    n_states = model.n_states
    trans = model.trans
    obs = model.obs
    cum_obs = np.cumsum(obs, axis=1)
    streams_out: list[np.ndarray] = []
    pow_cache: dict[int, np.ndarray] = {}
    for t in times:
        t = np.asarray(t, dtype=np.int64)
        m = t.shape[0]
        cum_pow: dict[int, np.ndarray] = {}
        s = int(rng.choice(n_states, p=model.prior))
        out = np.empty(m, dtype=np.int32)
        for j in range(m):
            if j > 0:
                dt = int(t[j] - t[j - 1])
                cp = cum_pow.get(dt)
                if cp is None:
                    P = pow_cache.get(dt)
                    if P is None:
                        P = np.linalg.matrix_power(trans, dt)
                        pow_cache[dt] = P
                    cp = np.cumsum(P, axis=1)
                    cum_pow[dt] = cp
                s = int(np.searchsorted(cp[s], rng.random()))
                if s >= n_states:
                    s = n_states - 1
            k = int(np.searchsorted(cum_obs[s], rng.random()))
            out[j] = min(k, obs.shape[1] - 1)
        streams_out.append(out)
    return streams_out


def generate_training_set(
    n_samples: int,
    n_states: int,
    n_streams: int,
    n_bursts: int = 150,
    burst_len: int = 80,
    mean_dt: float = 4.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate ``n_samples`` labelled datasets → ``(features, target)`` arrays."""
    rng = np.random.default_rng(seed)
    X: list[np.ndarray] = []
    Y: list[np.ndarray] = []
    for i in range(n_samples):
        model = _random_model(n_states, n_streams, rng)
        times = [
            np.concatenate([[0], np.cumsum(rng.poisson(mean_dt, burst_len - 1) + 1)]).astype(np.int64)
            for _ in range(n_bursts)
        ]
        streams = _fast_simulate(model, times, rng)
        data = prepare_bursts(times, streams, n_streams)
        X.append(extract_features(data))
        Y.append(_encode(model))
    return np.asarray(X), np.asarray(Y)


def train_surrogate(
    n_states: int = 2,
    n_streams: int = 2,
    n_samples: int = 2500,
    hidden_layer_sizes: tuple[int, ...] = (256, 256, 128),
    max_iter: int = 800,
    n_bursts: int = 150,
    burst_len: int = 80,
    mean_dt: float = 4.0,
    seed: int = 0,
) -> SurrogateModel:
    """Generate training data and fit the amortised MLP estimator.

    Raises
    ------
    RuntimeError
        If scikit-learn is not installed (:func:`surrogate_available` is False).
    """
    if not _HAVE_SKLEARN:
        raise RuntimeError("scikit-learn is required to train a surrogate")
    X, Y = generate_training_set(
        n_samples, n_states, n_streams,
        n_bursts=n_bursts, burst_len=burst_len, mean_dt=mean_dt, seed=seed,
    )
    x_scaler = StandardScaler().fit(X)
    y_scaler = StandardScaler().fit(Y)
    net = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation="relu",
        max_iter=max_iter,
        early_stopping=True,
        random_state=seed,
    )
    net.fit(x_scaler.transform(X), y_scaler.transform(Y))
    return SurrogateModel(
        net=net, x_scaler=x_scaler, y_scaler=y_scaler,
        n_states=n_states, n_streams=n_streams, features_version=FEATURES_VERSION,
        meta={"n_samples": n_samples, "n_bursts": n_bursts,
              "burst_len": burst_len, "mean_dt": mean_dt},
    )


# ---------------------------------------------------------------------------
# Estimation entry point
# ---------------------------------------------------------------------------


def estimate_model(
    data: BurstPhotons,
    n_states: int,
    surrogate: SurrogateModel | str | Path,
    refine_iters: int = 0,
    tol: float = 1e-7,
) -> H2mmModel:
    """Estimate an H2MM model with the surrogate, optionally polished by EM.

    Parameters
    ----------
    data : BurstPhotons
        Photon data in engine layout.
    n_states : int
        Number of hidden states; must match the surrogate.
    surrogate : SurrogateModel or path
        A trained surrogate, or a path to one saved by :meth:`SurrogateModel.save`.
    refine_iters : int
        If > 0, run this many Baum-Welch maps from the surrogate estimate to
        polish it toward the exact MLE (``0`` returns the pure one-shot estimate
        — the fast, approximate path).
    tol : float
        Convergence threshold for the optional refinement.

    Returns
    -------
    H2mmModel
        The estimated model (``loglik``/``n_phot`` populated when refined).
    """
    if isinstance(surrogate, (str, Path)):
        surrogate = SurrogateModel.load(surrogate)
    if surrogate.n_states != n_states:
        raise ValueError(
            f"surrogate trained for n_states={surrogate.n_states}, got {n_states}"
        )
    model = surrogate.predict(data)
    if refine_iters > 0:
        model = optimize(model, data, max_iter=refine_iters, tol=tol)
    return model
