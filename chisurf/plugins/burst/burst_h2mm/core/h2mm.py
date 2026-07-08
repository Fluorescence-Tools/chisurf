r"""Photon-by-photon Hidden Markov Model (H2MM) — Numba engine.

This module is a pure-``numpy``/``numba`` re-implementation of the H2MM
algorithm of Pirchi *et al.* (J. Phys. Chem. B 2016, 120, 13065) and the
reference ``H2MM_C`` library by P. D. Harris.  It is Qt-free and has no
ChiSurf dependencies so it can run on a head-less backend, in tests, or from
the CLI.

Model
-----
A model :math:`\lambda = \{\pi, A, B\}` over ``n_states`` hidden states and
``n_streams`` photon streams (detector categories):

* ``prior`` — ``(n_states,)`` initial-state probabilities, sums to 1.
* ``trans`` — ``(n_states, n_states)`` **row-stochastic** transition matrix for
  **one base time unit** (``trans[i, j] = P(state j at t+1 | state i at t)``).
* ``obs`` — ``(n_states, n_streams)`` **row-stochastic** emission matrix
  (``obs[i, k] = P(photon stream k | state i)``).

Variable inter-photon times
---------------------------
Photons arrive at integer macro-times ``t_1 < t_2 < ...``.  Between two
consecutive photons the hidden state performs :math:`\Delta t` unobserved
transitions, so wherever a standard HMM uses ``A`` this engine uses
``A**Δt``.  Powers (and the transition-count tensor ``ρ``) are computed once
per **unique** ``Δt`` and cached, so cost scales with the number of *photons*,
not the number of clock ticks.

The ``ρ`` tensor ``ρ[k, m, i, j](Δt)`` — the expected number of ``i→j``
transitions during an interval of length ``Δt`` whose endpoints are states
``k`` and ``m`` — is built together with ``A**Δt`` using the associative
"pair-power" recursion (paper eq. 26)::

    ρ(Δa+Δb)[k,m,i,j] = Σ_z ρ(Δa)[k,z,i,j]·A^Δb[z,m]
                      + Σ_z A^Δa[k,z]·ρ(Δb)[z,m,i,j]

so both are obtained by binary exponentiation of the base pair
``(A, ρ(1))`` in ``O(log Δt)`` steps per unique interval.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

try:  # numba is a first-class dependency; degrade gracefully if unavailable
    from numba import njit, prange

    _HAVE_NUMBA = True
except Exception:  # pragma: no cover - exercised only without numba
    _HAVE_NUMBA = False

    def njit(*args, **kwargs):  # type: ignore
        """No-op ``njit`` fallback used when numba is unavailable."""
        def _wrap(fn):
            return fn

        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return _wrap

    def prange(*args):  # type: ignore
        """Serial ``prange`` fallback used when numba is unavailable."""
        return range(*args)


# ---------------------------------------------------------------------------
# Model container
# ---------------------------------------------------------------------------


@dataclass
class H2mmModel:
    """An H2MM model plus its optimisation diagnostics.

    Attributes
    ----------
    prior : numpy.ndarray
        Initial-state probabilities, shape ``(n_states,)``.
    trans : numpy.ndarray
        One-step row-stochastic transition matrix, shape
        ``(n_states, n_states)``.
    obs : numpy.ndarray
        Row-stochastic emission matrix, shape ``(n_states, n_streams)``.
    loglik : float
        Log-likelihood of the data under the model (``-inf`` if not scored).
    n_iter : int
        Number of EM iterations performed.
    n_phot : int
        Total number of photons the model was scored on.
    converged : bool
        Whether the EM loop met the convergence criterion.
    """

    prior: np.ndarray
    trans: np.ndarray
    obs: np.ndarray
    loglik: float = -np.inf
    n_iter: int = 0
    n_phot: int = 0
    converged: bool = False

    @property
    def n_states(self) -> int:
        """Number of hidden states."""
        return int(self.prior.shape[0])

    @property
    def n_streams(self) -> int:
        """Number of photon streams (detector categories)."""
        return int(self.obs.shape[1])

    @property
    def n_free(self) -> int:
        """Number of free parameters ``k`` (used by BIC/ICL)."""
        n, p = self.n_states, self.n_streams
        return n * n + (p - 1) * n - 1

    @property
    def bic(self) -> float:
        """Bayesian information criterion ``-2·logL + k·ln(N_phot)``."""
        if not math.isfinite(self.loglik) or self.n_phot <= 0:
            return np.inf
        return -2.0 * self.loglik + self.n_free * math.log(self.n_phot)

    def normalize(self) -> "H2mmModel":
        """Renormalise ``prior``, and the rows of ``trans`` and ``obs``."""
        self.prior = _row_normalize(self.prior.reshape(1, -1)).ravel()
        self.trans = _row_normalize(self.trans)
        self.obs = _row_normalize(self.obs)
        return self

    def copy(self) -> "H2mmModel":
        """Return a deep copy of the model."""
        return H2mmModel(
            self.prior.copy(),
            self.trans.copy(),
            self.obs.copy(),
            self.loglik,
            self.n_iter,
            self.n_phot,
            self.converged,
        )


def _row_normalize(a: np.ndarray) -> np.ndarray:
    """Return ``a`` with every row rescaled to sum to 1 (zero rows → uniform)."""
    a = np.asarray(a, dtype=np.float64)
    out = a.copy()
    s = out.sum(axis=1)
    for i in range(out.shape[0]):
        if s[i] > 0:
            out[i] /= s[i]
        else:
            out[i] = 1.0 / out.shape[1]
    return out


# ---------------------------------------------------------------------------
# Burst-photon data container
# ---------------------------------------------------------------------------


@dataclass
class BurstPhotons:
    """Photon streams for a set of bursts in engine-ready (CSR) layout.

    Attributes
    ----------
    streams : numpy.ndarray
        Concatenated per-photon stream index, ``int32`` of length ``N``.
    gap_slot : numpy.ndarray
        For each photon ``n``, the cache slot of ``Δt`` between photon ``n``
        and ``n+1`` (``-1`` for the last photon of every burst), ``int32``.
    burst_offsets : numpy.ndarray
        CSR offsets, ``int64`` of length ``n_bursts + 1``.
    unique_dt : numpy.ndarray
        Sorted unique inter-photon ``Δt`` values, ``int64``.
    n_streams : int
        Number of photon streams.
    """

    streams: np.ndarray
    gap_slot: np.ndarray
    burst_offsets: np.ndarray
    unique_dt: np.ndarray
    n_streams: int

    @property
    def n_bursts(self) -> int:
        """Number of bursts."""
        return int(self.burst_offsets.shape[0] - 1)

    @property
    def n_photons(self) -> int:
        """Total number of photons across all bursts."""
        return int(self.streams.shape[0])


def prepare_bursts(
    times: Sequence[np.ndarray],
    streams: Sequence[np.ndarray],
    n_streams: int,
) -> BurstPhotons:
    """Pack per-burst photon arrays into the engine's CSR layout.

    Parameters
    ----------
    times : sequence of numpy.ndarray
        One monotonically non-decreasing integer macro-time array per burst.
    streams : sequence of numpy.ndarray
        Matching per-burst photon stream indices in ``[0, n_streams)``.
    n_streams : int
        Number of photon streams.

    Returns
    -------
    BurstPhotons
        Concatenated arrays, burst offsets, and the unique-``Δt`` table.
    """
    if len(times) != len(streams):
        raise ValueError("times and streams must have the same number of bursts")

    kept_times: List[np.ndarray] = []
    kept_streams: List[np.ndarray] = []
    for t, s in zip(times, streams):
        t = np.asarray(t).astype(np.int64, copy=False)
        s = np.asarray(s).astype(np.int32, copy=False)
        if t.shape[0] != s.shape[0]:
            raise ValueError("each burst needs equal-length times and streams")
        if t.shape[0] == 0:
            continue
        kept_times.append(t)
        kept_streams.append(s)

    if not kept_times:
        raise ValueError("no non-empty bursts provided")

    offsets = np.zeros(len(kept_times) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([t.shape[0] for t in kept_times])
    streams_concat = np.concatenate(kept_streams).astype(np.int32)

    # Inter-photon Δt per photon (0 at the last photon of each burst).
    all_dt: List[np.ndarray] = []
    for t in kept_times:
        if t.shape[0] > 1:
            all_dt.append(np.diff(t))
    if all_dt:
        unique_dt = np.unique(np.concatenate(all_dt)).astype(np.int64)
        unique_dt = unique_dt[unique_dt > 0]
    else:
        unique_dt = np.zeros(0, dtype=np.int64)

    gap_slot = np.full(streams_concat.shape[0], -1, dtype=np.int32)
    for b, t in enumerate(kept_times):
        if t.shape[0] < 2:
            continue
        start = offsets[b]
        dt = np.diff(t)
        slots = np.searchsorted(unique_dt, dt).astype(np.int32)
        gap_slot[start : start + dt.shape[0]] = slots

    return BurstPhotons(
        streams=streams_concat,
        gap_slot=gap_slot,
        burst_offsets=offsets,
        unique_dt=unique_dt,
        n_streams=int(n_streams),
    )


# ---------------------------------------------------------------------------
# A^Δt and ρ caches (associative pair-power)
# ---------------------------------------------------------------------------


@njit(cache=True)
def _matmul_norm(a, b):
    """Return the row-normalised matrix product ``a @ b``."""
    n = a.shape[0]
    out = np.zeros((n, n))
    for i in range(n):
        s = 0.0
        for j in range(n):
            v = 0.0
            for k in range(n):
                v += a[i, k] * b[k, j]
            out[i, j] = v
            s += v
        if s > 0.0:
            for j in range(n):
                out[i, j] /= s
    return out


@njit(cache=True)
def _rho_base(A):
    """Return ``ρ(1)[k,m,i,j] = δ_{k,i}·A[i,j]·δ_{j,m}``."""
    n = A.shape[0]
    R = np.zeros((n, n, n, n))
    for i in range(n):
        for j in range(n):
            R[i, j, i, j] = A[i, j]
    return R


@njit(cache=True)
def _pair_compose(Pa, Ra, Pb, Rb):
    """Compose interval propagators ``(Pa,Ra)`` then ``(Pb,Rb)``.

    Returns ``(P, R)`` with ``P`` the row-normalised product ``Pa @ Pb`` and
    ``R[k,m,i,j] = Σ_z Ra[k,z,i,j]·Pb[z,m] + Σ_z Pa[k,z]·Rb[z,m,i,j]``.
    """
    n = Pa.shape[0]
    P = _matmul_norm(Pa, Pb)
    R = np.zeros((n, n, n, n))
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for m in range(n):
                    v = 0.0
                    for z in range(n):
                        v += Ra[k, z, i, j] * Pb[z, m] + Pa[k, z] * Rb[z, m, i, j]
                    R[k, m, i, j] = v
    return P, R


@njit(cache=True)
def _pair_pow(A, R1, power):
    """Binary-exponentiate the base pair ``(A, R1)`` to ``(A^power, ρ(power))``."""
    n = A.shape[0]
    # Identity element: P = I, R = 0.
    Pres = np.eye(n)
    Rres = np.zeros((n, n, n, n))
    Pb = A.copy()
    Rb = R1.copy()
    e = power
    while e > 0:
        if e & 1:
            Pres, Rres = _pair_compose(Pres, Rres, Pb, Rb)
        e >>= 1
        if e > 0:
            Pb, Rb = _pair_compose(Pb, Rb, Pb, Rb)
    return Pres, Rres


@njit(parallel=True, cache=True)
def _build_caches(A, unique_dt, pow_cache, rho_cache):
    """Fill ``pow_cache[s]=A^Δt`` and ``rho_cache[s]=ρ(Δt)`` for each slot ``s``."""
    R1 = _rho_base(A)
    for s in prange(unique_dt.shape[0]):
        P, R = _pair_pow(A, R1, unique_dt[s])
        pow_cache[s] = P
        rho_cache[s] = R


# ---------------------------------------------------------------------------
# E-step: scaled forward-backward + Baum-Welch accumulation
# ---------------------------------------------------------------------------


@njit(cache=True)
def _estep(
    prior,
    obs,
    pow_cache,
    rho_cache,
    streams,
    gap_slot,
    offsets,
    alpha,
    beta,
    scale,
    xi_acc,
    gamma_obs_acc,
    gamma_i_acc,
    prior_acc,
):
    """Run scaled forward-backward over all bursts; accumulate BW statistics.

    Returns the total log-likelihood ``Σ_bursts Σ_n log(scale[n])``.
    """
    n_states = prior.shape[0]
    n_bursts = offsets.shape[0] - 1
    loglik = 0.0
    w = np.zeros(n_states)  # scratch: B[m, y]·β[n+1, m]

    for b in range(n_bursts):
        s = offsets[b]
        e = offsets[b + 1]

        # ---- forward ----
        y0 = streams[s]
        tot = 0.0
        for i in range(n_states):
            alpha[s, i] = prior[i] * obs[i, y0]
            tot += alpha[s, i]
        scale[s] = tot
        if tot > 0.0:
            for i in range(n_states):
                alpha[s, i] /= tot
            loglik += math.log(tot)

        for n in range(s + 1, e):
            slot = gap_slot[n - 1]
            P = pow_cache[slot]
            yn = streams[n]
            tot = 0.0
            for i in range(n_states):
                v = 0.0
                for k in range(n_states):
                    v += alpha[n - 1, k] * P[k, i]
                v *= obs[i, yn]
                alpha[n, i] = v
                tot += v
            scale[n] = tot
            if tot > 0.0:
                for i in range(n_states):
                    alpha[n, i] /= tot
                loglik += math.log(tot)

        # ---- backward ----
        for i in range(n_states):
            beta[e - 1, i] = 1.0
        for n in range(e - 2, s - 1, -1):
            slot = gap_slot[n]
            P = pow_cache[slot]
            yn1 = streams[n + 1]
            cc = scale[n + 1]
            for i in range(n_states):
                v = 0.0
                for k in range(n_states):
                    v += P[i, k] * obs[k, yn1] * beta[n + 1, k]
                beta[n, i] = v / cc if cc > 0.0 else 0.0

        # ---- accumulate γ (occupancy) ----
        for n in range(s, e):
            yn = streams[n]
            for i in range(n_states):
                g = alpha[n, i] * beta[n, i]
                gamma_i_acc[i] += g
                gamma_obs_acc[i, yn] += g
                if n == s:
                    prior_acc[i] += g

        # ---- accumulate ξ (transitions) over each gap ----
        for n in range(s, e - 1):
            slot = gap_slot[n]
            R = rho_cache[slot]
            yn1 = streams[n + 1]
            cc = scale[n + 1]
            if cc <= 0.0:
                continue
            inv = 1.0 / cc
            for m in range(n_states):
                w[m] = obs[m, yn1] * beta[n + 1, m]
            for i in range(n_states):
                for j in range(n_states):
                    acc = 0.0
                    for k in range(n_states):
                        ak = alpha[n, k]
                        if ak == 0.0:
                            continue
                        for m in range(n_states):
                            acc += ak * R[k, m, i, j] * w[m]
                    xi_acc[i, j] += acc * inv

    return loglik


# ---------------------------------------------------------------------------
# EM optimiser
# ---------------------------------------------------------------------------


def optimize(
    model: H2mmModel,
    data: BurstPhotons,
    max_iter: int = 500,
    tol: float = 1e-7,
    min_trans: float = 1e-12,
) -> H2mmModel:
    """Baum-Welch (EM) optimisation of an H2MM model.

    Parameters
    ----------
    model : H2mmModel
        Initial model; not modified in place.
    data : BurstPhotons
        Photon data in engine layout (from :func:`prepare_bursts`).
    max_iter : int
        Maximum number of EM iterations.
    tol : float
        Convergence threshold on the log-likelihood increment.
    min_trans : float
        Floor for off-diagonal transition probabilities, keeping ``trans``
        irreducible so rare transitions can still be discovered.

    Returns
    -------
    H2mmModel
        The optimised model with ``loglik``, ``n_iter``, ``n_phot`` and
        ``converged`` populated.
    """
    n = model.n_states
    p = data.n_streams
    n_dt = int(data.unique_dt.shape[0])

    prior = _row_normalize(model.prior.reshape(1, -1)).ravel()
    trans = _row_normalize(model.trans)
    obs = _row_normalize(model.obs)

    n_phot = data.n_photons
    alpha = np.zeros((n_phot, n))
    beta = np.zeros((n_phot, n))
    scale = np.zeros(n_phot)
    n_slots = max(n_dt, 1)
    pow_cache = np.zeros((n_slots, n, n))
    rho_cache = np.zeros((n_slots, n, n, n, n))

    prev_ll = -np.inf
    last_ll = -np.inf
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        if n_dt > 0:
            _build_caches(trans, data.unique_dt, pow_cache, rho_cache)

        xi_acc = np.zeros((n, n))
        gamma_obs_acc = np.zeros((n, p))
        gamma_i_acc = np.zeros(n)
        prior_acc = np.zeros(n)

        last_ll = _estep(
            prior,
            obs,
            pow_cache,
            rho_cache,
            data.streams,
            data.gap_slot,
            data.burst_offsets,
            alpha,
            beta,
            scale,
            xi_acc,
            gamma_obs_acc,
            gamma_i_acc,
            prior_acc,
        )

        # ---- M-step ----
        new_prior = prior_acc / data.n_bursts
        new_trans = _row_normalize(xi_acc)
        new_obs = _row_normalize(gamma_obs_acc)

        # Keep the chain irreducible: floor off-diagonal transitions.
        if min_trans > 0.0:
            for i in range(n):
                for j in range(n):
                    if i != j and new_trans[i, j] < min_trans:
                        new_trans[i, j] = min_trans
            new_trans = _row_normalize(new_trans)

        prior = _row_normalize(new_prior.reshape(1, -1)).ravel()
        trans = new_trans
        obs = new_obs

        if last_ll - prev_ll < tol and it > 1:
            converged = True
            prev_ll = last_ll
            break
        prev_ll = last_ll

    return H2mmModel(
        prior=prior,
        trans=trans,
        obs=obs,
        loglik=last_ll,
        n_iter=it,
        n_phot=n_phot,
        converged=converged,
    )


# ---------------------------------------------------------------------------
# Viterbi most-likely path + ICL
# ---------------------------------------------------------------------------


@njit(cache=True)
def _viterbi_burst(log_prior, log_obs, log_pow, streams, gap_slot, s, e, path):
    """Viterbi decode one burst into ``path[s:e]``; return its path log-lik."""
    n_states = log_prior.shape[0]
    m_len = e - s
    delta = np.empty((m_len, n_states))
    psi = np.zeros((m_len, n_states), dtype=np.int64)

    y0 = streams[s]
    for i in range(n_states):
        delta[0, i] = log_prior[i] + log_obs[i, y0]

    for rel in range(1, m_len):
        n = s + rel
        slot = gap_slot[n - 1]
        LP = log_pow[slot]
        yn = streams[n]
        for j in range(n_states):
            best = -np.inf
            arg = 0
            for i in range(n_states):
                cand = delta[rel - 1, i] + LP[i, j]
                if cand > best:
                    best = cand
                    arg = i
            delta[rel, j] = best + log_obs[j, yn]
            psi[rel, j] = arg

    best = -np.inf
    arg = 0
    for i in range(n_states):
        if delta[m_len - 1, i] > best:
            best = delta[m_len - 1, i]
            arg = i
    path[e - 1] = arg
    for rel in range(m_len - 1, 0, -1):
        arg = psi[rel, arg]
        path[s + rel - 1] = arg
    return best


def viterbi(model: H2mmModel, data: BurstPhotons) -> Tuple[np.ndarray, float]:
    """Most-likely hidden-state path per photon plus the ICL criterion.

    Parameters
    ----------
    model : H2mmModel
        A (usually optimised) model.
    data : BurstPhotons
        Photon data in engine layout.

    Returns
    -------
    path : numpy.ndarray
        Per-photon most-likely state index, ``int64`` of length ``N``.
    icl : float
        Integrated Complete Likelihood ``-2·logL_path + k·ln(N_phot)``.
    """
    n = model.n_states
    n_dt = int(data.unique_dt.shape[0])
    n_slots = max(n_dt, 1)

    pow_cache = np.zeros((n_slots, n, n))
    rho_cache = np.zeros((n_slots, n, n, n, n))
    if n_dt > 0:
        _build_caches(model.trans, data.unique_dt, pow_cache, rho_cache)

    tiny = np.finfo(np.float64).tiny
    log_prior = np.log(np.clip(model.prior, tiny, None))
    log_obs = np.log(np.clip(model.obs, tiny, None))
    log_pow = np.log(np.clip(pow_cache, tiny, None))

    path = np.zeros(data.n_photons, dtype=np.int64)
    path_ll = 0.0
    offsets = data.burst_offsets
    for b in range(data.n_bursts):
        path_ll += _viterbi_burst(
            log_prior, log_obs, log_pow, data.streams, data.gap_slot,
            int(offsets[b]), int(offsets[b + 1]), path,
        )

    icl = -2.0 * path_ll + model.n_free * math.log(max(data.n_photons, 1))
    return path, icl


# ---------------------------------------------------------------------------
# Model initialisation / simulation
# ---------------------------------------------------------------------------


def factory_model(
    n_states: int,
    n_streams: int,
    trans_scale: float = 1e-4,
    seed: int | None = None,
) -> H2mmModel:
    """Build a reasonable initial model for EM.

    Parameters
    ----------
    n_states : int
        Number of hidden states.
    n_streams : int
        Number of photon streams.
    trans_scale : float
        Off-diagonal transition probability of the initial ``trans`` matrix.
    seed : int, optional
        Seed for the small random spread applied to the emission matrix so
        states are not degenerate.

    Returns
    -------
    H2mmModel
        A row-stochastic initial model.
    """
    rng = np.random.default_rng(seed)
    prior = np.full(n_states, 1.0 / n_states)

    trans = np.full((n_states, n_states), trans_scale)
    for i in range(n_states):
        trans[i, i] = 1.0 - trans_scale * (n_states - 1)
    trans = _row_normalize(trans)

    # Spread emission profiles across the stream axis so states are distinct.
    obs = np.full((n_states, n_streams), 1.0 / n_streams)
    if n_states > 1 and n_streams > 1:
        for i in range(n_states):
            frac = (i + 1) / (n_states + 1)
            profile = np.linspace(1.0 - frac, frac, n_streams)
            profile = np.clip(profile + 0.05 * rng.standard_normal(n_streams), 1e-3, None)
            obs[i] = profile
    obs = _row_normalize(obs)

    return H2mmModel(prior=prior, trans=trans, obs=obs)


def simulate_bursts(
    model: H2mmModel,
    burst_times: Sequence[np.ndarray],
    seed: int | None = None,
) -> List[np.ndarray]:
    """Monte-Carlo sample photon streams from a model along given time axes.

    The hidden chain is advanced tick-by-tick with the one-step ``trans``
    matrix (so it exercises the exact ``A**Δt`` propagation the engine
    caches), and each photon emits a stream drawn from ``obs``.

    Parameters
    ----------
    model : H2mmModel
        The generative model.
    burst_times : sequence of numpy.ndarray
        One monotonically increasing integer macro-time array per burst.
    seed : int, optional
        Random seed.

    Returns
    -------
    list of numpy.ndarray
        Per-burst photon stream indices, matching ``burst_times`` in shape.
    """
    rng = np.random.default_rng(seed)
    n_states = model.n_states
    streams_out: List[np.ndarray] = []
    for t in burst_times:
        t = np.asarray(t).astype(np.int64)
        m = t.shape[0]
        s = np.empty(m, dtype=np.int32)
        state = rng.choice(n_states, p=model.prior)
        for n in range(m):
            if n > 0:
                for _ in range(int(t[n] - t[n - 1])):
                    state = rng.choice(n_states, p=model.trans[state])
            s[n] = rng.choice(model.n_streams, p=model.obs[state])
        streams_out.append(s)
    return streams_out


def fit_states(
    data: BurstPhotons,
    n_states: int,
    n_restarts: int = 1,
    max_iter: int = 500,
    tol: float = 1e-7,
    seed: int | None = 0,
) -> H2mmModel:
    """Fit an ``n_states`` H2MM model, keeping the best of ``n_restarts`` runs.

    Parameters
    ----------
    data : BurstPhotons
        Photon data in engine layout.
    n_states : int
        Number of hidden states to fit.
    n_restarts : int
        Number of random initialisations; the highest-likelihood fit wins.
    max_iter : int
        Maximum EM iterations per restart.
    tol : float
        Convergence threshold on the log-likelihood increment.
    seed : int, optional
        Base seed for restart initialisations.

    Returns
    -------
    H2mmModel
        The best optimised model.
    """
    best: H2mmModel | None = None
    for r in range(max(n_restarts, 1)):
        init = factory_model(
            n_states, data.n_streams,
            seed=None if seed is None else seed + r,
        )
        fit = optimize(init, data, max_iter=max_iter, tol=tol)
        if best is None or fit.loglik > best.loglik:
            best = fit
    assert best is not None
    return best
