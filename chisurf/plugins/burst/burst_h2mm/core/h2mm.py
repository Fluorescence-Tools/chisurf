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
import os
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

try:  # numba is a first-class dependency; degrade gracefully if unavailable
    from numba import get_num_threads, njit, prange

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
        """Return a serial range (``prange`` fallback without numba)."""
        return range(*args)

    def get_num_threads():  # type: ignore
        """Return a single thread (fallback without numba)."""
        return 1


def _sync_numba_threads() -> None:
    """Pin ``NUMBA_NUM_THREADS`` back to numba's already-launched pool size.

    ChiSurf's startup (``chisurf.core.settings.env_bootstrap``) may rewrite the
    ``NUMBA_NUM_THREADS`` environment variable *after* numba's threadpool has
    launched.  numba re-reads that variable on every fresh (cold) compile and
    raises if it no longer matches the launched pool — so the first new kernel
    specialisation compiled after such a rewrite (e.g. the ``float32`` E-step)
    would crash.  Rewriting the env back to the launched count keeps late cold
    compiles valid without touching the pool.
    """
    if not _HAVE_NUMBA:
        return
    try:
        from numba import config as _nb_config

        os.environ["NUMBA_NUM_THREADS"] = str(_nb_config.NUMBA_NUM_THREADS)
    except Exception:  # pragma: no cover - defensive only
        pass


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

    def normalize(self) -> H2mmModel:
        """Renormalise ``prior``, and the rows of ``trans`` and ``obs``."""
        self.prior = _row_normalize(self.prior.reshape(1, -1)).ravel()
        self.trans = _row_normalize(self.trans)
        self.obs = _row_normalize(self.obs)
        return self

    def copy(self) -> H2mmModel:
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

    kept_times: list[np.ndarray] = []
    kept_streams: list[np.ndarray] = []
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
    all_dt: list[np.ndarray] = []
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


@njit(cache=True, fastmath=True)
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


@njit(cache=True, fastmath=True)
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


@njit(parallel=True, cache=True, fastmath=True)
def _build_caches(A, unique_dt, pow_cache, rho_cache):
    """Fill ``pow_cache[s]=A^Δt`` and ``rho_cache[s]=ρ(Δt)`` for each slot ``s``.

    Robust ``O(log Δt)`` binary-exponentiation build (the associative pair-power
    of the paper); used for every model and as the fallback whenever the
    spectral build (:func:`_build_caches_eig`) declines.
    """
    R1 = _rho_base(A)
    for s in prange(unique_dt.shape[0]):
        P, R = _pair_pow(A, R1, unique_dt[s])
        pow_cache[s] = P
        rho_cache[s] = R


# Spectral build is preferred only when intervals are long enough that the
# ``O(log Δt)`` pair-power does meaningful work.  Benchmarks put the crossover
# past this Δt for ``n ≥ 3``; below it (dense high-count-rate data, tiny Δt) the
# pair-power is already trivially cheap and the eig setup would dominate.  For
# ``n == 2`` the crossover is far higher, so the spectral build is not used
# there (see :func:`_prefer_eig`).
_EIG_MIN_DT = 512


def _prefer_eig(n_states: int, unique_dt: np.ndarray) -> bool:
    """Whether the spectral cache build is expected to beat pair-power here."""
    return (
        unique_dt.shape[0] > 0
        and n_states >= 3
        and int(unique_dt.max()) >= _EIG_MIN_DT
    )


def _build_caches_eig(A, unique_dt, pow_cache, rho_cache):
    r"""Fill the ``A^Δt`` / ``ρ(Δt)`` caches from an eigendecomposition of ``A``.

    Uses the spectral closed form instead of binary exponentiation.  With
    ``A = V·diag(λ)·V⁻¹`` the τ-sum inside ``ρ`` collapses to a **divided
    difference**::

        ρ(Δt)[k,m,i,j] = A[i,j] · Re Σ_{a,b} V[k,a] V⁻¹[a,i] V[j,b] V⁻¹[b,m] · D[a,b]
        D[a,b] = (λ_a^Δt − λ_b^Δt)/(λ_a − λ_b)      (a≠b)
               = Δt · λ_a^{Δt−1}                     (a=b, confluent limit)

    and ``A^Δt = Re(V·diag(λ^Δt)·V⁻¹)``.  Cost is ``O(n³)`` for the one
    decomposition plus a vectorised per-slot contraction, with **no** dependence
    on ``Δt`` beyond the elementwise power — so it wins over pair-power when the
    unique intervals are long (sparse photon streams).

    Only valid when ``A`` is diagonalisable; the function returns ``False`` (and
    leaves the caches untouched) when the eigenvectors are ill-conditioned or the
    reconstruction is inaccurate, so the caller can fall back to
    :func:`_build_caches`.

    Returns
    -------
    bool
        ``True`` if the caches were filled, ``False`` if the caller must fall
        back to the pair-power build.
    """
    n = A.shape[0]
    if n == 1:
        return False  # trivial; pair-power handles it without eig machinery
    try:
        lam, V = np.linalg.eig(A)
        Vinv = np.linalg.inv(V)
    except np.linalg.LinAlgError:
        return False
    # Diagonalisability / conditioning guard: a defective or near-defective A
    # makes the divided differences blow up — fall back instead.
    recon = (V * lam) @ Vinv
    if np.linalg.cond(V) > 1e8 or np.abs(recon - A).max() > 1e-9:
        return False

    dt = unique_dt.astype(np.float64)
    lam_dt = lam[None, :] ** unique_dt[:, None]  # (S, n) complex
    lam_dtm1 = lam[None, :] ** (unique_dt[:, None] - 1)

    num = lam_dt[:, :, None] - lam_dt[:, None, :]  # (S, n, n)
    den = lam[None, :, None] - lam[None, None, :]  # (1, n, n)
    with np.errstate(invalid="ignore", divide="ignore"):
        D = num / np.where(den == 0.0, 1.0, den)
    confluent = dt[:, None] * lam_dtm1  # (S, n) = Δt·λ^{Δt−1}
    close = np.broadcast_to(np.abs(den) <= 1e-12, D.shape)
    D = np.where(close, np.broadcast_to(confluent[:, :, None], D.shape), D)

    # A^Δt = Re(V diag(λ^Δt) V⁻¹), then row-normalise for parity with pair-power.
    P = np.einsum("ia,sa,aj->sij", V, lam_dt, Vinv, optimize=True).real
    rs = P.sum(axis=2, keepdims=True)
    np.divide(P, rs, out=P, where=rs > 0.0)

    # ρ[s,k,m,i,j] = A[i,j]·Re Σ_{a,b} V[k,a]V⁻¹[a,i] V[j,b]V⁻¹[b,m] D[s,a,b]
    R = np.einsum("ka,ai,sab,jb,bm->skmij", V, Vinv, D, V, Vinv, optimize=True).real
    R *= A[None, None, None, :, :]

    pow_cache[:] = P
    rho_cache[:] = R
    return True


def _fill_caches(A, unique_dt, pow_cache, rho_cache, prefer_eig):
    """Fill both caches, using the spectral build when ``prefer_eig`` and valid.

    Falls back to the robust pair-power :func:`_build_caches` whenever the
    spectral build is disabled or declines (non-diagonalisable / ill-conditioned
    ``A``).
    """
    if prefer_eig and _build_caches_eig(A, unique_dt, pow_cache, rho_cache):
        return
    _build_caches(A, unique_dt, pow_cache, rho_cache)


# ---------------------------------------------------------------------------
# E-step: scaled forward-backward + Baum-Welch accumulation
# ---------------------------------------------------------------------------


@njit(parallel=True, fastmath=True)
def _estep(
    prior,
    obs,
    pow_cache,
    rho_cache,
    streams,
    gap_slot,
    offsets,
    alpha,
    scale,
    xi_acc,
    gamma_obs_acc,
    prior_acc,
):
    """Run scaled forward-backward over all bursts; accumulate BW statistics.

    Bursts are partitioned into contiguous chunks processed in parallel.  Each
    thread accumulates its Baum-Welch statistics into **thread-local** arrays
    (so the hot ``ξ``/``γ`` writes never touch memory another core is writing —
    no false sharing) and touches only disjoint photon ranges of
    ``alpha``/``scale``.  The per-thread partials are stored to shared buffers
    once per chunk and reduced serially.  Returns the total log-likelihood
    ``Σ_bursts Σ_n log(scale[n])``.

    Only two sweeps over each burst's photons are made: a **forward** pass
    filling ``alpha``/``scale``, then a **single backward** pass that folds in
    both the occupancy statistics (``γ``) and the transition weight (``W``,
    below).  ``β`` is held as two ``(n,)`` vectors (``β[n]`` needs only
    ``β[n+1]``), so the full ``(N, n)`` backward array is never materialised —
    halving the large-array memory traffic that dominates at high photon counts.

    Transition statistics are *not* contracted against the full ``ρ`` tensor per
    photon (that would cost ``O(N·n⁴)`` and stream ``rho_cache`` from memory once
    per gap).  Instead each thread accumulates the per-slot weight
    ``W[slot,k,m] = Σ_gaps α[n,k]·obs[m,y_{n+1}]·β[n+1,m]/c`` in ``O(N·n²)``; the
    single ``ξ[i,j] = Σ_{slot,k,m} W[slot,k,m]·ρ[slot,k,m,i,j]`` contraction
    (``O(n_slots·n⁴)``, ``n_slots`` = unique Δt ≪ N) is done once in the serial
    reduction.  Mathematically identical, but the hot loop is ``O(n²)`` and
    ``rho_cache`` is read only ``n_slots`` times.
    """
    n_states = prior.shape[0]
    n_streams = obs.shape[1]
    n_bursts = offsets.shape[0] - 1
    n_slots = pow_cache.shape[0]
    nthreads = get_num_threads()

    W_p = np.zeros((nthreads, n_slots, n_states, n_states))
    gobs_p = np.zeros((nthreads, n_states, n_streams))
    prior_p = np.zeros((nthreads, n_states))
    ll_p = np.zeros(nthreads)

    for c in prange(nthreads):
        b0 = c * n_bursts // nthreads
        b1 = (c + 1) * n_bursts // nthreads
        # Thread-local accumulators (own stack → no cross-core false sharing).
        w = np.empty(n_states)
        beta_next = np.empty(n_states)
        beta_cur = np.empty(n_states)
        W_local = np.zeros((n_slots, n_states, n_states))
        gobs_local = np.zeros((n_states, n_streams))
        prior_local = np.zeros(n_states)
        ll_local = 0.0
        for b in range(b0, b1):
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
                ll_local += math.log(tot)

            for n in range(s + 1, e):
                slot = gap_slot[n - 1]
                yn = streams[n]
                tot = 0.0
                for i in range(n_states):
                    v = 0.0
                    for k in range(n_states):
                        v += alpha[n - 1, k] * pow_cache[slot, k, i]
                    v *= obs[i, yn]
                    alpha[n, i] = v
                    tot += v
                scale[n] = tot
                if tot > 0.0:
                    for i in range(n_states):
                        alpha[n, i] /= tot
                    ll_local += math.log(tot)

            # ---- backward, with γ (occupancy) and W (transitions) fused in ----
            # β is carried as two (n,) vectors; the full (N,n) array is never
            # built.  ``ξ`` is not formed here — the ρ contraction is deferred to
            # the serial reduction so this hot loop stays ``O(n²)`` per gap and
            # ``rho_cache`` is untouched until then (see the docstring).
            yl = streams[e - 1]
            for i in range(n_states):
                beta_next[i] = 1.0  # β[e-1] = 1
                g = alpha[e - 1, i]
                gobs_local[i, yl] += g
                if e - 1 == s:
                    prior_local[i] += g
            for n in range(e - 2, s - 1, -1):
                slot = gap_slot[n]
                yn1 = streams[n + 1]
                cc = scale[n + 1]
                inv_c = 1.0 / cc if cc > 0.0 else 0.0
                # w = obs · β[n+1]; reused by both β[n] and the W accumulation.
                for k in range(n_states):
                    w[k] = obs[k, yn1] * beta_next[k]
                for i in range(n_states):
                    v = 0.0
                    for k in range(n_states):
                        v += pow_cache[slot, i, k] * w[k]
                    beta_cur[i] = v * inv_c
                # γ at photon n (occupancy α[n]·β[n]).
                yn = streams[n]
                for i in range(n_states):
                    g = alpha[n, i] * beta_cur[i]
                    gobs_local[i, yn] += g
                    if n == s:
                        prior_local[i] += g
                # W over gap n (skipped when the scale underflowed).
                if cc > 0.0:
                    for k in range(n_states):
                        ak = alpha[n, k] * inv_c
                        if ak == 0.0:
                            continue
                        for m in range(n_states):
                            W_local[slot, k, m] += ak * w[m]
                # advance β[n+1] ← β[n]
                for i in range(n_states):
                    beta_next[i] = beta_cur[i]

        W_p[c] = W_local
        gobs_p[c] = gobs_local
        prior_p[c] = prior_local
        ll_p[c] = ll_local

    # ---- reduce per-thread accumulators ----
    loglik = 0.0
    for c in range(nthreads):
        loglik += ll_p[c]
        for i in range(n_states):
            prior_acc[i] += prior_p[c, i]
            for k in range(n_streams):
                gamma_obs_acc[i, k] += gobs_p[c, i, k]

    # Reduce W across threads, then contract with ρ once per slot:
    # ξ[i,j] = Σ_{slot,k,m} W[slot,k,m]·ρ[slot,k,m,i,j].
    for slot in range(n_slots):
        for k in range(n_states):
            for m in range(n_states):
                wkm = 0.0
                for c in range(nthreads):
                    wkm += W_p[c, slot, k, m]
                if wkm == 0.0:
                    continue
                for i in range(n_states):
                    for j in range(n_states):
                        xi_acc[i, j] += wkm * rho_cache[slot, k, m, i, j]

    return loglik


# ---------------------------------------------------------------------------
# EM optimiser
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Parameter-vector helpers for SQUAREM acceleration
# ---------------------------------------------------------------------------


def _pack(prior: np.ndarray, trans: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """Flatten ``(prior, trans, obs)`` into one contiguous parameter vector."""
    return np.concatenate([prior.ravel(), trans.ravel(), obs.ravel()])


def _unpack(vec: np.ndarray, n: int, p: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split a parameter vector back into ``(prior, trans, obs)`` copies."""
    prior = vec[:n].copy()
    trans = vec[n : n + n * n].reshape(n, n).copy()
    obs = vec[n + n * n :].reshape(n, p).copy()
    return prior, trans, obs


def _project(vec: np.ndarray, n: int, p: int, min_trans: float) -> np.ndarray:
    """Map an extrapolated parameter vector back onto the feasible model set.

    Clips negatives, renormalises ``prior`` and every ``trans``/``obs`` row, and
    re-applies the off-diagonal ``min_trans`` floor so the projected model is a
    valid EM input (the SQUAREM step can otherwise overshoot outside the
    simplex).
    """
    prior = _row_normalize(np.clip(vec[:n], 0.0, None).reshape(1, -1)).ravel()
    trans = _row_normalize(np.clip(vec[n : n + n * n].reshape(n, n), 0.0, None))
    obs = _row_normalize(np.clip(vec[n + n * n :].reshape(n, p), 0.0, None))
    if min_trans > 0.0:
        for i in range(n):
            for j in range(n):
                if i != j and trans[i, j] < min_trans:
                    trans[i, j] = min_trans
        trans = _row_normalize(trans)
    return _pack(prior, trans, obs)


def _plain_em(em_step, prior, trans, obs, max_iter, tol):
    """Classic Baum-Welch loop: iterate the EM map until the logL increment < ``tol``."""
    prev_ll = -np.inf
    last_ll = -np.inf
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        prior, trans, obs, last_ll = em_step(prior, trans, obs)
        if last_ll - prev_ll < tol and it > 1:
            converged = True
            prev_ll = last_ll
            break
        prev_ll = last_ll
    return prior, trans, obs, last_ll, it, converged


def _squarem(em_step, prior, trans, obs, n, p, max_iter, tol, min_trans):
    r"""SQUAREM-accelerated EM (Varadhan & Roland 2008, scheme S3).

    Each outer step takes two ordinary EM maps ``θ→p1→p2``, forms the squared
    extrapolation ``θ' = θ − 2α r + α² v`` with ``r = p1−θ``, ``v = p2−2p1+θ``
    and steplength ``α = −‖r‖/‖v‖ ≤ −1``, projects ``θ'`` back onto the model
    simplex, and runs one stabilising EM map from it.  A monotonicity safeguard
    keeps the better of the accelerated point and the plain double-EM point, so
    the accepted log-likelihood is non-decreasing and the fixed point is exactly
    that of plain EM — only reached in far fewer maps.  ``n_iter`` counts EM-map
    evaluations, so it is directly comparable to the plain-EM iteration count.
    """
    def em_vec(vec):
        pr, tr, ob = _unpack(vec, n, p)
        npr, ntr, nob, ll = em_step(pr, tr, ob)
        return _pack(npr, ntr, nob), ll

    theta = _pack(prior, trans, obs)
    prev_ll = -np.inf
    last_ll = -np.inf
    converged = False
    evals = 0
    while evals < max_iter:
        p1, _l0 = em_vec(theta)
        evals += 1
        if evals >= max_iter:
            theta, last_ll = p1, _l0
            break
        r = p1 - theta
        p2, l1 = em_vec(p1)
        evals += 1
        v = (p2 - p1) - r
        rn = math.sqrt(float(r @ r))
        vn = math.sqrt(float(v @ v))
        if vn < 1e-12 or rn < 1e-12:
            # Already at (or numerically indistinguishable from) the fixed point.
            theta, last_ll = p2, l1
            if l1 - prev_ll < tol:
                converged = True
                break
            prev_ll = l1
            continue
        a = -rn / vn
        if a > -1.0:
            a = -1.0
        theta_e = _project(theta - 2.0 * a * r + (a * a) * v, n, p, min_trans)
        if evals >= max_iter:
            theta, last_ll = p2, l1
            break
        p3, l2 = em_vec(theta_e)
        evals += 1
        # Monotonicity safeguard: fall back to plain double-EM if the accelerated
        # point did not improve on it (or went non-finite).
        if (not math.isfinite(l2)) or l2 < l1:
            theta, last_ll = p2, l1
        else:
            theta, last_ll = p3, l2
        if last_ll - prev_ll < tol and evals > 2:
            converged = True
            break
        prev_ll = last_ll

    pr, tr, ob = _unpack(theta, n, p)
    return pr, tr, ob, last_ll, evals, converged


def optimize(
    model: H2mmModel,
    data: BurstPhotons,
    max_iter: int = 500,
    tol: float = 1e-7,
    min_trans: float = 1e-12,
    accelerate: bool = True,
    single_precision: bool = False,
) -> H2mmModel:
    """Baum-Welch (EM) optimisation of an H2MM model.

    Parameters
    ----------
    model : H2mmModel
        Initial model; not modified in place.
    data : BurstPhotons
        Photon data in engine layout (from :func:`prepare_bursts`).
    max_iter : int
        Maximum number of EM-map evaluations.
    tol : float
        Convergence threshold on the log-likelihood increment.
    min_trans : float
        Floor for off-diagonal transition probabilities, keeping ``trans``
        irreducible so rare transitions can still be discovered.
    accelerate : bool
        Use SQUAREM extrapolation (:func:`_squarem`) to reach the EM fixed point
        in fewer maps.  The fixed point is identical to plain EM; set ``False``
        for the unaccelerated loop.
    single_precision : bool
        Run the forward-backward hot loop and the ``A^Δt``/``ρ`` caches in
        ``float32`` to roughly halve their memory bandwidth (the model
        parameters and Baum-Welch reductions stay ``float64``).  This is an
        **approximate** fast mode: the log-likelihood carries ``float32`` round-off
        (~1e-2 at typical magnitudes), so it does *not* meet the ~1e-9 reference
        tolerance and the convergence threshold is floored accordingly.  Use for
        exploratory fits on very large datasets, not for final numbers.

    Returns
    -------
    H2mmModel
        The optimised model with ``loglik``, ``n_iter``, ``n_phot`` and
        ``converged`` populated.
    """
    _sync_numba_threads()  # keep the float32 kernel compilable after env rewrites
    n = model.n_states
    p = data.n_streams
    n_dt = int(data.unique_dt.shape[0])

    prior = _row_normalize(model.prior.reshape(1, -1)).ravel()
    trans = _row_normalize(model.trans)
    obs = _row_normalize(model.obs)

    # float32 round-off swamps a tight logL threshold, so raise the floor.
    cdt = np.float32 if single_precision else np.float64
    if single_precision:
        tol = max(tol, 1e-3)

    n_phot = data.n_photons
    alpha = np.zeros((n_phot, n), dtype=cdt)
    scale = np.zeros(n_phot, dtype=cdt)
    n_slots = max(n_dt, 1)
    pow_cache = np.zeros((n_slots, n, n), dtype=cdt)
    rho_cache = np.zeros((n_slots, n, n, n, n), dtype=cdt)

    # Spectral cache build pays off only when the unique intervals are long.
    prefer_eig = _prefer_eig(n, data.unique_dt)

    def em_step(prior_, trans_, obs_):
        """One EM map: returns ``(new_prior, new_trans, new_obs, logL(input))``."""
        if n_dt > 0:
            _fill_caches(trans_, data.unique_dt, pow_cache, rho_cache, prefer_eig)

        xi_acc = np.zeros((n, n))
        gamma_obs_acc = np.zeros((n, p))
        prior_acc = np.zeros(n)

        ll = _estep(
            prior_.astype(cdt, copy=False),
            obs_.astype(cdt, copy=False),
            pow_cache,
            rho_cache,
            data.streams,
            data.gap_slot,
            data.burst_offsets,
            alpha,
            scale,
            xi_acc,
            gamma_obs_acc,
            prior_acc,
        )

        # ---- M-step ----
        new_prior = _row_normalize((prior_acc / data.n_bursts).reshape(1, -1)).ravel()
        new_trans = _row_normalize(xi_acc)
        new_obs = _row_normalize(gamma_obs_acc)

        # Keep the chain irreducible: floor off-diagonal transitions.
        if min_trans > 0.0:
            for i in range(n):
                for j in range(n):
                    if i != j and new_trans[i, j] < min_trans:
                        new_trans[i, j] = min_trans
            new_trans = _row_normalize(new_trans)

        return new_prior, new_trans, new_obs, ll

    if accelerate:
        prior, trans, obs, last_ll, it, converged = _squarem(
            em_step, prior, trans, obs, n, p, max_iter, tol, min_trans
        )
    else:
        prior, trans, obs, last_ll, it, converged = _plain_em(
            em_step, prior, trans, obs, max_iter, tol
        )

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


@njit(parallel=True, cache=True)
def _viterbi_all(log_prior, log_obs, log_pow, streams, gap_slot, offsets, path):
    """Viterbi-decode every burst in parallel; return the summed path log-lik.

    Each burst writes a disjoint ``path[s:e]`` range and allocates its own
    ``delta``/``psi`` work arrays (in :func:`_viterbi_burst`), so the ``prange``
    over bursts has no shared-write hazard — the same partitioning the E-step
    uses.  The per-burst path log-likelihoods reduce into ``total``.
    """
    n_bursts = offsets.shape[0] - 1
    total = 0.0
    for b in prange(n_bursts):
        total += _viterbi_burst(
            log_prior, log_obs, log_pow, streams, gap_slot,
            offsets[b], offsets[b + 1], path,
        )
    return total


def viterbi(model: H2mmModel, data: BurstPhotons) -> tuple[np.ndarray, float]:
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
        _fill_caches(
            model.trans, data.unique_dt, pow_cache, rho_cache,
            _prefer_eig(n, data.unique_dt),
        )

    tiny = np.finfo(np.float64).tiny
    log_prior = np.log(np.clip(model.prior, tiny, None))
    log_obs = np.log(np.clip(model.obs, tiny, None))
    log_pow = np.log(np.clip(pow_cache, tiny, None))

    path = np.zeros(data.n_photons, dtype=np.int64)
    _sync_numba_threads()
    path_ll = _viterbi_all(
        log_prior, log_obs, log_pow, data.streams, data.gap_slot,
        data.burst_offsets, path,
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
) -> list[np.ndarray]:
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
    streams_out: list[np.ndarray] = []
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
    surrogate=None,
    refine_iters: int = 0,
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
    surrogate : SurrogateModel or path, optional
        If given, use the **optional** amortised neural estimator
        (:mod:`.surrogate`) instead of EM: the model is predicted in one forward
        pass (an *approximate* MLE — see that module), optionally polished by
        ``refine_iters`` Baum-Welch maps.  ``None`` (default) uses EM unchanged.
    refine_iters : int
        Baum-Welch maps to polish the surrogate estimate (ignored without a
        ``surrogate``).

    Returns
    -------
    H2mmModel
        The best optimised model.
    """
    if surrogate is not None:
        from .surrogate import estimate_model

        return estimate_model(
            data, n_states, surrogate, refine_iters=refine_iters, tol=tol
        )

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
