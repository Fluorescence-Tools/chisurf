"""Photon-stream simulator for 2D-FLC (port of ``TK_MyMain_Simu_PhotonStream``).

Generates a TTTR-like single-molecule photon stream from an ``n``-state exchange process:
each state has a fluorescence lifetime and brightness, and the states interconvert
according to a rate matrix. The output is exactly what the rest of the plugin consumes —
macro-time ticks, micro-time (TCSPC) ticks and the (ground-truth) state per photon — so it
closes the loop for validation: simulate, then recover the lifetimes and rate matrix.

The MATLAB reference advances a fixed ``Tstep`` and tests for a transition/emission every
step. This implementation is the statistically-equivalent **event-driven** form: state
sojourns are drawn from the continuous-time Markov chain (Gillespie) and photons within a
sojourn are a Poisson process, which is exact and far faster (``O(n_photons)`` rather than
``O(total_time / Tstep)``). Micro-times are sampled from the state lifetime plus an IRF
offset drawn from the (area-normalized) instrument response.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fit.kinetics import equilibrium_populations

__all__ = ["SimulatedStream", "simulate_photon_stream", "dwell_time_histogram"]


@dataclass
class SimulatedStream:
    """A simulated TTTR photon stream with ground-truth state labels."""

    macro_times: np.ndarray  # int64 macro ticks (resolution = macro_time_resolution_s)
    micro_times: np.ndarray  # int64 TCSPC channel indices (resolution = tstep_ns)
    states: np.ndarray  # int per-photon state (ground truth)
    macro_time_resolution_s: float
    micro_time_resolution_ns: float
    n_microtime_channels: int
    equilibrium_populations: np.ndarray


def simulate_photon_stream(
    rate_matrix: np.ndarray,
    lifetimes_ns,
    intensities_cps,
    *,
    total_time_s: float = 100.0,
    irf: np.ndarray | None = None,
    irf_time_ns: np.ndarray | None = None,
    macro_time_resolution_s: float = 1e-6,
    tstep_ns: float = 0.004,
    n_microtime_channels: int = 3127,
    seed: int = 0,
    max_photons: int = 50_000_000,
) -> SimulatedStream:
    """Simulate a single-molecule photon stream from an n-state exchange process.

    Parameters
    ----------
    rate_matrix
        ``K[n, m]`` = rate of ``n -> m`` (1/s); diagonal ignored.
    lifetimes_ns
        Fluorescence lifetime of each state (ns).
    intensities_cps
        Brightness of each state (counts per second).
    total_time_s
        Total acquisition time to simulate.
    irf, irf_time_ns
        Optional instrument response and its ns axis. Without an IRF a delta at t=0 is used.
    macro_time_resolution_s, tstep_ns, n_microtime_channels
        TTTR/TCSPC calibration of the output.
    seed
        RNG seed (deterministic output).
    max_photons
        Safety cap on the number of photons generated (stops early if exceeded).
    """
    rng = np.random.default_rng(seed)
    K = np.asarray(rate_matrix, dtype=float)
    n_states = K.shape[0]
    tau = np.asarray(lifetimes_ns, dtype=float)
    inten = np.asarray(intensities_cps, dtype=float)

    exit_rates = np.array([K[i].sum() - K[i, i] for i in range(n_states)])
    p_eq = equilibrium_populations(K)

    # IRF inverse-CDF sampling table (offset in ns).
    if irf is not None and irf_time_ns is not None:
        w = np.clip(np.asarray(irf, dtype=float), 0.0, None)
        w = w / w.sum() if w.sum() > 0 else np.ones_like(w) / w.size
        irf_cdf = np.cumsum(w)
        irf_t = np.asarray(irf_time_ns, dtype=float)
    else:
        irf_cdf = None
        irf_t = None

    macro_chunks: list[np.ndarray] = []
    micro_chunks: list[np.ndarray] = []
    state_chunks: list[np.ndarray] = []

    # initial state from equilibrium
    state = int(rng.choice(n_states, p=p_eq))
    t = 0.0
    n_total = 0
    while t < total_time_s and n_total < max_photons:
        rate = exit_rates[state]
        dwell = rng.exponential(1.0 / rate) if rate > 0 else (total_time_s - t)
        dwell = min(dwell, total_time_s - t)

        # photons in this sojourn: Poisson process at the state brightness
        n_ph = rng.poisson(inten[state] * dwell)
        if n_ph:
            arrivals = t + np.sort(rng.uniform(0.0, dwell, n_ph))
            macro = np.rint(arrivals / macro_time_resolution_s).astype(np.int64)
            # lifetime sample + IRF offset -> micro tick
            life_ns = rng.exponential(tau[state], n_ph)
            if irf_cdf is not None:
                u = rng.uniform(0.0, 1.0, n_ph)
                offset = irf_t[np.searchsorted(irf_cdf, u, side="left").clip(0, irf_t.size - 1)]
            else:
                offset = 0.0
            micro = np.rint((life_ns + offset) / tstep_ns).astype(np.int64)
            np.clip(micro, 0, n_microtime_channels - 1, out=micro)
            macro_chunks.append(macro)
            micro_chunks.append(micro)
            state_chunks.append(np.full(n_ph, state, dtype=np.int16))
            n_total += n_ph

        t += dwell
        if rate > 0:
            probs = np.array([K[state, m] if m != state else 0.0 for m in range(n_states)])
            probs = probs / probs.sum()
            state = int(rng.choice(n_states, p=probs))

    if not macro_chunks:
        macro = np.zeros(0, dtype=np.int64)
        micro = np.zeros(0, dtype=np.int64)
        states = np.zeros(0, dtype=np.int16)
    else:
        macro = np.concatenate(macro_chunks)
        micro = np.concatenate(micro_chunks)
        states = np.concatenate(state_chunks)
        order = np.argsort(macro, kind="stable")  # macro times must be ascending
        macro, micro, states = macro[order], micro[order], states[order]

    return SimulatedStream(
        macro_times=macro,
        micro_times=micro,
        states=states,
        macro_time_resolution_s=macro_time_resolution_s,
        micro_time_resolution_ns=tstep_ns,
        n_microtime_channels=n_microtime_channels,
        equilibrium_populations=p_eq,
    )


def dwell_time_histogram(
    states: np.ndarray,
    macro_times: np.ndarray,
    macro_time_resolution_s: float,
    *,
    bin_s: float = 0.01,
    n_states: int | None = None,
):
    """Per-transition dwell-time histograms (port of the simulator's dwell check).

    Returns ``(centers_s, hist)`` where ``hist[n, m]`` is the dwell-time histogram of
    sojourns in state ``n`` that end in a transition to state ``m``.
    """
    states = np.asarray(states)
    t = np.asarray(macro_times, dtype=float) * macro_time_resolution_s
    if n_states is None:
        n_states = int(states.max()) + 1 if states.size else 1
    change = np.flatnonzero(np.diff(states) != 0)
    dwells = {}  # (n, m) -> list of dwell times
    start_t = t[0] if t.size else 0.0
    prev = states[0] if states.size else 0
    for idx in change:
        n = int(prev)
        m = int(states[idx + 1])
        dwell = t[idx] - start_t
        dwells.setdefault((n, m), []).append(dwell)
        start_t = t[idx + 1]
        prev = m

    all_d = [d for lst in dwells.values() for d in lst]
    t_max = max(all_d) if all_d else bin_s
    edges = np.arange(0.0, t_max + bin_s, bin_s)
    centers = 0.5 * (edges[:-1] + edges[1:])
    hist = np.zeros((n_states, n_states, centers.size))
    for (n, m), lst in dwells.items():
        hist[n, m] = np.histogram(lst, bins=edges)[0]
    return centers, hist
