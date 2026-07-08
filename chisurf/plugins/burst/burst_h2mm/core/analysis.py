"""High-level H2MM analysis: state scan, model selection, and diagnostics.

Pure compute (numpy only) shared by the backend service and the CLI.  Given
engine-ready :class:`~chisurf.plugins.burst.burst_h2mm.core.h2mm.BurstPhotons`,
it fits a range of state counts, selects the best by BIC/ICL, and derives
Viterbi state paths, dwell times, and transition tables.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .h2mm import BurstPhotons, H2mmModel, fit_states, viterbi


@dataclass
class StateFit:
    """One fitted model at a given state count with selection scores."""

    n_states: int
    model: H2mmModel
    loglik: float
    bic: float
    icl: float


@dataclass
class Transition:
    """A within-burst state transition recovered by Viterbi decoding."""

    burst: int
    state_from: int
    state_to: int
    e_from: float
    e_to: float
    time: int


@dataclass
class H2mmAnalysis:
    """Full result of an H2MM analysis run.

    Attributes
    ----------
    best : StateFit
        The selected model (minimum BIC by default).
    scan : list of StateFit
        Every state count that was fitted, in ascending order.
    fret : numpy.ndarray
        Per-state apparent FRET efficiency, shape ``(n_states,)``.
    populations : numpy.ndarray
        Viterbi state populations (photon fraction per state).
    dwell_times : dict
        Maps ``state -> numpy.ndarray`` of dwell durations (base time units).
    transitions : list of Transition
        Within-burst transitions for the transition-density plot.
    trans_rates : numpy.ndarray
        Transition matrix converted to rates (1/s) using ``base_time_s``;
        diagonal is zero.
    base_time_s : float
        Seconds per base time unit.
    n_photons : int
        Total photons analysed.
    n_bursts : int
        Total bursts analysed.
    """

    best: StateFit
    scan: List[StateFit]
    fret: np.ndarray
    populations: np.ndarray
    dwell_times: Dict[int, np.ndarray]
    transitions: List[Transition]
    trans_rates: np.ndarray
    base_time_s: float
    n_photons: int
    n_bursts: int


def state_fret(model: H2mmModel, acceptor_stream: int = 1, donor_stream: int = 0) -> np.ndarray:
    """Return apparent per-state FRET ``E = A / (A + D)`` from the emission matrix."""
    a = model.obs[:, acceptor_stream]
    d = model.obs[:, donor_stream]
    denom = a + d
    with np.errstate(divide="ignore", invalid="ignore"):
        e = np.where(denom > 0, a / denom, np.nan)
    return e


def scan_states(
    data: BurstPhotons,
    state_counts: Sequence[int],
    n_restarts: int = 2,
    max_iter: int = 500,
    tol: float = 1e-7,
    seed: int = 0,
) -> List[StateFit]:
    """Fit a model for each requested state count and score BIC/ICL."""
    fits: List[StateFit] = []
    for k in state_counts:
        model = fit_states(
            data, n_states=int(k), n_restarts=n_restarts,
            max_iter=max_iter, tol=tol, seed=seed,
        )
        _, icl = viterbi(model, data)
        fits.append(
            StateFit(
                n_states=int(k),
                model=model,
                loglik=float(model.loglik),
                bic=float(model.bic),
                icl=float(icl),
            )
        )
    return fits


def _dwells_and_transitions(
    model: H2mmModel,
    data: BurstPhotons,
    fret: np.ndarray,
) -> Tuple[Dict[int, List[int]], List[Transition], np.ndarray]:
    """Derive dwell times, transitions, and photon populations via Viterbi."""
    path, _ = viterbi(model, data)
    n_states = model.n_states
    offsets = data.burst_offsets

    dwells: Dict[int, List[int]] = {s: [] for s in range(n_states)}
    transitions: List[Transition] = []
    populations = np.zeros(n_states, dtype=np.float64)

    # We need macro times to measure dwell durations; reconstruct per burst
    # from gap_slot + unique_dt (cumulative), which mirrors the input times.
    unique_dt = data.unique_dt
    for b in range(data.n_bursts):
        s = int(offsets[b])
        e = int(offsets[b + 1])
        seg = path[s:e]
        for st in seg:
            populations[st] += 1

        # Rebuild relative macro times within the burst.
        t = np.zeros(e - s, dtype=np.int64)
        for rel in range(1, e - s):
            slot = data.gap_slot[s + rel - 1]
            t[rel] = t[rel - 1] + (int(unique_dt[slot]) if slot >= 0 else 0)

        run_start = 0
        for rel in range(1, e - s):
            if seg[rel] != seg[rel - 1]:
                dwells[int(seg[rel - 1])].append(int(t[rel] - t[run_start]))
                transitions.append(
                    Transition(
                        burst=b,
                        state_from=int(seg[rel - 1]),
                        state_to=int(seg[rel]),
                        e_from=float(fret[seg[rel - 1]]),
                        e_to=float(fret[seg[rel]]),
                        time=int(t[rel]),
                    )
                )
                run_start = rel
        # Trailing dwell of the final run.
        dwells[int(seg[-1])].append(int(t[e - s - 1] - t[run_start]))

    if populations.sum() > 0:
        populations /= populations.sum()
    return dwells, transitions, populations


def analyze(
    data: BurstPhotons,
    state_counts: Sequence[int] = (1, 2, 3),
    criterion: str = "bic",
    base_time_s: float = 1.0,
    acceptor_stream: int = 1,
    donor_stream: int = 0,
    n_restarts: int = 2,
    max_iter: int = 500,
    tol: float = 1e-7,
    seed: int = 0,
) -> H2mmAnalysis:
    """Fit, select, and characterise an H2MM model over a range of states.

    Parameters
    ----------
    data : BurstPhotons
        Photon data in engine layout.
    state_counts : sequence of int
        State counts to scan (e.g. ``(1, 2, 3, 4)``).
    criterion : {"bic", "icl"}
        Model-selection criterion (minimised).
    base_time_s : float
        Seconds per base time unit, used to convert transition probabilities
        to rates.
    acceptor_stream, donor_stream : int
        Stream indices used to compute apparent per-state FRET.
    n_restarts, max_iter, tol, seed
        Passed through to the optimiser.

    Returns
    -------
    H2mmAnalysis
        The selected model plus its diagnostics.
    """
    scan = scan_states(
        data, state_counts, n_restarts=n_restarts,
        max_iter=max_iter, tol=tol, seed=seed,
    )
    key = (lambda f: f.icl) if criterion.lower() == "icl" else (lambda f: f.bic)
    best = min(scan, key=key)

    fret = state_fret(best.model, acceptor_stream, donor_stream)
    dwells, transitions, populations = _dwells_and_transitions(best.model, data, fret)
    dwell_arrays = {s: np.asarray(v, dtype=np.float64) for s, v in dwells.items()}

    # Transition probabilities → rates (1/s); diagonal set to zero.
    trans = best.model.trans
    rates = (trans / base_time_s if base_time_s > 0 else trans).copy()
    np.fill_diagonal(rates, 0.0)

    return H2mmAnalysis(
        best=best,
        scan=scan,
        fret=fret,
        populations=populations,
        dwell_times=dwell_arrays,
        transitions=transitions,
        trans_rates=rates,
        base_time_s=base_time_s,
        n_photons=data.n_photons,
        n_bursts=data.n_bursts,
    )
