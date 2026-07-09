"""High-level H2MM analysis: state scan, model selection, and diagnostics.

Pure compute (numpy only) shared by the backend service and the CLI.  Given
engine-ready :class:`~chisurf.plugins.burst.burst_h2mm.core.h2mm.BurstPhotons`,
it fits a range of state counts, selects the best by BIC/ICL, and derives
Viterbi state paths, dwell times, and transition tables.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .engines import fit_one
from .h2mm import BurstPhotons, H2mmModel, viterbi


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
    scan: list[StateFit]
    fret: np.ndarray
    populations: np.ndarray
    dwell_times: dict[int, np.ndarray]
    transitions: list[Transition]
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
    engine: str = "em",
    surrogates: dict[int, object] | None = None,
    refine_iters: int = 20,
    criterion: str = "bic",
    patience: int | None = None,
) -> list[StateFit]:
    """Fit a model for each requested state count and score BIC/ICL.

    Each state count is fitted independently with ``n_restarts`` random restarts
    using the selected compute ``engine`` (see :mod:`.engines`); model selection
    always scores the fitted models by BIC/ICL.

    When ``patience`` is set, the scan stops fitting higher state counts once the
    ``criterion`` has risen for ``patience + 1`` consecutive counts past the
    running best.  The model-selection curve is typically U-shaped and the
    **over-fit high-``k`` fits are the most expensive** — a redundant state
    creates a flat likelihood ridge, so those fits usually run to ``max_iter``
    without converging.  ``patience=None`` (default) fits every requested count
    (exact, unchanged behaviour); ``patience=1`` gives a safe, ~1.6× faster scan.

    (State-splitting / warm-starting the ``k``-state fit from the ``(k-1)``-state
    solution was evaluated as a speed-up but rejected: a single split cannot undo
    the state merging in the smaller fit, so it reliably reached *worse* optima
    than random restarts on well-separated data — the robust version needs full
    split+merge SMEM, which is out of scope here.)
    """
    use_icl = criterion.lower() == "icl"
    ordered = sorted(int(k) for k in state_counts)
    fits: list[StateFit] = []
    best_score = np.inf
    worse = 0
    for k in ordered:
        model = fit_one(
            data, k, engine,
            surrogates=surrogates, refine_iters=refine_iters,
            n_restarts=n_restarts, max_iter=max_iter, tol=tol, seed=seed,
        )
        _, icl = viterbi(model, data)
        fits.append(
            StateFit(
                n_states=k,
                model=model,
                loglik=float(model.loglik),
                bic=float(model.bic),
                icl=float(icl),
            )
        )
        if patience is not None:
            score = float(icl) if use_icl else float(model.bic)
            if score < best_score:
                best_score = score
                worse = 0
            else:
                worse += 1
                if worse > patience:
                    break
    return fits


def _dwells_and_transitions(
    model: H2mmModel,
    data: BurstPhotons,
    fret: np.ndarray,
) -> tuple[dict[int, list[int]], list[Transition], np.ndarray]:
    """Derive dwell times, transitions, and photon populations via Viterbi."""
    path, _ = viterbi(model, data)
    n_states = model.n_states
    offsets = data.burst_offsets

    dwells: dict[int, list[int]] = {s: [] for s in range(n_states)}
    transitions: list[Transition] = []
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
    engine: str = "em",
    surrogates: dict[int, object] | None = None,
    refine_iters: int = 20,
    patience: int | None = None,
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
    engine : str
        Compute engine for the per-state-count fits (see :mod:`.engines`):
        ``"em"`` (exact, default), ``"em-float32"``, ``"surrogate"``, or
        ``"surrogate-refine"``.
    surrogates : dict, optional
        Mapping ``n_states -> SurrogateModel`` for the surrogate engines;
        missing entries fall back to exact EM.
    refine_iters : int
        EM polish maps for the ``surrogate-refine`` engine.
    patience : int, optional
        Early-stop the state-count scan once the criterion has risen for
        ``patience + 1`` consecutive counts (see :func:`scan_states`); ``None``
        scans every count.

    Returns
    -------
    H2mmAnalysis
        The selected model plus its diagnostics.
    """
    scan = scan_states(
        data, state_counts, n_restarts=n_restarts,
        max_iter=max_iter, tol=tol, seed=seed,
        engine=engine, surrogates=surrogates, refine_iters=refine_iters,
        criterion=criterion, patience=patience,
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
