"""Data models for the H2MM API."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class StreamSettings:
    """One H2MM photon-stream (detector category) definition.

    Attributes
    ----------
    name : str
        Human-readable stream name.
    channels : list of int
        TCSPC routing channel numbers assigned to this stream.
    micro_time_ranges : list of tuple[int, int]
        Inclusive micro-time windows (empty accepts any micro time).
    """

    name: str = "stream"
    channels: list[int] = field(default_factory=list)
    micro_time_ranges: list[tuple[int, int]] = field(default_factory=list)


@dataclass
class H2mmSettings:
    """Settings for an H2MM burst analysis.

    Attributes
    ----------
    streams : list of StreamSettings
        Photon-stream definitions. The first two are treated as donor and
        acceptor for apparent-FRET reporting.
    min_states : int
        Smallest state count to scan.
    max_states : int
        Largest state count to scan.
    criterion : str
        Model-selection criterion, ``"bic"`` or ``"icl"``.
    n_restarts : int
        Random initialisations per state count.
    max_iter : int
        Maximum EM iterations per fit.
    tol : float
        Convergence threshold on the log-likelihood increment.
    time_scale : int
        Integer down-scaling of macro times (coarser base unit).
    min_photons : int
        Minimum stream-assigned photons for a burst to be analysed.
    file_type : str
        tttrlib container name (e.g. ``"SPC-130"`` or ``"auto"``).
    seed : int
        Base RNG seed for reproducible restarts.
    """

    streams: list[StreamSettings] = field(
        default_factory=lambda: [
            StreamSettings("green", [0, 8], []),
            StreamSettings("red", [1, 9], []),
        ]
    )
    min_states: int = 1
    max_states: int = 3
    criterion: str = "bic"
    n_restarts: int = 2
    max_iter: int = 500
    tol: float = 1e-7
    time_scale: int = 1
    min_photons: int = 5
    file_type: str = "SPC-130"
    seed: int = 0

    @property
    def state_counts(self) -> list[int]:
        """The list of state counts to scan."""
        return list(range(int(self.min_states), int(self.max_states) + 1))


@dataclass
class StateFitSummary:
    """Per-state-count model-selection summary."""

    n_states: int
    loglik: float
    bic: float
    icl: float
    converged: bool
    n_iter: int


@dataclass
class H2mmResult:
    """Result of an H2MM analysis.

    Attributes
    ----------
    n_states : int
        Selected number of hidden states.
    criterion : str
        Model-selection criterion used.
    scan : list of StateFitSummary
        Scores for every fitted state count.
    prior : list of float
        Selected model initial-state probabilities.
    trans : list of list of float
        One-step transition matrix (row-stochastic).
    obs : list of list of float
        Emission matrix (row-stochastic).
    trans_rates : list of list of float
        Transition rates in 1/s (diagonal zeroed).
    fret : list of float
        Apparent per-state FRET efficiency.
    populations : list of float
        Viterbi photon fraction per state.
    dwell_mean_s : list of float
        Mean per-state dwell time in seconds.
    n_transitions : int
        Number of within-burst transitions found.
    n_bursts : int
        Bursts analysed.
    n_photons : int
        Photons analysed.
    base_time_s : float
        Seconds per base time unit.
    output_paths : dict
        Files written by the analysis.
    settings_applied : dict
        The settings that were used.
    """

    n_states: int = 0
    criterion: str = "bic"
    scan: list[StateFitSummary] = field(default_factory=list)
    prior: list[float] = field(default_factory=list)
    trans: list[list[float]] = field(default_factory=list)
    obs: list[list[float]] = field(default_factory=list)
    trans_rates: list[list[float]] = field(default_factory=list)
    fret: list[float] = field(default_factory=list)
    populations: list[float] = field(default_factory=list)
    dwell_mean_s: list[float] = field(default_factory=list)
    n_transitions: int = 0
    n_bursts: int = 0
    n_photons: int = 0
    base_time_s: float = 1.0
    output_paths: dict = field(default_factory=dict)
    settings_applied: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return the result as a JSON-compatible dictionary."""
        return asdict(self)
