"""JSON-compatible request and result models for MaxEnt MEM services."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(slots=True)
class MEMSettings:
    """Settings for a MaxEnt MEM job."""

    mode: Literal["lifetime", "fret"] = "lifetime"
    nu: float = 1e-3
    max_iter: int = 200
    tol: float = 1e-4
    tau_min: float = 0.01
    tau_max: float = 6.0
    tau_bins: int = 192
    tau0: float = 4.1
    R0: float = 52.0
    r_min_frac: float = 0.1
    r_max_frac: float = 3.0
    r_bins: int = 96
    timeshift: float = 0.0
    background: float = 0.0
    lamp_scatter: float = 0.0
    irf_background: float | None = None
    optimize_nuisance: bool = False
    period: float | None = None
    x_donly: float = 0.0
    fit_start_fraction: float = 0.9
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class MEMRequest:
    """Request payload for a MaxEnt MEM job."""

    decay: list[float]
    irf: list[float]
    dt: float
    fitrange: tuple[int, int] | None = None
    settings: MEMSettings = field(default_factory=MEMSettings)
    prior: list[float] | None = None
    donly: list[float] | None = None


@dataclass(slots=True)
class MEMResult:
    """Normalized result returned by MaxEnt MEM jobs."""

    p: list[float]
    axis: list[float]
    chisq: float
    S: float
    nu: float
    timeshift: float
    background: float
    fit_curve: list[float]
    residuals: list[float]
    fitrange: tuple[int, int]
    history: list[tuple[float, float, float, float]]
    mode: str
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class LCurveResult:
    """L-curve sweep result."""

    log10_nu: list[float]
    chi2r: list[float]
    sol_norm: list[float]
    corner_index: int | None


@dataclass(slots=True)
class ContractDescriptor:
    """Workflow contract descriptor."""

    plugin_id: str = "maxent_decay"
    contract_version: str = "1.0.0"
    methods: tuple[str, ...] = (
        "maxent_decay.jobs.run_lifetime_mem",
        "maxent_decay.jobs.run_fret_mem",
        "maxent_decay.jobs.run_lcurve",
        "maxent_decay.contract.describe",
    )
