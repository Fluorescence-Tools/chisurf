from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class PchSettings:
    channels: list[int] = field(default_factory=lambda: [0, 2])
    bin_time_us: float = 100.0
    micro_time_min: int = 0
    micro_time_max: int = 65535
    reading_routine: str = "PTU"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PchSettings:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class PchFitSettings:
    n_components: int = 1
    initial_epsilons: list[float] = field(default_factory=lambda: [2.0])
    initial_Ns: list[float] = field(default_factory=lambda: [3.0])
    fit_low: int = 0
    fit_high: int = 100

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PchFitSettings:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class PchResult:
    k_vals: list[float]
    p_exp: list[float]
    hist_counts: list[int]
    total_bins: int
    trace_t: list[float]
    trace_counts: list[float]
    filename: str = ""
    settings: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PchResult:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class FitResult:
    epsilons: list[float]
    avg_Ns: list[float]
    fractions: list[float]
    chi2: float
    reduced_chi2: float
    dof: int
    fit_low: int
    fit_high: int
    p_fit: list[float]
    n_components: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> FitResult:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class TttrInfo:
    filename: str
    n_photons: int
    routing_channels: list[int]
    macro_time_resolution: float
    micro_time_range: tuple[int, int]
    has_micro_times: bool

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["micro_time_range"] = list(self.micro_time_range)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TttrInfo:
        if isinstance(d.get("micro_time_range"), list):
            d["micro_time_range"] = tuple(d["micro_time_range"])
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
