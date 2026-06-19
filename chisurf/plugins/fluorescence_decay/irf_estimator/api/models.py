from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class IRFEstimationSettings:
    """Settings for IRF estimation."""
    window_length: int = 11
    polyorder: int = 3
    rl_iterations: int = 500
    regularization: int = 3
    manual_background: float = 0.0
    use_range_selection: bool = False
    range_bounds: list[float] = field(default_factory=lambda: [0.0, 100.0])


@dataclass
class IRFEstimationResult:
    """Result of an IRF estimation run."""
    irf: list[float]
    params: dict[str, Any]
    time_axis: list[float]
    dt: float
    lifetime_ns: float
    decay_rate_ns: float
    amplitude: float
    offset: float


@dataclass
class DecayData:
    """Decay data loaded from a file or dataset."""
    time_axis: list[float]
    intensity: list[float]
    intensity_original: list[float]
    dt: float
    source: str = ""
    filename: str | None = None


def dataclass_to_dict(value: Any) -> dict[str, Any]:
    """Convert a dataclass to a JSON-safe dictionary."""
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    if isinstance(value, list):
        return [dataclass_to_dict(item) for item in value]
    if isinstance(value, dict):
        return {str(key): dataclass_to_dict(val) for key, val in value.items()}
    return value
