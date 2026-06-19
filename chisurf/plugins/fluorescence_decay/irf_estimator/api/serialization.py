from __future__ import annotations

from typing import Any

from .models import IRFEstimationSettings


def settings_from_dict(data: dict[str, Any] | None) -> IRFEstimationSettings:
    """Build IRFEstimationSettings from a JSON-safe dict, falling back to defaults."""
    if not data:
        return IRFEstimationSettings()
    return IRFEstimationSettings(
        window_length=int(data.get("window_length", 11)),
        polyorder=int(data.get("polyorder", 3)),
        rl_iterations=int(data.get("rl_iterations", 500)),
        regularization=int(data.get("regularization", 3)),
        manual_background=float(data.get("manual_background", 0.0)),
        use_range_selection=bool(data.get("use_range_selection", False)),
        range_bounds=[float(v) for v in data.get("range_bounds", [0.0, 100.0])],
    )
