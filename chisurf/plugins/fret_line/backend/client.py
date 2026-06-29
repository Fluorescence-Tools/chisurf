"""In-process client for the FRET line generator.

Delegates to ``core.algorithms`` directly.
"""

from __future__ import annotations

from typing import Any

from ..core.algorithms import (
    compute_fret_line,
    get_model_parameters,
    list_models,
    list_sweep_targets,
)


class FRETLineClient:
    """Thin wrapper over the core algorithms."""

    def list_models(self) -> list[str]:
        """Return the available component-model display names."""
        return list_models()

    def get_model_parameters(self, model_name: str, n_components: int = 1) -> list[dict]:
        """Return all parameters of a freshly built model."""
        return get_model_parameters(model_name, n_components)

    def list_sweep_targets(self, components: list[dict]) -> list[dict]:
        """Enumerate sweepable parameters and fractions for *components*."""
        return list_sweep_targets(components)

    def compute(
        self,
        components: list[dict],
        sweep: dict,
        param_min: float,
        param_max: float,
        n_points: int = 100,
        fractions: list[float] | None = None,
        tau_d0: float | None = None,
        log_scale: bool = False,
    ) -> dict[str, Any]:
        """Compute a FRET line for the given mixture and sweep specification."""
        return compute_fret_line(
            components=components,
            sweep=sweep,
            param_min=param_min,
            param_max=param_max,
            n_points=n_points,
            fractions=fractions,
            tau_d0=tau_d0,
            log_scale=log_scale,
        )
