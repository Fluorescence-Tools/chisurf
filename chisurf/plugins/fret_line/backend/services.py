"""ServiceDispatcher-compatible RPC handlers for the FRET line generator."""

from __future__ import annotations

from typing import Any

from ..core.algorithms import (
    compute_fret_line,
    get_model_parameters,
    list_models,
    list_sweep_targets,
)


def register_services(dispatcher: Any) -> None:
    """Register all FRET line RPC methods with *dispatcher*."""
    dispatcher.register("fret_line.list_models", lambda _: list_models())
    dispatcher.register(
        "fret_line.get_model_parameters",
        lambda p: get_model_parameters(**p),
    )
    dispatcher.register(
        "fret_line.list_sweep_targets",
        lambda p: list_sweep_targets(**p),
    )
    dispatcher.register(
        "fret_line.compute",
        lambda p: compute_fret_line(**p),
    )
