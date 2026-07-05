"""ServiceDispatcher-compatible RPC handlers for the FRET line generator."""

from __future__ import annotations

from typing import Any

from ..core.algorithms import (
    compute_fret_line,
    fret_line_overlays,
    get_model_parameters,
    list_fret_line_projections,
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
    # Shared overlay-lines interface (PRD-56): FRET lines in the same LineSet shape
    # as ``phasor.overlays``, so ndXplorer draws both uniformly.
    dispatcher.register(
        "fret_line.overlays",
        lambda p: fret_line_overlays(**p),
    )
    dispatcher.register(
        "fret_line.list_projections",
        lambda _: {"ok": True, "result": {"projections": list_fret_line_projections()}},
    )
