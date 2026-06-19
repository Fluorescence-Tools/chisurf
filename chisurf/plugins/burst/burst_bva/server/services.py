"""Compatibility shim — registers legacy method names for backward compat."""

from __future__ import annotations

from typing import Any

from ..backend.services import (
    compute_bva_handler,
    list_methods as _list_methods,
)


def register_bva_services(dispatcher: Any) -> None:
    dispatcher.register(
        "burst_bva.compute",
        lambda params: compute_bva_handler(**params),
    )


def list_methods() -> dict[str, str]:
    return _list_methods()
