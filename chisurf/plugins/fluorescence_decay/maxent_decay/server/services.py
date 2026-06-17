"""Compatibility shim for MaxEnt MEM server services."""

from __future__ import annotations

from typing import Any

from ..backend.services import (
    contract_handler,
    list_methods as _list_methods,
    run_fret_handler,
    run_lcurve_handler,
    run_lifetime_handler,
)
from ..api.contract import METHOD_DESCRIBE, METHOD_RUN_FRET, METHOD_RUN_LCURVE, METHOD_RUN_LIFETIME


def register_services(dispatcher: Any) -> None:
    """Register MaxEnt MEM RPC handlers."""
    dispatcher.register(METHOD_RUN_LIFETIME, lambda params: run_lifetime_handler(**(params or {})))
    dispatcher.register(METHOD_RUN_FRET, lambda params: run_fret_handler(**(params or {})))
    dispatcher.register(METHOD_RUN_LCURVE, lambda params: run_lcurve_handler(**(params or {})))
    dispatcher.register(METHOD_DESCRIBE, lambda params: contract_handler())


def list_methods() -> dict[str, str]:
    """Return the MaxEnt MEM RPC method catalogue."""
    return _list_methods()
