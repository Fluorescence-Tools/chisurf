"""Server service registration for Burst Selection."""

from __future__ import annotations

from typing import Any

from ..backend.services import list_methods as _list_methods
from ..backend.services import register_services as _register_services


def register_burst_selection_services(dispatcher: Any) -> None:
    """Register Burst Selection RPC handlers."""
    _register_services(dispatcher)


def list_methods() -> dict[str, str]:
    """Return the Burst Selection RPC method catalogue."""
    return _list_methods()
