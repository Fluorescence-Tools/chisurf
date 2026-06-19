"""Compatibility shim for legacy method name registration.

All new code should import from ``..backend`` instead.
"""

from __future__ import annotations

from typing import Any

from ..backend.services import analyze_files_handler, list_methods as _list_methods


def register_time_window_services(dispatcher: Any) -> None:
    """Register Time Window Bins RPC handlers with legacy method names."""
    dispatcher.register(
        "tttr_time_windows.analyze_files",
        lambda params: analyze_files_handler(**params),
    )


def list_methods() -> dict[str, str]:
    """Return the Time Window Bins RPC method catalogue."""
    return _list_methods()
