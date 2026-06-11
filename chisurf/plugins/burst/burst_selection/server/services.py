"""Compatibility shim — registers legacy method names for backward compat.

All new code should import from ``..backend`` instead.
"""

from __future__ import annotations

from typing import Any

from ..backend.services import (
    analyze_files_handler,
    fit_gmm_handler,
    inspect_bur_handler,
    list_methods as _list_methods,
)


def register_burst_selection_services(dispatcher: Any) -> None:
    """Register Burst Selection RPC handlers with legacy method names."""
    dispatcher.register(
        "burst_selection.analyze_files",
        lambda params: analyze_files_handler(**params),
    )
    dispatcher.register(
        "burst_selection.inspect_bur",
        lambda params: inspect_bur_handler(**params),
    )
    dispatcher.register(
        "burst_selection.fit_gmm_from_bur",
        lambda params: fit_gmm_handler(**params),
    )


def list_methods() -> dict[str, str]:
    """Return the Burst Selection RPC method catalogue."""
    return _list_methods()
