"""ServiceDispatcher-compatible RPC handlers for {{ cookiecutter.plugin_display_name }}.

Thin adapters: accept JSON-compatible params, delegate to core, return JSON-safe results.
"""

from __future__ import annotations

from typing import Any


def register_services(dispatcher: Any) -> None:
    """Register RPC handlers with a ServiceDispatcher.

    Parameters
    ----------
    dispatcher : ServiceDispatcher
        The server's service dispatcher.

    """
    dispatcher.register(
        "{{ cookiecutter.plugin_name }}.ping",
        lambda params: {"ok": True, "result": "pong"},
    )


def list_methods() -> dict[str, str]:
    """Return the RPC method catalogue."""
    return {
        "{{ cookiecutter.plugin_name }}.ping": "Liveness check.",
    }
