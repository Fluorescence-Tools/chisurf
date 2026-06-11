"""Tests for Burst Selection service registration."""

from __future__ import annotations

from chisurf.plugins.burst.burst_selection.server.services import (
    list_methods,
    register_burst_selection_services,
)
from chisurf.plugins.burst.burst_selection.backend.services import (
    contract_handler,
    list_methods as list_backend_methods,
)


def test_register_burst_selection_services() -> None:
    """Burst Selection services should register with ServiceDispatcher."""

    class Dispatcher:
        """Minimal dispatcher double."""

        def __init__(self) -> None:
            self.names = []

        def register(self, name: str, handler):
            self.names.append(name)

    dispatcher = Dispatcher()
    register_burst_selection_services(dispatcher)
    assert dispatcher.names == [
        "burst_selection.analyze_files",
        "burst_selection.inspect_bur",
        "burst_selection.fit_gmm_from_bur",
    ]


def test_list_methods() -> None:
    """The method catalogue should expose all Burst Selection RPC methods."""
    methods = list_methods()
    # Both new dotted names and legacy aliases
    assert "burst_selection.jobs.analyze_files" in methods
    assert "burst_selection.results.inspect_bur" in methods
    assert "burst_selection.gmm.fit" in methods
    assert "burst_selection.contract.describe" in methods
    # Legacy aliases still present
    assert "burst_selection.analyze_files" in methods
    assert "burst_selection.inspect_bur" in methods
    assert "burst_selection.fit_gmm_from_bur" in methods
    # All methods have descriptions
    for name, desc in methods.items():
        assert isinstance(desc, str) and len(desc) > 0


def test_backend_contract_handler_returns_service_result() -> None:
    """Workflow clients should be able to discover the contract through RPC."""
    result = contract_handler()
    assert result["ok"] is True
    assert result["result"]["plugin_id"] == "burst_selection"
    assert "burst_selection.jobs.analyze_files" in result["result"]["rpc_methods"]


def test_backend_method_catalogue_includes_contract_method() -> None:
    """The backend catalogue should publish the contract method."""
    methods = list_backend_methods()
    assert methods["burst_selection.contract.describe"] == "Return the Burst Selection workflow contract."
