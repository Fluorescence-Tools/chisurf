"""Headless tests for the central ``detector_setups.*`` RPC service.

These exercise the live session-active channel (``set_current`` / ``current``)
and the persistent listing path without any Qt/GUI involvement.
"""

from __future__ import annotations

from chisurf.server.dispatcher import ServiceDispatcher
from chisurf.server.session import SessionState


def _dispatcher() -> ServiceDispatcher:
    state = SessionState()
    dispatcher = ServiceDispatcher(state)
    dispatcher._build_default_registry()
    return dispatcher


def test_no_gui_import_in_server():
    """SV-01 guard: chisurf.server must not import chisurf.gui (which brings Qt)."""
    import ast
    import chisurf.server.services.detector_setups as svc
    import inspect
    source = inspect.getsource(svc)
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                name = alias.name if isinstance(node, ast.Import) else node.module or ""
                if "chisurf.gui" in name:
                    raise AssertionError(
                        f"chisurf.server.services.detector_setups must not import "
                        f"chisurf.gui (found: import {name})"
                    )


def test_methods_registered():
    d = _dispatcher()
    for method in (
        "detector_setups.list",
        "detector_setups.get",
        "detector_setups.save",
        "detector_setups.current",
        "detector_setups.set_current",
    ):
        assert d.has_method(method), method


def test_set_and_get_current_roundtrip():
    d = _dispatcher()
    payload = {
        "windows": {"PIE": [0, 100]},
        "detectors": {"green": {"chs": [0, 1], "mtr": [[0, 100]]}},
    }
    res = d.dispatch("detector_setups.set_current", {"settings": payload})
    assert res["ok"] is True

    got = d.dispatch("detector_setups.current", {})
    assert got["ok"] is True
    assert got["result"] == payload


def test_set_current_requires_settings_or_name():
    d = _dispatcher()
    res = d.dispatch("detector_setups.set_current", {})
    assert res["ok"] is False
    assert res["error_code"] == "INVALID_INPUT"


def test_list_returns_store_shape():
    d = _dispatcher()
    res = d.dispatch("detector_setups.list", {})
    assert res["ok"] is True
    assert "setups" in res["result"]
    assert "last_used" in res["result"]
    assert isinstance(res["result"]["setups"], list)


def test_active_state_isolated_per_session():
    d1 = _dispatcher()
    d2 = _dispatcher()
    d1.dispatch("detector_setups.set_current", {"settings": {"detectors": {"a": {}}}})
    # A fresh session must not see d1's live definition.
    got2 = d2.dispatch("detector_setups.current", {})
    assert got2["ok"] is True
    assert got2["result"].get("detectors", {}).get("a") is None
