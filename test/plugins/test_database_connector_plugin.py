"""Tests for the core database connector plugin."""

from __future__ import annotations

from pathlib import Path

from chisurf.core.plugin.manifest import load_manifest
from chisurf.core.plugin.registry import PluginRegistry
from chisurf.core.plugins.database_connector.services import register_services
from chisurf.server.dispatcher import ServiceDispatcher
from chisurf.server.session import SessionState


def test_database_connector_manifest_is_valid():
    """Validate the core database connector plugin manifest."""
    manifest_path = (
        Path(__file__).resolve().parents[2]
        / "chisurf"
        / "core"
        / "plugins"
        / "database_connector"
        / "manifest.json"
    )
    manifest = load_manifest(manifest_path)
    assert manifest is not None
    assert manifest.id == "database_connector"
    assert manifest.statefulness.enabled is False
    assert "database_connector.status" in [method.name for method in manifest.rpc_methods]
    assert "database_connector.repository" in [method.name for method in manifest.rpc_methods]


def test_database_connector_services_register():
    """Register database connector services with the dispatcher."""
    dispatcher = ServiceDispatcher(SessionState())
    register_services(dispatcher)
    methods = set(dispatcher.list_methods())
    assert "database_connector.status" in methods
    assert "database_connector.open" in methods
    assert "database_connector.repository" in methods


def test_database_connector_registry_discovers_core_plugin():
    """Discover the core database connector plugin from core plugin path."""
    registry = PluginRegistry()
    core_plugin_path = (
        Path(__file__).resolve().parents[2]
        / "chisurf"
        / "core"
        / "plugins"
    )
    manifests = registry.discover(search_paths=[core_plugin_path])
    assert any(manifest.id == "database_connector" for manifest in manifests)
