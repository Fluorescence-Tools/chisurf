"""Test suite for {{ cookiecutter.plugin_display_name }}."""

from __future__ import annotations

import json
from pathlib import Path


def test_manifest_is_valid():
    """The manifest.json must parse and contain required fields."""
    manifest_path = Path(__file__).parents[1] / "manifest.json"
    with open(manifest_path) as fh:
        data = json.load(fh)

    assert "id" in data
    assert "version" in data
    assert "display_name" in data
    assert "statefulness" in data
    assert data["statefulness"]["enabled"] is True
    assert data["statefulness"]["window"]["enabled"] is True
    assert "entrypoints" in data
    assert "rpc_methods" in data


def test_client_ping():
    """The plugin client can ping the backend."""
    from chisurf.plugins.{{ cookiecutter.plugin_name }}.gui.client import (
        {{ cookiecutter.widget_class_name }}Client,
    )

    client = {{ cookiecutter.widget_class_name }}Client()
    result = client.ping()
    assert result == "pong"
