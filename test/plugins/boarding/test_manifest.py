"""Tests for the boarding wizard manifest."""

import json
import pathlib

import chisurf.plugins
from chisurf.core.plugin.manifest import load_manifest, validate_manifest


def test_boarding_manifest_loads_and_hides_from_menu():
    """Validate the boarding wizard manifest metadata."""
    manifest_path = (
        pathlib.Path(__file__).resolve().parents[3]
        / "chisurf"
        / "plugins"
        / "core"
        / "boarding"
        / "manifest.json"
    )
    data = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert validate_manifest(data) == []

    manifest = load_manifest(manifest_path)
    assert manifest is not None
    assert manifest.id == "boarding"
    assert manifest.display_name == "Help:Boarding Wizard"
    assert manifest.entrypoints.gui == "chisurf.plugins.core.boarding.wizard:WelcomeToChiSurfWizard"
    assert manifest.menu_hidden is True


def test_boarding_iter_plugins_uses_manifest_metadata():
    """Ensure manifest metadata drives plugin discovery."""
    boarding_plugins = [
        info
        for info in chisurf.plugins.iter_plugins()
        if info.get("manifest_id") == "boarding"
    ]

    assert len(boarding_plugins) == 1
    info = boarding_plugins[0]
    assert info["module_path"] == "chisurf.plugins.core.boarding"
    assert info["plugin_name"] == "Help:Boarding Wizard"
    assert info["menu_hidden"] is True
