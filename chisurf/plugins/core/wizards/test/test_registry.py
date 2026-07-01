"""Headless tests for the Wizards hub registry (no Qt)."""

from __future__ import annotations

import importlib
from pathlib import Path

from chisurf.core.plugin import load_manifest
from chisurf.plugins.core.wizards.core.registry import default_wizards


def test_default_wizards_nonempty_and_unique():
    entries = default_wizards()
    assert entries
    ids = [e.id for e in entries]
    assert len(ids) == len(set(ids))  # unique ids
    labels = [e.label for e in entries]
    assert len(labels) == len(set(labels))  # unique labels


def test_entries_point_at_importable_widgets():
    for entry in default_wizards():
        assert ":" in entry.widget, f"{entry.id} widget path must be 'module:Class'"
        module_name, attr = entry.widget.split(":", 1)
        module = importlib.import_module(module_name)
        assert hasattr(module, attr), f"{module_name} has no {attr!r}"


def test_manifest_loads():
    manifest = load_manifest(Path(__file__).parents[1] / "manifest.json")
    assert manifest is not None
    assert manifest.id == "wizards"
    assert manifest.entrypoints.gui.endswith("WizardHub")
