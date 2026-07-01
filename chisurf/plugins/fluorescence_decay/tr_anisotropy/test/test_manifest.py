"""Manifest sanity checks for the anisotropy plugin."""

from __future__ import annotations

from pathlib import Path

from chisurf.core.plugin import load_manifest


def test_manifest_loads():
    manifest = load_manifest(Path(__file__).parents[1] / "manifest.json")
    assert manifest is not None
    assert manifest.id == "tr_anisotropy"
    assert manifest.entrypoints.gui.endswith("AnisotropyWizard")
    cmd, _, module = manifest.entrypoints.cli.partition("=")
    assert cmd == "anisotropy"
    assert module.endswith(".cli:cli")
