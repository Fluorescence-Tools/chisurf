"""Manifest sanity checks for the Batch-Analysis plugin."""

from __future__ import annotations

from pathlib import Path

from chisurf.core.plugin import load_manifest


def test_manifest_loads():
    manifest = load_manifest(Path(__file__).parents[1] / "manifest.json")
    assert manifest is not None
    assert manifest.id == "batch_analysis"
    assert manifest.entrypoints.gui.endswith("BatchProcessingWizard")
    assert manifest.entrypoints.cli is not None


def test_cli_entrypoint_format():
    manifest = load_manifest(Path(__file__).parents[1] / "manifest.json")
    cmd, _, module = manifest.entrypoints.cli.partition("=")
    assert cmd == "batch-analysis"
    assert module.endswith(".cli:cli")
