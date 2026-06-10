"""Tests for burst_selection manifest.json."""

from __future__ import annotations

import pathlib

from chisurf.core.plugin.manifest import load_manifest, validate_manifest


_MANIFEST_PATH = pathlib.Path(__file__).resolve().parents[3] / "chisurf" / "plugins" / "burst" / "burst_selection" / "manifest.json"


class TestBurstSelectionManifest:
    """Validate the burst_selection manifest."""

    def test_manifest_exists(self):
        assert _MANIFEST_PATH.exists()

    def test_manifest_loads(self):
        manifest = load_manifest(_MANIFEST_PATH)
        assert manifest is not None

    def test_manifest_id(self):
        manifest = load_manifest(_MANIFEST_PATH)
        assert manifest is not None
        assert manifest.id == "burst_selection"

    def test_manifest_version(self):
        manifest = load_manifest(_MANIFEST_PATH)
        assert manifest is not None
        assert manifest.version == "2.0.0"

    def test_manifest_display_name(self):
        manifest = load_manifest(_MANIFEST_PATH)
        assert manifest is not None
        assert "Burst" in manifest.display_name

    def test_manifest_has_entrypoints(self):
        manifest = load_manifest(_MANIFEST_PATH)
        assert manifest is not None
        assert manifest.entrypoints.gui is not None
        assert manifest.entrypoints.cli is not None
        assert manifest.entrypoints.services is not None

    def test_manifest_has_rpc_methods(self):
        manifest = load_manifest(_MANIFEST_PATH)
        assert manifest is not None
        assert len(manifest.rpc_methods) == 4

    def test_manifest_rpc_method_names(self):
        manifest = load_manifest(_MANIFEST_PATH)
        assert manifest is not None
        names = [m.name for m in manifest.rpc_methods]
        assert "burst_selection.jobs.analyze_files" in names
        assert "burst_selection.results.inspect_bur" in names
        assert "burst_selection.gmm.fit" in names

    def test_manifest_has_state_namespace(self):
        manifest = load_manifest(_MANIFEST_PATH)
        assert manifest is not None
        assert manifest.state_namespace == "burst_selection"

    def test_manifest_validates(self):
        import json
        data = json.loads(_MANIFEST_PATH.read_text())
        errors = validate_manifest(data)
        assert errors == []

    def test_manifest_events(self):
        manifest = load_manifest(_MANIFEST_PATH)
        assert manifest is not None
        assert len(manifest.events) == 2
