"""Tests for PluginManifest dataclass, loading, and validation."""

from __future__ import annotations

import json
import pathlib
import tempfile

import pytest

from chisurf.core.plugin.manifest import (
    PluginManifest,
    RPCMethodSpec,
    PluginEntrypoints,
    load_manifest,
    validate_manifest,
)


class TestPluginManifest:
    """PluginManifest dataclass construction and roundtrips."""

    def test_minimal_manifest(self):
        """Minimal manifest requires only id and version."""
        m = PluginManifest(id="test", version="1.0.0")
        assert m.id == "test"
        assert m.version == "1.0.0"
        assert m.display_name == ""

    def test_full_manifest_defaults(self):
        """Full manifest with all fields."""
        m = PluginManifest(
            id="burst_selection",
            version="2.0.0",
            display_name="Burst Selection",
            description="Analyze bursts",
            authors=["Alice"],
            categories=["Spectroscopy"],
            state_namespace="burst_selection",
            entrypoints=PluginEntrypoints(
                gui="pkg.mod:Widget",
                cli="burst-sel=pkg.mod:cli",
                services="pkg.mod:register",
            ),
            rpc_methods=[
                RPCMethodSpec(
                    name="burst_selection.jobs.analyze_files",
                    summary="Analyze files",
                    long_running=True,
                    cancelable=True,
                    events=["burst_selection.jobs.progress"],
                ),
            ],
        )
        assert m.id == "burst_selection"
        assert m.entrypoints.gui == "pkg.mod:Widget"
        assert len(m.rpc_methods) == 1
        assert m.rpc_methods[0].name == "burst_selection.jobs.analyze_files"
        assert m.rpc_methods[0].long_running is True

    def test_to_dict_roundtrip(self):
        """to_dict() produces a dict that can rebuild the manifest."""
        m1 = PluginManifest(
            id="test",
            version="1.0.0",
            display_name="Test",
            description="A test plugin",
            authors=["Author"],
            categories=["Tools"],
            entrypoints=PluginEntrypoints(cli="test=mod:cli"),
            rpc_methods=[RPCMethodSpec(name="test.ping")],
        )
        d = m1.to_dict()
        m2 = PluginManifest.from_dict(d)
        assert m2.id == m1.id
        assert m2.version == m1.version
        assert m2.display_name == m1.display_name
        assert m2.entrypoints.cli == m1.entrypoints.cli
        assert m2.rpc_methods[0].name == m1.rpc_methods[0].name

    def test_to_json_roundtrip(self):
        """to_json() produces valid JSON that can be loaded back."""
        m1 = PluginManifest(id="test", version="1.0.0")
        json_str = m1.to_json()
        data = json.loads(json_str)
        m2 = PluginManifest.from_dict(data)
        assert m2.id == "test"
        assert m2.version == "1.0.0"


class TestManifestFromDict:
    """PluginManifest.from_dict() parsing."""

    MINIMAL = {"id": "test", "version": "1.0.0"}

    def test_minimal(self):
        m = PluginManifest.from_dict(self.MINIMAL)
        assert m.id == "test"
        assert m.version == "1.0.0"

    def test_with_entrypoints(self):
        data = {
            "id": "burst",
            "version": "2.0.0",
            "entrypoints": {
                "gui": "pkg:Widget",
                "cli": "burst=pkg:cli",
                "services": "pkg:register",
            },
        }
        m = PluginManifest.from_dict(data)
        assert m.entrypoints.gui == "pkg:Widget"
        assert m.entrypoints.cli == "burst=pkg:cli"
        assert m.entrypoints.services == "pkg:register"

    def test_with_rpc_methods(self):
        data = {
            "id": "burst",
            "version": "1.0.0",
            "rpc_methods": [
                {
                    "name": "burst.jobs.run",
                    "summary": "Run analysis",
                    "long_running": True,
                    "cancelable": True,
                    "events": ["burst.jobs.progress"],
                },
                {"name": "burst.jobs.status"},
            ],
        }
        m = PluginManifest.from_dict(data)
        assert len(m.rpc_methods) == 2
        assert m.rpc_methods[0].name == "burst.jobs.run"
        assert m.rpc_methods[0].long_running is True
        assert m.rpc_methods[0].events == ["burst.jobs.progress"]
        assert m.rpc_methods[1].name == "burst.jobs.status"
        assert m.rpc_methods[1].long_running is False

    def test_with_legacy_fields(self):
        data = {
            "id": "old_plugin",
            "version": "1.0.0",
            "menu_hidden": True,
            "deprecated": True,
            "deprecation_message": "Use new plugin instead.",
        }
        m = PluginManifest.from_dict(data)
        assert m.menu_hidden is True
        assert m.deprecated is True
        assert m.deprecation_message == "Use new plugin instead."

    def test_missing_id_raises(self):
        with pytest.raises(KeyError):
            PluginManifest.from_dict({"version": "1.0.0"})


class TestLoadManifest:
    """load_manifest() file loading."""

    def test_load_valid(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "manifest.json"
            path.write_text(json.dumps({"id": "test", "version": "1.0.0"}))
            m = load_manifest(path)
            assert m is not None
            assert m.id == "test"

    def test_load_missing(self):
        m = load_manifest(pathlib.Path("/nonexistent/manifest.json"))
        assert m is None

    def test_load_invalid_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "manifest.json"
            path.write_text("not json")
            m = load_manifest(path)
            assert m is None

    def test_load_missing_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "manifest.json"
            path.write_text(json.dumps({"version": "1.0.0"}))
            m = load_manifest(path)
            assert m is None


class TestValidateManifest:
    """validate_manifest() schema validation."""

    def test_valid_manifest(self):
        data = {"id": "test", "version": "1.0.0"}
        errors = validate_manifest(data)
        assert errors == []

    def test_missing_id(self):
        data = {"version": "1.0.0"}
        errors = validate_manifest(data)
        assert any("id" in e for e in errors)

    def test_empty_id(self):
        data = {"id": "", "version": "1.0.0"}
        errors = validate_manifest(data)
        assert any("id" in e for e in errors)

    def test_rpc_method_missing_name(self):
        data = {"id": "test", "version": "1.0.0", "rpc_methods": [{"summary": "no name"}]}
        errors = validate_manifest(data)
        assert any("name" in e for e in errors)

    def test_non_dict_input(self):
        errors = validate_manifest("not a dict")
        assert errors


class TestRPCMethodSpec:
    """RPCMethodSpec defaults."""

    def test_defaults(self):
        spec = RPCMethodSpec(name="test.ping")
        assert spec.summary == ""
        assert spec.params_schema is None
        assert spec.result_schema is None
        assert spec.long_running is False
        assert spec.cancelable is False
        assert spec.events == []

    def test_full_spec(self):
        spec = RPCMethodSpec(
            name="test.run",
            summary="Run",
            params_schema={"type": "object"},
            result_schema={"type": "object"},
            long_running=True,
            cancelable=True,
            events=["test.progress"],
        )
        assert spec.long_running is True
        assert spec.cancelable is True
        assert spec.params_schema == {"type": "object"}
