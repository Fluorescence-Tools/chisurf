"""Tests for PluginRegistry discovery and registration."""

from __future__ import annotations

import json
import pathlib
import sys
import tempfile
import types
from unittest.mock import MagicMock, patch

from chisurf.core.plugin.manifest import PluginManifest, PluginStatefulness, PluginWindowState
from chisurf.core.plugin.registry import (
    PluginRegistry,
    _apply_manifest_statefulness,
    _read_legacy_metadata,
    resolve_plugin_statefulness,
)


def _make_plugin_dir(base: pathlib.Path, plugin_id: str, manifest: dict) -> pathlib.Path:
    """Create a plugin directory with manifest.json."""
    plugin_dir = base / plugin_id
    plugin_dir.mkdir(parents=True, exist_ok=True)
    (plugin_dir / "__init__.py").write_text("")
    (plugin_dir / "manifest.json").write_text(json.dumps(manifest))
    return plugin_dir


class TestPluginRegistryDiscovery:
    """PluginRegistry.discover() with manifest.json files."""

    def test_discover_empty_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = pathlib.Path(tmp)
            reg = PluginRegistry()
            manifests = reg.discover(search_paths=[base])
            assert manifests == []

    def test_discover_single_plugin(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = pathlib.Path(tmp)
            _make_plugin_dir(base, "burst", {
                "id": "burst_selection",
                "version": "2.0.0",
                "display_name": "Burst Selection",
                "entrypoints": {
                    "services": "chisurf.plugins.burst.burst_selection.server.services:register_burst_selection_services",
                },
            })
            reg = PluginRegistry()
            manifests = reg.discover(search_paths=[base])
            assert len(manifests) == 1
            assert manifests[0].id == "burst_selection"
            assert manifests[0].version == "2.0.0"
            assert reg.get_manifest("burst_selection") is not None

    def test_discover_multiple_plugins(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = pathlib.Path(tmp)
            _make_plugin_dir(base, "burst", {"id": "burst_selection", "version": "1.0.0"})
            _make_plugin_dir(base, "fcs", {"id": "fcs", "version": "1.0.0"})
            reg = PluginRegistry()
            manifests = reg.discover(search_paths=[base])
            assert len(manifests) == 2
            ids = {m.id for m in manifests}
            assert ids == {"burst_selection", "fcs"}

    def test_skip_dir_without_init(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = pathlib.Path(tmp)
            d = base / "no_init"
            d.mkdir()
            reg = PluginRegistry()
            manifests = reg.discover(search_paths=[base])
            assert len(manifests) == 0

    def test_skip_invalid_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = pathlib.Path(tmp)
            plugin_dir = base / "bad"
            plugin_dir.mkdir()
            (plugin_dir / "__init__.py").write_text("")
            (plugin_dir / "manifest.json").write_text("not json")
            reg = PluginRegistry()
            manifests = reg.discover(search_paths=[base])
            assert len(manifests) == 0

    def test_discover_dedup(self):
        """Duplicate paths should not produce duplicate manifests."""
        with tempfile.TemporaryDirectory() as tmp:
            base = pathlib.Path(tmp)
            _make_plugin_dir(base, "test", {"id": "test", "version": "1.0.0"})
            reg = PluginRegistry()
            manifests = reg.discover(search_paths=[base, base])
            assert len(manifests) == 1


class TestPluginRegistryServices:
    """PluginRegistry.register_services()."""

    def test_register_services_called(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = pathlib.Path(tmp)
            _make_plugin_dir(base, "burst", {
                "id": "burst_selection",
                "version": "1.0.0",
                "entrypoints": {
                    "services": "test_register_fn",
                },
            })
            dispatcher = MagicMock()
            register_fn = MagicMock()

            reg = PluginRegistry()
            reg.discover(search_paths=[base])

            with patch.object(reg._loader, "load", return_value=register_fn):
                reg.register_services(dispatcher)

            register_fn.assert_called_once_with(dispatcher)

    def test_register_services_excludes_startup_entrypoints(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = pathlib.Path(tmp)
            _make_plugin_dir(base, "mfdb", {
                "id": "mfdb_plugin",
                "version": "1.0.0",
                "entrypoints": {
                    "services": "mfdb_register_fn",
                },
            })
            _make_plugin_dir(base, "other", {
                "id": "other_plugin",
                "version": "1.0.0",
                "entrypoints": {
                    "services": "other_register_fn",
                },
            })
            dispatcher = MagicMock()

            reg = PluginRegistry()
            reg.discover(search_paths=[base])

            with patch.object(reg._loader, "load") as load:
                load.side_effect = lambda entrypoint: MagicMock()
                reg.register_services(
                    dispatcher,
                    exclude_entrypoints={"mfdb_register_fn"},
                )

            loaded = [call.args[0] for call in load.call_args_list]
            assert loaded == ["other_register_fn"]

    def test_register_services_skip_no_entrypoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = pathlib.Path(tmp)
            _make_plugin_dir(base, "test", {"id": "test", "version": "1.0.0"})
            dispatcher = MagicMock()

            reg = PluginRegistry()
            reg.discover(search_paths=[base])
            reg.register_services(dispatcher)

            # No entrypoint = no call
            dispatcher.register.assert_not_called()


class TestPluginRegistryCLI:
    """PluginRegistry.register_cli()."""

    def test_register_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = pathlib.Path(tmp)
            _make_plugin_dir(base, "burst", {
                "id": "burst_selection",
                "version": "1.0.0",
                "entrypoints": {
                    "cli": "burst-selection=pkg:cli",
                },
            })
            group = MagicMock()
            cli_obj = MagicMock()
            cli_obj.name = "burst-selection"

            reg = PluginRegistry()
            reg.discover(search_paths=[base])

            with patch.object(reg._loader, "load", return_value=cli_obj):
                reg.register_cli(group)

            group.add_command.assert_called_once()


class _Event:
    """Minimal event object for plugin statefulness tests."""

    def __init__(self):
        self.accepted = False

    def accept(self):
        self.accepted = True


class _FakeWindow:
    """Minimal window-like object for plugin statefulness tests."""

    def __init__(self):
        self.show_events = []
        self.close_events = []
        self.geometry = None
        self.state = None

    def showEvent(self, event):
        self.show_events.append(event)

    def closeEvent(self, event):
        self.close_events.append(event)

    def saveGeometry(self):
        return b"geometry"

    def restoreGeometry(self, geometry):
        self.geometry = geometry

    def saveState(self):
        return b"state"

    def restoreState(self, state):
        self.state = state


class TestPluginRegistryStatefulness:
    """PluginRegistry manifest-driven window statefulness."""

    def test_resolve_plugin_statefulness_uses_manifest_default(self):
        manifest = PluginManifest(
            id="test_plugin",
            version="1.0.0",
            statefulness=PluginStatefulness(enabled=True),
        )
        assert resolve_plugin_statefulness(manifest, {"statefulness": {}}) is True

        manifest.statefulness.enabled = False
        assert resolve_plugin_statefulness(manifest, {"statefulness": {}}) is False

    def test_resolve_plugin_statefulness_global_override(self):
        manifest = PluginManifest(
            id="test_plugin",
            version="1.0.0",
            statefulness=PluginStatefulness(enabled=False),
        )
        assert resolve_plugin_statefulness(
            manifest,
            {"statefulness": {"mode": "enabled"}},
        ) is True
        assert resolve_plugin_statefulness(
            manifest,
            {"statefulness": {"mode": "disabled"}},
        ) is False

    def test_resolve_plugin_statefulness_per_plugin_override(self):
        manifest = PluginManifest(
            id="test_plugin",
            version="1.0.0",
            state_namespace="namespace_plugin",
            statefulness=PluginStatefulness(enabled=True),
        )
        plugin_settings = {
            "statefulness": {
                "mode": "enabled",
                "per_plugin": {
                    "test_plugin": False,
                    "namespace_plugin": "plugin_default",
                },
            }
        }
        assert resolve_plugin_statefulness(manifest, plugin_settings) is False

        manifest.id = "namespace_plugin"
        assert resolve_plugin_statefulness(manifest, plugin_settings) is True

    def test_apply_manifest_statefulness_wraps_window_events(self, monkeypatch):
        calls = []
        fake_helpers = types.SimpleNamespace(
            restore_plugin_window_state=lambda window, key: calls.append(("restore", key)),
            save_plugin_window_state=lambda window, key: calls.append(("save", key)),
        )
        monkeypatch.setitem(sys.modules, "chisurf.gui.misc_helpers", fake_helpers)
        manifest = PluginManifest(
            id="test_plugin",
            version="1.0.0",
            statefulness=PluginStatefulness(
                enabled=True,
                window=PluginWindowState(settings_key="test_window"),
            ),
        )
        window = _FakeWindow()
        show_event = _Event()
        close_event = _Event()

        _apply_manifest_statefulness(window, manifest)
        window.showEvent(show_event)
        window.closeEvent(close_event)

        assert calls == [("restore", "test_window"), ("save", "test_window")]
        assert show_event in window.show_events
        assert close_event in window.close_events

    def test_apply_manifest_statefulness_skips_existing_persistence(self, monkeypatch):
        calls = []
        fake_helpers = types.SimpleNamespace(
            restore_plugin_window_state=lambda window, key: calls.append(key),
            save_plugin_window_state=lambda window, key: calls.append(key),
        )
        monkeypatch.setitem(sys.modules, "chisurf.gui.misc_helpers", fake_helpers)
        manifest = PluginManifest(
            id="test_plugin",
            version="1.0.0",
            statefulness=PluginStatefulness(enabled=True),
        )
        window = _FakeWindow()
        window._persist_plugin_name = "legacy_key"

        _apply_manifest_statefulness(window, manifest)
        window.showEvent(_Event())

        assert calls == []


class TestPluginRegistryState:
    """PluginRegistry state inspection."""

    def test_all_manifests(self):
        reg = PluginRegistry()
        assert reg.all_manifests == []

    def test_get_manifest_nonexistent(self):
        reg = PluginRegistry()
        assert reg.get_manifest("nonexistent") is None


class TestLegacyMetadata:
    """_read_legacy_metadata() fallback."""

    def test_legacy_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = pathlib.Path(tmp)
            plugin_dir = base / "legacy_plugin"
            plugin_dir.mkdir()
            init_py = plugin_dir / "__init__.py"
            init_py.write_text(
                'name = "Category:Legacy Plugin"\n'
                'cli_entrypoint = "legacy=pkg:cli"\n'
                'menu_hidden = True\n'
            )
            meta = _read_legacy_metadata(plugin_dir)
            assert meta is not None
            assert meta["id"] == "legacy_plugin"
            assert meta["display_name"] == "Category:Legacy Plugin"
            assert meta["cli_entrypoint"] == "legacy=pkg:cli"
            assert meta["menu_hidden"] is True

    def test_legacy_fallback_no_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = pathlib.Path(tmp)
            plugin_dir = base / "no_name"
            plugin_dir.mkdir()
            (plugin_dir / "__init__.py").write_text("# no name variable")
            meta = _read_legacy_metadata(plugin_dir)
            assert meta is None
