from __future__ import annotations

import importlib
import logging
import pathlib
from typing import Any, Protocol

from chisurf.core.plugin.manifest import (
    PluginManifest,
    load_manifest,
)

_log = logging.getLogger(__name__)


class _EntrypointLoader(Protocol):
    """Protocol for objects that can load a plugin entrypoint."""

    def load(self, entrypoint: str) -> Any:
        ...


class _DefaultLoader:
    """Default entrypoint loader using ``importlib``."""

    def load(self, entrypoint: str) -> Any:
        if ":" in entrypoint:
            module_path, attr_name = entrypoint.rsplit(":", 1)
            module = importlib.import_module(module_path)
            return getattr(module, attr_name)
        return importlib.import_module(entrypoint)


class PluginRegistry:
    """Discovers and manages plugins from ``manifest.json`` files.

    Supports both the new ``manifest.json`` format and the legacy AST-based
    metadata parsing from ``__init__.py`` as fallback.

    """

    def __init__(self, loader: _EntrypointLoader | None = None):
        self._loader = loader or _DefaultLoader()
        self._manifests: dict[str, PluginManifest] = {}
        self._legacy_plugins: list[dict[str, Any]] = []

    # ── discovery ──────────────────────────────────────────────────

    def discover(
        self,
        search_paths: list[pathlib.Path] | None = None,
    ) -> list[PluginManifest]:
        """Walk plugin directories and load manifests.

        Parameters
        ----------
        search_paths : list of pathlib.Path, optional
            Directories to search. Defaults to ``chisurf.plugins.__path__``
            and ``~/.chisurf/plugins/``.

        Returns
        -------
        list of PluginManifest
            All successfully loaded manifests (deduplicated by directory).

        """
        if search_paths is None:
            search_paths = self._default_search_paths()

        manifests: list[PluginManifest] = []
        seen_dirs: set[pathlib.Path] = set()

        for base_path in search_paths:
            if not base_path.exists():
                continue
            for init_py_path in sorted(base_path.rglob("__init__.py")):
                plugin_dir = init_py_path.parent
                if any(part.startswith(".") or part == "__pycache__" or "{{" in part for part in plugin_dir.parts):
                    continue
                resolved = plugin_dir.resolve()
                if resolved in seen_dirs:
                    continue
                seen_dirs.add(resolved)

                # Prefer manifest.json over legacy metadata
                manifest_path = plugin_dir / "manifest.json"
                manifest = load_manifest(manifest_path)

                if manifest is not None:
                    manifests.append(manifest)
                    self._manifests[manifest.id] = manifest
                else:
                    # Legacy fallback: parse __init__.py AST
                    legacy = _read_legacy_metadata(plugin_dir)
                    if legacy is not None:
                        self._legacy_plugins.append(legacy)

        return manifests

    def get_manifest(self, plugin_id: str) -> PluginManifest | None:
        """Return the manifest for a given plugin ID."""
        return self._manifests.get(plugin_id)

    @property
    def all_manifests(self) -> list[PluginManifest]:
        """Return all registered manifests."""
        return list(self._manifests.values())

    # ── registration ───────────────────────────────────────────────

    def register_services(
        self,
        dispatcher: Any,
        exclude_entrypoints: set[str] | None = None,
    ) -> None:
        """Call each plugin's ``services`` entrypoint.

        Parameters
        ----------
        dispatcher : ServiceDispatcher
            The server's service dispatcher.
        exclude_entrypoints : set of str, optional
            Entrypoints already registered through central startup config.

        """
        excluded = exclude_entrypoints or set()
        for manifest in self._manifests.values():
            entrypoint = manifest.entrypoints.services
            if not entrypoint:
                continue
            if entrypoint in excluded:
                continue
            try:
                register_fn = self._loader.load(entrypoint)
                register_fn(dispatcher)
                _log.info("Registered services for plugin %r", manifest.id)
            except Exception:
                _log.exception(
                    "Failed to register services for plugin %r (%s)",
                    manifest.id,
                    entrypoint,
                )

    def register_cli(self, main_group: Any) -> None:
        """Attach each plugin's CLI command to a Click group.

        Parameters
        ----------
        main_group : click.Group
            The main CLI group.

        """
        for manifest in self._manifests.values():
            entrypoint = manifest.entrypoints.cli
            if not entrypoint:
                continue
            try:
                # entrypoint format: "cmd=module:attr"
                cmd_part, _, module_part = entrypoint.partition("=")
                cmd_name = cmd_part.strip()
                if module_part:
                    cli_obj = self._loader.load(module_part.strip())
                else:
                    cli_obj = self._loader.load(entrypoint)
                if hasattr(cli_obj, "name"):
                    cmd_name = cli_obj.name
                main_group.add_command(cli_obj, name=cmd_name)
                _log.info("Registered CLI for plugin %r as %r", manifest.id, cmd_name)
            except Exception:
                _log.exception(
                    "Failed to register CLI for plugin %r (%s)",
                    manifest.id,
                    entrypoint,
                )

    def register_gui(self, menu: Any) -> None:
        """Register plugin GUI entrypoints (placeholder for Qt menu).

        Parameters
        ----------
        menu : QMenu
            The plugins menu.

        """
        for manifest in self._manifests.values():
            entrypoint = manifest.entrypoints.gui
            if not entrypoint or manifest.menu_hidden:
                continue
            try:
                widget_class = self._loader.load(entrypoint)
                action = menu.addAction(manifest.display_name or manifest.id)
                action.triggered.connect(
                    lambda checked, wc=widget_class, manifest=manifest: _show_plugin(wc, manifest)
                )
                _log.info("Registered GUI for plugin %r", manifest.id)
            except Exception:
                _log.exception(
                    "Failed to register GUI for plugin %r (%s)",
                    manifest.id,
                    entrypoint,
                )

    # ── internals ──────────────────────────────────────────────────

    @staticmethod
    def _default_search_paths() -> list[pathlib.Path]:
        from chisurf.plugins import __path__ as plugin_paths
        from chisurf.plugins import user_plugins_dir

        paths = [pathlib.Path(p) for p in plugin_paths]
        paths.append(user_plugins_dir)
        return paths


def _read_legacy_metadata(plugin_dir: pathlib.Path) -> dict[str, Any] | None:
    """Read legacy plugin metadata from ``__init__.py`` AST."""
    try:
        from chisurf.plugins import _read_plugin_metadata  # type: ignore

        init_py = plugin_dir / "__init__.py"
        name, description, cli_entrypoint, cli_only, menu_hidden = _read_plugin_metadata(
            init_py
        )
        if not name:
            return None
        return {
            "id": plugin_dir.name,
            "display_name": name,
            "description": description,
            "cli_entrypoint": cli_entrypoint,
            "cli_only": cli_only,
            "menu_hidden": menu_hidden,
            "package_dir": plugin_dir,
        }
    except Exception:
        return None


def _show_plugin(widget_class: Any, manifest: PluginManifest) -> None:
    """Show a plugin widget and apply manifest statefulness if enabled."""
    try:
        widget = widget_class()
        apply_manifest_statefulness(widget, manifest)
        widget.show()
        widget.raise_()
        widget.activateWindow()
    except Exception:
        _log.exception("Failed to show plugin widget %s", widget_class)


def resolve_plugin_statefulness(
    manifest: PluginManifest,
    plugin_settings: dict[str, Any] | None = None,
) -> bool:
    """Resolve whether a plugin should persist window state."""
    if plugin_settings is None:
        try:
            import chisurf as cs

            plugin_settings = cs.core.settings.cs_settings.get("plugins", {})
        except Exception:
            plugin_settings = {}
    if not isinstance(plugin_settings, dict):
        plugin_settings = {}

    statefulness_settings = plugin_settings.get("statefulness", {})
    if not isinstance(statefulness_settings, dict):
        statefulness_settings = {}

    overrides = statefulness_settings.get("per_plugin", {})
    if not isinstance(overrides, dict):
        overrides = {}

    for key in (manifest.id, manifest.state_namespace, manifest.display_name):
        if key not in overrides:
            continue
        override = overrides[key]
        if isinstance(override, str) and override.lower() == "plugin_default":
            continue
        return bool(override)

    mode = str(statefulness_settings.get("mode", "plugin_default")).lower()
    if mode in {"enabled", "force_enabled", "true"}:
        return True
    if mode in {"disabled", "force_disabled", "false"}:
        return False
    return manifest.statefulness.enabled


def apply_manifest_statefulness(widget: Any, manifest: PluginManifest) -> None:
    """Apply manifest-declared window-state persistence to a widget."""
    _apply_manifest_statefulness(widget, manifest)


def _apply_manifest_statefulness(widget: Any, manifest: PluginManifest) -> None:
    """Apply window-state persistence declared in a plugin manifest."""
    if not resolve_plugin_statefulness(manifest):
        return
    if not manifest.statefulness.window.enabled:
        return
    if getattr(widget, "_manifest_statefulness_applied", False):
        return
    if getattr(widget, "_persist_plugin_name", None):
        return

    settings_key = (
        manifest.statefulness.window.settings_key
        or manifest.state_namespace
        or manifest.id
    )

    try:
        from PyQt5.QtCore import Qt as _Qt
        widget.setAttribute(_Qt.WA_DeleteOnClose)
    except Exception:
        pass

    try:
        from chisurf.gui.misc_helpers import (
            restore_plugin_window_state,
            save_plugin_window_state,
        )
    except Exception:
        _log.debug(
            "Failed to import plugin window-state helpers for %r",
            manifest.id,
            exc_info=True,
        )
        return

    orig_show = getattr(widget, "showEvent", None)
    orig_close = getattr(widget, "closeEvent", None)

    def _restore_state(event: Any) -> None:
        if not getattr(widget, "_manifest_state_restored", False):
            setattr(widget, "_manifest_state_restored", True)
            try:
                restore_plugin_window_state(widget, settings_key)
            except Exception:
                _log.debug(
                    "Failed to restore window state for plugin %r",
                    manifest.id,
                    exc_info=True,
                )
        if orig_show is not None:
            orig_show(event)
        elif hasattr(event, "accept"):
            event.accept()

    def _save_state(event: Any) -> None:
        try:
            save_plugin_window_state(widget, settings_key)
        except Exception:
            _log.debug(
                "Failed to save window state for plugin %r",
                manifest.id,
                exc_info=True,
            )
        if orig_close is not None:
            orig_close(event)
        elif hasattr(event, "accept"):
            event.accept()

    setattr(widget, "_manifest_statefulness_applied", True)
    setattr(widget, "_manifest_statefulness_key", settings_key)
    setattr(widget, "_manifest_state_restored", False)
    widget.showEvent = _restore_state
    widget.closeEvent = _save_state
