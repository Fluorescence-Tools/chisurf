"""TttrToolboxTool — unified TTTR file toolbox using NavigationPanelTool.

A single window with a left navigation list and a lazily-loaded right panel area.
Each panel embeds one of the existing TTTR tools **unchanged**:

    1. ALEX Creator         — AlexPTUCreator
    2. Micro-time Shifter   — MicrotimeShifterTool
    3. PTU Header Editor    — TagsEditor
    ───────────────────────  (separator)
    4. Split / Convert      — PTUSplitter

Panels are imported lazily inside their factory functions so the combined window
opens fast and a sub-tool whose heavy dependencies are missing only breaks its own
panel (NavigationPanelTool renders an error panel) rather than the whole window.
"""

from __future__ import annotations

import importlib
import json
import logging
import pathlib

from qtpy import QtCore, QtWidgets

from chisurf.gui.widgets.navigation import NavigationPanelTool

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Embedding helper
# ---------------------------------------------------------------------------
# Several sub-tools are full ``QMainWindow``s (their own toolbar/menu + a custom
# central widget). Reparenting a QMainWindow into the aggregator's stacked area
# is a fragile Qt pattern — on macOS a nested QMainWindow's tab bars stop
# receiving mouse clicks. To avoid this we lift the tool's *central widget* into
# a plain container and re-expose its toolbar/menu actions as a button row, so no
# nested QMainWindow remains. (Mirrors structure_tools' helper of the same name.)


def _embed_mainwindow(mw: QtWidgets.QWidget) -> QtWidgets.QWidget:
    """Return an embeddable plain-``QWidget`` view of a ``QMainWindow`` tool.

    If ``mw`` is not a ``QMainWindow`` it is returned unchanged. Otherwise its
    central widget is reparented into a container, prefixed by a button row that
    mirrors the window's toolbar actions (or, if it has none, its top-level menu
    actions). A reference to the original window is kept on the container so its
    Python object (and any signal connections) stays alive.
    """
    if not isinstance(mw, QtWidgets.QMainWindow):
        return mw

    container = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(4)

    # Re-expose actions: prefer the window's OWN toolbars (not toolbars that
    # belong to nested panels inside the central widget), fall back to the menu.
    actions: list[QtWidgets.QAction] = []
    for tb in mw.findChildren(QtWidgets.QToolBar):
        if tb.parent() is mw:
            actions.extend(tb.actions())
    if not actions:
        mbar = mw.menuBar()
        if mbar is not None:
            for menu_action in mbar.actions():
                menu = menu_action.menu()
                if menu is not None:
                    actions.extend(menu.actions())
    seen: set[int] = set()
    button_row = QtWidgets.QHBoxLayout()
    button_row.setContentsMargins(6, 4, 6, 0)
    n_buttons = 0
    for act in actions:
        if act is None or act.isSeparator() or not act.text().strip():
            continue
        if id(act) in seen:
            continue
        seen.add(id(act))
        btn = QtWidgets.QToolButton()
        btn.setDefaultAction(act)
        btn.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        button_row.addWidget(btn)
        n_buttons += 1
    if n_buttons:
        button_row.addStretch(1)
        layout.addLayout(button_row)

    central = mw.centralWidget()
    if central is not None:
        central.setParent(container)
        layout.addWidget(central, 1)

    # Keep the originating window alive (owns the model/signals).
    container._embedded_mainwindow = mw  # type: ignore[attr-defined]
    return container


# ---------------------------------------------------------------------------
# Data-driven panel list
# ---------------------------------------------------------------------------
# Panels are declared in ``panels.json`` (name / icon / description / role plus a
# ``entrypoint`` "module:Class" string and an optional ``embed`` flag). The
# factory for each panel is generated here: it lazily imports the entrypoint,
# instantiates it, and flattens QMainWindow tools via ``_embed_mainwindow`` when
# ``embed`` is set. Adding a tool is therefore a single JSON entry — no Python.

_PANELS_JSON = pathlib.Path(__file__).with_name("panels.json")


def _resolve_entrypoint(entrypoint: str):
    """Import ``"pkg.module:Attr"`` and return the referenced attribute."""
    module_name, _, attr = entrypoint.partition(":")
    module = importlib.import_module(module_name)
    return getattr(module, attr)


def _make_factory(entrypoint: str, embed: bool):
    """Build a lazy panel factory from an entrypoint string + embed flag."""

    def factory(parent: QtWidgets.QWidget) -> QtWidgets.QWidget:
        widget = _resolve_entrypoint(entrypoint)()
        return _embed_mainwindow(widget) if embed else widget

    return factory


def _load_panels(spec: dict) -> list[dict]:
    """Translate the ``panels.json`` spec into NavigationPanelTool panel dicts."""
    panels: list[dict] = []
    for entry in spec.get("panels", []):
        if entry.get("separator"):
            panels.append(
                {
                    "name": "────────",
                    "icon": "",
                    "separator": True,
                    "role": entry.get("role", "separator"),
                }
            )
            continue
        panels.append(
            {
                "name": entry["name"],
                "icon": entry.get("icon", ""),
                "description": entry.get("description", ""),
                "role": entry["role"],
                "factory": _make_factory(entry["entrypoint"], bool(entry.get("embed", False))),
            }
        )
    return panels


_PANEL_SPEC = json.loads(_PANELS_JSON.read_text())
TTTR_PANELS: list[dict] = _load_panels(_PANEL_SPEC)


class TttrToolboxTool(NavigationPanelTool):
    """Unified TTTR file toolbox with a left-navigation panel."""

    def __init__(self, parent=None):
        super().__init__(
            title=_PANEL_SPEC.get("title", "🧰 TTTR Tools"),
            panels=TTTR_PANELS,
            parent=parent,
            minimum_size=(900, 600),
            initial_size=(1200, 750),
            navigation_width=210,
        )


__all__ = ["TttrToolboxTool"]
