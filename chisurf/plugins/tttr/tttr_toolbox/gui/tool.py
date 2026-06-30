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

import logging

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
# Panel factory functions — each imported lazily to keep startup fast.
# QMainWindow-based tools are flattened via ``_embed_mainwindow``.
# ---------------------------------------------------------------------------


def _alex_creator(parent: TttrToolboxTool) -> QtWidgets.QWidget:
    from chisurf.plugins.tttr.ptu_alex_creator.wizard import AlexPTUCreator

    return _embed_mainwindow(AlexPTUCreator())


def _microtime_shifter(parent: TttrToolboxTool) -> QtWidgets.QWidget:
    from chisurf.plugins.tttr.tttr_microtime_shifter.gui.tool import MicrotimeShifterTool

    return _embed_mainwindow(MicrotimeShifterTool())


def _ptu_header_editor(parent: TttrToolboxTool) -> QtWidgets.QWidget:
    from chisurf.plugins.tttr.ptu_header_edit.wizard import TagsEditor, json_data

    return _embed_mainwindow(TagsEditor(json_data))


def _split_convert(parent: TttrToolboxTool) -> QtWidgets.QWidget:
    from chisurf.plugins.tttr.tttr_splitter.gui.tool import PTUSplitter

    return PTUSplitter()


# ---------------------------------------------------------------------------
# Panel list
# ---------------------------------------------------------------------------

TTTR_PANELS: list[dict] = [
    {
        "name": "ALEX Creator",
        "icon": "🔀",
        "description": "Convert ALEX macro-time modulation into micro-time so ALEX data runs through PIE pipelines.",
        "factory": _alex_creator,
        "role": "alex_creator",
    },
    {
        "name": "Micro-time Shifter",
        "icon": "⏱️",
        "description": "Apply global and per-channel micro-time shifts to TTTR files.",
        "factory": _microtime_shifter,
        "role": "microtime_shifter",
    },
    {
        "name": "PTU Header Editor",
        "icon": "🏷️",
        "description": "View, edit, add and remove header tags in PicoQuant PTU files.",
        "factory": _ptu_header_editor,
        "role": "ptu_header_editor",
    },
    {
        "name": "────────",
        "icon": "",
        "separator": True,
        "role": "separator_split",
    },
    {
        "name": "Split / Convert",
        "icon": "✂️",
        "description": "Split large TTTR files into segments and convert between container formats.",
        "factory": _split_convert,
        "role": "split_convert",
    },
]


class TttrToolboxTool(NavigationPanelTool):
    """Unified TTTR file toolbox with a left-navigation panel."""

    def __init__(self, parent=None):
        super().__init__(
            title="🧰 TTTR Tools",
            panels=TTTR_PANELS,
            parent=parent,
            minimum_size=(900, 600),
            initial_size=(1200, 750),
            navigation_width=210,
        )


__all__ = ["TttrToolboxTool"]
