"""StructureToolsTool — unified structure-modelling panel using NavigationPanelTool.

A single window with a left navigation list and a lazily-loaded right panel area.
Each panel embeds one of the existing structure tools **unchanged**:

    1. FPS JSON Editor          — FpsJsonEditorTool
    2. FRET Docking & Screening — FretDockingTool
    3. Kappa2 Distribution      — Kappa2Dist
    ───────────────────────────  (separator)
    4. QuEst                    — QuEstWindow
    5. HydroPro                 — HydroGui
    ───────────────────────────  (separator)
    6. Trajectory Tools         — TrajectoryToolsTool

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
# ``DockArea`` central widget). Reparenting a QMainWindow into the aggregator's
# stacked area is a fragile Qt pattern — on macOS the nested QMainWindow's
# DockArea tab bar stops receiving mouse clicks. To avoid this we lift the
# tool's *central widget* into a plain container and re-expose its toolbar/menu
# actions as a button row, so no nested QMainWindow remains.

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
# QMainWindow-based tools are flattened via ``_embed_mainwindow`` so their
# DockArea tabs remain clickable when embedded (see helper above).
# ---------------------------------------------------------------------------

def _fps_json_editor(parent: "StructureToolsTool") -> QtWidgets.QWidget:
    from chisurf.plugins.modelling.fps_json_editor.gui.tool import FpsJsonEditorTool
    return _embed_mainwindow(FpsJsonEditorTool())


def _docking(parent: "StructureToolsTool") -> QtWidgets.QWidget:
    from chisurf.plugins.modelling.fret.gui.dock_tool import FretDockingTool
    return FretDockingTool()


def _kappa2(parent: "StructureToolsTool") -> QtWidgets.QWidget:
    from chisurf.plugins.kappa2_dist.gui.tool import Kappa2Dist
    return Kappa2Dist()


def _quest(parent: "StructureToolsTool") -> QtWidgets.QWidget:
    from chisurf.plugins.quenching_estimator import QuEstWindow
    return _embed_mainwindow(QuEstWindow())


def _hydropro(parent: "StructureToolsTool") -> QtWidgets.QWidget:
    from chisurf.plugins.modelling.hydropro.gui.tool import HydroProTool
    return _embed_mainwindow(HydroProTool())


def _traj_tools(parent: "StructureToolsTool") -> QtWidgets.QWidget:
    from chisurf.plugins.traj.traj_tools.gui.tool import TrajectoryToolsTool
    return TrajectoryToolsTool()


# ---------------------------------------------------------------------------
# Panel list
# ---------------------------------------------------------------------------

STRUCTURE_PANELS: list[dict] = [
    {
        "name": "1. FPS JSON Editor",
        "icon": "📝",
        "description": "Edit fps.json files for FRET accessible-volume modelling and fetch reference PDBs.",
        "factory": _fps_json_editor,
        "role": "fps_json_editor",
    },
    {
        "name": "2. Docking & Screening",
        "icon": "🎯",
        "description": "FRET-restrained rigid-body docking, refinement and structure-library screening (IMP + IMP.bff).",
        "factory": _docking,
        "role": "docking",
    },
    {
        "name": "3. Kappa2 Distribution",
        "icon": "📐",
        "description": "Calculate and visualise the κ² orientation-factor distribution for FRET.",
        "factory": _kappa2,
        "role": "kappa2",
    },
    {
        "name": "────────",
        "icon": "",
        "separator": True,
        "role": "separator_compute",
    },
    {
        "name": "QuEst",
        "icon": "💡",
        "description": "Quenching estimator — dye-diffusion simulation of fluorescence quenching and decays.",
        "factory": _quest,
        "role": "quest",
    },
    {
        "name": "HydroPro",
        "icon": "🌊",
        "description": "Hydrodynamic property prediction (HydroPro) from atomic structures.",
        "factory": _hydropro,
        "role": "hydropro",
    },
    {
        "name": "────────",
        "icon": "",
        "separator": True,
        "role": "separator_traj",
    },
    {
        "name": "Trajectory Tools",
        "icon": "🎞️",
        "description": "Combined workspace for trajectory alignment, conversion, energy calculation and FRET.",
        "factory": _traj_tools,
        "role": "traj_tools",
    },
]


class StructureToolsTool(NavigationPanelTool):
    """Unified structure-modelling toolbox with a left-navigation panel."""

    def __init__(self, parent=None):
        super().__init__(
            title="🧬 Structure Tools",
            panels=STRUCTURE_PANELS,
            parent=parent,
            minimum_size=(900, 600),
            initial_size=(1200, 750),
            navigation_width=210,
        )


__all__ = ["StructureToolsTool"]
