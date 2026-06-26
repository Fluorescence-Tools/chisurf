"""FCS Toolbox — a meta tool that hosts several FCS tools behind a left icon rail.

A QMainWindow with a left vertical rail of (emoji) icons; selecting one shows the
corresponding tool on the right. Tools are imported and instantiated lazily on
first selection so opening the toolbox stays cheap.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple

from qtpy import QtCore, QtWidgets


def _make_2dflcs() -> QtWidgets.QWidget:
    from chisurf.plugins.fcs.fcs_2d import TwoDFCSPlugin
    return TwoDFCSPlugin()


def _make_burst_fcs() -> QtWidgets.QWidget:
    from chisurf.plugins.burst.burst_fcs_correlator.gui.tool import BurstFcsTool
    return BurstFcsTool()


def _make_diffusion_calc() -> QtWidgets.QWidget:
    from chisurf.plugins.fcs.fcs_calculator.wizard import ConfocalCalcWidget
    return ConfocalCalcWidget()


# (emoji, label, factory)
TOOLS: List[Tuple[str, str, Callable[[], QtWidgets.QWidget]]] = [
    ("🟦", "2D-FLCS", _make_2dflcs),
    ("🔬", "Burst-wise\nFCS", _make_burst_fcs),
    ("🧮", "Diffusion\nCalc", _make_diffusion_calc),
]


class FcsToolboxTool(QtWidgets.QMainWindow):
    """Meta FCS tool: left icon rail + the selected tool on the right."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("🧰 FCS Toolbox")
        self.resize(1000, 660)

        self._factories: List[Callable[[], QtWidgets.QWidget]] = []
        self._instances: List[Optional[QtWidgets.QWidget]] = []

        central = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # ── left icon rail ────────────────────────────────────────────
        rail = QtWidgets.QWidget()
        rail.setFixedWidth(96)
        rail.setAutoFillBackground(True)
        rail_l = QtWidgets.QVBoxLayout(rail)
        rail_l.setContentsMargins(4, 6, 4, 6)
        rail_l.setSpacing(4)

        self._group = QtWidgets.QButtonGroup(self)
        self._group.setExclusive(True)
        for i, (emoji, label, factory) in enumerate(TOOLS):
            btn = QtWidgets.QToolButton()
            btn.setText(f"{emoji}\n{label}")
            btn.setCheckable(True)
            btn.setAutoRaise(True)
            btn.setToolButtonStyle(QtCore.Qt.ToolButtonTextOnly)
            btn.setToolTip(label.replace("\n", " "))
            btn.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
            btn.setMinimumHeight(64)
            btn.setStyleSheet("QToolButton { font-size: 11px; } ")
            self._group.addButton(btn, i)
            rail_l.addWidget(btn)
            self._factories.append(factory)
            self._instances.append(None)
        rail_l.addStretch(1)
        layout.addWidget(rail)

        # ── right content stack ───────────────────────────────────────
        self._stack = QtWidgets.QStackedWidget()
        for _ in TOOLS:
            self._stack.addWidget(QtWidgets.QWidget())  # lazy placeholder
        layout.addWidget(self._stack, 1)

        self.setCentralWidget(central)

        self._group.idClicked.connect(self._select_tool)
        # Select the first tool by default.
        first = self._group.button(0)
        if first is not None:
            first.setChecked(True)
            self._select_tool(0)

    def _select_tool(self, index: int) -> None:
        if not (0 <= index < len(self._factories)):
            return
        btn = self._group.button(index)
        if btn is not None and not btn.isChecked():
            btn.setChecked(True)
        if self._instances[index] is None:
            try:
                widget = self._factories[index]()
            except Exception as exc:  # show a readable placeholder on failure
                widget = QtWidgets.QLabel(f"Failed to load tool:\n{exc}")
                widget.setAlignment(QtCore.Qt.AlignCenter)
                widget.setWordWrap(True)
            self._instances[index] = widget
            old = self._stack.widget(index)
            self._stack.insertWidget(index, widget)
            self._stack.removeWidget(old)
            old.deleteLater()
        self._stack.setCurrentIndex(index)
