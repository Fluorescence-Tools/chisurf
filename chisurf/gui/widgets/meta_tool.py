"""Reusable "meta tool" window: a left icon rail hosting several tools.

A :class:`MetaToolWindow` shows a vertical rail of (emoji) icon buttons on the
left; selecting one shows the corresponding tool on the right. Tools are
imported / instantiated lazily on first selection so opening the window stays
cheap. A ``SEPARATOR`` sentinel renders a horizontal divider between groups.

This is the construction pattern of the FCS Tools window, factored out so other
category windows (Image Tools, Burst Analysis, Decay Analysis, Settings, …) can
reuse the exact same look and behaviour by passing their own tool list.

Example
-------
::

    TOOLS = [
        ("🎛️", "Detector\\nDef", _make_detector_def),
        SEPARATOR,
        ("🧮", "Calc", _make_calc),
    ]
    win = MetaToolWindow("🧰 FCS Tools", TOOLS)
"""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple

from qtpy import QtCore, QtWidgets

#: A rail entry: ``(emoji, label, factory)``. ``factory`` builds the tool widget.
ToolEntry = Tuple[str, str, Optional[Callable[[], QtWidgets.QWidget]]]

#: Sentinel entry (factory is ``None``) that renders a horizontal divider.
SEPARATOR: ToolEntry = ("—", "", None)


class MetaToolWindow(QtWidgets.QMainWindow):
    """A meta tool: a left icon rail + the selected tool shown on the right."""

    def __init__(
        self,
        title: str,
        tools: List[ToolEntry],
        parent: QtWidgets.QWidget = None,
        rail_width: int = 96,
        default_size: Tuple[int, int] = (1000, 660),
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(*default_size)

        self._tools = list(tools)
        self._factories: List[Callable[[], QtWidgets.QWidget]] = []
        self._instances: List[Optional[QtWidgets.QWidget]] = []

        central = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # ── left icon rail ────────────────────────────────────────────
        rail = QtWidgets.QWidget()
        rail.setFixedWidth(rail_width)
        rail.setAutoFillBackground(True)
        rail_l = QtWidgets.QVBoxLayout(rail)
        rail_l.setContentsMargins(4, 6, 4, 6)
        rail_l.setSpacing(4)

        self._group = QtWidgets.QButtonGroup(self)
        self._group.setExclusive(True)
        for emoji, label, factory in self._tools:
            if factory is None:  # SEPARATOR
                line = QtWidgets.QFrame()
                line.setFrameShape(QtWidgets.QFrame.HLine)
                line.setFrameShadow(QtWidgets.QFrame.Sunken)
                rail_l.addWidget(line)
                continue
            tool_index = len(self._factories)
            btn = QtWidgets.QToolButton()
            btn.setText(f"{emoji}\n{label}")
            btn.setCheckable(True)
            btn.setAutoRaise(True)
            btn.setToolButtonStyle(QtCore.Qt.ToolButtonTextOnly)
            btn.setToolTip(label.replace("\n", " "))
            btn.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
            btn.setMinimumHeight(64)
            btn.setStyleSheet("QToolButton { font-size: 11px; } ")
            self._group.addButton(btn, tool_index)
            rail_l.addWidget(btn)
            self._factories.append(factory)
            self._instances.append(None)
        rail_l.addStretch(1)
        layout.addWidget(rail)

        # ── right content stack ───────────────────────────────────────
        self._stack = QtWidgets.QStackedWidget()
        for _ in self._factories:
            self._stack.addWidget(QtWidgets.QWidget())  # lazy placeholder
        layout.addWidget(self._stack, 1)

        self.setCentralWidget(central)

        self._group.idClicked.connect(self._select_tool)
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
            except Exception as exc:  # readable placeholder on failure
                widget = QtWidgets.QLabel(f"Failed to load tool:\n{exc}")
                widget.setAlignment(QtCore.Qt.AlignCenter)
                widget.setWordWrap(True)
            self._instances[index] = widget
            old = self._stack.widget(index)
            self._stack.insertWidget(index, widget)
            self._stack.removeWidget(old)
            old.deleteLater()
        self._stack.setCurrentIndex(index)
