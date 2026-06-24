"""CollapsibleBox — reusable collapsible section widget.

Usage::

    box = CollapsibleBox("Fit parameters", expanded=True, auto_fold=True)
    box.add_widget(my_widget)
    box.add_row("Alpha:", alpha_spin)
    layout.addWidget(box)
"""
from __future__ import annotations

from qtpy import QtCore, QtGui, QtWidgets


class CollapsibleBox(QtWidgets.QWidget):
    """A section with a clickable header that shows/hides its content.

    Parameters
    ----------
    title : str
        Text shown in the header bar.
    expanded : bool
        Initial state (True = open, False = closed).
    auto_fold : bool
        When True, the section automatically folds after the mouse leaves
        it and the fold timer expires.
    auto_fold_delay_ms : int
        Milliseconds of inactivity before auto-fold fires (default 1200 ms).
    """

    toggled = QtCore.Signal(bool)  # True = expanded, False = collapsed

    def __init__(
        self,
        title: str,
        parent: QtWidgets.QWidget | None = None,
        *,
        expanded: bool = True,
        auto_fold: bool = False,
        auto_fold_delay_ms: int = 1200,
    ) -> None:
        super().__init__(parent)
        self._title = title
        self._expanded = expanded
        self._auto_fold_enabled = auto_fold
        self._auto_fold_delay = auto_fold_delay_ms

        self._fold_timer = QtCore.QTimer(self)
        self._fold_timer.setSingleShot(True)
        self._fold_timer.timeout.connect(self._on_fold_timer)

        self.setMouseTracking(True)
        self._build_ui()
        self._apply_state(animate=False)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Header button — arrow lives on the very left via text alignment
        self._btn = QtWidgets.QToolButton(self)
        self._btn.setCheckable(True)
        self._btn.setChecked(self._expanded)
        self._btn.setToolButtonStyle(QtCore.Qt.ToolButtonTextOnly)
        self._btn.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed
        )
        self._btn.setStyleSheet(
            "QToolButton {"
            "  background: #272b33;"
            "  color: #c8ccd4;"
            "  border: none;"
            "  border-bottom: 1px solid #1a1d22;"
            "  font-size: 10px;"
            "  font-weight: bold;"
            "  text-align: left;"
            "  padding: 4px 6px 4px 0px;"
            "}"
            "QToolButton:hover { background: #32363f; }"
            "QToolButton:checked { color: #7eb8f7; }"
        )
        self._btn.clicked.connect(self._on_header_clicked)
        self._update_header_text()
        root.addWidget(self._btn)

        # Content container
        self._content = QtWidgets.QWidget(self)
        self._content.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )
        self._content_layout = QtWidgets.QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(6, 4, 6, 6)
        self._content_layout.setSpacing(3)
        root.addWidget(self._content, 1)  # stretch=1: fills available vertical space

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_widget(self, widget: QtWidgets.QWidget) -> None:
        """Append *widget* to the content area."""
        self._content_layout.addWidget(widget)

    def add_row(self, label: str, widget: QtWidgets.QWidget) -> None:
        """Append a label + widget row to the content area."""
        row = QtWidgets.QWidget(self._content)
        rl = QtWidgets.QHBoxLayout(row)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(8)
        lbl = QtWidgets.QLabel(label, row)
        lbl.setFixedWidth(96)
        lbl.setStyleSheet("color: #9ba3af; font-size: 10px;")
        rl.addWidget(lbl)
        rl.addWidget(widget, 1)
        self._content_layout.addWidget(row)

    def set_expanded(self, expanded: bool) -> None:
        """Programmatically expand or collapse the section."""
        if expanded != self._expanded:
            self._expanded = expanded
            self._btn.setChecked(expanded)
            self._apply_state(animate=True)

    def is_expanded(self) -> bool:
        return self._expanded

    @property
    def auto_fold(self) -> bool:
        return self._auto_fold_enabled

    @auto_fold.setter
    def auto_fold(self, value: bool) -> None:
        self._auto_fold_enabled = bool(value)
        if not self._auto_fold_enabled:
            self._fold_timer.stop()

    # ------------------------------------------------------------------
    # Internal slots & event handlers
    # ------------------------------------------------------------------

    def _on_header_clicked(self) -> None:
        self._expanded = not self._expanded
        self._btn.setChecked(self._expanded)
        self._apply_state(animate=True)

    def _apply_state(self, *, animate: bool = True) -> None:
        self._content.setVisible(self._expanded)
        self._update_header_text()
        # Switch size policy so the parent layout can give us vertical space when
        # expanded and treat us as fixed-height when collapsed.
        if self._expanded:
            self.setSizePolicy(
                QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
            )
        else:
            self.setSizePolicy(
                QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed
            )
        if self.parent() is not None:
            try:
                self.parent().updateGeometry()  # type: ignore[union-attr]
            except Exception:
                pass
        self.updateGeometry()
        self.toggled.emit(self._expanded)

    def _update_header_text(self) -> None:
        arrow = "▼" if self._expanded else "▶"
        self._btn.setText(f"{arrow}  {self._title}")

    # Auto-fold on mouse leave -------------------------------------------

    def leaveEvent(self, event: QtCore.QEvent) -> None:  # type: ignore[override]
        if self._auto_fold_enabled and self._expanded:
            self._fold_timer.start(self._auto_fold_delay)
        super().leaveEvent(event)

    def enterEvent(self, event: QtCore.QEvent) -> None:  # type: ignore[override]
        self._fold_timer.stop()
        super().enterEvent(event)

    def _on_fold_timer(self) -> None:
        if self._auto_fold_enabled and self._expanded and not self.underMouse():
            self.set_expanded(False)
