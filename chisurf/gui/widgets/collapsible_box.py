"""CollapsibleBox — reusable collapsible section widget.

Usage::

    box = CollapsibleBox("Fit parameters", expanded=True, auto_fold=True)
    box.add_widget(my_widget)
    box.add_row("Alpha:", alpha_spin)
    layout.addWidget(box)
"""
from __future__ import annotations

from qtpy import QtCore, QtWidgets


def _resolve_auto_fold_delay(delay: int | None = None) -> int:
    """Resolve auto-fold delay from central settings if not explicitly given.

    Returns
    -------
    int
        Delay in milliseconds.  A value < 0 means auto-fold is globally
        disabled.
    """
    if delay is not None:
        return delay
    try:
        from chisurf.core.settings import cs_settings
        cfg = cs_settings.get("gui", {}).get("collapsible_box", {})
        return int(cfg.get("auto_fold_timeout_ms", 1200))
    except Exception:
        return 1200


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
    auto_fold_delay_ms : int or None
        Milliseconds of inactivity before auto-fold fires.  ``None`` (the
        default) reads the value from the global setting
        ``gui.collapsible_box.auto_fold_timeout_ms`` in
        ``settings_chisurf.yaml``.  A value < 0 disables auto-fold.
    """

    toggled = QtCore.Signal(bool)  # True = expanded, False = collapsed

    def title(self) -> str:
        """Return the header text of this collapsible section."""
        return self._title

    def __init__(
        self,
        title: str,
        parent: QtWidgets.QWidget | None = None,
        *,
        expanded: bool = True,
        auto_fold: bool = False,
        auto_fold_delay_ms: int | None = None,
    ) -> None:
        super().__init__(parent)
        self._title = title
        self._expanded = expanded

        delay = _resolve_auto_fold_delay(auto_fold_delay_ms)
        self._auto_fold_enabled = auto_fold and delay >= 0
        self._auto_fold_delay = max(delay, 0)  # non-negative timer interval

        self._fold_timer = QtCore.QTimer(self)
        self._fold_timer.setSingleShot(True)
        self._fold_timer.timeout.connect(self._on_fold_timer)

        self.setMouseTracking(True)
        self._build_ui()
        self._apply_state(animate=False)

    def _build_ui(self) -> None:
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # QPushButton (not QToolButton) correctly honours CSS text-align:left.
        # Styled via gui/styles/widgets/collapsible_box.qss (objectName target).
        self._btn = QtWidgets.QPushButton(self)
        self._btn.setObjectName("CollapsibleBoxHeader")
        self._btn.setCheckable(True)
        self._btn.setChecked(self._expanded)
        self._btn.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed
        )
        self._btn.clicked.connect(self._on_header_clicked)
        self._update_header_text()
        root.addWidget(self._btn)

        self._content = QtWidgets.QWidget(self)
        # Preferred (not Expanding) so the box takes exactly the space it needs;
        # Expanding caused equal height distribution among all sibling panels.
        self._content.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred
        )
        self._content_layout = QtWidgets.QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(2, 2, 2, 2)
        self._content_layout.setSpacing(1)
        root.addWidget(self._content)  # no stretch — parent decides allocation

    def add_widget(self, widget: QtWidgets.QWidget, stretch: int = 0) -> None:
        """Append *widget* to the content area."""
        self._content_layout.addWidget(widget, stretch)

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

    def _on_header_clicked(self) -> None:
        self._expanded = not self._expanded
        self._btn.setChecked(self._expanded)
        self._apply_state(animate=True)

    def _apply_state(self, *, animate: bool = True) -> None:
        self._content.setVisible(self._expanded)
        self._update_header_text()
        if self._expanded:
            self.setSizePolicy(
                QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred
            )
        else:
            self.setSizePolicy(
                QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed
            )
        if self.parent() is not None:
            try:
                self.parent().updateGeometry()
            except Exception:
                pass
        self.updateGeometry()
        self.toggled.emit(self._expanded)

    def _update_header_text(self) -> None:
        arrow = "▼" if self._expanded else "▶"
        self._btn.setText(f"{arrow}  {self._title}")

    def leaveEvent(self, event: QtCore.QEvent) -> None:
        if self._auto_fold_enabled and self._expanded:
            self._fold_timer.start(self._auto_fold_delay)
        super().leaveEvent(event)

    def enterEvent(self, event: QtCore.QEvent) -> None:
        self._fold_timer.stop()
        super().enterEvent(event)

    def _on_fold_timer(self) -> None:
        if self._auto_fold_enabled and self._expanded and not self.underMouse():
            self.set_expanded(False)
