"""Two-column directed-wizard renderer for a :class:`WizardSection`.

:class:`WizardWidget` lays a wizard out the same way the Settings tool does — a
left navigation list of steps and a right content pane — but *directed*: a
Back / Next / Finish bar walks the steps in order, completed steps show a ✓, and
when the wizard is ``linear`` the *Next* button is gated on the current step
being complete. Prior steps stay clickable so the user can revisit them.

The step *bodies* are built by the surrounding :class:`~chisurf.gui.autoform.auto_form.AutoForm`
(via ``_emit_sections``) and handed in as ready widgets, so the steps bind to the
same model as the rest of the form and this module needs no per-control logic.
"""

from __future__ import annotations

from qtpy import QtCore, QtWidgets

from chisurf import typing

#: Prefix shown in the nav list for a completed step.
_CHECK = "✓ "


class WizardWidget(QtWidgets.QWidget):
    """A directed, two-column stepper hosting pre-built step bodies.

    Parameters
    ----------
    section : chisurf.core.dataspec.WizardSection
        The declarative wizard (titles, icons, ``linear``, ``nav_width``, ``persist``).
    pages : list of QtWidgets.QWidget
        One ready-built body widget per step, in step order.
    is_complete : callable
        ``is_complete(step_index) -> bool`` — re-evaluated on demand to drive the
        ✓ marks and (when ``linear``) the *Next* gate.
    parent : QtWidgets.QWidget, optional
    """

    #: Let AutoForm.rebuild() give the wizard the spare vertical space.
    _autoform_expanding = True
    #: Refresh nav ✓ / Next-gate when the model changes (AutoForm.refresh_plots).
    AUTOFORM_REFRESH = True

    def __init__(
        self,
        section,
        pages: typing.List[QtWidgets.QWidget],
        is_complete: typing.Callable[[int], bool],
        parent=None,
    ):
        super().__init__(parent)
        self._section = section
        self._steps = list(getattr(section, "steps", ()))
        self._pages = list(pages)
        self._is_complete = is_complete
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

        nav_width = int(getattr(section, "nav_width", 200) or 200)

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        outer.addWidget(self.splitter, 1)

        # -- left: navigation list ------------------------------------------
        self.nav_list = QtWidgets.QListWidget()
        self.nav_list.setMinimumWidth(nav_width)
        self.nav_list.setSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.nav_list.setSpacing(4)
        self.nav_list.setStyleSheet(
            """
            QListWidget {
                border: none;
                border-right: 1px solid rgba(128, 128, 128, 0.3);
                padding-top: 5px;
            }
            QListWidget::item {
                height: 32px;
                padding-left: 10px;
                border-radius: 8px;
                margin: 2px 10px;
                font-weight: bold;
                font-size: 14px;
            }
            """
        )
        for step in self._steps:
            item = QtWidgets.QListWidgetItem(self._nav_label(step, False))
            if getattr(step, "description", ""):
                item.setToolTip(str(step.description))
            self.nav_list.addItem(item)
        self.splitter.addWidget(self.nav_list)

        # -- right: header + stacked pages + button bar ---------------------
        right = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right)
        right_layout.setContentsMargins(10, 8, 10, 8)
        right_layout.setSpacing(6)

        self.title_label = QtWidgets.QLabel()
        self.title_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        self.subtitle_label = QtWidgets.QLabel()
        self.subtitle_label.setWordWrap(True)
        self.subtitle_label.setStyleSheet("color: rgba(128,128,128,1);")
        right_layout.addWidget(self.title_label)
        right_layout.addWidget(self.subtitle_label)

        self.stack = QtWidgets.QStackedWidget()
        for page in self._pages:
            scroll = QtWidgets.QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
            scroll.setWidget(page)
            self.stack.addWidget(scroll)
        right_layout.addWidget(self.stack, 1)

        # button bar
        bar = QtWidgets.QHBoxLayout()
        bar.setContentsMargins(0, 0, 0, 0)
        bar.addStretch(1)
        self.back_btn = QtWidgets.QPushButton("‹ Back")
        self.next_btn = QtWidgets.QPushButton("Next ›")
        self.finish_btn = QtWidgets.QPushButton("Finish")
        bar.addWidget(self.back_btn)
        bar.addWidget(self.next_btn)
        bar.addWidget(self.finish_btn)
        right_layout.addLayout(bar)

        self.splitter.addWidget(right)
        self.splitter.setCollapsible(0, False)
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setSizes([nav_width, max(600, nav_width * 3)])

        # -- wiring ---------------------------------------------------------
        self.nav_list.currentRowChanged.connect(self._on_nav_changed)
        self.back_btn.clicked.connect(self._go_back)
        self.next_btn.clicked.connect(self._go_next)
        self.finish_btn.clicked.connect(self._finish)

        self._restore_index()

    # -- helpers ------------------------------------------------------------
    @staticmethod
    def _nav_label(step, complete: bool) -> str:
        icon = getattr(step, "icon", "") or ""
        title = getattr(step, "title", "") or "Step"
        prefix = _CHECK if complete else ""
        core = f"{icon} {title}".strip() if icon else title
        return f"{prefix}{core}"

    @property
    def current_index(self) -> int:
        """Index of the currently selected step (``-1`` when none)."""
        return self.nav_list.currentRow()

    def _step_complete(self, index: int) -> bool:
        step = self._steps[index]
        if getattr(step, "optional", False):
            return True
        try:
            return bool(self._is_complete(index))
        except Exception:  # pragma: no cover - defensive
            return False

    # -- navigation ---------------------------------------------------------
    def _on_nav_changed(self, row: int) -> None:
        if row < 0 or row >= len(self._steps):
            return
        self.stack.setCurrentIndex(row)
        step = self._steps[row]
        self.title_label.setText(getattr(step, "title", "") or "")
        subtitle = getattr(step, "subtitle", "") or ""
        self.subtitle_label.setText(subtitle)
        self.subtitle_label.setVisible(bool(subtitle))
        self.refresh()

    def _go_back(self) -> None:
        if self.current_index > 0:
            self.nav_list.setCurrentRow(self.current_index - 1)

    def _go_next(self) -> None:
        if self.current_index < len(self._steps) - 1:
            self.nav_list.setCurrentRow(self.current_index + 1)

    def _finish(self) -> None:
        self._store_index()
        win = self.window()
        if win is not None:
            win.close()

    # -- live state ---------------------------------------------------------
    def refresh(self) -> None:
        """Update ✓ marks and the Back/Next/Finish bar for the current model state."""
        for i, step in enumerate(self._steps):
            item = self.nav_list.item(i)
            if item is not None:
                item.setText(self._nav_label(step, self._step_complete(i)))
        idx = self.current_index
        last = idx >= len(self._steps) - 1
        self.back_btn.setEnabled(idx > 0)
        self.next_btn.setVisible(not last)
        self.finish_btn.setVisible(last)
        gated = bool(getattr(self._section, "linear", False)) and not self._step_complete(idx)
        self.next_btn.setEnabled(not last and not gated)

    # -- persistence (best-effort, Qt-native) -------------------------------
    def _persist_key(self) -> str:
        return str(getattr(self._section, "persist", "") or "")

    def _restore_index(self) -> None:
        key = self._persist_key()
        idx = 0
        if key:
            try:
                settings = QtCore.QSettings("chisurf", "autoform-wizard")
                idx = int(settings.value(f"{key}/step", 0))
            except Exception:
                idx = 0
        idx = max(0, min(idx, len(self._steps) - 1)) if self._steps else 0
        self.nav_list.setCurrentRow(idx)
        # setCurrentRow(0) on an already-0 row emits no signal; force the header/bar sync.
        self._on_nav_changed(idx)

    def _store_index(self) -> None:
        key = self._persist_key()
        if not key:
            return
        try:
            settings = QtCore.QSettings("chisurf", "autoform-wizard")
            settings.setValue(f"{key}/step", int(self.current_index))
        except Exception:
            pass


__all__ = ["WizardWidget"]
