"""background_startup.py — deferred post-startup stage runner (adapter).

This module provides a ``QTimer``-driven runner that executes ``post_gui_show``
app startup services one-at-a-time so the Qt event loop stays responsive.
It is a thin adapter over ``chisurf.startup.services.AppStartupServiceManager``.
"""

from __future__ import annotations

import chisurf as cs
import traceback
from typing import Any, Callable, Optional

from qtpy import QtCore

from chisurf.startup.services import (
    AppStartupContext,
    AppStartupServiceManager,
    AppStartupServiceSpec,
    _DefaultEntrypointLoader,
)


class BackgroundStartupRunner(QtCore.QObject):
    """Run post-show app startup services sequentially on the GUI thread.

    Parameters
    ----------
    window:
        The ``Main`` window instance.  Must have ``progress_bar``
        (``QProgressBar``) and ``status_label`` (``QLabel``) attributes.
    manager:
        An ``AppStartupServiceManager`` with loaded specs.
    on_complete:
        Optional callback invoked after all stages finish.
    """

    finished = QtCore.Signal()

    def __init__(
        self,
        window,
        manager: AppStartupServiceManager,
        on_complete: Optional[Callable[[], None]] = None,
        parent: Optional[QtCore.QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._window = window
        self._manager = manager
        self._specs: list[AppStartupServiceSpec] = []
        self._on_complete = on_complete
        self._index = 0
        self._loader = _DefaultEntrypointLoader()

    def start(self) -> None:
        """Evaluate conditions, report skips, and schedule the first stage."""
        self._specs = self._manager.resolve_enabled(surface="gui", phase="post_gui_show")
        if not self._specs:
            self._finalize()
            return
        self._index = 0
        self._schedule_next()

    def _schedule_next(self) -> None:
        QtCore.QTimer.singleShot(0, self._run_next)

    def _run_next(self) -> None:
        if self._index >= len(self._specs):
            self._finalize()
            return

        spec = self._specs[self._index]

        self._set_status(spec.label, self._index, len(self._specs))

        dependencies = {
            dep: self._manager.get_service_result(dep)
            for dep in spec.depends_on
        }
        context = AppStartupContext(
            dispatcher=self._manager.dispatcher,
            state=self._manager.state,
            event_bus=self._manager.event_bus,
            job_manager=self._manager.job_manager,
            stop_event=self._manager._stop_event,
            dependencies=dependencies,
            surface=spec.surface,
            phase=spec.phase,
        )

        try:
            register_fn = self._loader.load(spec.entrypoint)
            register_fn(context)
        except Exception:
            try:
                cs.logging.error(
                    f"Background startup stage '{spec.id}' failed:\n"
                    + traceback.format_exc()
                )
            except Exception:
                pass

        self._index += 1
        self._schedule_next()

    def _finalize(self) -> None:
        self._set_ready()
        self.finished.emit()
        if callable(self._on_complete):
            try:
                self._on_complete()
            except Exception:
                pass

    def _set_status(self, label: str, index: int, total: int) -> None:
        try:
            cs.logging.info(f"[BG] {label}")
        except Exception:
            pass
        try:
            pb = getattr(self._window, "progress_bar", None)
            if pb is not None:
                pct = int(100 * index / total) if total else 0
                pb.setVisible(True)
                pb.setValue(pct)
        except Exception:
            pass
        try:
            lbl = getattr(self._window, "status_label", None)
            if lbl is not None:
                lbl.setText(label)
        except Exception:
            pass
        try:
            status = getattr(self._window, "status", None)
            if status is not None:
                status.showMessage(label)
        except Exception:
            pass
        try:
            from qtpy import QtWidgets
            app = QtWidgets.QApplication.instance()
            if app is not None:
                app.processEvents()
        except Exception:
            pass

    def _set_ready(self) -> None:
        try:
            pb = getattr(self._window, "progress_bar", None)
            if pb is not None:
                pb.setValue(100)
                pb.setVisible(False)
        except Exception:
            pass
        try:
            lbl = getattr(self._window, "status_label", None)
            if lbl is not None:
                lbl.setText("Ready")
        except Exception:
            pass
        try:
            status = getattr(self._window, "status", None)
            if status is not None:
                status.showMessage("Ready", 3000)
        except Exception:
            pass
