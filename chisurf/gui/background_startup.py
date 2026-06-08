"""background_startup.py — deferred post-startup stage runner.

import chisurf as cs
After the main window becomes visible, certain non-critical initialisation
tasks (plugin discovery, Jupyter start, update check, module warm-up) are
scheduled here so they do not block the splash-screen phase.

Usage
-----
    runner = BackgroundStartupRunner(window, stages, on_complete=None)
    runner.start()

Each *stage* is a tuple ``(label: str, callable: Callable[[], None])``.
The runner executes them one-at-a-time via ``QTimer.singleShot(0, …)`` so the
Qt event loop stays responsive between stages. Progress is reflected on:

* ``window.progress_bar``  — a ``QProgressBar`` in the status bar
* ``window.status_label``  — a ``QLabel`` next to the progress bar
"""
from __future__ import annotations

import chisurf as cs
import traceback
from typing import Callable, List, Optional, Tuple

from qtpy import QtCore


class BackgroundStartupRunner(QtCore.QObject):
    """Run a list of (label, callable) stages sequentially on the GUI thread.

    Each stage is kicked off via ``QTimer.singleShot(0, ...)`` so control
    returns to the event loop between stages and the UI stays responsive.

    Parameters
    ----------
    window:
        The ``Main`` window instance.  Must have ``progress_bar``
        (``QProgressBar``) and ``status_label`` (``QLabel``) attributes.
    stages:
        Ordered list of ``(label, callable)`` pairs.  The callable receives
        no arguments.
    on_complete:
        Optional callback invoked (with no arguments) after all stages finish.
    """

    # Emitted when all stages are done
    finished = QtCore.Signal()

    def __init__(
        self,
        window,
        stages: List[Tuple[str, Callable[[], None]]],
        on_complete: Optional[Callable[[], None]] = None,
        parent: Optional[QtCore.QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._window = window
        self._stages = list(stages)
        self._on_complete = on_complete
        self._index = 0
        self._total = len(self._stages)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Schedule the first stage and return immediately."""
        if not self._stages:
            self._finalize()
            return
        self._schedule_next()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _schedule_next(self) -> None:
        QtCore.QTimer.singleShot(0, self._run_next)

    def _run_next(self) -> None:
        if self._index >= self._total:
            self._finalize()
            return

        label, fn = self._stages[self._index]

        # --- update status bar ---
        self._set_status(label, self._index, self._total)

        # --- run stage ---
        try:
            fn()
        except Exception:
            try:
                cs.logging.error(
                    f"Background startup stage '{label}' failed:\n"
                    + traceback.format_exc()
                )
            except Exception:
                pass

        self._index += 1
        self._schedule_next()

    def _finalize(self) -> None:
        """Called after all stages finish."""
        self._set_ready()
        self.finished.emit()
        if callable(self._on_complete):
            try:
                self._on_complete()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Status-bar helpers
    # ------------------------------------------------------------------

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

        # Also update the status bar message
        try:
            status = getattr(self._window, "status", None)
            if status is not None:
                status.showMessage(label)
        except Exception:
            pass

        # Keep the UI responsive
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
                # Hide progress bar when done — status bar has more space for messages
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
