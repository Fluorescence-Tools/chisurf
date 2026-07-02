"""GUI entrypoint for the IRF & BG calibration step (AutoForm + view.json)."""

from __future__ import annotations

import logging

from qtpy import QtCore, QtWidgets

from chisurf.gui.autoform import AutoForm

from .view_model import CalibrationViewModel

logger = logging.getLogger(__name__)


class _HistSignals(QtCore.QObject):
    """Signal emitted when the background histogram binning finishes."""

    done = QtCore.Signal()


class _HistTask(QtCore.QRunnable):
    """Bin the (slow) data/IRF histograms off the UI thread (bincount frees the GIL)."""

    def __init__(self, model, signals: _HistSignals):
        super().__init__()
        self._model = model
        self._signals = signals
        self.setAutoDelete(True)

    def run(self) -> None:  # noqa: N802 (Qt override)
        try:
            self._model.ensure_histograms()
        except Exception:
            logger.debug("background histogram binning failed", exc_info=True)
        self._signals.done.emit()


class ImgCalibrationTool(QtWidgets.QWidget):
    """Per-detector IRF / background calibration tool (AutoForm-hosted)."""

    def __init__(self, parent=None, embedded: bool = False, view_model=None):
        super().__init__(parent)
        self._embedded = bool(embedded)
        self.model = view_model or CalibrationViewModel()
        self.setWindowTitle("IRF & BG")
        self.setMinimumSize(560, 360)
        self.setAcceptDrops(True)  # drop an IRF file to set it for the current detector
        self._binning = False

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        self.auto_form = AutoForm(self.model)
        layout.addWidget(self.auto_form)
        self.model.add_observer(self._on_model_event)

    def apply_setup_settings(self, payload: dict) -> None:
        """Forward the shared detector definition (rebuilds the table rows)."""
        self.model.apply_setup_settings(payload)

    def apply_pipeline_context(self, payload: dict) -> None:
        """Forward the shared pipeline context to the view-model (no-op here)."""
        self.model.apply_pipeline_context(payload)

    def _on_model_event(self, event: str) -> None:
        for fn in (self.auto_form.sync_fields, self.auto_form.refresh_plots):
            try:
                fn()
            except Exception:
                logger.debug("calibration tool refresh failed", exc_info=True)
        self._ensure_histograms_async()

    def _ensure_histograms_async(self) -> None:
        """Bin the histograms in the background so the UI never blocks."""
        if self._binning or not self.model.needs_histograms():
            return
        self._binning = True
        self.model.status_text = "Binning decay/IRF histograms…"
        signals = _HistSignals()
        signals.done.connect(self._on_hist_done)
        self._hist_signals = signals  # keep a ref
        QtCore.QThreadPool.globalInstance().start(_HistTask(self.model, signals))

    def _on_hist_done(self) -> None:
        self._binning = False
        self.model.status_text = ""
        self.model.notify("run")  # redraw the decay with the freshly binned histograms

    # ── IRF file drops ──
    def dragEnterEvent(self, event) -> None:  # noqa: N802 (Qt signature)
        """Accept drags carrying a file URL (an IRF TTTR file)."""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:  # noqa: N802 (Qt signature)
        """Append dropped files to the selected detector's IRF list + redraw."""
        paths = [u.toLocalFile() for u in event.mimeData().urls() if u.toLocalFile()]
        if not paths:
            return
        current = list(self.model.sel_irf_files)
        for path in paths:
            if path not in current:
                current.append(path)
        self.model.sel_irf_files = current  # setter redraws
        self.model.status_text = f"IRF ({len(current)} file(s)) for {self.model.display_detector}"
        self.model.notify("changed")


__all__ = ["ImgCalibrationTool"]
