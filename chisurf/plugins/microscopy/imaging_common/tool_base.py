"""Qt host widget for per-pixel imaging map tools (toolbar + file drops).

:class:`ImagingMapTool` wraps a single :class:`~chisurf.gui.autoform.AutoForm`
bound to an :class:`~.base.ImagingMapViewModel`, puts the Run / HDF5 / ndxplorer
actions in a **toolbar**, and accepts **file drops** (drop a TTTR file to load +
run). Concrete tools subclass it with their view-model.
"""

from __future__ import annotations

import logging

from qtpy import QtCore, QtWidgets

from chisurf.gui.autoform import AutoForm

logger = logging.getLogger(__name__)


class _ComputeSignals(QtCore.QObject):
    """Cross-thread signals for a background compute task."""

    progress = QtCore.Signal(float, str)
    finished = QtCore.Signal(bool)


class _ComputeTask(QtCore.QRunnable):
    """Run a view-model's Qt-free ``compute`` off the UI thread.

    The heavy read + CLSM fills hold the GIL, so ``compute`` routes them to a
    worker process (see ``compute_windows``); this QThread only waits, so the UI
    stays responsive.
    """

    def __init__(self, model):
        super().__init__()
        self._model = model
        self.signals = _ComputeSignals()
        self.setAutoDelete(True)

    def run(self) -> None:  # noqa: N802 (Qt override)
        ok = False
        try:
            ok = bool(self._model.compute(progress=lambda f, t: self.signals.progress.emit(float(f), str(t))))
        except Exception:
            logger.debug("background compute failed", exc_info=True)
            ok = False
        self.signals.finished.emit(ok)


class ImagingMapTool(QtWidgets.QWidget):
    """Host widget: action toolbar + file drops + ``AutoForm(view_model)``."""

    def __init__(self, view_model, title: str = "Imaging", parent=None, embedded: bool = False):
        super().__init__(parent)
        self._embedded = bool(embedded)
        self.model = view_model
        self.setWindowTitle(title)
        self.setMinimumSize(700, 520)
        self.setAcceptDrops(True)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)

        toolbar = QtWidgets.QToolBar()
        toolbar.setToolButtonStyle(QtCore.Qt.ToolButtonTextOnly)
        a_run = toolbar.addAction("▶ Run")
        a_run.setToolTip("Compute the per-pixel maps from the loaded file.")
        a_run.triggered.connect(self.run_with_progress)
        a_h5 = toolbar.addAction(getattr(self.model, "HDF5_ACTION_LABEL", "➕ Add to HDF5"))
        a_h5.setToolTip("Add these fields to a standard imaging HDF5 (or create it).")
        a_h5.triggered.connect(self.model.add_to_hdf5)
        a_ndx = toolbar.addAction("🧭 ndxplorer")
        a_ndx.setToolTip("Explore the per-pixel table in ndxplorer (shared, live dataset).")
        a_ndx.triggered.connect(self._open_ndxplorer)
        self._ndx_window = None
        toolbar.addSeparator()
        a_next = toolbar.addAction("Next ▶")
        a_next.setToolTip("Go to the next analysis step (steps are freely navigable on the left).")
        a_next.triggered.connect(self._on_next)
        self.toolbar = toolbar
        layout.addWidget(toolbar)

        self.auto_form = AutoForm(self.model)
        layout.addWidget(self.auto_form)
        self.model.add_observer(self._on_model_event)

    def run_with_progress(self) -> None:
        """Compute in the background (UI stays responsive), reporting to the status bar.

        The GIL-holding read + fills run in a worker process, so this never blocks
        the UI — there is no modal dialog; progress goes to the status bar. Skips
        work entirely when nothing changed since the last compute.
        """
        model = self.model
        if not getattr(model, "filename", ""):
            model.run()
            return
        if hasattr(model, "needs_recompute") and not model.needs_recompute():
            model.run()  # instant re-render, no compute
            return
        if getattr(self, "_computing", False):
            return  # a background compute is already running for this tool
        self._computing = True
        task = _ComputeTask(model)
        self._compute_task = task  # keep a ref so its signals survive
        task.signals.progress.connect(self._on_compute_progress)
        task.signals.finished.connect(self._on_compute_finished)
        self._set_status(f"Computing {self.windowTitle()} …")
        QtCore.QThreadPool.globalInstance().start(task)

    def _on_compute_progress(self, fraction: float, text: str) -> None:
        self._set_status(f"{self.windowTitle()}: {text} ({int(fraction * 100)}%)")

    def _on_compute_finished(self, ok: bool) -> None:
        self._computing = False
        self._compute_task = None
        self._set_status("")
        self.model.notify("run")  # refresh on the UI thread

    def _set_status(self, message: str) -> None:
        """Show background activity in the main-window status bar (best-effort)."""
        try:
            window = self.window()
            status = window.statusBar() if hasattr(window, "statusBar") else None
            if status is not None:
                status.showMessage(message)
        except Exception:
            logger.debug("status bar update failed", exc_info=True)

    def apply_setup_settings(self, payload: dict) -> None:
        """Forward the shared imaging detector setup to the view-model."""
        self.model.apply_setup_settings(payload)

    def apply_pipeline_context(self, payload: dict) -> None:
        """Forward the shared pipeline context (source + HDF5) to the view-model."""
        self.model.apply_pipeline_context(payload)

    def _on_next(self) -> None:
        """Remember the current source/HDF5 and advance to the next step."""
        coordinator = getattr(self, "_coordinator", None)
        role = getattr(self, "_pipeline_role", None)
        if coordinator is None or role is None:
            return
        coordinator.set_pipeline(
            source=self.model.filename or None,
            hdf5=getattr(self.model, "pipeline_hdf5", "") or None,
        )
        coordinator.advance_from(role)

    def _on_model_event(self, event: str) -> None:
        for fn in (self.auto_form.sync_fields, self.auto_form.refresh_plots):
            try:
                fn()
            except Exception:
                logger.debug("imaging tool refresh failed", exc_info=True)
        # Keep an open ndxplorer view in sync with the shared dataset (dynamic).
        if event == "run":
            self._sync_ndxplorer()

    def _open_ndxplorer(self) -> None:
        """Open (or focus) ndxplorer on this tool's shared, live per-pixel dataset."""
        df = self.model.to_dataframe()
        if df is None or df.empty:
            self.model.results_text = "Nothing to explore — press Run first."
            self.model.notify("run")
            return
        try:
            try:
                from ndxplorer import NDXplorer
            except Exception:
                from ndxplorer.core.plot_main import NDXplorer

            from .base import build_ndx_data_source

            ds = build_ndx_data_source(df)
            win = self._ndx_window
            if win is None:
                try:
                    from chisurf.plugins.ndxplorer.rpc_bridge import make_ndxplorer

                    win = self._ndx_window = make_ndxplorer()
                except Exception:
                    win = self._ndx_window = NDXplorer()
                win.setWindowTitle("NDXplorer — imaging")
            win.show()
            win.raise_()
            win.activateWindow()
            # ndxplorer builds its plot widgets in a deferred (QTimer) init after
            # the window is shown; let that run before loading data, else the
            # data lands before the UI exists and the window shows up empty.
            QtWidgets.QApplication.processEvents()
            self._load_ndx_data_source(win, ds)
        except Exception as exc:
            logger.warning("could not open ndxplorer: %s", exc, exc_info=True)
            self.model.results_text = f"ndxplorer failed: {exc}"
            self.model.notify("changed")

    @staticmethod
    def _load_ndx_data_source(win, ds) -> None:
        """Assign a DataSource and put ndxplorer into image mode, then replot.

        Mirrors ndxplorer's own file-load path: assign data, run image-axis
        detection (``X pixel``/``Y pixel`` → pixel axes + frame selection), then
        recompute + replot. Without the detection call the per-pixel table is
        treated as generic burst data instead of an image.
        """
        win.data_source = ds
        applied = False
        detect = getattr(win, "check_and_set_image_axes", None)
        if callable(detect):
            try:
                applied = bool(detect())
            except Exception:
                logger.debug("ndxplorer image-axis detection failed", exc_info=True)
        if not applied:
            refresh = getattr(win, "refresh_axis_comboboxes_preserving_selection", None)
            if callable(refresh):
                try:
                    refresh()
                except Exception:
                    logger.debug("ndxplorer axis refresh failed", exc_info=True)
        upd = getattr(win, "update_ui_data", None)
        if callable(upd):
            try:
                upd()
            except Exception:
                logger.debug("ndxplorer update_ui_data failed", exc_info=True)

    def _sync_ndxplorer(self) -> None:
        win = self._ndx_window
        if win is None or not win.isVisible():
            return
        df = self.model.to_dataframe()
        if df is None or df.empty:
            return
        try:
            from .base import build_ndx_data_source

            self._load_ndx_data_source(win, build_ndx_data_source(df))
        except Exception:
            logger.debug("ndxplorer live-sync failed", exc_info=True)

    # ── file drops ──
    def dragEnterEvent(self, event) -> None:  # noqa: N802 (Qt signature)
        """Accept drags that carry file URLs."""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:  # noqa: N802 (Qt signature)
        """Load the first dropped local file into the view-model."""
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if path:
                self.model.load_file(path)
                break
