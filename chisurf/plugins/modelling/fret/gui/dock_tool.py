"""AutoForm GUI for IMP + IMP.bff FRET docking.

Declarative, layout-driven widget (PRD-40 AutoForm) backed by
:mod:`...api.operations`. The form fields are described in
``fret_dock.view.json``; the operation runs in a worker thread so the UI does
not freeze during sampling.
"""

from __future__ import annotations

import glob
import os
import pathlib
import threading
import time
import traceback

import pyqtgraph as pg
from qtpy import QtCore, QtGui, QtWidgets

from chisurf.core.dataspec import load_view_spec


class _NumericItem(QtWidgets.QTableWidgetItem):
    """Table item that sorts by a numeric value while showing formatted text."""

    def __init__(self, text: str, value: float) -> None:
        super().__init__(text)
        self._value = value

    def __lt__(self, other) -> bool:
        if isinstance(other, _NumericItem):
            return self._value < other._value
        return super().__lt__(other)


def _fmt_eta(seconds: float) -> str:
    """Format a remaining-time estimate like the chisurf fitting dialog."""
    s = max(0, int(seconds))
    if s >= 3600:
        return f"ETA {s // 3600}h {(s % 3600) // 60}m"
    if s >= 60:
        return f"ETA {s // 60}m {s % 60}s"
    return f"ETA {s}s"

try:
    from chisurf.gui.misc_helpers import persist_plugin_state
except Exception:  # pragma: no cover - fallback when helper unavailable
    def persist_plugin_state(name):  # type: ignore
        return lambda c: c

from ..core import stat as _stat

_GUI_DIR = pathlib.Path(__file__).parent


class _DockingModel:
    """Backing model for the FRET docking AutoForm.

    Attributes mirror the fields in ``fret_dock.view.json``; AutoForm reads and
    writes them directly. Output fields are filled in after a run.
    """

    def __init__(self) -> None:
        # inputs
        self.pdb_paths = ""
        self.fps_json = ""
        self.output_dir = ""
        self.operation = "dock"
        self.method = "minimize"
        self.score_set = ""
        # sampling
        self.n_frames = 500
        self.mc_steps = 10
        self.n_best = 20
        self.n_repeats = 1
        self.refine_av_cycles = 0
        self.fixed_body = 0
        self.sigma_da = 6.0
        self.ev_weight = 1.0
        self.simulated_annealing = False
        self.save_distributions = False
        self.av_backend = "auto"
        # results
        self.status = ""
        self.score = 0.0
        self.n_distances = 0
        # structure(s) shown in the 3D preview (str or list of PDB paths)
        self.preview_models = []

    def view_spec(self):
        """Return the AutoForm view spec from ``fret_dock.view.json``."""
        return load_view_spec(_GUI_DIR / "fret_dock.view.json")

    # -- helpers -----------------------------------------------------------
    def pdb_list(self):
        return [p.strip() for p in self.pdb_paths.split(",") if p.strip()]

    def ensure_output_dir(self):
        """Fill ``output_dir`` with a sensible default when the user left it blank.

        Defaults to a ``dock_out`` folder next to the fps.json (or the first
        PDB) so the example runs without forcing the user to pick a directory.
        """
        if self.output_dir and self.output_dir.strip():
            return self.output_dir
        anchor = self.fps_json or (self.pdb_list()[0] if self.pdb_list() else "")
        base = pathlib.Path(anchor).parent if anchor else pathlib.Path.cwd()
        self.output_dir = str(base / "dock_out")
        return self.output_dir

    def build_params(self):
        """Build the api.operations request dict for the selected operation.

        ``dock`` with ``n_repeats > 1`` becomes an ``errors`` request (repeated
        independent docking trials), mirroring the FPS repeated-run workflow.
        """
        op = self.operation
        if op == "dock":
            req = {
                "pdb_paths": self.pdb_list(), "fps_json": self.fps_json,
                "output_dir": self.output_dir, "n_frames": self.n_frames,
                "mc_steps": self.mc_steps, "n_best": self.n_best,
                "fixed_body": self.fixed_body, "sigma_da": self.sigma_da,
                "simulated_annealing": self.simulated_annealing,
                "score_set": self.score_set, "method": self.method,
                "refine_av_cycles": self.refine_av_cycles,
                "ev_weight": self.ev_weight,
                "save_distributions": self.save_distributions,
                "av_backend": self.av_backend,
            }
            if int(self.n_repeats) > 1:
                req["n_trials"] = int(self.n_repeats)
                return "errors", req
            return op, req
        if op == "refine":
            return op, {
                "pdb_paths": self.pdb_list(), "fps_json": self.fps_json,
                "output_dir": self.output_dir, "score_set": self.score_set,
            }
        if op == "screen":
            return op, {
                "pdb_inputs": self.pdb_list(), "fps_json": self.fps_json,
                "score_set": self.score_set,
                "output_csv": str(pathlib.Path(self.output_dir) / "screen.csv")
                if self.output_dir else None,
            }
        # score
        return op, {
            "pdb_paths": self.pdb_list(), "fps_json": self.fps_json,
            "score_set": self.score_set, "mean_position_restraint": True,
            "sigma_da": self.sigma_da,
        }


def _run_op_child(op, params, q):
    """Run one operation in a worker process and put ``(status, payload)`` on q."""
    try:
        from ..api import operations as ops
        fn = {"dock": ops.dock, "refine": ops.refine,
              "screen": ops.screen, "score": ops.score}[op]
        q.put(("ok", fn(params)))
    except Exception:
        q.put(("err", traceback.format_exc()))


class _Worker(QtCore.QObject):
    """Runs one api.operations call off the UI thread."""

    finished = QtCore.Signal(dict)
    failed = QtCore.Signal(str)
    stopped = QtCore.Signal()

    def __init__(self, op: str, params: dict, stop_check=None) -> None:
        super().__init__()
        self._op = op
        self._params = params
        self._stop_check = stop_check

    def run(self) -> None:
        try:
            from ..api import operations as ops
            if self._op == "dock":
                # Single dock runs in a terminable child process so Stop works
                # for *any* method (PMI's Monte-Carlo execute_macro can't be
                # interrupted cooperatively).
                self._run_in_subprocess()
            elif self._op == "errors":
                # estimate_errors itself terminates its trial pool on stop.
                res = ops.estimate_errors(self._params, stop_check=self._stop_check)
                self.finished.emit(res)
            else:
                res = {"refine": ops.refine, "screen": ops.screen,
                       "score": ops.score}[self._op](self._params)
                self.finished.emit(res)
        except Exception:  # pragma: no cover - surfaced to the UI
            self.failed.emit(traceback.format_exc())

    def _run_in_subprocess(self) -> None:
        import multiprocessing as mp
        import queue as _queue

        os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")
        try:
            ctx = mp.get_context("fork")
        except ValueError:  # no fork (e.g. Windows) -> run inline, no cancel
            from ..api import operations as ops
            self.finished.emit(ops.dock(self._params, stop_check=self._stop_check))
            return
        q = ctx.Queue()
        proc = ctx.Process(target=_run_op_child, args=(self._op, self._params, q),
                           daemon=True)
        proc.start()
        while True:
            if self._stop_check is not None and self._stop_check():
                proc.terminate()
                proc.join(3)
                self.stopped.emit()
                return
            try:
                status, payload = q.get(timeout=0.2)
                break
            except _queue.Empty:
                if not proc.is_alive():
                    self.failed.emit("Docking process exited unexpectedly.")
                    return
        proc.join(3)
        if status == "ok":
            self.finished.emit(payload)
        else:
            self.failed.emit(payload)


@persist_plugin_state("fret_docking")
class FretDockingTool(QtWidgets.QWidget):
    """FRET-restrained docking tool (IMP + IMP.bff), rendered with AutoForm."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.setWindowTitle("FRET Docking (IMP.bff)")
        self._model = _DockingModel()
        self._thread = None
        self._worker = None
        self._stop_event = threading.Event()
        self._dialog = None
        self._t0 = 0.0
        self._total = 1
        self._pending_pdb = None    # models queued by the preview debounce timer
        self._build_ui()

    def _build_ui(self) -> None:
        from chisurf.gui.autoform import AutoForm

        layout = QtWidgets.QVBoxLayout(self)

        # Toolbar with emoji actions (replaces the old button rows).
        tb = QtWidgets.QToolBar(self)
        self._act_load = tb.addAction("📂 Project", self._load_project)
        self._act_load.setToolTip("Load a docking project (.json): PDBs, fps.json and parameters.")
        self._act_save = tb.addAction("💾 Save", self._save_project)
        self._act_save.setToolTip("Save the current inputs and parameters as a docking project.")
        tb.addSeparator()
        tb.addAction("➕ Add PDB", self._pick_pdbs).setToolTip(
            "Add one or more PDB files (one rigid body per file).")
        tb.addAction("➖ Remove PDB", self._remove_pdb).setToolTip(
            "Remove the selected PDB(s) from the list.")
        tb.addAction("🏷 fps.json", self._pick_fps).setToolTip(
            "Choose the labelling/distance fps.json (or FPS LPs .txt) file.")
        tb.addAction("📁 Output", self._pick_out).setToolTip("Choose the output directory.")
        tb.addSeparator()
        self._act_run = tb.addAction("▶️ Run", self._on_run)
        self._act_run.setToolTip("Run docking; results are appended to the table.")
        self._act_clear = tb.addAction("🧹 Clear", self._clear_results)
        self._act_clear.setToolTip("Clear the results table and score plot.")
        layout.addWidget(tb)

        # PDB rigid bodies as an editable list (one file per rigid body).
        pdb_box = QtWidgets.QGroupBox("🧬 PDB rigid bodies (one per body)")
        pv = QtWidgets.QVBoxLayout(pdb_box)
        pv.setContentsMargins(6, 2, 6, 4)
        self._pdb_list = QtWidgets.QListWidget()
        self._pdb_list.setSelectionMode(QtWidgets.QListWidget.ExtendedSelection)
        self._pdb_list.setMaximumHeight(90)
        self._pdb_list.setToolTip("PDB files, one per rigid body; order = body_id 0,1,2…")
        pv.addWidget(self._pdb_list)
        layout.addWidget(pdb_box)

        self._form = AutoForm(self._model, parent=self)
        layout.addWidget(self._form)

        # Results live in dockable tabs: table, score plot, structure preview.
        from chisurf.gui.widgets.dock_area import DockArea
        self._dock_area = DockArea(self)
        layout.addWidget(self._dock_area, 1)

        self._table = QtWidgets.QTableWidget(0, 5)
        self._table.setHorizontalHeaderLabels(
            ["Trial", "Type", "Score", "Distances", "Best PDB"])
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.setEditTriggers(QtWidgets.QTableWidget.NoEditTriggers)
        self._table.setSelectionBehavior(QtWidgets.QTableWidget.SelectRows)
        self._table.setAlternatingRowColors(True)
        self._table.verticalHeader().setVisible(False)
        self._table.itemSelectionChanged.connect(self._on_row_selected)
        self._table.setSortingEnabled(True)  # click a header (e.g. Score) to sort
        self._table.setToolTip(
            "Per-trial results — click a column header to sort (e.g. by Score); "
            "select a row to show that structure.")

        self._plot = pg.PlotWidget()
        self._plot.setLabel("bottom", "Step")
        self._plot.setLabel("left", "Total score")
        self._plot.showGrid(x=True, y=True)
        self._plot.setToolTip(
            "Total restraint score vs step (CG iteration or MC frame; one curve per trial).")

        # Structure preview: reusable AutoForm ChiMol section (viewer + frame
        # slider) bound to the model's ``preview_models`` list.
        from chisurf.gui.autoform.sections.chimol_section import ChiMolSectionWidget
        self._structure_view = ChiMolSectionWidget(self._model, "preview_models")

        self._dock_area.addTab(self._table, "📊 Results")
        self._dock_area.addTab(self._plot, "📈 Score")
        self._dock_area.addTab(self._structure_view, "🧬 Structure")

        # Polls the trace file(s) to drive the progress dialog and score plot.
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(400)
        self._timer.timeout.connect(self._poll_progress)
        self._stat_path = None
        self._run_op = None
        self._curves = {}

        # Debounce structure-preview loads: rapid row changes (e.g. arrow keys)
        # coalesce into a single load once the selection settles.
        self._preview_timer = QtCore.QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(180)
        self._preview_timer.timeout.connect(self._load_pending_structure)

    # -- structure preview (AutoForm ChiMol section) -----------------------
    def _show_structure(self, paths) -> None:
        """Queue structure(s) for the 3D preview (debounced via the model attr).

        ``paths`` may be a single path or a list; a list of same-shape docked
        models becomes a frame-stepped trajectory in the ChiMol section.
        """
        models = [paths] if isinstance(paths, (str, pathlib.Path)) else list(paths or [])
        if models == list(self._model.preview_models):
            return
        self._pending_pdb = models
        self._preview_timer.start()

    def _load_pending_structure(self) -> None:
        self._model.preview_models = list(self._pending_pdb or [])
        try:
            self._structure_view.refresh()
        except Exception:
            pass

    def _on_row_selected(self) -> None:
        items = self._table.selectedItems()
        if not items:
            return
        pdb = self._table.item(items[0].row(), 4).data(QtCore.Qt.UserRole)
        if pdb:
            self._show_structure(pdb)

    # -- file pickers ------------------------------------------------------
    def _pick_pdbs(self) -> None:
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Select PDB file(s)", "", "PDB (*.pdb);;All files (*)")
        if files:
            for f in files:
                self._add_pdb_item(f)
            self._sync_pdb_model()
            self._show_structure(files[0])  # preview the chosen structure

    def _add_pdb_item(self, path: str) -> None:
        item = QtWidgets.QListWidgetItem(pathlib.Path(path).name)
        item.setToolTip(path)
        item.setData(QtCore.Qt.UserRole, path)
        self._pdb_list.addItem(item)

    def _pdb_paths_from_list(self):
        return [self._pdb_list.item(i).data(QtCore.Qt.UserRole)
                for i in range(self._pdb_list.count())]

    def _sync_pdb_model(self) -> None:
        """Mirror the PDB list into the model (comma-joined for the engine)."""
        self._model.pdb_paths = ", ".join(self._pdb_paths_from_list())

    def _set_pdb_list(self, paths) -> None:
        self._pdb_list.clear()
        for p in paths:
            if p:
                self._add_pdb_item(p)
        self._sync_pdb_model()

    def _remove_pdb(self) -> None:
        for item in self._pdb_list.selectedItems():
            self._pdb_list.takeItem(self._pdb_list.row(item))
        self._sync_pdb_model()

    def _pick_fps(self) -> None:
        f, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select fps.json or FPS LPs .txt", "",
            "FPS labelling (*.json *.txt);;All files (*)")
        if f:
            self._model.fps_json = f
            self._form.sync_fields()

    def _pick_out(self) -> None:
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Output directory")
        if d:
            self._model.output_dir = d
            self._form.sync_fields()

    # -- project load/save -------------------------------------------------
    def _model_params(self) -> dict:
        m = self._model
        return {
            "n_frames": m.n_frames, "mc_steps": m.mc_steps, "n_best": m.n_best,
            "fixed_body": m.fixed_body, "sigma_da": m.sigma_da,
            "simulated_annealing": m.simulated_annealing,
        }

    def _load_project(self) -> None:
        f, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load docking project", "", "Docking project (*.json);;All files (*)")
        if not f:
            return
        try:
            from ..api.project import load_docking_project
            proj = load_docking_project(f)
        except Exception:
            QtWidgets.QMessageBox.critical(
                self, "Load failed", traceback.format_exc()[-2000:])
            return
        m = self._model
        self._set_pdb_list(proj.pdb_paths)  # populates the list and m.pdb_paths
        m.fps_json = proj.fps_json
        m.output_dir = proj.output_dir
        m.operation = proj.operation or "dock"
        m.method = proj.method or "minimize"
        m.score_set = proj.score_set
        p = proj.params or {}
        m.n_frames = int(p.get("n_frames", m.n_frames))
        m.mc_steps = int(p.get("mc_steps", m.mc_steps))
        m.n_best = int(p.get("n_best", m.n_best))
        m.fixed_body = int(p.get("fixed_body", m.fixed_body))
        m.sigma_da = float(p.get("sigma_da", m.sigma_da))
        m.simulated_annealing = bool(p.get("simulated_annealing", m.simulated_annealing))
        m.status = f"loaded {pathlib.Path(f).name}"
        self._form.sync_fields()

    def _save_project(self) -> None:
        f, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save docking project", "docking_project.json",
            "Docking project (*.json);;All files (*)")
        if not f:
            return
        try:
            from ..api.project import save_docking_project
            save_docking_project(
                f,
                pdb_paths=self._model.pdb_list(),
                fps_json=self._model.fps_json,
                output_dir=self._model.output_dir,
                operation=self._model.operation,
                method=self._model.method,
                score_set=self._model.score_set,
                params=self._model_params(),
            )
        except Exception:
            QtWidgets.QMessageBox.critical(
                self, "Save failed", traceback.format_exc()[-2000:])
            return
        self._model.status = f"saved {pathlib.Path(f).name}"
        self._form.sync_fields()

    # -- results plot / table ---------------------------------------------
    def _clear_results(self) -> None:
        self._plot.clear()
        self._curves = {}
        self._table.setRowCount(0)

    def _trace_name(self) -> str:
        """Per-iteration score file written by the chosen docking method."""
        return "convergence.csv" if self._model.method == "minimize" else "stat.0.out"

    def _stat_files(self) -> list:
        """Trace files to plot: one for a single dock, one per trial for repeats."""
        out = self._model.output_dir
        if not out:
            return []
        name = self._trace_name()
        if self._run_op == "errors":
            return sorted(glob.glob(str(pathlib.Path(out) / "trial_*" / name)))
        return [str(pathlib.Path(out) / name)]

    def _update_plot(self) -> None:
        for i, sf in enumerate(self._stat_files()):
            frames, scores = _stat.read_score_series(sf)
            if not scores:
                continue
            curve = self._curves.get(sf)
            if curve is None:
                pen = pg.mkPen(pg.intColor(i, hues=max(1, int(self._model.n_repeats))), width=2)
                curve = self._plot.plot(frames, scores, pen=pen)
                self._curves[sf] = curve
            else:
                curve.setData(frames, scores)

    def _fill_table(self, rows: list, best_trial=None, kind="dock") -> None:
        """Append ``[(trial, score, n_dist, pdb)]`` rows to the results table.

        Runs accumulate — Run appends solutions, the Clear button empties the
        table. ``kind`` is the sampling type shown in the Type column; numeric
        columns sort numerically and the overall best score is highlighted.
        """
        self._table.setSortingEnabled(False)
        for (trial, score, n_dist, pdb) in rows:
            r = self._table.rowCount()
            self._table.insertRow(r)
            score_txt = f"{score:.2f}" if score == score else "nan"
            items = [
                _NumericItem(str(trial), float(trial)),
                QtWidgets.QTableWidgetItem(kind),
                _NumericItem(score_txt, score if score == score else float("inf")),
                _NumericItem(str(n_dist), float(n_dist)),
                QtWidgets.QTableWidgetItem(pathlib.Path(pdb).name if pdb else ""),
            ]
            if pdb:
                items[4].setToolTip(pdb)
                items[4].setData(QtCore.Qt.UserRole, str(pdb))
            for c, item in enumerate(items):
                self._table.setItem(r, c, item)
        self._table.setSortingEnabled(True)
        self._table.sortItems(2, QtCore.Qt.AscendingOrder)  # best score first
        self._highlight_best()

    def _highlight_best(self) -> None:
        """Mark the best-scoring (top) row after the best-first sort.

        A muted dark-green fill with white text (readable on light and dark
        themes); other rows keep the view's default colours.
        """
        best_bg = QtGui.QColor(27, 94, 32)   # green 900 — contrasts with white
        best_fg = QtGui.QColor(255, 255, 255)
        clear = QtGui.QBrush()               # empty brush -> theme default
        for r in range(self._table.rowCount()):
            is_best = r == 0
            for c in range(self._table.columnCount()):
                it = self._table.item(r, c)
                if it is None:
                    continue
                it.setBackground(QtGui.QBrush(best_bg) if is_best else clear)
                it.setForeground(QtGui.QBrush(best_fg) if is_best else clear)

    # -- run / modal progress ----------------------------------------------
    def _make_dialog(self, op: str, n_trials: int):
        """Modal progress dialog with ETA + Cancel (reuses the chisurf fitting one)."""
        try:
            from chisurf.gui.widgets.progress import EnhancedProgressDialog
        except Exception:  # pragma: no cover - run without a dialog if unavailable
            return None
        label = (f"Docking {n_trials} trials…" if op == "errors" else "Docking…")
        dlg = EnhancedProgressDialog("FRET Docking", label, 0, 100, parent=self)
        # Cancel -> cooperative stop (signal is robust even if the dialog closes).
        dlg.canceled.connect(self._stop_event.set)
        dlg.show()
        return dlg

    def _start_progress(self, op: str) -> None:
        """Prime the trace files and open the modal dialog (results are appended)."""
        self._run_op = op
        self._stop_event.clear()
        out = self._model.output_dir
        n_frames = max(1, int(self._model.n_frames))
        n_trials = max(1, int(self._model.n_repeats)) if op == "errors" else 1
        self._total = n_frames * n_trials
        if out:  # stale trace files would make progress jump to 100% instantly
            for sf in glob.glob(str(pathlib.Path(out) / "**" / self._trace_name()),
                                recursive=True):
                try:
                    pathlib.Path(sf).unlink()
                except OSError:
                    pass
        self._stat_path = out if op in ("dock", "errors") else None
        self._t0 = time.perf_counter()
        self._dialog = self._make_dialog(op, n_trials)
        self._timer.start()

    def _best_score_so_far(self):
        best = None
        for sf in self._stat_files():
            _frames, scores = _stat.read_score_series(sf)
            if scores:
                m = min(scores)
                best = m if best is None else min(best, m)
        return best

    def _poll_progress(self) -> None:
        # user pressed Cancel -> request a cooperative stop
        if self._dialog is not None and self._dialog.wasCanceled():
            self._stop_event.set()
        self._update_plot()
        if self._dialog is None:
            return
        if not self._stat_path:
            self._dialog.update_text("running…")
            return
        done = sum(_stat.count_frames(sf) for sf in self._stat_files())
        total = max(1, self._total)
        pct = int(100.0 * min(done, total) / total)
        parts = []
        elapsed = time.perf_counter() - self._t0
        if done > 0:
            parts.append(_fmt_eta(elapsed / float(done) * (float(total) - done)))
        best = self._best_score_so_far()
        if best is not None:
            parts.append(f"best score {best:.1f}")
        self._dialog.update_progress(pct, text="  |  ".join(parts) or "starting…")

    def _stop_progress(self) -> None:
        self._timer.stop()
        self._update_plot()
        if self._dialog is not None:
            try:
                self._dialog.finish(auto_close=True, close_delay_ms=500)
            except Exception:
                try:
                    self._dialog.close()
                except Exception:
                    pass
            self._dialog = None
        self._stat_path = None

    def _on_run(self) -> None:
        if self._model.operation in ("dock", "refine", "screen"):
            self._model.ensure_output_dir()
        op, params = self._model.build_params()
        self._model.status = "running…"
        self._form.sync_fields()
        self._act_run.setEnabled(False)
        self._start_progress(op)

        self._thread = QtCore.QThread(self)
        self._worker = _Worker(op, params, stop_check=self._stop_event.is_set)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.stopped.connect(self._on_stopped)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._worker.stopped.connect(self._thread.quit)
        self._thread.start()

    def _on_stopped(self) -> None:
        self._stop_progress()
        self._model.status = "stopped"
        self._form.sync_fields()
        self._act_run.setEnabled(True)

    def _sampling_kind(self) -> str:
        """Label for the Type column based on operation + method."""
        if self._model.operation == "refine":
            return "refine"
        return "dock" if self._model.method == "minimize" else "mc"

    def _on_finished(self, result: dict) -> None:
        stopped = self._stop_event.is_set()
        self._stop_progress()
        data = result.get("data", {})
        self._model.status = result.get("status", "ok")
        kind = self._sampling_kind()
        if "trial_details" in data:  # repeated docking (error estimation)
            details = data["trial_details"]
            best = data.get("best_trial")
            self._fill_table(
                [(d["trial"], d["score"], d["n_distances"], d.get("best_pdb")) for d in details],
                best_trial=best, kind=kind,
            )
            if details:
                self._model.score = float(min(d["score"] for d in details))
                self._model.n_distances = int(details[0]["n_distances"])
            workers = data.get("n_workers", 1)
            par = f", {workers}x parallel" if workers and workers > 1 else ""
            done = len(details)
            head = f"stopped after {done}/{data['n_trials']} trials" if stopped \
                else f"{data['n_trials']} trials{par}"
            spread = (f": mean {data['score_mean']:.1f} ± {data['score_std']:.1f}"
                      if details else "")
            unc = data.get("uncertainty") or {}
            prec = (f"; precision {unc['mobile_rmsf_mean']:.1f} Å"
                    if unc.get("mobile_rmsf_mean") == unc.get("mobile_rmsf_mean")
                    and unc.get("n_models", 0) >= 2 else "")
            self._model.status = head + spread + prec
            # step through all docked solutions with the ChiMol frame slider
            models = [d["best_pdb"] for d in details if d.get("best_pdb")]
            if models:
                self._show_structure(models)
        elif "score" in data:  # single dock / refine / score
            self._model.score = float(data.get("score") or 0.0)
            self._model.n_distances = int(data.get("n_distances") or 0)
            self._fill_table(
                [(0, self._model.score, self._model.n_distances,
                  (data.get("best_pdbs") or [None])[0])],
                best_trial=0, kind=kind,
            )
            best_pdbs = data.get("best_pdbs") or []
            if best_pdbs:
                self._show_structure(best_pdbs)  # n_best models -> frames
            if stopped or data.get("extra", {}).get("stopped"):
                self._model.status = f"stopped (score {self._model.score:.1f})"
        elif "ranked" in data:
            self._model.n_distances = len(data["ranked"])
            self._model.status = f"ranked {len(data['ranked'])} structures"
        self._form.sync_fields()
        self._act_run.setEnabled(True)

    def _on_failed(self, tb: str) -> None:
        self._stop_progress()
        self._model.status = "error (see message)"
        self._form.sync_fields()
        self._act_run.setEnabled(True)
        QtWidgets.QMessageBox.critical(self, "FRET docking failed", tb[-2000:])
