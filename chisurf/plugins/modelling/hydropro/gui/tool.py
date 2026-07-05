"""AutoForm GUI for HYDROPRO / HYDRO++.

The parameter form is described declaratively in ``hydropro.view.json`` and
rendered by :class:`chisurf.gui.autoform.AutoForm`; the calculation itself runs
off the UI thread via :func:`...core.run_hydro`.
"""

from __future__ import annotations

import csv
import pathlib
from pathlib import Path
from typing import List, Optional, Tuple

from qtpy import QtCore, QtWidgets
from qtpy.QtCore import QSettings, Qt, QUrl
from qtpy.QtGui import QDesktopServices

from chisurf.core.dataspec import load_view_spec

from ..core import HydroProSettings, HydroResult, run_hydro
from .dialogs import DownloadInfoDialog, OutputDialog

try:
    from chisurf.gui.misc_helpers import persist_plugin_state
except Exception:  # pragma: no cover - fallback when helper unavailable
    def persist_plugin_state(name):  # type: ignore
        return lambda c: c

_GUI_DIR = pathlib.Path(__file__).parent
_DOWNLOAD_URL = "https://leonardo.inf.um.es/macromol/programs/hydro%2B%2B/hydro%2B%2B.htm"


class _HydroModel:
    """Backing model for the HydroPro AutoForm.

    Attributes mirror the fields in ``hydropro.view.json``; AutoForm reads and
    writes them directly. ``indmode`` is held as a string for the combo box and
    converted back to ``int`` in :meth:`to_settings`.
    """

    def __init__(self) -> None:
        self.exe_path = ""
        self.struct_files = ""
        # primary model
        self.indmode = "1"
        self.aer = 2.9
        self.nsig = 6
        self.sigmin = 1.0
        self.sigmax = 2.0
        # solvent & macromolecule
        self.t = 20.0
        self.eta = 0.01
        self.rm = 100000.0
        self.vbar = 0.74
        self.rho = 1.0
        # optional calculations
        self.nq = -1
        self.qmax = 0.0
        self.ns = -1
        self.rmax = 0.0
        self.ntrials = 0
        self.idif = True
        # results
        self.status = ""

    def view_spec(self):
        """Return the AutoForm view spec from ``hydropro.view.json``."""
        return load_view_spec(_GUI_DIR / "hydropro.view.json")

    # -- conversions -------------------------------------------------------
    def struct_list(self) -> List[Path]:
        return [Path(p.strip()) for p in self.struct_files.split(",") if p.strip()]

    def to_settings(self) -> HydroProSettings:
        return HydroProSettings(
            indmode=int(float(self.indmode)),
            aer=float(self.aer), nsig=int(self.nsig),
            sigmin=float(self.sigmin), sigmax=float(self.sigmax),
            t=float(self.t), eta=float(self.eta), rm=float(self.rm),
            vbar=float(self.vbar), rho=float(self.rho),
            nq=int(self.nq), qmax=float(self.qmax),
            ns=int(self.ns), rmax=float(self.rmax),
            ntrials=int(self.ntrials), idif=1 if self.idif else 0,
        )

    def load_settings(self, s: HydroProSettings) -> None:
        self.indmode = str(s.indmode)
        self.aer, self.nsig = s.aer, s.nsig
        self.sigmin, self.sigmax = s.sigmin, s.sigmax
        self.t, self.eta, self.rm = s.t, s.eta, s.rm
        self.vbar, self.rho = s.vbar, s.rho
        self.nq, self.qmax = s.nq, s.qmax
        self.ns, self.rmax = s.ns, s.rmax
        self.ntrials = s.ntrials
        self.idif = bool(s.idif)


class _RunWorker(QtCore.QObject):
    """Runs :func:`core.run_hydro` off the UI thread, streaming log lines."""

    log = QtCore.Signal(str)
    progress = QtCore.Signal(int, int)
    finished = QtCore.Signal(list)
    failed = QtCore.Signal(str)

    def __init__(self, struct_files, settings, exe_path) -> None:
        super().__init__()
        self._struct_files = struct_files
        self._settings = settings
        self._exe_path = exe_path
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:
        try:
            results = run_hydro(
                self._struct_files, self._settings, self._exe_path,
                on_log=self.log.emit,
                on_progress=self.progress.emit,
                should_cancel=lambda: self._cancel,
            )
            self.finished.emit(results)
        except Exception as exc:  # pragma: no cover - surfaced to the UI
            self.failed.emit(str(exc))


@persist_plugin_state("hydropro")
class HydroProTool(QtWidgets.QMainWindow):
    """HYDROPRO / HYDRO++ diffusion-coefficient calculator (AutoForm UI)."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("HYDRO++ / HYDROPRO Diffusion Coefficient Calculator")
        self.resize(640, 720)

        self._model = _HydroModel()
        self._qsettings = QSettings("ChiSurf", "HydroPRO")
        self._results: List[HydroResult] = []
        self._thread: Optional[QtCore.QThread] = None
        self._worker: Optional[_RunWorker] = None
        self._out_dlg: Optional[OutputDialog] = None

        self._load_persisted()
        self._build_ui()

    # -- UI ----------------------------------------------------------------
    def _build_ui(self) -> None:
        from chisurf.gui.autoform import AutoForm

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        self._form = AutoForm(self._model, parent=self)
        layout.addWidget(self._form)

        btn_row = QtWidgets.QHBoxLayout()
        self._select_btn = QtWidgets.QPushButton("Select files…")
        self._select_btn.clicked.connect(self._select_files)
        self._exe_btn = QtWidgets.QPushButton("Executable…")
        self._exe_btn.clicked.connect(self._select_exe)
        self._run_btn = QtWidgets.QPushButton("Run")
        self._run_btn.clicked.connect(self._on_run)
        self._save_btn = QtWidgets.QPushButton("Save CSV")
        self._save_btn.clicked.connect(self._save_csv)
        self._save_btn.setEnabled(False)
        self._clear_btn = QtWidgets.QPushButton("Clear")
        self._clear_btn.clicked.connect(self._clear)
        self._dl_btn = QtWidgets.QPushButton("Download page")
        self._dl_btn.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(_DOWNLOAD_URL)))
        for b in (self._select_btn, self._exe_btn, self._run_btn,
                  self._save_btn, self._clear_btn, self._dl_btn):
            btn_row.addWidget(b)
        layout.addLayout(btn_row)

        self.table = QtWidgets.QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["File", "Diffusion coefficient (cm²/s)"])
        self.table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table, 1)

    # -- persistence -------------------------------------------------------
    def _load_persisted(self) -> None:
        try:
            self._model.exe_path = self._qsettings.value("hydro_exe", "", type=str) or ""
            stored = {}
            for name in HydroProSettings().to_dict():
                key = f"hp.{name}"
                if self._qsettings.contains(key):
                    stored[name] = self._qsettings.value(key)
            if stored:
                self._model.load_settings(HydroProSettings.from_dict(stored))
        except Exception:
            pass

    def _save_persisted(self) -> None:
        try:
            self._qsettings.setValue("hydro_exe", self._model.exe_path)
            for name, value in self._model.to_settings().to_dict().items():
                self._qsettings.setValue(f"hp.{name}", value)
        except Exception:
            pass

    # -- file pickers ------------------------------------------------------
    def _select_files(self) -> None:
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Select structural files", str(Path.home()),
            "Structural files (*.pdb *.txt *.bea *.*)",
        )
        if paths:
            self._model.struct_files = ", ".join(paths)
            self._form.rebuild()
            self._refresh_table_files()

    def _select_exe(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select HYDRO executable", str(Path.home()),
            "Executables (*.exe);;All files (*.*)",
        )
        if path:
            self._model.exe_path = path
            self._form.rebuild()

    # -- run ---------------------------------------------------------------
    def _ensure_exe(self) -> Optional[Path]:
        exe = Path(self._model.exe_path) if self._model.exe_path else None
        if exe and exe.exists():
            return exe
        dlg = DownloadInfoDialog(_DOWNLOAD_URL, self)
        if dlg.exec_() == QtWidgets.QDialog.Accepted and dlg.selected_path:
            if dlg.selected_path.exists():
                self._model.exe_path = str(dlg.selected_path)
                self._form.rebuild()
                return dlg.selected_path
        return None

    def _on_run(self) -> None:
        struct_files = self._model.struct_list()
        if not struct_files:
            QtWidgets.QMessageBox.warning(self, "No files", "Please select one or more files first.")
            return
        try:
            settings = self._model.to_settings()
            settings.validate()
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "Invalid settings", str(exc))
            return
        exe = self._ensure_exe()
        if not exe:
            QtWidgets.QMessageBox.information(
                self, "Executable required", "Configure the HYDRO executable before running.")
            return
        self._save_persisted()

        total = len(struct_files)
        self._out_dlg = OutputDialog(self, total_steps=total)
        self._out_dlg.set_status(f"Starting HYDRO for {total} file(s)…")
        self._out_dlg.append(f"Executable: {exe}")
        self._out_dlg.show()

        self._thread = QtCore.QThread(self)
        self._worker = _RunWorker(struct_files, settings, exe)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.log.connect(self._out_dlg.append)
        self._worker.progress.connect(lambda i, n: self._out_dlg.set_progress(i))
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._out_dlg.cancel_btn.clicked.connect(self._worker.cancel)
        self._run_btn.setEnabled(False)
        self._thread.start()

    def _teardown_thread(self) -> None:
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait()
            self._thread = None
        self._worker = None
        self._run_btn.setEnabled(True)

    def _on_finished(self, results: list) -> None:
        self._results = list(results)
        self._model.status = f"Finished: {len(self._results)} file(s)."
        if self._out_dlg is not None:
            self._out_dlg.set_status("Finished.")
            self._out_dlg.cancel_btn.setEnabled(False)
        self._save_btn.setEnabled(bool(self._results))
        self._populate_results()
        self._form.rebuild()
        self._teardown_thread()

    def _on_failed(self, message: str) -> None:
        self._model.status = f"Error: {message}"
        if self._out_dlg is not None:
            self._out_dlg.append(f"\nERROR: {message}")
            self._out_dlg.set_status("Failed.")
        self._form.rebuild()
        self._teardown_thread()

    # -- results table -----------------------------------------------------
    def _refresh_table_files(self) -> None:
        self.table.setRowCount(0)
        for path in self._model.struct_list():
            row = self.table.rowCount()
            self.table.insertRow(row)
            self.table.setItem(row, 0, QtWidgets.QTableWidgetItem(str(path)))
            self.table.setItem(row, 1, QtWidgets.QTableWidgetItem(""))

    def _populate_results(self) -> None:
        self._refresh_table_files()
        by_file = {r.struct_file: r.diffusion_coefficient for r in self._results}
        for row in range(self.table.rowCount()):
            name = self.table.item(row, 0).text()
            if name in by_file:
                value = by_file[name]
                item = QtWidgets.QTableWidgetItem(
                    f"{value:.3e}" if value is not None else "N/A")
                item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.table.setItem(row, 1, item)

    def _save_csv(self) -> None:
        if not self._results:
            QtWidgets.QMessageBox.warning(self, "No results", "There are no results to save.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save CSV", str(Path.home() / "hydro_results.csv"), "CSV files (*.csv)")
        if not path:
            return
        try:
            with open(path, "w", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                writer.writerow(["File", "DiffusionCoefficient(cm^2/s)"])
                for r in self._results:
                    writer.writerow(
                        [r.struct_file,
                         f"{r.diffusion_coefficient:.3e}" if r.diffusion_coefficient is not None else ""])
            QtWidgets.QMessageBox.information(self, "Saved", f"Results saved to {path}")
        except OSError as exc:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to save CSV: {exc}")

    def _clear(self) -> None:
        self._model.struct_files = ""
        self._model.status = ""
        self._results = []
        self._save_btn.setEnabled(False)
        self.table.setRowCount(0)
        self._form.rebuild()


# Backwards-compatibility alias (former class name / entry point).
HydroGui = HydroProTool


__all__ = ["HydroProTool", "HydroGui"]
