"""H2MM analysis tool: toolbar, tabbed settings, and pyqtgraph result plots."""

from __future__ import annotations

import pathlib
from typing import Any

import numpy as np
import pyqtgraph as pg
from qtpy.QtCore import QCoreApplication, QSettings, QSize, Qt, Signal
from qtpy.QtGui import QDragEnterEvent, QDropEvent
from qtpy.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QSizePolicy,
    QSpinBox,
    QTextEdit,
    QToolBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from chisurf import logging
from chisurf.gui.misc_helpers import get_plugin_settings_path, persist_plugin_state
from chisurf.gui.widgets.dock_area.dock_area import DockArea
from chisurf.gui.widgets.wizard import DetectorWizardPage

from ..api.models import H2mmSettings, StreamSettings
from ..backend.services import run_analysis

_STATE_COLORS = [
    "#4e79a7", "#f28e2b", "#59a14f", "#e15759",
    "#b07aa1", "#76b7b2", "#edc948", "#ff9da7",
]


class _FolderLineEdit(QLineEdit):
    """QLineEdit that accepts a folder drop."""

    folderDropped = Signal(str)

    def __init__(self, placeholder: str = "", parent=None):
        super().__init__(parent)
        self.setPlaceholderText(placeholder)
        self.setReadOnly(True)
        self.setAcceptDrops(True)
        self.setStyleSheet("color: #aaa; padding: 0 4px; background: transparent; border: none;")

    def dragEnterEvent(self, event: QDragEnterEvent):
        """Accept a drag that carries file URLs."""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        """Emit the dropped folder path."""
        urls = event.mimeData().urls()
        if urls and urls[0].toLocalFile():
            self.setText(urls[0].toLocalFile())
            self.folderDropped.emit(urls[0].toLocalFile())


class _ProgressDialog(QDialog):
    """A modal busy indicator shown while a fit runs."""

    def __init__(self, title="Progress", message="Processing...", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setWindowModality(Qt.WindowModal)
        layout = QVBoxLayout(self)
        self.label = QLabel(message)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)  # busy indicator
        layout.addWidget(self.label)
        layout.addWidget(self.progress)

    def set_message(self, msg: str):
        """Update the progress message."""
        self.label.setText(msg)
        QCoreApplication.processEvents()


class HelpDialog(QDialog):
    """About/help dialog with a CLI reference."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("About H2MM")
        self.resize(620, 520)
        layout = QVBoxLayout(self)
        text = QTextEdit(self)
        text.setReadOnly(True)
        cli_text = ""
        try:
            from click.testing import CliRunner

            from ..cli.main import cli

            cli_text = "<pre>\n" + CliRunner().invoke(cli, ["compute", "--help"]).output + "</pre>"
        except Exception as exc:  # pragma: no cover
            cli_text = f"<p>CLI help unavailable: {exc}</p>"
        text.setHtml(
            """
            <h2>Photon-by-photon HMM (H2MM)</h2>
            <p>H2MM fits a Hidden Markov Model directly to photon arrival times and
            colours within bursts, resolving sub-burst FRET-state dynamics on the
            microsecond scale (Pirchi <i>et al.</i>, J. Phys. Chem. B 2016).</p>
            <h3>Workflow</h3>
            <ol>
              <li>Select a folder of <code>.bur</code> burst files.</li>
              <li>Define donor/acceptor detector channels.</li>
              <li>Choose the state range and BIC/ICL selection.</li>
              <li><b>Run</b> to fit models and view FRET states, a transition-density
                  plot, model selection, and dwell-time distributions.</li>
            </ol>
            <hr><h3>CLI reference</h3>
            """
            + cli_text
        )
        layout.addWidget(text, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)


@persist_plugin_state("burst_h2mm")
class H2mmTool(QMainWindow):
    """H2MM analysis widget with toolbar, tabbed settings, and result plots."""

    def __init__(self, parent=None, *, embedded: bool = False):
        super().__init__(parent)
        self._embedded = embedded
        self.setWindowTitle("smFRET H2MM Analysis")
        self.data_folder: pathlib.Path | None = None
        self.file_type = "SPC-130"
        self._result = None
        self._bundle = None
        self._build_ui()

    # ── UI build ─────────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._setup_toolbar()
        layout.addWidget(self.toolbar)

        self.dock_area = DockArea()
        self.dock_area.addTab(self._build_settings_tab(), "H2MM Settings", close_mode="hide")
        self.dock_area.addTab(self._build_channels_tab(), "Channel Definitions", close_mode="hide")
        self.dock_area.addTab(self._build_plots(), "Results", close_mode="hide")
        layout.addWidget(self.dock_area, 1)

        self._status_label = QLabel("Ready")
        self._status_label.setStyleSheet("color: #888; font-style: italic; padding: 0 8px;")
        self._status_label.setFixedHeight(22)
        layout.addWidget(self._status_label)

        self._connect_signals()
        self._load_settings()

    def _setup_toolbar(self):
        self.toolbar = QToolBar("Main")
        self.toolbar.setObjectName("h2mmMainToolbar")
        self.toolbar.setMovable(False)
        self.toolbar.setIconSize(QSize(16, 16))
        self.toolbar.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)

        def _tbtn(text):
            btn = QToolButton()
            btn.setText(text)
            btn.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
            return btn

        self.btn_folder = _tbtn("\U0001f4c2  Data")
        self._folder_field = _FolderLineEdit(placeholder="No folder selected")
        self._folder_field.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.btn_run = _tbtn("▶  Run")
        self.btn_save = _tbtn("\U0001f4be  Save plot")
        self.btn_help = _tbtn("ℹ  Help")

        self.toolbar.addWidget(self.btn_folder)
        self.toolbar.addWidget(self._folder_field)
        self.toolbar.addSeparator()
        self.toolbar.addWidget(self.btn_run)
        self.toolbar.addWidget(self.btn_save)
        self.toolbar.addSeparator()
        self.toolbar.addWidget(self.btn_help)

    def _build_settings_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setSpacing(8)

        group = QGroupBox("Model Selection")
        f = QFormLayout(group)
        self.sb_min_states = QSpinBox()
        self.sb_min_states.setRange(1, 8)
        self.sb_min_states.setValue(1)
        self.sb_max_states = QSpinBox()
        self.sb_max_states.setRange(1, 8)
        self.sb_max_states.setValue(3)
        self.cb_criterion = QComboBox()
        self.cb_criterion.addItems(["bic", "icl"])
        f.addRow("Min states:", self.sb_min_states)
        f.addRow("Max states:", self.sb_max_states)
        f.addRow("Criterion:", self.cb_criterion)
        layout.addWidget(group)

        opt = QGroupBox("Optimisation")
        of = QFormLayout(opt)
        self.sb_restarts = QSpinBox()
        self.sb_restarts.setRange(1, 20)
        self.sb_restarts.setValue(2)
        self.sb_max_iter = QSpinBox()
        self.sb_max_iter.setRange(10, 5000)
        self.sb_max_iter.setValue(500)
        self.sb_min_photons = QSpinBox()
        self.sb_min_photons.setRange(2, 1000)
        self.sb_min_photons.setValue(5)
        self.sb_time_scale = QSpinBox()
        self.sb_time_scale.setRange(1, 100000)
        self.sb_time_scale.setValue(1)
        of.addRow("Restarts:", self.sb_restarts)
        of.addRow("Max iterations:", self.sb_max_iter)
        of.addRow("Min photons/burst:", self.sb_min_photons)
        of.addRow("Macro-time scale:", self.sb_time_scale)
        layout.addWidget(opt)
        layout.addStretch()
        return w

    def _build_channels_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(4, 4, 4, 4)

        sel = QGroupBox("FRET Pair Assignment")
        sl = QVBoxLayout(sel)
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Donor detector:"))
        self.cb_donor = QComboBox()
        row1.addWidget(self.cb_donor, 1)
        sl.addLayout(row1)
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Acceptor detector:"))
        self.cb_acceptor = QComboBox()
        row2.addWidget(self.cb_acceptor, 1)
        sl.addLayout(row2)
        layout.addWidget(sel)

        self.detector_page = DetectorWizardPage(parent=self)
        layout.addWidget(self.detector_page, 1)
        self.detector_page.detectorsChanged.connect(self._refresh_detector_combos)
        self._refresh_detector_combos()
        return w

    def _refresh_detector_combos(self):
        settings = self.detector_page.get_settings()
        names = list(settings.get("detectors", {}).keys())
        donor_cur, acc_cur = self.cb_donor.currentText(), self.cb_acceptor.currentText()
        self.cb_donor.clear()
        self.cb_acceptor.clear()
        self.cb_donor.addItems(names)
        self.cb_acceptor.addItems(names)
        if donor_cur in names:
            self.cb_donor.setCurrentText(donor_cur)
        if acc_cur in names:
            self.cb_acceptor.setCurrentText(acc_cur)
        elif len(names) > 1:
            self.cb_acceptor.setCurrentIndex(1)

    def _build_plots(self) -> QWidget:
        self.plot_widget = pg.GraphicsLayoutWidget()
        self._p_fret = self.plot_widget.addPlot(row=0, col=0, title="FRET states")
        self._p_fret.setLabels(bottom="Apparent FRET E", left="Population")
        self._p_fret.setXRange(0, 1)
        self._p_tdp = self.plot_widget.addPlot(row=0, col=1, title="Transition-density plot")
        self._p_tdp.setLabels(bottom="E before", left="E after")
        self._p_tdp.setRange(xRange=(0, 1), yRange=(0, 1))
        self._tdp_img = pg.ImageItem(axisOrder="col-major")
        self._p_tdp.addItem(self._tdp_img)
        self._p_sel = self.plot_widget.addPlot(row=1, col=0, title="Model selection")
        self._p_sel.setLabels(bottom="Number of states", left="Criterion")
        self._p_sel.addLegend()
        self._p_dwell = self.plot_widget.addPlot(row=1, col=1, title="Dwell-time distributions")
        self._p_dwell.setLabels(bottom="Dwell time (ms)", left="Counts")
        return self.plot_widget

    def _connect_signals(self):
        self.btn_folder.clicked.connect(self._select_folder)
        self._folder_field.folderDropped.connect(self._set_folder)
        self.btn_run.clicked.connect(self._run_analysis)
        self.btn_save.clicked.connect(self._save_plot)
        self.btn_help.clicked.connect(lambda: HelpDialog(self).exec_())

    # ── settings ─────────────────────────────────────────────────────

    def _detector_streams(self) -> list[StreamSettings]:
        settings = self.detector_page.get_settings()
        detectors = settings.get("detectors", {})
        self.file_type = settings.get("tttr_reading", {}).get("file_type", self.file_type)

        def _stream(name: str) -> StreamSettings:
            d = detectors.get(name, {})
            ranges = [(int(a), int(b)) for a, b in d.get("micro_time_ranges", [])]
            return StreamSettings(name=name or "stream", channels=list(d.get("chs", [])), micro_time_ranges=ranges)

        return [_stream(self.cb_donor.currentText()), _stream(self.cb_acceptor.currentText())]

    def _gather_settings(self) -> H2mmSettings:
        return H2mmSettings(
            streams=self._detector_streams(),
            min_states=self.sb_min_states.value(),
            max_states=max(self.sb_max_states.value(), self.sb_min_states.value()),
            criterion=self.cb_criterion.currentText(),
            n_restarts=self.sb_restarts.value(),
            max_iter=self.sb_max_iter.value(),
            min_photons=self.sb_min_photons.value(),
            time_scale=self.sb_time_scale.value(),
            file_type=self.file_type,
        )

    # ── run ──────────────────────────────────────────────────────────

    def _select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select burst (.bur) folder")
        if folder:
            self._set_folder(folder)

    def _set_folder(self, path: str):
        p = pathlib.Path(path)
        if p.is_dir():
            self.data_folder = p
            self._folder_field.setText(str(p))
            self._status(f"Data folder: {p}")

    def _run_analysis(self):
        if not self.data_folder:
            QMessageBox.warning(self, "No data", "Please select a folder of .bur files first.")
            return
        settings = self._gather_settings()
        progress = _ProgressDialog("H2MM", "Fitting H2MM models (this may take a while)...", self)
        progress.show()
        QCoreApplication.processEvents()
        try:
            result, bundle = run_analysis(settings, analysis_folder=str(self.data_folder))
        except Exception as exc:
            progress.close()
            QMessageBox.critical(self, "H2MM error", str(exc))
            logging.error(f"H2MM analysis failed: {exc}")
            return
        progress.close()
        self._result = result
        self._bundle = bundle
        self._update_plots()
        self._status(
            f"Selected {result.n_states} states "
            f"({result.criterion.upper()}) from {result.n_bursts} bursts / {result.n_photons} photons"
        )

    # ── plotting ─────────────────────────────────────────────────────

    def _update_plots(self):
        if self._result is None or self._bundle is None:
            return
        res = self._result
        ana = self._bundle.analysis

        # FRET states as vertical bars sized by population.
        self._p_fret.clear()
        for i, (e, pop) in enumerate(zip(res.fret, res.populations)):
            if not np.isfinite(e):
                continue
            color = _STATE_COLORS[i % len(_STATE_COLORS)]
            bar = pg.BarGraphItem(x=[e], height=[pop], width=0.03, brush=color)
            self._p_fret.addItem(bar)

        # Transition-density plot: E_before vs E_after 2D histogram.
        if ana.transitions:
            eb = np.array([t.e_from for t in ana.transitions])
            ea = np.array([t.e_to for t in ana.transitions])
            good = np.isfinite(eb) & np.isfinite(ea)
            hist, _, _ = np.histogram2d(
                eb[good], ea[good], bins=(41, 41), range=[[0, 1], [0, 1]]
            )
            self._tdp_img.setImage(hist)
            self._tdp_img.setRect(0, 0, 1, 1)
            try:
                self._tdp_img.setColorMap(pg.colormap.get("CET-L4"))
            except Exception:
                pass

        # Model-selection curve: BIC and ICL vs number of states.
        self._p_sel.clear()
        ns = [f.n_states for f in res.scan]
        bic = [f.bic for f in res.scan]
        icl = [f.icl for f in res.scan]
        self._p_sel.plot(ns, bic, pen=pg.mkPen("#4e79a7", width=2), symbol="o", name="BIC")
        self._p_sel.plot(ns, icl, pen=pg.mkPen("#e15759", width=2), symbol="s", name="ICL")

        # Dwell-time distributions per state (ms).
        self._p_dwell.clear()
        base_ms = ana.base_time_s * 1e3
        for i, (state, arr) in enumerate(sorted(ana.dwell_times.items())):
            if arr.size == 0:
                continue
            dwell_ms = arr * base_ms
            counts, edges = np.histogram(dwell_ms, bins=30)
            centers = (edges[:-1] + edges[1:]) / 2
            color = _STATE_COLORS[i % len(_STATE_COLORS)]
            self._p_dwell.plot(centers, counts, pen=pg.mkPen(color, width=2))

    def _save_plot(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save plot", "h2mm.png", "PNG (*.png)")
        if path:
            self.plot_widget.grab().save(path)

    # ── workflow integration ─────────────────────────────────────────

    def apply_workflow_context(self, context: dict[str, Any]) -> None:
        """Apply a burst-workflow context (folder + channel settings)."""
        folder = context.get("burst_folder") or context.get("analysis_folder")
        if folder:
            self._set_folder(str(folder))
        channel_settings = context.get("channel_settings") or {}
        file_type = (channel_settings.get("tttr_reading") or {}).get("file_type")
        if file_type:
            self.file_type = file_type

    # ── persistence ──────────────────────────────────────────────────

    def _status(self, msg: str):
        self._status_label.setText(msg)
        QCoreApplication.processEvents()

    def _save_settings(self):
        ini = QSettings(str(get_plugin_settings_path("burst_h2mm")), QSettings.IniFormat)
        ini.setValue("min_states", self.sb_min_states.value())
        ini.setValue("max_states", self.sb_max_states.value())
        ini.setValue("criterion", self.cb_criterion.currentText())
        if self.data_folder is not None:
            ini.setValue("last_folder", str(self.data_folder))

    def _load_settings(self):
        ini = QSettings(str(get_plugin_settings_path("burst_h2mm")), QSettings.IniFormat)
        if (v := ini.value("min_states")) is not None:
            self.sb_min_states.setValue(int(v))
        if (v := ini.value("max_states")) is not None:
            self.sb_max_states.setValue(int(v))
        if (v := ini.value("criterion")) is not None:
            idx = self.cb_criterion.findText(str(v))
            if idx >= 0:
                self.cb_criterion.setCurrentIndex(idx)
        lf = ini.value("last_folder")
        if lf and pathlib.Path(str(lf)).is_dir():
            self._set_folder(str(lf))

    def closeEvent(self, event):
        """Persist settings when the window closes."""
        self._save_settings()
        super().closeEvent(event)
