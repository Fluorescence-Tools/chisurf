"""Burst Background Estimation Plugin

This plugin estimates background count rates from TTTR data by fitting the
exponential tail of the interphoton time distribution, analogous to
PAM's `Estimate_Background_From_Burst.m`.

It uses the detector and PIE-window definitions from the
:class:`chisurf.gui.widgets.wizard.tttr_channel_definition.DetectorWizardPage`
so that the same setups can be shared with other TTTR tools.
"""

name = "Spectroscopy:Single-Molecule:Burst Background Estimation"

# Expose the plugin CLI through chisurf.core.cli
cli_entrypoint = "burst-background=chisurf.plugins.burst.burst_background.cli:cli"

import os
import sys
from typing import Dict

import numpy as np
import tttrlib

from qtpy.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QTabWidget,
    QPushButton,
    QFileDialog,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QMessageBox,
)
from qtpy.QtCore import Qt
from qtpy.QtGui import QDragEnterEvent, QDropEvent

from chisurf.gui.widgets.wizard.tttr_channeldefinition import DetectorWizardPage
import chisurf.core.fluorescence.burst


class BurstBackgroundEstimator(QWidget):
    """Main widget for the Burst Background Estimation plugin."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Burst Background Estimation")

        # Data storage
        self.tttr_files = []  # type: ignore[var-annotated]
        self.backgrounds: Dict[str, Dict[str, float]] = {}

        # Enable drag and drop
        self.setAcceptDrops(True)

        # Build UI
        self._init_ui()

    # ------------------------------------------------------------------
    # Qt Events (drag & drop)
    # ------------------------------------------------------------------
    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # type: ignore[override]
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:  # type: ignore[override]
        urls = event.mimeData().urls()
        file_paths = [url.toLocalFile() for url in urls]
        self._add_tttr_files(file_paths)
        event.acceptProposedAction()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _init_ui(self) -> None:
        main_layout = QVBoxLayout()

        tab_widget = QTabWidget()

        # Tab 1: detector / PIE-window definition
        self.detector_wizard_page = DetectorWizardPage(
            show_edit_json=False,
            show_save=False,
            show_setups_file=True,
            show_setup_selection=True,
            show_help=True,
            show_tttr_reading=True,
            show_tables=True,
            show_add_inputs=True,
        )
        tab_widget.addTab(self.detector_wizard_page, "Channel Definition")

        # Tab 2: files and background estimation results
        files_tab = QWidget()
        files_layout = QVBoxLayout(files_tab)

        controls_layout = QHBoxLayout()

        self.load_button = QPushButton("Load TTTR Files")
        self.load_button.clicked.connect(self._load_tttr_files)
        controls_layout.addWidget(self.load_button)

        self.clear_button = QPushButton("Clear Files")
        self.clear_button.clicked.connect(self._clear_files)
        controls_layout.addWidget(self.clear_button)

        self.estimate_button = QPushButton("Estimate Background")
        self.estimate_button.clicked.connect(self._estimate_background)
        controls_layout.addWidget(self.estimate_button)

        files_layout.addLayout(controls_layout)

        # Table listing TTTR files and status
        self.file_table = QTableWidget()
        self.file_table.setColumnCount(2)
        self.file_table.setHorizontalHeaderLabels(["File", "Status"])
        self.file_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.file_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        files_layout.addWidget(self.file_table)

        # Results table: one row per (file, detector)
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(3)
        self.results_table.setHorizontalHeaderLabels([
            "File",
            "Detector",
            "Background (kHz)",
        ])
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        files_layout.addWidget(self.results_table)

        tab_widget.addTab(files_tab, "Files & Results")

        main_layout.addWidget(tab_widget)
        self.setLayout(main_layout)
        self.resize(700, 500)

    # ------------------------------------------------------------------
    # File handling helpers
    # ------------------------------------------------------------------
    def _load_tttr_files(self) -> None:
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Load TTTR Files",
            "",
            "All Files (*)",
        )
        if file_paths:
            self._add_tttr_files(file_paths)

    def _add_tttr_files(self, file_paths) -> None:
        new_files_added = False
        for path in file_paths:
            if path and path not in self.tttr_files:
                self.tttr_files.append(path)
                new_files_added = True

        if not new_files_added:
            return

        # Sort lexically by basename and rebuild the table
        self.tttr_files.sort(key=lambda p: os.path.basename(p).lower())
        self.file_table.setRowCount(0)
        for path in self.tttr_files:
            row = self.file_table.rowCount()
            self.file_table.insertRow(row)
            self.file_table.setItem(row, 0, QTableWidgetItem(os.path.basename(path)))
            self.file_table.setItem(row, 1, QTableWidgetItem("Loaded"))

    def _clear_files(self) -> None:
        self.tttr_files = []
        self.backgrounds.clear()
        self.file_table.setRowCount(0)
        self.results_table.setRowCount(0)

    # ------------------------------------------------------------------
    # Core calculation
    # ------------------------------------------------------------------
    def _estimate_background(self) -> None:
        if not self.tttr_files:
            QMessageBox.warning(self, "No Files", "Please load TTTR files first.")
            return

        settings = self.detector_wizard_page.get_settings()
        detectors = settings.get("detectors", {})
        if not detectors:
            QMessageBox.warning(
                self,
                "No Detectors",
                "Please define at least one detector in the channel definition tab.",
            )
            return

        self.backgrounds.clear()
        self.results_table.setRowCount(0)

        for file_idx, path in enumerate(self.tttr_files):
            # Update status
            self.file_table.setItem(file_idx, 1, QTableWidgetItem("Processing..."))
            QApplication.processEvents()

            try:
                tttr = tttrlib.TTTR(path)
            except Exception as exc:  # pragma: no cover - GUI error path
                self.file_table.setItem(file_idx, 1, QTableWidgetItem(f"Error: {exc}"))
                continue

            bg = chisurf.core.fluorescence.burst.estimate_background_from_bursts(
                tttr,
                detectors,
            )
            self.backgrounds[path] = bg

            self.file_table.setItem(file_idx, 1, QTableWidgetItem("Done"))

        self._update_results_table()

    def _update_results_table(self) -> None:
        # Count rows required
        n_rows = 0
        for bg in self.backgrounds.values():
            n_rows += len(bg)

        self.results_table.setRowCount(n_rows)
        row = 0
        for path, bg in self.backgrounds.items():
            fname = os.path.basename(path)
            for det_name, rate in bg.items():
                self.results_table.setItem(row, 0, QTableWidgetItem(fname))
                self.results_table.setItem(row, 1, QTableWidgetItem(str(det_name)))
                self.results_table.setItem(row, 2, QTableWidgetItem(f"{rate:.3f}"))
                row += 1


if __name__ == "__main__":  # pragma: no cover - manual GUI entry
    app = QApplication(sys.argv)
    window = BurstBackgroundEstimator()
    window.show()
    sys.exit(app.exec())

elif __name__ == "plugin":  # pragma: no cover - used by ChiSurf plugin loader
    window = BurstBackgroundEstimator()
    window.show()
