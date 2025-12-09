from __future__ import annotations

import pathlib
from typing import List

import numpy as np
from qtpy import QtWidgets, QtCore
import pyqtgraph as pg

import chisurf
from chisurf.fluorescence.fcs.filtered import calc_ffcs_filters


class FcsFilterCalculatorWidget(QtWidgets.QWidget):
    """GUI widget for filtered-FCS lifetime filter calculation.

    Workflow
    --------
    1. Load a total decay histogram (one column text file).
    2. Load one or more species decay histograms (one column text files).
       All histograms must have the same number of bins.
    3. Press "Compute filters" to calculate species filters, reconstructed
       decay, and weighted residuals using the same weighted least-squares
       scheme as PAM's Calc_fFCS_Filters.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Filtered FCS: Lifetime Filter Calculator")
        self.resize(900, 700)

        self._total_path: pathlib.Path | None = None
        self._species_paths: List[pathlib.Path] = []

        self._setup_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _setup_ui(self) -> None:
        main = QtWidgets.QVBoxLayout(self)

        # Input group
        input_group = QtWidgets.QGroupBox("Input decay histograms")
        input_layout = QtWidgets.QFormLayout(input_group)

        # Total decay
        self.le_total = QtWidgets.QLineEdit()
        self.le_total.setReadOnly(True)
        btn_total = QtWidgets.QPushButton("Browse…")
        btn_total.clicked.connect(self._on_browse_total)
        w_total = QtWidgets.QWidget()
        hl_total = QtWidgets.QHBoxLayout(w_total)
        hl_total.setContentsMargins(0, 0, 0, 0)
        hl_total.addWidget(self.le_total)
        hl_total.addWidget(btn_total)
        input_layout.addRow("Total decay:", w_total)

        # Species decays (multiple files)
        self.le_species = QtWidgets.QLineEdit()
        self.le_species.setReadOnly(True)
        btn_species = QtWidgets.QPushButton("Browse…")
        btn_species.clicked.connect(self._on_browse_species)
        w_species = QtWidgets.QWidget()
        hl_species = QtWidgets.QHBoxLayout(w_species)
        hl_species.setContentsMargins(0, 0, 0, 0)
        hl_species.addWidget(self.le_species)
        hl_species.addWidget(btn_species)
        input_layout.addRow("Species decays:", w_species)

        # List of individual species decay files
        self.lw_species = QtWidgets.QListWidget()
        self.lw_species.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        input_layout.addRow("Files:", self.lw_species)

        # Compute button
        self.btn_compute = QtWidgets.QPushButton("Compute filters")
        self.btn_compute.clicked.connect(self._on_compute)
        input_layout.addRow(self.btn_compute)

        main.addWidget(input_group)

        # Plots
        self.plot_filters = pg.PlotWidget()
        self.plot_filters.setLabel("bottom", "TAC bin")
        self.plot_filters.setLabel("left", "Filter value")
        self.plot_filters.addLegend()

        self.plot_recon = pg.PlotWidget()
        self.plot_recon.setLabel("bottom", "TAC bin")
        self.plot_recon.setLabel("left", "Counts / Residuals")
        self.plot_recon.addLegend()

        splitter = QtWidgets.QSplitter()
        splitter.setOrientation(QtCore.Qt.Vertical)
        splitter.addWidget(self.plot_filters)
        splitter.addWidget(self.plot_recon)
        splitter.setSizes([350, 350])

        main.addWidget(splitter, 1)

        # Status label
        self.status_label = QtWidgets.QLabel()
        main.addWidget(self.status_label)

    # ------------------------------------------------------------------
    # File selection handlers
    # ------------------------------------------------------------------
    def _on_browse_total(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select total decay histogram",
            "",
            "Text files (*.txt *.dat *.csv);;All files (*)",
        )
        if not path:
            return
        self._total_path = pathlib.Path(path)
        self.le_total.setText(self._total_path.as_posix())
        self._update_status()

    def _on_browse_species(self) -> None:
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "Select species decay histograms",
            "",
            "Text files (*.txt *.dat *.csv);;All files (*)",
        )
        if not paths:
            return
        self._species_paths = [pathlib.Path(p) for p in paths]
        if len(self._species_paths) == 1:
            txt = self._species_paths[0].as_posix()
        else:
            txt = f"{len(self._species_paths)} files selected"
        self.le_species.setText(txt)

        # Update list widget with filenames of individual species decays
        self.lw_species.clear()
        for p in self._species_paths:
            self.lw_species.addItem(p.name)

        self._update_status()

    def _update_status(self, msg: str | None = None) -> None:
        if msg is None:
            if self._total_path is None:
                msg = "Load a total decay histogram and at least one species histogram."
            elif not self._species_paths:
                msg = "Select one or more species decay histograms."
            else:
                msg = "Ready: press 'Compute filters'."
        self.status_label.setText(msg)

    # ------------------------------------------------------------------
    # Core computation
    # ------------------------------------------------------------------
    def _on_compute(self) -> None:
        try:
            if self._total_path is None or not self._species_paths:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Missing input",
                    "Please select a total decay histogram and at least one species histogram.",
                )
                return

            total = self._load_vector(self._total_path)
            species = [self._load_vector(p) for p in self._species_paths]

            n_bins = total.size
            for i, v in enumerate(species):
                if v.size != n_bins:
                    QtWidgets.QMessageBox.critical(
                        self,
                        "Size mismatch",
                        f"Species histogram {self._species_paths[i].name} has {v.size} bins, "
                        f"but total decay has {n_bins} bins.",
                    )
                    return

            filters, recon, wres = calc_ffcs_filters(total, species)
            self._update_plots(total, species, filters, recon, wres)
            self._update_status("Filters computed successfully.")
        except Exception as exc:  # pragma: no cover - GUI safety
            chisurf.logging.error(f"Error computing fFCS filters: {exc}")
            QtWidgets.QMessageBox.critical(self, "Error", str(exc))

    @staticmethod
    def _load_vector(path: pathlib.Path) -> np.ndarray:
        """Load a 1D histogram from a text file.

        Expects a single column of numeric values. Empty lines and comments
        starting with '#' are ignored.
        """
        data = np.loadtxt(path, ndmin=1)
        if data.ndim != 1:
            # Flatten multi-column input
            data = np.asarray(data).ravel()
        if data.size == 0:
            raise ValueError(f"Histogram file '{path.name}' is empty.")
        return data.astype(float)

    # ------------------------------------------------------------------
    # Plot updating
    # ------------------------------------------------------------------
    def _update_plots(
        self,
        total: np.ndarray,
        species: List[np.ndarray],
        filters: np.ndarray,
        recon: np.ndarray,
        wres: np.ndarray,
    ) -> None:
        n_bins = total.size
        x = np.arange(n_bins)

        # Filters plot
        self.plot_filters.clear()
        self.plot_filters.addLegend()
        n_species = filters.shape[0]
        for i in range(n_species):
            name = f"Filter {i + 1}"
            self.plot_filters.plot(
                x,
                filters[i, :],
                pen=pg.intColor(i, max(n_species, 1)),
                name=name,
            )

        # Reconstruction / residuals plot
        self.plot_recon.clear()
        self.plot_recon.addLegend()
        self.plot_recon.plot(x, total, pen="w", name="Total")
        self.plot_recon.plot(x, recon, pen="r", name="Reconstruction")
        self.plot_recon.plot(x, wres, pen="y", name="Weighted residuals")
