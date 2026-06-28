"""Spectrum viewer widget for the fluorophore database."""

from __future__ import annotations

from typing import Any

import numpy as np
import pyqtgraph as pg
from qtpy import QtCore, QtWidgets


class SpectrumView(QtWidgets.QWidget):
    """Displays absorption and emission spectra for a fluorophore probe."""

    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.setLayout(QtWidgets.QVBoxLayout(self))
        self.layout().setContentsMargins(0, 0, 0, 0)

        self.plot = pg.PlotWidget()
        self.plot.setBackground("w")
        self.plot.addLegend()
        self.plot.setLabel("bottom", "Wavelength", units="nm")
        self.plot.setLabel("left", "Normalized Intensity", units="a.u.")
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.layout().addWidget(self.plot)

        self._abs_curve: pg.PlotDataItem | None = None
        self._em_curve: pg.PlotDataItem | None = None
        self._empty_label: QtWidgets.QLabel | None = None

    def clear(self) -> None:
        """Clear all plotted curves."""
        self.plot.clear()
        self._abs_curve = None
        self._em_curve = None

    def display(self, data: dict[str, Any]) -> None:
        """Display spectra for a probe.

        Parameters
        ----------
        data : dict
            Response from ``fluorophores.get`` containing ``spectra`` list.
        """
        self.clear()
        spectra = data.get("spectra", [])
        if not spectra:
            self._show_empty()
            return

        for spec in spectra:
            stype = spec.get("spectrum_type", "")
            wl = np.array(spec.get("wavelengths", []), dtype=float)
            iv = np.array(spec.get("intensity", []), dtype=float)
            if len(wl) == 0 or len(iv) == 0:
                continue
            if np.nanmax(iv) > 0:
                iv = iv / np.nanmax(iv)
            if stype == "absorption":
                color = (0, 100, 200)
                name = "Absorption"
            elif stype == "emission":
                color = (200, 0, 0)
                name = "Emission"
            else:
                color = (100, 100, 100)
                name = stype.capitalize()
            curve = self.plot.plot(
                wl, iv, pen=pg.mkPen(color, width=2), name=name,
            )
            if stype == "absorption":
                self._abs_curve = curve
            elif stype == "emission":
                self._em_curve = curve

        if not self._abs_curve and not self._em_curve:
            self._show_empty()

    def _show_empty(self) -> None:
        """Show a placeholder when no spectra are available."""
        if self._empty_label is None:
            self._empty_label = QtWidgets.QLabel("No spectra data found.")
            self._empty_label.setAlignment(QtCore.Qt.AlignCenter)
            self._empty_label.setStyleSheet("color: #999; font-style: italic;")
        self.plot.addItem(pg.TextItem(
            "No spectra data found.",
            color=(150, 150, 150),
            anchor=(0.5, 0.5),
        ))
