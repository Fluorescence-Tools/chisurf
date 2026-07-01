"""Interactive IRF background-region selector (embedded in the wizard).

A pyqtgraph plot of the raw and corrected VV/VH IRFs with a draggable
:class:`~pyqtgraph.LinearRegionItem` for the background window, mirrored by two
spin boxes. Dragging the region recomputes the corrected IRFs through the model
(which calls the Qt-free :func:`...core.irf.correct_irfs`). Embedded via the
``embed`` section with ``pass_model=True``.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from qtpy import QtCore, QtWidgets


class IrfNormalizationWidget(QtWidgets.QWidget):
    """Plot + region selector bound to an :class:`AnisotropyViewModel`."""

    _autoform_expanding = True

    def __init__(self, model, parent=None):
        super().__init__(parent)
        self._model = model
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        bar = QtWidgets.QHBoxLayout()
        load_btn = QtWidgets.QToolButton()
        load_btn.setText("🔄 Load / reload data")
        load_btn.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        load_btn.setToolTip("Load the polarised IRF/data files selected on the previous step.")
        load_btn.clicked.connect(self._on_load)
        bar.addWidget(load_btn)
        bar.addStretch(1)
        bar.addWidget(QtWidgets.QLabel("BG from"))
        self._lb = QtWidgets.QSpinBox()
        self._lb.setMaximum(1_000_000)
        bar.addWidget(self._lb)
        bar.addWidget(QtWidgets.QLabel("to"))
        self._ub = QtWidgets.QSpinBox()
        self._ub.setMaximum(1_000_000)
        bar.addWidget(self._ub)
        layout.addLayout(bar)

        self._plot = pg.PlotWidget()
        self._plot.setLogMode(x=False, y=True)
        self._plot.addLegend()
        layout.addWidget(self._plot, 1)

        self._region = pg.LinearRegionItem()
        self._plot.addItem(self._region)

        self._region.sigRegionChangeFinished.connect(self._on_region)
        self._lb.editingFinished.connect(self._on_spin)
        self._ub.editingFinished.connect(self._on_spin)

        self._sync_region_from_model()
        self._refresh_plot()

    # ── actions ─────────────────────────────────────────────────────────
    def _on_load(self) -> None:
        try:
            ok = self._model.load_data()
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Load failed", str(exc))
            return
        if not ok:
            QtWidgets.QMessageBox.warning(
                self, "Load failed", "Could not load data — check the file paths and setup."
            )
            return
        self._sync_region_from_model()
        self._refresh_plot()

    def _on_region(self) -> None:
        lb, ub = self._region.getRegion()
        self._lb.blockSignals(True)
        self._ub.blockSignals(True)
        self._lb.setValue(int(lb))
        self._ub.setValue(int(ub))
        self._lb.blockSignals(False)
        self._ub.blockSignals(False)
        self._model.apply_region(int(lb), int(ub))
        self._refresh_plot(keep_region=True)

    def _on_spin(self) -> None:
        lb, ub = self._lb.value(), self._ub.value()
        self._region.blockSignals(True)
        self._region.setRegion((lb, ub))
        self._region.blockSignals(False)
        self._model.apply_region(lb, ub)
        self._refresh_plot(keep_region=True)

    # ── rendering ───────────────────────────────────────────────────────
    def _sync_region_from_model(self) -> None:
        lb, ub = int(self._model.region_lb), int(self._model.region_ub)
        for w, v in ((self._lb, lb), (self._ub, ub)):
            w.blockSignals(True)
            w.setValue(v)
            w.blockSignals(False)
        self._region.blockSignals(True)
        self._region.setRegion((lb, ub))
        self._region.blockSignals(False)

    def _refresh_plot(self, keep_region: bool = False) -> None:
        # clear only the data curves, keep the region item
        for item in list(self._plot.getPlotItem().listDataItems()):
            self._plot.removeItem(item)
        for s in self._model.plot_series():
            self._plot.plot(
                x=np.asarray(s["x"]),
                y=np.asarray(s["y"]),
                pen=pg.mkPen(s.get("color", "w"), width=s.get("width", 1)),
                name=s.get("name"),
            )


__all__ = ["IrfNormalizationWidget"]
