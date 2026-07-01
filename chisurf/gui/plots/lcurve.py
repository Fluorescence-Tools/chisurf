from __future__ import annotations

"""Read-only L-curve plot for regularised model-free fits (e.g. DEER).

Plots the discrete L-curve — residual norm ``||K P - F||`` versus solution
roughness ``||L P||`` in log-log space — for a model that exposes a
``compute_lcurve()`` method returning ``{'rho', 'eta', 'corner', 'used'}``.
The automatically selected corner is highlighted. Purely diagnostic: it does
not modify the fit.
"""

import numpy as np
import pyqtgraph as pg

from chisurf.gui.plots import plotbase


class LCurvePlot(plotbase.Plot):
    """L-curve diagnostic for regularisation-parameter selection."""

    name = "L-Curve"

    def __init__(self, fit, **kwargs):
        """Build the log-log L-curve plot with a highlighted corner marker."""
        super().__init__(fit=fit, **kwargs)
        self._pw = pg.PlotWidget()
        self.layout.addWidget(self._pw)
        item = self._pw.getPlotItem()
        item.setLogMode(x=True, y=True)
        item.showGrid(x=True, y=True, alpha=0.3)
        self._pw.setLabel("bottom", "residual ||K P - F||")
        self._pw.setLabel("left", "roughness ||L P||")
        self._curve = self._pw.plot(
            [], [], pen=pg.mkPen("#2f80ed", width=2),
            symbol="o", symbolSize=6, symbolBrush="#2f80ed", symbolPen="w")
        self._corner = self._pw.plot(
            [], [], pen=None, symbol="x", symbolSize=16,
            symbolBrush="#ffd166", symbolPen="#ffd166")

    def _model(self):
        """Return the selected fit's model, or ``None``."""
        fit = getattr(self.fit, "selected_fit", self.fit)
        return getattr(fit, "model", None)

    def update(self, *args, **kwargs) -> None:
        """Recompute and redraw the L-curve from the current model state."""
        model = self._model()
        fn = getattr(model, "compute_lcurve", None)
        if not callable(fn):
            return
        try:
            data = fn()
        except Exception:
            data = None
        if not data:
            self._curve.setData([], [])
            self._corner.setData([], [])
            return
        rho = np.asarray(data.get("rho"), dtype=float)
        eta = np.asarray(data.get("eta"), dtype=float)
        ok = np.isfinite(rho) & np.isfinite(eta) & (rho > 0) & (eta > 0)
        self._curve.setData(rho[ok], eta[ok])
        c = data.get("corner")
        if c is not None and 0 <= int(c) < rho.size and ok[int(c)]:
            self._corner.setData([rho[int(c)]], [eta[int(c)]])
        else:
            self._corner.setData([], [])
