from __future__ import annotations

import numpy as np
import pyqtgraph as pg

import chisurf.core.fitting
from chisurf.gui.plots import plotbase

color_scheme = chisurf.core.settings.colors


class ResidualPlot(plotbase.Plot):

    name = "Residuals"

    def __init__(self, fit: chisurf.core.fitting.fit.FitGroup, *args, **kwargs):
        super().__init__(*args, fit=fit, **kwargs)
        self.data_x, self.data_y = None, None

        curves = list()
        lw = chisurf.core.settings.gui['plot']['line_width']

        p = pg.PlotWidget()
        self.layout.addWidget(p)

        try:
            p.getPlotItem().setLabel('left', 'w.res.')
        except Exception:
            pass

        for i, f in enumerate(fit):
            color = chisurf.core.settings.colors[i % len(chisurf.core.settings.colors)]['hex']
            c = pg.PlotCurveItem(pen=pg.mkPen(color, width=lw), name=f.data.name)
            p.addItem(c)
            c.setPos(0, i*6)
            curves.append(c)
        self.curves = curves

    def update(self, *args, **kwargs) -> None:
        super().update(*args, **kwargs)
        # Get parameters from plot-control
        fits = self.fit
        for ci, fi in zip(self.curves, fits):
            w_res = fi.model.weighted_residuals
            x = np.arange(len(w_res))
            ci.setData(x, w_res)
