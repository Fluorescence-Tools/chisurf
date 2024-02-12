from __future__ import annotations

import numpy as np
import pyqtgraph as pg

import chisurf.fitting
from chisurf.plots import plotbase

color_scheme = chisurf.settings.colors


class ResidualPlot(plotbase.Plot):

    name = "Residuals"

    def __init__(self, fit: chisurf.fitting.fit.FitGroup, *args, **kwargs):
        super().__init__(*args, fit=fit, **kwargs)
        self.data_x, self.data_y = None, None

        curves = list()
        lw = chisurf.settings.gui['plot']['line_width']

        p = pg.PlotWidget()
        self.layout.addWidget(p)

        for i, f in enumerate(fit):
            color = chisurf.settings.colors[i % len(chisurf.settings.colors)]['hex']
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
