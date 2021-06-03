from __future__ import annotations

import numpy as np
import pyqtgraph as pg

import chisurf.fitting
from chisurf.plots import plotbase

color_scheme = chisurf.settings.colors


class ResidualPlot(plotbase.Plot):
    """
    Started off as a plotting class to display TCSPC-data displaying the IRF,
    the experimental data, the residuals
    and the autocorrelation of the residuals. Now it is also used also for fcs-data.

    In case the models is a :py:class:`~experiment.models.tcspc.LifetimeModel` it takes the irf and displays it:

        irf = fit.models.convolve.irf
        irf_y = irf.y

    The models data and the weighted residuals are taken directly from the fit:

        model_x, model_y = fit[:]
        wres_y = fit.weighted_residuals

    """

    name = "Residuals"

    def __init__(
            self,
            fit: chisurf.fitting.fit.FitGroup,
            *args,
            **kwargs
    ):
        super().__init__(
            *args,
            fit=fit,
            **kwargs
        )
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
            w_res = fi.weighted_residuals
            x = np.arange(len(w_res.y))
            ci.setData(
                x=x,
                y=w_res.y
            )
