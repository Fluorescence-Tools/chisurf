import numpy as np
import pyqtgraph as pg
from PyQt5 import QtWidgets

import mfm
from mfm.plots import plotbase

pyqtgraph_settings = mfm.settings.cs_settings['gui']['plot']["pyqtgraph"]
color_scheme = mfm.settings.colors


class ResidualPlot(plotbase.Plot):
    """
    Started off as a plotting class to display TCSPC-data displaying the IRF, the experimental data, the residuals
    and the autocorrelation of the residuals. Now it is also used also for FCS-data.

    In case the model is a :py:class:`~experiment.model.tcspc.LifetimeModel` it takes the irf and displays it:

        irf = fit.model.convolve.irf
        irf_y = irf.y

    The model data and the weighted residuals are taken directly from the fit:

        model_x, model_y = fit[:]
        wres_y = fit.weighted_residuals

    """

    name = "Residuals"

    def __init__(self, fit, *args, **kwargs):
        mfm.plots.Plot.__init__(self, fit)
        self.layout = QtWidgets.QVBoxLayout(self)
        self.data_x, self.data_y = None, None

        curves = list()
        lw = mfm.cs_settings['gui']['plot']['line_width']
        self.curves = curves

        p = pg.PlotWidget(useOpenGL=pyqtgraph_settings['useOpenGL'])
        self.layout.addWidget(p)

        for i, f in enumerate(fit):
            color = mfm.colors[i % len(mfm.colors)]['hex']
            c = pg.PlotCurveItem(pen=pg.mkPen(color, width=lw), name=f.data.name)
            p.addItem(c)
            c.setPos(0, i*6)
            curves.append(c)

    def update_all(self, *args, **kwargs):
        # Get parameters from plot-control
        fits = self.fit
        for ci, fi in zip(self.curves, fits):
            y_res = fi.weighted_residuals
            x_res = np.arange(0, len(y_res))
            ci.setData(x=x_res, y=y_res)

