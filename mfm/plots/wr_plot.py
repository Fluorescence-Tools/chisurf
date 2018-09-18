from PyQt4 import QtGui, QtCore, uic, Qt
import numpy as np
import mfm
from mfm.plots import plotbase
import pyqtgraph as pg
from pyqtgraph.dockarea import DockArea, Dock
pyqtgraph_settings = mfm.settings['gui']['plot']["pyqtgraph"]
for setting in mfm.settings['gui']['plot']['pyqtgraph']:
    pg.setConfigOption(setting, mfm.settings['gui']['plot']['pyqtgraph'][setting])
color_scheme = mfm.colors
import matplotlib.colors as mpl_colors


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
        self.layout = QtGui.QVBoxLayout(self)
        self.data_x, self.data_y = None, None

        docks = list()
        plots = list()
        curves = list()
        colors = mfm.settings['gui']['plot']['colors']
        lw = mfm.settings['gui']['plot']['line_width']
        residual_color = colors['residuals_inactive']
        self.curves = curves

        area = DockArea()
        self.layout.addWidget(area)

        downsample = mfm.settings['gui']['plot']['downsample']
        downsampleMethod = mfm.settings['gui']['plot']['downsampleMethod']
        docks = list()

        for i, f in enumerate(fit):
            di = Dock(f.data.name, size=(200, 40), hideTitle=True)
            docks.append(di)

            pi = pg.PlotWidget(useOpenGL=pyqtgraph_settings['useOpenGL'])
            ci = pi.plot(x=[0], y=[9], pen=pg.mkPen(residual_color, width=lw), name='residues')
            rp = pi.getPlotItem()
            rp.setDownsampling(ds=downsample, mode=downsampleMethod, auto=True)
            if mfm.settings['gui']['plot']['show_residual_grid']:
                rp.showGrid(True, True, 1.0)
            label = QtGui.QLineEdit()
            label.setText(f.data.name)
            label.setEnabled(False)
            di.addWidget(label)
            di.addWidget(pi)
            area.addDock(di, 'bottom')
            docks.append(di)
            plots.append(pi)
            curves.append(ci)

    def update_all(self, *args, **kwargs):
        # Get parameters from plot-control
        fits = self.fit

        for ci, fi in zip(self.curves, fits):
            y_res = fi.weighted_residuals
            x_res = np.arange(0, len(y_res))
            ci.setData(x=x_res, y=y_res)

