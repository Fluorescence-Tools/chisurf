import pyqtgraph as pg
from PyQt5 import QtWidgets
from pyqtgraph.dockarea import DockArea, Dock

import mfm
from mfm.plots import plotbase

plot_settings = mfm.cs_settings['gui']['plot']
pyqtgraph_settings = plot_settings["pyqtgraph"]
for setting in pyqtgraph_settings:
    pg.setConfigOption(setting, pyqtgraph_settings[setting])
colors = plot_settings['colors']
color_scheme = mfm.colors
lw = plot_settings['line_width']


class DistributionPlotControl(QtWidgets.QWidget):

    @property
    def distribution_type(self):
        return str(self.selector.currentText())

    def __init__(self, *args, **kwargs):
        QtWidgets.QWidget.__init__(self)
        self.layout = QtWidgets.QVBoxLayout()
        self.setLayout(self.layout)
        self.parent = kwargs.get('parent', None)
        self.selector = QtWidgets.QComboBox()
        self.layout.addWidget(self.selector)
        self.selector.addItems(['Distance', 'Lifetime', 'FRET-rate'])

        self.selector.currentIndexChanged[int].connect(self.parent.update_all)


class DistributionPlot(plotbase.Plot):

    name = "Distribution"

    def __init__(self, fit, **kwargs):
        mfm.plots.Plot.__init__(self, fit)
        self.layout = QtWidgets.QVBoxLayout(self)
        self.data_x, self.data_y = None, None

        self.pltControl = DistributionPlotControl(self, parent=self, **kwargs)

        area = DockArea()
        self.layout.addWidget(area)
        hide_title = plot_settings['hideTitle']
        d2 = Dock("Fit", size=(500, 400), hideTitle=hide_title)

        self.p1 = QtWidgets.QPlainTextEdit()
        p2 = pg.PlotWidget(useOpenGL=pyqtgraph_settings['useOpenGL'])

        d2.addWidget(p2)

        area.addDock(d2, 'top')

        distribution_plot = p2.getPlotItem()

        self.distribution_plot = distribution_plot
        self.distribution_curve = distribution_plot.plot(x=[0.0], y=[0.0], pen=pg.mkPen(colors['data'], width=lw), name='Data')

    def update_all(self, *args, **kwargs):
        if self.pltControl.distribution_type == 'Distance':
            d = self.fit.model.distance_distribution
            y = d[0][0]
            x = d[0][1]
        elif self.pltControl.distribution_type == 'Lifetime':
            ls = self.fit.model.lifetime_spectrum
            y, x = mfm.fluorescence.interleaved_to_two_columns(ls, sort=True)
        elif self.pltControl.distribution_type == 'FRET-rate':
            ls = self.fit.model.fret_rate_spectrum
            y, x = mfm.fluorescence.interleaved_to_two_columns(ls, sort=True)
        self.distribution_curve.setData(x=x, y=y)
