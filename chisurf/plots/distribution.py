from __future__ import annotations

import pyqtgraph as pg
from qtpy import QtWidgets
from pyqtgraph.dockarea import DockArea, Dock

import chisurf.settings as mfm
import chisurf.fitting
import chisurf.fluorescence
import chisurf.math.datatools
from chisurf.plots import plotbase

plot_settings = chisurf.settings.gui['plot']
pyqtgraph_settings = chisurf.settings.pyqtgraph_settings
colors = plot_settings['colors']
color_scheme = chisurf.settings.colors
lw = plot_settings['line_width']


class DistributionPlotControl(QtWidgets.QWidget):

    @property
    def distribution_type(self):
        return str(self.selector.currentText())

    def __init__(
            self,
            *args,
            parent: QtWidgets.QWidget = None,
            **kwargs
    ):
        super().__init__(
            *args,
            **kwargs
        )
        self.layout = QtWidgets.QVBoxLayout()
        self.setLayout(self.layout)
        self.selector = QtWidgets.QComboBox()
        self.layout.addWidget(self.selector)
        self.selector.addItems(
            [
                'Distance',
                'Lifetime',
                'FRET-rate'
            ]
        )

        self.selector.currentIndexChanged[int].connect(parent.update_all)


class DistributionPlot(plotbase.Plot):

    name = "Distribution"

    def __init__(
            self,
            fit: chisurf.fitting.fit.FitGroup,
            parent: QtWidgets.QWidget,
            **kwargs
    ):
        super().__init__(
            fit=fit,
            parent=parent
        )
        self.layout = QtWidgets.QVBoxLayout(self)
        self.data_x, self.data_y = None, None

        self.pltControl = DistributionPlotControl(
            self,
            parent=self,
            **kwargs
        )

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
        self.distribution_curve = distribution_plot.plot(
            x=[0.0],
            y=[0.0],
            pen=pg.mkPen(
                colors['data'],
                width=lw
            ),
            name='Data'
        )

    def update_all(self, *args, **kwargs):
        if self.pltControl.distribution_type == 'Distance':
            d = self.fit.model.distance_distribution
            y = d[0][0]
            x = d[0][1]
        elif self.pltControl.distribution_type == 'FRET-rate':
            lifetime_spectrum = self.fit.model.fret_rate_spectrum
            y, x = chisurf.math.datatools.interleaved_to_two_columns(
                lifetime_spectrum,
                sort=True
            )
        else: #elif self.pltControl.distribution_type == 'Lifetime':
            lifetime_spectrum = self.fit.model.lifetime_spectrum
            y, x = chisurf.math.datatools.interleaved_to_two_columns(
                lifetime_spectrum,
                sort=True
            )
        self.distribution_curve.setData(
            x=x,
            y=y
        )
