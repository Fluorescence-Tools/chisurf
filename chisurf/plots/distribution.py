from __future__ import annotations

import pyqtgraph as pg
from qtpy import QtWidgets

import chisurf.fitting
import chisurf.fluorescence
import chisurf.math.datatools
from chisurf.plots import plotbase

plot_settings = chisurf.settings.gui['plot']
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
        super().__init__(*args, **kwargs)
        self.layout = QtWidgets.QVBoxLayout()
        self.setLayout(self.layout)
        self.selector = QtWidgets.QComboBox(None)
        self.layout.addWidget(self.selector)
        self.selector.addItems(
            [
                'Distance',
                'Lifetime',
                'FRET-rate'
            ]
        )
        self.selector.currentIndexChanged[int].connect(parent.update)


class DistributionPlot(plotbase.Plot):

    name = "Distribution"

    def __init__(
            self,
            fit: chisurf.fitting.fit.FitGroup,
            parent: QtWidgets.QWidget = None,
            **kwargs
    ):
        super().__init__(fit=fit, parent=parent)
        self.data_x, self.data_y = None, None

        self.plot_controller = DistributionPlotControl(
            self,
            parent=self,
            **kwargs
        )

        pw = pg.PlotWidget()
        self.layout.addWidget(pw)

        self.distribution_plot = pw.getPlotItem()
        pen = pg.mkPen(colors['data'], width=lw)
        self.distribution_curve = self.distribution_plot.plot(
            x=[0.0],
            y=[0.0],
            pen=pen,
            stepMode='right',
            fillLevel=0, fillBrush=colors['data']
        )

    def update(self, *args, **kwargs) -> None:
        super().update(*args, **kwargs)
        if self.plot_controller.distribution_type == 'Distance':
            d = self.fit.model.distance_distribution
            y = d[0][0]
            x = d[0][1]
        elif self.plot_controller.distribution_type == 'FRET-rate':
            lifetime_spectrum = self.fit.model.fret_rate_spectrum
            y, x = chisurf.math.datatools.interleaved_to_two_columns(
                lifetime_spectrum,
                sort=True
            )
        else: #elif self.plot_controller.distribution_type == 'Lifetime':
            lifetime_spectrum = self.fit.model.lifetime_spectrum
            y, x = chisurf.math.datatools.interleaved_to_two_columns(
                lifetime_spectrum,
                sort=True
            )
        self.distribution_curve.setData(x, y)
