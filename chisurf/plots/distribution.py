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


# Define some distribution options: If an attribute in a model is there it will be plotted.
# accessor: a function that accesses the attribute and returns a pair (y, x)
# that is plotted
# accessor_kwargs: are kwargs that are passed to the accessor function
# plot_options: default options used for plotting
distribution_options = {
    'Distance': {
        'attribute': 'distance_distribution',
        'accessor': lambda x: (x[0][0], x[0][1]),
        'accessor_kwargs': {},
        'plot_options': {
            'stepMode': False, #'right',
            'connect': False, #'all',
            'symbol': "t"
        }
    },
    'FRET-rate': {
        'attribute': 'fret_rate_spectrum',
        'accessor': chisurf.math.datatools.interleaved_to_two_columns,
        'accessor_kwargs': {'sort': True},
        'plot_options': {
            'stepMode': False, #'right',
            'connect': False, # 'all',
            'symbol': "x"
        }
    },
    'Lifetime': {
        'attribute': 'lifetime_spectrum',
        'accessor': chisurf.math.datatools.interleaved_to_two_columns,
        'accessor_kwargs': {'sort': True},
        'plot_options': {
            'stepMode': False,
            'connect': False,
            'symbol': "o"
        }
    }
}


class DistributionPlotControl(QtWidgets.QWidget):

    @property
    def distribution_type(self):
        return str(self.selector.currentText())

    def add_distribution_choices(self):
        model = self.parent().fit.model
        items = list()
        for distribution_type in distribution_options.keys():
            d = distribution_options[distribution_type]
            try:
                attr = model.__getattribute__(d['attribute'])
                items.append(distribution_type)
            except AttributeError:
                pass
        self.selector.addItems(items)

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
        self.selector.blockSignals(True)
        self.layout.addWidget(self.selector)
        self.add_distribution_choices()
        self.selector.currentIndexChanged[int].connect(parent.update)
        self.selector.blockSignals(False)


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
        d = distribution_options[self.plot_controller.distribution_type]
        y, x = d['accessor'](self.fit.model.__getattribute__(d['attribute']), **d['accessor_kwargs'])
        self.distribution_curve.setData(x, y, **d['plot_options'])
