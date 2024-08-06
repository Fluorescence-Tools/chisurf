from __future__ import annotations

import pyqtgraph as pg
import copy

from chisurf.gui import QtWidgets
from chisurf.gui.tools.parameter_editor import ParameterEditor

import chisurf.fitting
import chisurf.fluorescence
import chisurf.math.datatools
from chisurf.plots import plotbase

plot_settings = chisurf.settings.gui['plot']
colors = plot_settings['colors']
color_scheme = chisurf.settings.colors
lw = plot_settings['line_width']

"""
For plotting
"""
def d21(x, **kwargs):
    return x[0][0], x[0][1]

# Define some distribution options: If an attribute in a model is there it will be plotted.
# accessor: a function that accesses the attribute and returns a pair (y, x)
# that is plotted
# accessor_kwargs: are kwargs that are passed to the accessor function
# plot_options: default options used for plotting
distribution_options = {
    'Distance': {
        'attribute': 'distance_distribution',
        'accessor': lambda x, **kwargs: (x[0][0], x[0][1]),
        'accessor_kwargs': {'sort': False},
        'curve_options': {
            'stepMode': False,  # 'right'
            'connect': False,   # 'all'
            'symbol': "t",
            'multi_curve': False
        }
    },
    'FRET-rate': {
        'attribute': 'fret_rate_spectrum',
        'accessor': chisurf.math.datatools.interleaved_to_two_columns,
        'accessor_kwargs': {'sort': True},
        'curve_options': {
            'stepMode': False, #'right',
            'connect': False, # 'all',
            'symbol': "x",
            'multi_curve': False
        }
    },
    'Lifetime': {
        'attribute': 'lifetime_spectrum',
        'accessor': chisurf.math.datatools.interleaved_to_two_columns,
        'accessor_kwargs': {
            'sort': True
        },
        'curve_options': {
            'stepMode': False,
            'connect': False,
            'symbol': "o",
            'multi_curve': False
        }
    }
}


class DistributionPlotControl(QtWidgets.QWidget):

    @property
    def distribution_type(self):
        return str(self.selector.currentText())

    def add_distribution_choices(self, options: dict = None) -> None:
        if options is None:
            options = distribution_options

        model = self.parent.fit.model
        items = list()

        for distribution_type in options.keys():
            d = options[distribution_type]
            try:
                attr = model.__getattribute__(d['attribute'])
                items.append(distribution_type)
            except AttributeError:
                pass
        self.selector.addItems(items)

    def update_parameter(self):
        self.parameter_editor._dict = self.parent.distribution_options[self.distribution_type]
        self.parameter_editor.update()
        self.parent.update()

    def __init__(
            self,
            *args,
            parent: QtWidgets.QWidget = None,
            distribution_options: dict = None,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.parent = parent
        self.layout = QtWidgets.QVBoxLayout()
        self.setLayout(self.layout)
        self.selector = QtWidgets.QComboBox(None)
        self.selector.blockSignals(True)
        self.layout.addWidget(self.selector)
        self.add_distribution_choices(distribution_options)
        self.selector.currentIndexChanged[int].connect(self.update_parameter)
        self.selector.blockSignals(False)
        d = copy.deepcopy(parent.distribution_options[self.distribution_type])
        self.parameter_editor = ParameterEditor(json_file='', target=d, callback=parent.update)
        self.layout.addWidget(self.parameter_editor)


class DistributionPlot(plotbase.Plot):

    name = "Distribution"

    def __init__(
            self,
            fit: chisurf.fitting.fit.FitGroup,
            parent: QtWidgets.QWidget = None,
            distribution_options: dict = None,
            **kwargs
    ):
        super().__init__(fit=fit, parent=parent)
        self.data_x, self.data_y = None, None
        self.distribution_options = distribution_options
        self.plot_controller = DistributionPlotControl(
            self,
            parent=self,
            distribution_options=distribution_options,
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
            fillLevel=0,
            fillBrush=colors['data']
        )

    def update(self, *args, **kwargs) -> None:
        super().update(*args, **kwargs)
        # Clear curve and recreate plots
        self.distribution_plot.clear()

        # Get distribution
        ds = self.plot_controller.parameter_editor.dict
        r = ds['accessor'](
            self.fit.model.__getattribute__(ds['attribute']),
            **ds['accessor_kwargs']
        )

        p = ds.get('curve_options', {})
        if p.get('multi_curve', False):
            n_curves = len(r)
            pen = p.pop('pen', ['r', 'b', 'g', 'y', 'c', 'm', 'k'])
            symbols = p.pop('symbol', ['o', 'x', 'v', '^', '<'])
            if len(pen) < n_curves:
                pen = [pen[i % len(pen)] for i in range(n_curves)]
                symbols = [symbols[i % len(symbols)] for i in range(n_curves)]
            for i in range(len(r)):
                y, x = r[i]
                c, s = pen[i], symbols[i]
                self.distribution_plot.plot(x, y, **p, pen=pg.mkPen(c, width=lw), symbol=s)
        else:
            y, x = r
            c = p.pop('pen', 'b')
            s = p.pop('symbol', 'o')
            self.distribution_plot.plot(x, y, **p, pen=pg.mkPen(c, width=lw), symbol=s)
