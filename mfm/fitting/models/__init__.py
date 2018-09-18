"""
This module is responsible contains all fitting modules for experimental data

The :py:mod:`.models`

1. :py:mod:`.models.tcspc`
2. :py:mod:`.models.fcs`
3. :py:mod:`.models.gloablfit`
4. :py:mod:`.models.parse`
5. :py:mod:`.models.proteinMC`
6. :py:mod:`.models.stopped_flow`


"""
from PyQt4 import QtGui
from mfm import plots
from mfm.curve import Curve
from mfm.parameter import AggregatedParameters, FittingParameterWidget


class Model(AggregatedParameters):

    def __init__(self, fit,  **kwargs):
        AggregatedParameters.__init__(self, model=self, **kwargs)
        self.fit = fit
        self.flatten_weighted_residuals = True

    @property
    def parameters(self):
        return [p for p in self._parameters if not (p.fixed or p.is_linked)]

    @property
    def parameter_bounds(self):
        return [pi.bounds for pi in self.parameters]

    @property
    def n_free(self):
        return len(self.parameters)

    def finalize(self):
        if mfm.verbose:
            print "Finalize model: %s" % self.name
        self.update()
        for a in self.aggregated_parameters:
            if a is not self:
                a.finalize()
        for p in self.parameters_all:
            p.finalize()

    @property
    def weighted_residuals(self):
        return self.get_wres(self.fit)

    def get_wres(self, fit, **kwargs):
        f = fit
        xmin = kwargs.get('xmin', f.xmin)
        xmax = kwargs.get('xmax', f.xmax)
        x, m = f.model[xmin:xmax]
        x, d, w = f.data[xmin:xmax]
        ml = min([len(m), len(d)])
        wr = np.array((d[:ml] - m[:ml]) * w[:ml], dtype=np.float64)
        return wr

    def update_model(self):
        pass

    def update(self):
        self.find_parameters()
        self.update_model()

    def load(self, filename):
        with open(filename, 'r') as fp:
            txt = fp.read()
            self.from_json(txt)

    def __str__(self):
        s = ""
        s += "Model: %s\n" % str(self.name)

        pd = self.parameters_all_dict
        keylist = pd.keys()
        keylist.sort()

        s += "Parameter\tValue\tBounds\tFixed\tLinked\n"
        for k in keylist:
            p = (pd[k].name, pd[k].value, pd[k].bounds, pd[k].fixed, pd[k].is_linked)
            s += "%s\t%.4e\t%s\t%s\t%s\n" % p
        return s


class ModelCurve(Model, Curve):

    @property
    def n_points(self):
        return self.fit.xmax - self.fit.xmin

    def __init__(self, fit, *args, **kwargs):
        x = fit.data.x
        y = np.zeros_like(fit.data.y)
        Model.__init__(self, fit, **kwargs)
        Curve.__init__(self, *args, x=x, y=y, **kwargs)

    def __getitem__(self, key):
        start = key.start
        stop = key.stop
        step = 1 if key.step is None else key.step
        x, y = self._x[start:stop:step], self._y[start:stop:step]
        return x, y


class ModelWidget(QtGui.QWidget, Model):
    plot_classes = [
        (plots.LinePlot, {'d_scalex': 'lin', 'd_scaley': 'log', 'r_scalex': 'lin', 'r_scaley': 'lin',
                          'x_label': 'x', 'y_label': 'y', 'plot_irf': True}),
        (plots.FitInfo, {}), (plots.ParameterScanPlot, {}),
        (plots.ResidualPlot, {})
        #(plots.FitInfo, {}), (plots.AvPlot, {})
    ]

    def update_plots(self, *args, **kwargs):
        for p in self.fit.plots:
            p.update_all(*args, **kwargs)
            p.update()

    def update_widgets(self):
        #[p.update() for p in self._aggregated_parameters if isinstance(p, AggregatedParameters)]
        [p.update() for p in self._parameters if isinstance(p, FittingParameterWidget)]

    def update(self, *__args):
        QtGui.QWidget.update(self, *__args)
        self.update_widgets()
        Model.update(self)
        self.update_plots()

    def __init__(self, fit, **kwargs):
        QtGui.QWidget.__init__(self)
        Model.__init__(self, fit, **kwargs)
        self.plots = list()
        self.icon = kwargs.get('icon', QtGui.QIcon(":/icons/document-open.png"))


import fcs
import tcspc
import stopped_flow
from mfm.fitting.models.structure import proteinMC
import parse
from globalfit import *


