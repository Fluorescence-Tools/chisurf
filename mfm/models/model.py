from __future__ import annotations
from typing import List, Tuple

import numpy as np
from qtpy import  QtWidgets, QtGui

import mfm.curve
import mfm.fitting.parameter
import mfm.fitting.fitting_widgets
import mfm.plots
from mfm.curve import Curve


class Model(mfm.fitting.parameter.FittingParameterGroup):

    def __init__(self, fit,  **kwargs):
        super(Model, self).__init__(model=self, **kwargs)
        self.fit = fit
        self.flatten_weighted_residuals = True
        self.model_number = kwargs.get('model_number', 0)

    @property
    def x(self) -> np.array:
        return self._kw['x']

    @x.setter
    def x(
            self,
            v: np.array
    ):
        self._kw['x'] = v

    @property
    def y(self) -> np.array:
        return self._kw['y']

    @y.setter
    def y(
            self,
            v: np.array
    ):
        self._kw['y'] = v

    @property
    def parameters(self) -> List[mfm.fitting.parameter.FittingParameter]:
        return [p for p in self.parameters_all if not (p.fixed or p.is_linked)]

    @property
    def parameter_bounds(
            self
    ) -> List[
        Tuple[float, float]
    ]:
        return [pi.bounds for pi in self.parameters]

    @property
    def n_free(self) -> int:
        return len(self.parameters)

    def finalize(self):
        self.update()
        for a in self.aggregated_parameters:
            if a is not self:
                a.finalize()
        for pa in mfm.fitting.parameter.FittingParameter.getinstances():
            pa.finalize()
        #for p in self.parameters_all:
        #    p.finalize()

    @property
    def weighted_residuals(self) -> np.array:
        return self.get_wres(self.fit)

    @property
    def outputs(self):
        out = {
            'wres': self.weighted_residuals,
            'x': self.x,
            'y': self.y
        }
        return out

    def get_wres(
            self,
            fit: mfm.fitting.fit.Fit,
            xmin: int = None,
            xmax: int = None,
            **kwargs
    ) -> np.array:
        if xmin is None:
            xmin = fit.xmin
        if xmax is None:
            xmax = fit.xmax
        x, m = fit.model[xmin:xmax]
        _, d, e = fit.data[xmin:xmax]
        ml = min([len(m), len(d)])
        wr = np.array((d[:ml] - m[:ml]) / e[:ml], dtype=np.float64)
        return wr

    def update_model(self, **kwargs):
        pass

    def update(self):
        #self.find_parameters()
        self.update_model()

    def __str__(self):
        s = ""
        s += "Model: %s\n" % str(self.name)

        pd = self.parameters_all_dict
        keylist = list(pd.keys())
        keylist.sort()

        s += "Parameter\tValue\tBounds\tFixed\tLinked\n"
        for k in keylist:
            p = (pd[k].name, pd[k].value, pd[k].bounds, pd[k].fixed, pd[k].is_linked)
            s += "%s\t%.4e\t%s\t%s\t%s\n" % p
        return s


class ModelCurve(Model, mfm.curve.Curve):

    @property
    def n_points(self) -> int:
        return self.fit.xmax - self.fit.xmin

    def __init__(
            self,
            fit: mfm.fitting.fit.Fit,
            *args, **kwargs
    ):
        x = fit.data.x
        y = np.zeros_like(fit.data.y)
        Model.__init__(self, fit, **kwargs)
        Curve.__init__(
            self,
            *args,
            x=x,
            y=y,
            **kwargs
        )

    def __getitem__(self, key):
        start = key.start
        stop = key.stop
        step = 1 if key.step is None else key.step
        x, y = self.x[start:stop:step], self.y[start:stop:step]
        return x, y


class ModelWidget(Model, QtWidgets.QWidget):

    plot_classes = [
        (
            mfm.plots.LinePlot, {
                'scale_x': 'lin',
                'd_scaley': 'log',
                'r_scaley': 'lin',
                'x_label': 'x',
                'y_label': 'y',
                'plot_irf': True
            }
        ),
        (mfm.plots.FitInfo, {}),
        (mfm.plots.ParameterScanPlot, {}),
        (mfm.plots.ResidualPlot, {})
        #(plots.FitInfo, {}), (plots.AvPlot, {})
    ]

    def update_plots(self, *args, **kwargs):
        for p in self.fit.plots:
            p.update_all(*args, **kwargs)
            p.update()

    def update_widgets(self):
        #[p.update() for p in self._aggregated_parameters if isinstance(p, AggregatedParameters)]
        [p.update() for p in self._parameters if isinstance(p, mfm.fitting.fitting_widgets.FittingParameterWidget)]

    def update(self, *args, **kwargs):
        QtWidgets.QWidget.update(self, *args)
        self.update_widgets()
        Model.update(self)
        self.update_plots()

    def __getattr__(self, item):
        pass

    def __init__(self, fit, *args, **kwargs):
        super(ModelWidget, self).__init__(fit, *args, **kwargs)
        self.plots = list()
        self.icon = kwargs.get('icon', QtGui.QIcon(":/icons/document-open.png"))
