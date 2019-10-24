from __future__ import annotations
from typing import List, Tuple

import numpy as np
from qtpy import QtWidgets, QtGui

import chisurf.parameter
import chisurf.curve
from chisurf.fitting.parameter import FittingParameterGroup
import chisurf.fitting.widgets
import chisurf.plots


class Model(
    FittingParameterGroup
):
    name = "Model name not available"

    def __init__(
            self,
            fit: chisurf.fitting.fit.Fit,
            model_number: int = 0,
            **kwargs
    ):
        super().__init__(
            model=self,
            **kwargs
        )
        self.fit = fit
        self.flatten_weighted_residuals = True
        self.model_number = model_number

    @property
    def parameters(
            self
    ) -> List[
        chisurf.fitting.parameter.FittingParameter
    ]:
        return [p for p in self.parameters_all if not (p.fixed or p.is_linked)]

    @property
    def parameter_bounds(
            self
    ) -> List[
        Tuple[float, float]
    ]:
        return [pi.bounds for pi in self.parameters]

    @property
    def n_free(
            self
    ) -> int:
        return len(self.parameters)

    def finalize(self):
        self.update()
        for a in self.aggregated_parameters:
            if a is not self:
                a.finalize()
        #for pa in fitting.parameter.FittingParameter.get_instances():
        #    pa.finalize()

    @property
    def weighted_residuals(
            self
    ) -> np.array:
        return self.get_wres(
            self.fit,
            xmin=self.fit.xmin,
            xmax=self.fit.xmax
        )

    def get_wres(
            self,
            fit: chisurf.fitting.fit.Fit,
            xmin: int = None,
            xmax: int = None,
            **kwargs
    ) -> np.array:
        if xmin is None:
            xmin = fit.xmin
        if xmax is None:
            xmax = fit.xmax
        return chisurf.fitting.calculate_weighted_residuals(
            fit.data,
            fit.model,
            xmin=xmin,
            xmax=xmax
        )

    def update_model(
            self,
            **kwargs
    ):
        pass

    def update(
            self,
            **kwargs
    ) -> None:
        self.find_parameters()
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


class ModelCurve(
    Model,
    chisurf.curve.Curve
):

    @property
    def n_points(
            self
    ) -> int:
        return self.fit.xmax - self.fit.xmin

    @property
    def x(self) -> np.array:
        return self.__dict__['_x']

    @x.setter
    def x(
            self,
            v: np.array
    ):
        self.__dict__['_x'] = v

    @property
    def y(self) -> np.array:
        return self.__dict__['_y']

    @y.setter
    def y(
            self,
            v: np.array
    ):
        self.__dict__['_y'] = v

    def __init__(
            self,
            fit: chisurf.fitting.fit.Fit,
            *args, **kwargs
    ):
        super().__init__(
            fit,
            *args,
            **kwargs
        )
        chisurf.curve.Curve.__init__(
            self,
            x=fit.data.x,
            y=np.zeros_like(fit.data.y),
            *args,
            **kwargs
        )

    def __getitem__(
            self,
            key
    ) -> Tuple[
        np.ndarray,
        np.ndarray
    ]:
        start = key.start
        stop = key.stop
        step = 1 if key.step is None else key.step
        x, y = self.x[start:stop:step], self.y[start:stop:step]
        return x, y


class ModelWidget(
    Model,
    QtWidgets.QWidget
):

    plot_classes = [
        (
            chisurf.plots.LinePlot, {
                'scale_x': 'lin',
                'd_scaley': 'log',
                'r_scaley': 'lin',
                'x_label': 'x',
                'y_label': 'y',
                'plot_irf': True
            }
        ),
        (chisurf.plots.FitInfo, {}),
        (chisurf.plots.ParameterScanPlot, {}),
        (chisurf.plots.ResidualPlot, {})
    ]

    def update_plots(
            self,
            *args,
            **kwargs
    ) -> None:
        for p in self.fit.plots:
            p.update_all(
                *args,
                **kwargs
            )
            p.update()

    def update_widgets(
            self
    ) -> None:
        for parameter in self.parameters:
            if isinstance(
                    parameter,
                    chisurf.fitting.widgets.FittingParameterWidget
            ):
                parameter.update()

    def update(
            self
    ) -> None:
        super().update()
        self.update_widgets()
        self.update_plots()

    def __getattr__(self, item):
        pass

    def __init__(
            self,
            fit: chisurf.fitting.fit.FitGroup,
            icon: QtGui.QIcon = None,
            *args,
            **kwargs
    ):
        super().__init__(
            fit,
            *args,
            **kwargs
        )
        self.plots = list()
        if icon is None:
            icon = QtGui.QIcon(":/icons/document-open.png")
        self.icon = icon
