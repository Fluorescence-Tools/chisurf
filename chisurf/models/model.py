from __future__ import annotations
from chisurf import typing

import numpy as np

import chisurf.parameter
import chisurf.curve
import chisurf.gui.widgets.fitting.widgets
import chisurf.plots

from qtpy import QtWidgets, QtGui
from chisurf.fitting.parameter import FittingParameterGroup


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
    def n_free(
            self
    ) -> int:
        return len(self.parameters)

    @property
    def weighted_residuals(
            self
    ) -> np.ndarray:
        return self.get_wres(
            self.fit,
            xmin=self.fit.xmin,
            xmax=self.fit.xmax
        )

    def get_wres(
            self,
            fit: chisurf.fitting.fit.Fit,
            xmin: int = None,
            xmax: int = None
    ) -> np.ndarray:
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
            p = pd[k]
            if isinstance(p, chisurf.fitting.parameter.FittingParameter):
                s += "%s\t%.4e\t%s\t%s\t%s\n" % (p.name, p.value, p.bounds, p.fixed, p.is_linked)
            else:
                chisurf.logging.warning(
                    "The object is of type %s and is not a FittingParameter" % p.__class__.__name__
                )
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
    def x(self) -> np.ndarray:
        return self.__dict__['_x']

    @x.setter
    def x(self,v: np.ndarray):
        self.__dict__['_x'] = v

    @property
    def y(self) -> np.array:
        return self.__dict__['_y']

    @y.setter
    def y(self, v: np.ndarray):
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

    def get_curves(
            self,
            copy_curves: bool = False
    ) -> typing.Dict[str, chisurf.curve.Curve]:
        xmin = self.fit.xmin
        xmax = self.fit.xmax
        return {
            'model': chisurf.curve.Curve(
                x=self.x[xmin:xmax],
                y=self.y[xmin:xmax],
                copy_array=copy_curves
            )
        }

    def __getitem__(
            self,
            key
    ) -> typing.Tuple[
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
            p.update(*args, **kwargs)

    def update_widgets(
            self
    ) -> None:
        for parameter in self.parameters:
            if isinstance(
                    parameter,
                    chisurf.gui.widgets.fitting.widgets.FittingParameterWidget
            ):
                parameter.update()

    def update(
            self
    ) -> None:
        super().update()
        self.update_widgets()
        self.update_plots()

    # def __getattr__(self, item):
    #     try:
    #         return super().__getattr__(item)
    #     except KeyError:
    #         return self.model.__getattr__(item)

    def __init__(
            self,
            fit: chisurf.fitting.fit.FitGroup,
            icon: QtGui.QIcon = None,
            *args,
            **kwargs
    ):
        super().__init__(fit, *args, **kwargs)
        self.plots = list()
        if icon is None:
            icon = QtGui.QIcon(":/icons/document-open.png")
        self.icon = icon
