from __future__ import annotations

import abc

from chisurf import typing

from collections import OrderedDict

import numpy as np

import chisurf.parameter
import chisurf.curve
import chisurf.plots

from qtpy import QtWidgets, QtGui
from chisurf.fitting.parameter import FittingParameterGroup


class Model(FittingParameterGroup):

    name = "Model name not available"

    @property
    def n_free(self) -> int:
        return len(self.parameters)

    @property
    def weighted_residuals(self) -> np.ndarray:
        return self.get_wres(
            self.fit,
            xmin=self.fit.xmin,
            xmax=self.fit.xmax
        )

    @abc.abstractmethod
    def update_model(self, **kwargs):
        pass

    @abc.abstractmethod
    def update(self, **kwargs) -> None:
        self.find_parameters()

        # Update ParameterGroups
        d = [v for v in self.__dict__.values() if v is not self]
        pgs = chisurf.base.find_objects(
            search_iterable=d,
            searched_object_type=chisurf.fitting.parameter.FittingParameterGroup
        )
        for pg in pgs:
            try:
                pg.update()
            except:
                continue

        self.update_model()

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

    def __init__(self, fit: chisurf.fitting.fit.Fit, model_number: int = 0, **kwargs):
        # Set model to none otherwise will result in self reference
        super().__init__(model=None, **kwargs)
        self.fit = fit
        self.flatten_weighted_residuals = True
        self.model_number = model_number

    def __getstate__(self):
        state = super().__getstate__()
        return state
    
    def __setstate__(self, state):
        super().__setstate__(state)
        model = self
        for key in state:
            if key in model.parameters_all_dict.keys():
                target = model.parameters_all_dict.get(key)
                target.__setstate__(state[key])

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
                s += f"{p.name}\t{p.value:.4e}\t{p.bounds}\t{p.fixed}\t{p.is_linked}\n"
            else:
                chisurf.logging.warning("The object is of type %s and is not a FittingParameter" % p.__class__.__name__)
        return s


class ModelCurve(Model, chisurf.curve.Curve):

    @property
    def n_points(self) -> int:
        return self.fit.xmax - self.fit.xmin

    @property
    def x(self) -> np.ndarray:
        return self.__dict__['d'][0]

    @x.setter
    def x(self,v: np.ndarray):
        self.__dict__['d'][0] = v

    @property
    def y(self) -> np.array:
        return self.__dict__['d'][1]

    @y.setter
    def y(self, v: np.ndarray):
        self.__dict__['d'][1] = v

    def __init__(self, fit: chisurf.fitting.fit.Fit, *args, **kwargs):
        super().__init__(fit, *args, **kwargs)
        if fit.data.x is None:
            x = np.array([], dtype=np.float64)
        else:
            x = fit.data.x
        chisurf.curve.Curve.__init__(
            self,
            x=x, y=np.zeros_like(x),
            *args,
            **kwargs
        )

    def get_curves(self, copy_curves: bool = False) -> typing.OrderedDict[str, chisurf.curve.Curve]:
        #xmin = self.fit.xmin
        #xmax = self.fit.xmax
        d = OrderedDict()
        #d['model'] = chisurf.curve.Curve(x=self.x[xmin:xmax], y=self.y[xmin:xmax], copy_array=copy_curves)
        d['model'] = chisurf.curve.Curve(x=self.x, y=self.y, copy_array=copy_curves)
        return d

    def __getitem__(self, key) -> typing.Tuple[np.ndarray, np.ndarray]:
        start = key.start
        stop = key.stop
        step = 1 if key.step is None else key.step
        x, y = self.x[start:stop:step], self.y[start:stop:step]
        return x, y


class ModelWidget(Model, QtWidgets.QWidget):

    plot_classes = [
        (
            chisurf.plots.LinePlot, {
                'scale_x': 'lin',
                'd_scaley': 'log',
                'r_scaley': 'lin',
                'x_label': 'x',
                'y_label': 'y'
            }
        ),
        (chisurf.plots.FitInfo, {}),
        (chisurf.plots.ParameterScanPlot, {}),
        (chisurf.plots.ResidualPlot, {})
    ]

    def update_plots(self, *args, **kwargs) -> None:
        for p in self.fit.plots:
            p.update(*args, **kwargs)

    @abc.abstractmethod
    def update_widgets(self) -> None:
        for parameter in self.parameters:
            parameter.update()

    @abc.abstractmethod
    def update(self) -> None:
        super().update()
        self.update_widgets()
        self.update_plots()

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
