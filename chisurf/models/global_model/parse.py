from __future__ import annotations
from chisurf import typing

import numpy as np
import chinet as cn

import chisurf.decorators
import chisurf.parameter
import chisurf.fitting.fit

from chisurf.curve import Curve
from chisurf.models import model
from chisurf.fitting.parameter import GlobalFittingParameter


class ParameterTransformModel(model.Model):

    name = "Parameter Transform"

    def finalize(self):
        self.update_model()
        for p in self.parameters_all:
            p.controller.finalize()

    def update_model(self, **kwargs):
        self._model._node.evaluate()

    @property
    def n_points(self):
        return 1

    @property
    def n_free(self):
        return 0

    @property
    def weighted_residuals(self) -> np.ndarray:
        return np.array([], dtype=np.float64)

    @property
    def function(self) -> str:
        return self._function

    @function.setter
    def function(self, fun: str):
        self._function = fun
        m = chisurf.models.function_to_model_decorator(name=self.name)
        self._model = m(fun)(self.fit)

    @property
    def _parameters(self) -> typing.List[chisurf.fitting.parameter.FittingParameter]:
        return self._model.parameters_all

    @_parameters.setter
    def _parameters(self, v):
        pass

    def __init__(
            self,
            fit: chisurf.fitting.fit.Fit,
            function: typing.Callable = None,
            *args,
            **kwargs
    ):
        if function is None:
            function = 'def f(x): return x'
        self.fit = fit
        self.function = function
        super().__init__(fit, *args, **kwargs)

    def __str__(self):
        s = "\n"
        s += "Model: Parameter transform\n"
        s += "\n"
        s += "Function:\n"
        s += str(self._function)
        s += "\n"
        s += "Parameter \t Value \t Bounds \t Output \t Linked\n"
        for p in self.parameters_all:
            s += f"{p.name} \t {p.value:.4f} \t {p.bounds} \t {p.fixed} \t {p.is_linked} \n"
        s += "\n"
        return s

    def __getitem__(self, key):
        start = key.start
        stop = key.stop
        step = 1 if key.step is None else key.step
        return self.x[start:stop:step], self.y[start:stop:step]
