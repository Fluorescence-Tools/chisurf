from __future__ import annotations

from numpy import *
from re import Scanner

import chisurf.fio
import chisurf.decorators
import chisurf.widgets
import chisurf.parameter
import chisurf.widgets.fitting.widgets
from chisurf.fitting.parameter import FittingParameter, FittingParameterGroup
from chisurf.models.model import ModelWidget, ModelCurve


class ParseModel(
    ModelCurve,
    FittingParameterGroup
):

    name = "Parse-Model"

    @property
    def func(self) -> str:
        return self._func

    @func.setter
    def func(self, v):
        self._func = v
        self.parse_code()

    def parse_code(self):

        def var_found(
                scanner,
                name: str
        ):
            if 'scipy' in name:
                return name
            elif 'numpy' in name:
                return name
            elif 'np' in name:
                return name
            elif name not in self._keys:
                self._keys.append(name)
                ret = 'a[%d]' % self._count
                self._count += 1
            else:
                ret = 'a[%d]' % (self._keys.index(name))
            return ret

        code = self._func
        scanner = Scanner([
            (r"x", lambda y, x: x),
            (r"[a-zA-Z]+\.", lambda y, x: x),
            (r"[a-z]+\(", lambda y, x: x),
            (r"[a-zA-Z_]\w*", var_found),
            (r"\d+\.\d*", lambda y, x: x),
            (r"\d+", lambda y, x: x),
            (r"\+|-|\*|/", lambda y, x: x),
            (r"\s+", None),
            (r"\)+", lambda y, x: x),
            (r"\(+", lambda y, x: x),
            (r",", lambda y, x: x),
        ])
        self._count = 0
        self._keys = list()
        parsed, rubbish = scanner.scan(code)
        parsed = ''.join(parsed)
        if rubbish != '':
            raise Exception('parsed: %s, rubbish %s' % (parsed, rubbish))
        self.code = parsed

        # Define parameters
        self._parameters = list()
        for key in self._keys:
            p = FittingParameter(name=key, value=1.0)
            self._parameters.append(p)

    # def find_parameters(
    #         self,
    #         parameter_type=chisurf.parameter.Parameter
    # ):
    #     # do nothing
    #     pass

    def __init__(
            self,
            fit: chisurf.fitting.fit.Fit = None,
            *args,
            **kwargs,
    ):
        super().__init__(
            fit,
            *args,
            **kwargs
        )
        self._keys = list()
        self._models = dict()
        self._count = 0
        self._func = "x*0"
        self.code = self._func

    def update_model(self, **kwargs):
        a = [p.value for p in self.parameters_all]
        x = self.fit.data.x
        # TODO: better evaluate when the func is set
        y = eval(self.parse.code)
        self.y = y

