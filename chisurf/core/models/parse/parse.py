from __future__ import annotations

from numpy import *
from re import Scanner

import chisurf.core.fio
import chisurf.core.decorators
import chisurf.core.parameter
from chisurf.core.fitting.parameter import FittingParameter, FittingParameterGroup
from chisurf.core.models.model import ModelCurve


class ParseModel(ModelCurve, FittingParameterGroup):

    name = "Parse-Model"

    @property
    def func(self) -> str:
        """The equation string that defines the model."""
        return self._func

    @func.setter
    def func(self, v):
        """Set the equation string and trigger parsing."""
        self._func = v
        self.parse_code()

    def parse_code(self):
        """Parse the equation string and create fitting parameters for variables."""
        def var_found(scanner, name: str):
            """Handle a variable name found by the scanner.

            Parameters
            ----------
            scanner : Scanner
                The scanner instance.
            name : str
                The variable name found.

            Returns
            -------
            str
                The replacement token (``a[index]``) or the original name.
            """
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
        self._parameters_equation.clear()
        for key in self._keys:
            p = FittingParameter(name=key, value=1.0)
            self._parameters_equation.append(p)
        self.find_parameters()

    def __init__(
            self,
            fit: chisurf.core.fitting.fit.Fit = None,
            *args,
            **kwargs,
    ):
        """Initialize the ParseModel.

        Parameters
        ----------
        fit : chisurf.core.fitting.fit.Fit, optional
            Fit object this model is attached to.
        """
        super().__init__(fit,*args, **kwargs)
        self._keys = list()
        self._models = dict()
        self._count = 0
        self._func = "x*0"
        self._parameters_equation = list()
        self._func_listeners = []
        self.code = self._func

    def update_model(self, **kwargs):
        """Evaluate the parsed equation and update the model curve."""
        super().update_model(**kwargs)
        a = [p.value for p in self._parameters_equation]
        x = self.fit.data.x
        # TODO: better evaluate when the func is set
        y = eval(self.code)
        self.y = y

