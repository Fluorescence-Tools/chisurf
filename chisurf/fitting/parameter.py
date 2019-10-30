"""

"""
from __future__ import annotations
from typing import Tuple, List
import deprecation

from qtpy import QtWidgets

import numpy as np

import chisurf.settings
import chisurf.fitting
import chisurf.base
import chisurf.parameter
import chisurf.decorators
import chisurf.models.model

parameter_settings = chisurf.settings.parameter


class FittingParameter(
    chisurf.parameter.Parameter
):

    def __init__(
            self,
            model: chisurf.models.Model = None,
            fixed: bool = False,
            *args,
            **kwargs
    ):
        """

        :param model:
        :param fixed:
        :param args:
        :param kwargs:
        """
        super().__init__(
            *args,
            **kwargs
        )
        self.model = model
        self._fixed = fixed
        self._error_estimate = None
        self._chi2s = None
        self._values = None

    @property
    def parameter_scan(
            self
    ) -> Tuple[
        np.array,
        np.array
    ]:
        return self._values, self._chi2s

    @parameter_scan.setter
    def parameter_scan(
            self,
            v: Tuple[np.array, np.array]
    ):
        self._values, self._chi2s = v

    @property
    def error_estimate(
            self
    ) -> float:
        if self.is_linked:
            return self._link.error_estimate
        else:
            if isinstance(self._error_estimate, float):
                return self._error_estimate
            else:
                return float('nan')

    @error_estimate.setter
    def error_estimate(
            self,
            v: float
    ):
        self._error_estimate = v

    @property
    def fixed(
            self
    ) -> bool:
        return self._fixed

    @fixed.setter
    def fixed(
            self,
            v: bool
    ):
        self._fixed = v

    def scan(
            self,
            fit: chisurf.fitting.fit.Fit,
            rel_range: float = None,
            **kwargs
    ) -> None:
        fit.chi2_scan(
            parameter_name=self.name,
            rel_range=rel_range,
            **kwargs
        )

    def __str__(self):
        s = "\nVariable\n"
        s += "name: %s\n" % self.name
        s += "internal-value: %s\n" % self._value
        if self.bounds_on:
            s += "bounds: %s\n" % self.bounds
        if self.is_linked:
            s += "linked to: %s\n" % self.link.name
            s += "link-value: %s\n" % self.value
        return s

    @deprecation.deprecated(
        deprecated_in="19.08.23",
        removed_in="20.01.01",
        current_version="19.08.23",
        details="use the fitting.widget.make_fitting_widget function instead"
    )
    def make_widget(
            self,
            text: str = None,
            layout: QtWidgets.QLayout = None,
            decimals: int = 4,
            **kwargs
    ) -> chisurf.fitting.widgets.FittingParameterWidget:
        if text is None:
            text = self.name
        update_widget = kwargs.get('update_widget', lambda x: x)
        kw = {
            'name': text,
            'decimals': decimals,
            'layout': layout
        }
        widget = chisurf.fitting.widgets.FittingParameterWidget(
            self,
            **kw
        )
        self.controller = widget
        return widget


class GlobalFittingParameter(
    FittingParameter
):
    """

    """

    @property
    def value(self) -> float:
        g = self.g
        f = self.f
        r = eval(self.formula)
        return r.value

    @value.setter
    def value(
            self,
            v: float
    ):
        pass

    @property
    def name(self) -> str:
        return self.formula

    @name.setter
    def name(
            self,
            v: str
    ):
        pass

    def __init__(
            self,
            f,
            g,
            formula,
            **kwargs
    ):
        """

        :param f:
        :param g:
        :param formula:
        :param kwargs:
        """
        args = [f, g, formula]
        super().__init__(*args, **kwargs)
        self.f, self.g = f, g
        self.formula = formula


class FittingParameterGroup(
    chisurf.parameter.ParameterGroup
):
    """

    """

    @property
    def parameter_bounds(
            self
    ) -> List[
        Tuple[float, float]
    ]:
        return [pi.bounds for pi in self.parameters]

    @property
    def parameters_all(
            self
    ) -> List[
        chisurf.fitting.parameter.FittingParameter
    ]:
        return self._parameters

    @property
    def parameters(
            self
    ) -> List[
        chisurf.fitting.parameter.FittingParameter
    ]:
        return [
            p for p in self.parameters_all if not (p.fixed or p.is_linked)
        ]

    @property
    def parameters_all_dict(self):
        return dict([(p.name, p) for p in self.parameters_all])

    @property
    def parameters_dict(self):
        return dict([(p.name, p) for p in self.parameters])

    @property
    def aggregated_parameters(self):
        a = list()
        for value in self.__dict__.values():
            if isinstance(value, FittingParameterGroup):
                a.append(value)
        return list(set(a))

    @property
    def parameter_dict(
            self
    ) -> dict:
        re = dict()
        for p in self.parameters:
            re[p.name] = p
        return re

    @property
    def parameter_names(
            self
    ) -> List[str]:
        return [p.name for p in self.parameters]

    @property
    def parameter_values(
            self
    ) -> List[float]:
        return [p.value for p in self.parameters]

    @parameter_values.setter
    def parameter_values(
            self,
            vs: List[float]
    ):
        ps = self.parameters
        for i, v in enumerate(vs):
            ps[i].value = v

    def to_dict(
            self
    ) -> dict:
        s = dict()
        parameters = dict()
        s['parameter'] = parameters
        for parameter in self._parameters:
            parameters[parameter.name] = parameter.to_dict()
        return s

    def from_dict(
            self,
            v: dict
    ):
        self.find_parameters()
        parameter_target = self.parameters_all_dict
        parameter = v['parameter']
        for parameter_name in parameter:
            pn = str(parameter_name)
            try:
                parameter_target[pn].from_dict(parameter[pn])
            except KeyError:
                print("Key %s not found skipping" % pn)

    def find_parameters(
            self,
            parameter_type=chisurf.parameter.Parameter
    ) -> None:
        """

        :param parameter_type:
        :return:
        """
        self._aggregated_parameters = None
        self._parameters = None
        d = [v for v in self.__dict__.values() if v is not self]
        ag = chisurf.base.find_objects(
            d,
            chisurf.fitting.parameter.FittingParameterGroup
        )
        self._aggregated_parameters = ag

        ap = list()
        for o in set(ag):
            if not isinstance(
                    o,
                    chisurf.models.Model
            ):
                o.find_parameters()
                self.__dict__[o.name] = o
                ap += o.parameters_all

        mp = chisurf.base.find_objects(
            d, parameter_type
        )
        self._parameters = list(set(mp + ap))

    def append_parameter(
            self,
            p: chisurf.parameter.Parameter
    ):
        self._parameters.append(p)

    def finalize(self):
        pass

    # def __getattribute__(
    #         self,
    #         item_key
    # ):
    #     item = chisurf.base.Base.__getattribute__(
    #         self,
    #         item_key
    #     )
    #     if isinstance(
    #             item,
    #             chisurf.parameter.Parameter
    #     ):
    #         return item.value
    #     else:
    #         return item

    def __len__(self):
        return len(self.parameters_all)

    def __init__(
            self,
            fit: chisurf.fitting.fit.Fit = None,
            model: chisurf.models.Model = None,
            short: str = '',
            parameters: List[
                chisurf.fitting.parameter.FittingParameter
            ] = None,
            *args, **kwargs):
        """

        :param fit: the fit to which the parameter group is associated to
        :param model: the model to which the parameter group is associated to
        :param short: a short name for the parameter group
        :param parameters: a list of the fitting parameters that are grouped
        by the fitting parameter group
        :param args:
        :param kwargs:
        """
        super().__init__(
            *args,
            **kwargs
        )
        if chisurf.verbose:
            print("---------------")
            print("Class: %s" % self.__class__.name)
            print(kwargs)
            print("---------------")

        self.short = short
        self.model = model
        self.fit = fit

        if parameters is None:
            parameters = list()
        self._parameters = parameters
        self._aggregated_parameters = list()

        # Copy parameters from provided ParameterGroup
        if len(args) > 0:
            p0 = args[0]
            if isinstance(p0, FittingParameterGroup):
                self.__dict__ = p0.__dict__

    @deprecation.deprecated(
        deprecated_in="19.08.23",
        removed_in="20.01.01",
        current_version="19.08.23",
        details="use the fitting.widget.make_fitting_parameter_group_widget function instead"
    )
    def to_widget(
            self,
            *args,
            **kwargs
    ) -> chisurf.fitting.widgets.FittingParameterGroupWidget:
        return chisurf.fitting.widgets.FittingParameterGroupWidget(
            self, *args,
            **kwargs
        )


