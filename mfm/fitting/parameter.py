from __future__ import annotations
from typing import Tuple, List
import deprecation

import numpy as np

import mfm
import mfm.base
import mfm.parameter
import mfm.fitting.widgets
import mfm.decorators

parameter_settings = mfm.settings.cs_settings['parameter']


class FittingParameter(mfm.parameter.Parameter):

    def __init__(
            self,
            link: FittingParameter = None,
            model: mfm.models.model.Model = None,
            lb: float = -10000,
            ub: float = 10000,
            fixed: bool = False,
            bounds_on: bool = False,
            error_estimate: bool = None,
            **kwargs
    ):
        super(FittingParameter, self).__init__(**kwargs)
        self._link = link
        self.model = model

        self._lb = lb
        self._ub = ub
        self._fixed = fixed
        self._bounds_on = bounds_on
        self._error_estimate = error_estimate

        self._chi2s = None
        self._values = None

    @property
    def parameter_scan(self) -> Tuple[np.array, np.array]:
        return self._values, self._chi2s

    @parameter_scan.setter
    def parameter_scan(
            self,
            v: Tuple[np.array, np.array]
    ):
        self._values, self._chi2s = v

    @property
    def error_estimate(self) -> float:
        if self.is_linked:
            return self._link.error_estimate
        else:
            if isinstance(self._error_estimate, float):
                return self._error_estimate
            else:
                return float('nan')

    @error_estimate.setter
    def error_estimate(self, v):
        self._error_estimate = v

    @property
    def bounds(self) -> Tuple[float, float]:
        if self.bounds_on:
            return self._lb, self._ub
        else:
            return None, None

    @bounds.setter
    def bounds(
            self,
            b: Tuple[float, float]
    ):
        self._lb, self._ub = b

    @property
    def bounds_on(self) -> bool:
        return self._bounds_on

    @bounds_on.setter
    def bounds_on(
            self,
            v: bool
    ):
        self._bounds_on = bool(v)

    @property
    def value(self) -> float:
        v = self._value
        if callable(v):
            return v()
        else:
            if self.is_linked:
                return self.link.value
            else:
                if self.bounds_on:
                    lb, ub = self.bounds
                    if lb is not None:
                        v = max(lb, v)
                    if ub is not None:
                        v = min(ub, v)
                    return v
                else:
                    return v

    @value.setter
    def value(self, value):
        self._value = float(value)
        if self.is_linked:
            self.link.value = value

    @property
    def fixed(self) -> bool:
        return self._fixed

    @fixed.setter
    def fixed(
            self,
            v: bool
    ):
        self._fixed = v

    @property
    def link(self):
        return self._link

    @link.setter
    def link(
            self,
            link: mfm.fitting.parameter.FittingParameter
    ):
        if isinstance(link, FittingParameter):
            self._link = link
        elif link is None:
            try:
                self._value = self._link.value
            except AttributeError:
                pass
            self._link = None

    @property
    def is_linked(self) -> bool:
        return isinstance(self._link, FittingParameter)

    def to_dict(self) -> dict:
        d = mfm.parameter.Parameter.to_dict(self)
        d['lb'], d['ub'] = self.bounds
        d['fixed'] = self.fixed
        d['bounds_on'] = self.bounds_on
        d['error_estimate'] = self.error_estimate
        return d

    def from_dict(
            self,
            d: dict
    ):
        mfm.parameter.Parameter.from_dict(self, d)
        self._lb, self._ub = d['lb'], d['ub']
        self._fixed = d['fixed']
        self._bounds_on = d['bounds_on']
        self._error_estimate = d['error_estimate']

    def scan(
            self,
            fit: mfm.fitting.fit.Fit,
            **kwargs) -> None:
        fit.chi2_scan(self.name, **kwargs)

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
        details="use the mfm.fitting.widget.make_fitting_widget function instead"
    )
    def make_widget(
            self,
            **kwargs
    ) -> mfm.fitting.widgets.FittingParameterWidget:
        text = kwargs.get('text', self.name)
        layout = kwargs.get('layout', None)
        update_widget = kwargs.get('update_widget', lambda x: x)
        decimals = kwargs.get('decimals', self.decimals)
        kw = {
            'text': text,
            'decimals': decimals,
            'layout': layout
        }
        widget = mfm.fitting.widgets.FittingParameterWidget(self, **kw)
        self.controller = widget
        return widget


class GlobalFittingParameter(FittingParameter):

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
        args = [f, g, formula]
        super(GlobalFittingParameter, self).__init__(*args, **kwargs)
        self.f, self.g = f, g
        self.formula = formula


class FittingParameterGroup(mfm.base.Base):

    @property
    def parameters_all(self) -> List[mfm.fitting.parameter.FittingParameter]:
        return self._parameters

    @property
    def parameters_all_dict(self):
        return dict([(p.name, p) for p in self.parameters_all])

    @property
    def parameters(self) -> List[mfm.parameter.Parameter]:
        return self.parameters_all

    @property
    def aggregated_parameters(self):
        d = self.__dict__
        a = list()
        for key, value in d.items():
            if isinstance(value, FittingParameterGroup):
                a.append(value)
        return list(set(a))

    @property
    def parameter_dict(self) -> dict:
        re = dict()
        for p in self.parameters:
            re[p.name] = p
        return re

    @property
    def parameter_names(self) -> List[str]:
        return [p.name for p in self.parameters]

    @property
    def parameter_values(self) -> List[float]:
        return [p.value for p in self.parameters]

    @parameter_values.setter
    def parameter_values(
            self,
            vs: List[float]
    ):
        ps = self.parameters
        for i, v in enumerate(vs):
            ps[i].value = v

    @property
    def outputs(self) -> dict:
        """The outputs of a ParameterGroup are a dictionary

        Returns
        -------

        """
        a = dict()
        return a

    def to_dict(self) -> dict:
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
            parameter_type=mfm.parameter.Parameter
    ) -> None:
        self._aggregated_parameters = None
        self._parameters = None

        ag = mfm.base.find_objects(
            self.__dict__.values(), FittingParameterGroup
        )
        self._aggregated_parameters = ag

        ap = list()
        for o in set(ag):
            if not isinstance(o, mfm.models.model.Model):
                o.find_parameters()
                self.__dict__[o.name] = o
                ap += o._parameters

        mp = mfm.base.find_objects(
            self.__dict__.values(), parameter_type
        )
        self._parameters = list(set(mp + ap))

    def append_parameter(
            self,
            p: mfm.parameter.Parameter
    ):
        self._parameters.append(p)

    def finalize(self):
        pass

    def __getattr__(self, item):
        item = mfm.base.Base.__getattr__(self, item)
        if isinstance(item, mfm.parameter.Parameter):
            return item.value
        else:
            return item

    def __len__(self):
        return len(self.parameters_all)

    def __init__(
            self,
            fit: mfm.fitting.fit.Fit = None,
            model: mfm.models.model.Model = None,
            short: str = '',
            parameters: List[mfm.fitting.parameter.FittingParameter] = None,
            *args, **kwargs):
        """

        :param fit: the fit to which the parameter group is associated to
        :param model: the model to which the parameter group is associated to
        :param short: a short name for the parameter group
        :param parameters: a list of the fitting parameters that are grouped by the fitting parameter group
        :param args:
        :param kwargs:
        """
        super(FittingParameterGroup, self).__init__(*args, **kwargs)
        if mfm.verbose:
            print("---------------")
            print("Class: %s" % self.__class__.name)
            print(kwargs)
            print("---------------")
        # super(mfm.Base, self).__init__(*args, **kwargs)
        self.short = short
        self.model = model
        self.fit = fit

        if parameters is None:
            parameters = list()
        self._parameters = parameters
        self._aggregated_parameters = list()
        self._parameter_names = None

        # Copy parameters from provided ParameterGroup
        if len(args) > 0:
            p0 = args[0]
            if isinstance(p0, FittingParameterGroup):
                self.__dict__ = p0.__dict__

    @deprecation.deprecated(
        deprecated_in="19.08.23",
        removed_in="20.01.01",
        current_version="19.08.23",
        details="use the mfm.fitting.widget.make_fitting_parameter_group_widget function instead"
    )
    def to_widget(self, *args, **kwargs) -> mfm.fitting.widgets.FittingParameterGroupWidget:
        return mfm.fitting.widgets.FittingParameterGroupWidget(self, *args, **kwargs)


