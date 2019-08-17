from __future__ import annotations

import weakref
from typing import List
from PyQt5 import QtWidgets

import mfm
import mfm.base
import mfm.fitting.models
import mfm.fitting.fit

parameter_settings = mfm.cs_settings['parameter']


class Parameter(mfm.base.Base):

    _instances = set()

    @property
    def decimals(self) -> float:
        """
        The number of decimals that are displayed
        :return:
        """
        return self._decimals

    @decimals.setter
    def decimals(
            self,
            v: float
    ):
        self._decimals = v

    @property
    def value(self):
        v = self._value
        if callable(v):
            return v()
        else:
            return v

    @value.setter
    def value(self, value):
        self._value = float(value)

    @classmethod
    def getinstances(cls):
        dead = set()
        for ref in cls._instances:
            obj = ref()
            if obj is not None:
                yield obj
            else:
                dead.add(ref)
        cls._instances -= dead

    def __add__(
            self,
            other
    ):
        if isinstance(other, (int, float)):
            a = self.value + other
        else:
            a = self.value + other.value
        return Parameter(value=a)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            a = self.value * other
        else:
            a = self.value * other.value
        return Parameter(value=a)

    def __sub__(self, other):
        if isinstance(other, (int, float)):
            a = self.value - other
        else:
            a = self.value - other.value
        return Parameter(value=a)

    def __div__(self, other):
        if isinstance(other, (int, float)):
            a = self.value / other
        else:
            a = self.value / other.value
        return Parameter(value=a)

    def __float__(self):
        return float(self.value)

    def __invert__(self):
        return Parameter(value=float(1.0 / self.value))

    def __init__(self, *args, **kwargs):
        self.controller = None
        self._instances.add(weakref.ref(self))
        super(Parameter, self).__init__(*args, **kwargs)
        self._link = kwargs.get('link', None)
        self.model = kwargs.get('model', None)
        value = args[0] if len(args) > 0 else 1.0
        self._value = kwargs.get('value', value)
        self._decimals = kwargs.get('decimals', mfm.cs_settings['parameter']['decimals'])

    def to_dict(self) -> dict:
        v = mfm.Base.to_dict(self)
        v['value'] = self.value
        v['decimals'] = self.decimals
        return v

    def from_dict(
            self,
            v: dict
    ):
        mfm.base.Base.from_dict(self, v)
        self._value = v['value']
        self._decimals = v['decimals']

    def update(self):
        pass

    def finalize(self):
        if self.controller:
            self.controller.finalize()


"""
class ParameterGroup(mfm.base.Base):

    def __init__(
            self,
            fit: mfm.fitting.fit.Fit,
            **kwargs
    ):
        super(ParameterGroup, self).__init__(**kwargs)
        self.fit = fit
        self._activeRuns = list()
        self._chi2 = list()
        self._parameter = list()
        self.parameter_names = list()

    def clear(self):
        self._chi2 = list()
        self._parameter = list()

    def save_txt(
            self,
            filename: str,
            sep: str = '\t'
    ):
        fp = open(filename, 'w')
        s = ""
        for ph in self.parameter_names:
            s += ph + sep
        s += "\n"
        for l in self.values.T:
            for p in l:
                s += "%.5f%s" % (p, sep)
            s += "\n"
        fp.write(s)
        fp.close()

    @property
    def values(self) -> np.array:
        try:
            re = np.vstack(self._parameter)
            re = np.column_stack((re, self.chi2s))
            return re.T
        except ValueError:
            return np.array([[0], [0]]).T

    @property
    def chi2s(self) -> np.array:
        return np.hstack(self._chi2)
"""


class FittingParameterGroup(mfm.base.Base):

    @property
    def parameters_all(self):
        return self._parameters

    @property
    def parameters_all_dict(self):
        return dict([(p.name, p) for p in self.parameters_all])

    @property
    def parameters(self):
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
    def parameter_dict(self):
        re = dict()
        for p in self.parameters:
            re[p.name] = p
        return re

    @property
    def parameter_names(self):
        return [p.name for p in self.parameters]

    @property
    def parameter_values(self):
        return [p.value for p in self.parameters]

    @parameter_values.setter
    def parameter_values(self, vs):
        ps = self.parameters
        for i, v in enumerate(vs):
            ps[i].value = v

    @property
    def outputs(self):
        """The outputs of a ParameterGroup are a dictionary
        
        Returns
        -------
        
        """
        a = dict()
        return a

    def to_dict(self):
        s = dict()
        parameters = dict()
        s['parameter'] = parameters
        for parameter in self._parameters:
            parameters[parameter.name] = parameter.to_dict()
        return s

    def from_dict(self, v):
        self.find_parameters()
        parameter_target = self.parameters_all_dict
        parameter = v['parameter']
        for parameter_name in parameter:
            pn = str(parameter_name)
            try:
                parameter_target[pn].from_dict(parameter[pn])
            except KeyError:
                print("Key %s not found skipping" % pn)

    def find_parameters(self, parameter_type=Parameter):
        self._aggregated_parameters = None
        self._parameters = None

        ag = mfm.find_objects(self.__dict__.values(), FittingParameterGroup)
        self._aggregated_parameters = ag

        ap = list()
        for o in set(ag):
            if not isinstance(o, mfm.fitting.models.Model):
                o.find_parameters()
                self.__dict__[o.name] = o
                ap += o._parameters

        mp = mfm.find_objects(self.__dict__.values(), parameter_type)
        self._parameters = list(set(mp + ap))

    def append_parameter(self, p):
        if isinstance(p, Parameter):
            self._parameters.append(p)

    def finalize(self):
        pass

    def __getattr__(self, item):
        item = mfm.base.Base.__getattr__(self, item)
        if isinstance(item, Parameter):
            return item.value
        else:
            return item

    def __len__(self):
        return len(self.parameters_all)

    def __init__(
            self,
            fit: mfm.fitting.Fit = None,
            model: mfm.fitting.models.Model = None,
            short: str = '',
            parameters: List[mfm.parameter.Parameter] = list(),
            *args, **kwargs):
        """

        :param fit: the fit to which the parameter group is associated to
        :param model: the model to which the parameter group is associated to
        :param short: a short name for the parameter group
        :param parameters: a list of the fitting parameters that are grouped by the fitting parameter group
        :param args:
        :param kwargs:
        """
        if mfm.verbose:
            print("---------------")
            print("Class: %s" % self.__class__.name)
            print(kwargs)
            print("---------------")
        #super(mfm.Base, self).__init__(*args, **kwargs)
        self.short = short
        self.model = model
        self.fit = fit
        self._parameters = parameters
        self._aggregated_parameters = list()
        self._parameter_names = None

        # Copy parameters from provided ParameterGroup
        if len(args) > 0:
            p0 = args[0]
            if isinstance(p0, FittingParameterGroup):
                self.__dict__ = p0.__dict__

    def to_widget(self, *args, **kwargs) -> mfm.parameter.ParameterGroupWidget:
        return ParameterGroupWidget(self, *args, **kwargs)


class ParameterGroupWidget(FittingParameterGroup, QtWidgets.QGroupBox):

    def __init__(self, parameter_group, *args, **kwargs):
        self.parameter_group = parameter_group
        super(ParameterGroupWidget, self).__init__(*args, **kwargs)
        self.setTitle(self.name)

        self.n_col = kwargs.get('n_cols', mfm.cs_settings['gui']['fit_models']['n_columns'])
        self.n_row = 0
        l = QtWidgets.QGridLayout()
        l.setSpacing(0)
        l.setContentsMargins(0, 0, 0, 0)
        self.setLayout(l)

        for i, p in enumerate(parameter_group.parameters_all):
            pw = p.make_widget()
            col = i % self.n_col
            row = i // self.n_col
            l.addWidget(pw, row, col)


