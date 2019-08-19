from __future__ import annotations

import weakref
from typing import List
from PyQt5 import QtWidgets

import mfm
import mfm.base

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


