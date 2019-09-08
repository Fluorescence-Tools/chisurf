from __future__ import annotations
from typing import List, TypeVar, Tuple

import weakref
import numpy as np

import mfm
import mfm.base

T = TypeVar('T', bound='Parameter')


class Parameter(mfm.base.Base):

    _instances = set()

    @property
    def value(self) -> float:
        v = self._value
        if callable(v):
            return v()
        else:
            return v

    @value.setter
    def value(
            self,
            value: float
    ):
        self._value = float(value)

    @classmethod
    def getinstances(cls) -> List[Parameter]:
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
            other: T
    ) -> Parameter:
        if isinstance(other, (int, float)):
            a = self.value.__add__(other)
        else:
            a = self.value.__add__(other.value)
        return Parameter(value=a)

    def __mul__(
            self,
            other: T
    ) -> Parameter:
        if isinstance(other, (int, float)):
            a = self.value * other
        else:
            a = self.value * other.value
        return Parameter(value=a)

    def __sub__(
            self,
            other: T
    ) -> Parameter:
        if isinstance(other, (int, float)):
            a = self.value.__sub__(other)
        else:
            a = self.value.__sub__(other.value)
        return Parameter(value=a)

    def __divmod__(
            self,
            other: T
    ) -> Tuple[Parameter, Parameter]:
        if isinstance(other, (int, float)):
            d, m = self.value.__divmod__(other)
        else:
            d, m = self.value.__divmod__(other.value)
        return Parameter(value=d), Parameter(value=m)

    def __invert__(self) -> Parameter:
        return Parameter(
            value=float(1.0 / self.value)
        )

    def __float__(self):
        return float(self.value)

    def __eq__(
            self,
            other: Parameter
    ) -> bool:
        if isinstance(other, Parameter):
            return self.value == other.value
        return NotImplemented

    def __ne__(
            self,
            other: Parameter
    ):
        result = self.__eq__(other)
        if result is NotImplemented:
            return result
        return not result

    def __hash__(self):
        return super(Parameter, self).__hash__()

    def __repr__(self):
        s = super(Parameter, self).__repr__()
        s += "\n"
        s += self.__str__()
        return s

    def __init__(self,
                 value: float = 1.0,
                 link: Parameter = None,
                 *args, **kwargs
                 ):
        self._instances.add(weakref.ref(self))
        super(Parameter, self).__init__(*args, **kwargs)
        self._link = link
        self._value = value

    def to_dict(self) -> dict:
        v = super(Parameter, self).to_dict()
        v['value'] = self.value
        v['decimals'] = self.decimals
        return v

    def from_dict(
            self,
            v: dict
    ):
        super(Parameter, self).from_dict(v)
        self._value = v['value']


class ParameterGroup(mfm.base.Base):

    def __init__(
            self,
            fit: mfm.fitting.fit.Fit,
            *args,
            **kwargs
    ):
        super(ParameterGroup, self).__init__(*args, **kwargs)
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
        with open(filename, 'w') as fp:
            s = ""
            for ph in self.parameter_names:
                s += ph + sep
            s += "\n"
            for l in self.values.T:
                for p in l:
                    s += "%.5f%s" % (p, sep)
                s += "\n"
            fp.write(s)

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


