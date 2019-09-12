from __future__ import annotations
from typing import List, TypeVar, Tuple

import weakref
import numpy as np

import mfm
import mfm.base

T = TypeVar('T', bound='Parameter')


class Parameter(
    mfm.base.Base
):

    _instances = set()

    @property
    def bounds(
            self
    ) -> Tuple[float, float]:
        if self.bounds_on:
            return self._lb, self._ub
        else:
            return float("-inf"), float("inf")

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
    def link(self):
        return self._link

    @link.setter
    def link(
            self,
            link: Parameter
    ):
        if isinstance(link, Parameter):
            self._link = link
        elif link is None:
            try:
                self._value = self._link.value
            except AttributeError:
                pass
            self._link = None

    @property
    def is_linked(self) -> bool:
        return isinstance(self._link, Parameter)

    @classmethod
    def get_instances(cls) -> List[Parameter]:
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
        a = self.value
        b = other.value if isinstance(other, Parameter) else other
        return Parameter(
            value=a.__add__(b)
        )

    def __mul__(
            self,
            other: T
    ) -> Parameter:
        a = self.value
        b = other.value if isinstance(other, Parameter) else other
        return Parameter(
            value=a.__mul__(b)
        )

    def __truediv__(
            self,
            other: T
    ) -> Parameter:
        a = self.value
        b = other.value if isinstance(other, Parameter) else other
        return Parameter(
            value=a.__truediv__(b)
        )

    def __floordiv__(
            self,
            other: T
    ) -> Parameter:
        a = self.value
        b = other.value if isinstance(other, Parameter) else other
        return Parameter(
            value=a.__floordiv__(b)
        )

    def __sub__(
            self,
            other: T
    ) -> Parameter:
        a = self.value
        b = other.value if isinstance(other, Parameter) else other
        return Parameter(
            value=a.__sub__(b)
        )

    def __mod__(
            self,
            other: T
    ) -> Parameter:
        a = self.value
        b = other.value if isinstance(other, Parameter) else other
        return Parameter(
            value=a.__mod__(b)
        )

    def __pow__(
            self,
            other: T
    ) -> Parameter:
        a = self.value
        b = other.value if isinstance(other, Parameter) else other
        return Parameter(
            value=a.__pow__(b)
        )

    def __invert__(self) -> Parameter:
        a = self.value
        return Parameter(
            value=(1./a)
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

    def __init__(
            self,
            value: float = 1.0,
            link: Parameter = None,
            lb: float = float("-inf"),
            ub: float = float("inf"),
            bounds_on: bool = False,
            *args,
            **kwargs
    ):
        super(Parameter, self).__init__(*args, **kwargs)
        self._bounds_on = bounds_on
        self._instances.add(weakref.ref(self))
        self._link = link
        self._value = value
        self._lb = lb
        self._ub = ub

    def to_dict(self) -> dict:
        v = super(Parameter, self).to_dict()
        v['value'] = self.value
        v['decimals'] = self.decimals
        v['lb'], v['ub'] = self.bounds
        v['bounds_on'] = self.bounds_on
        return v

    def from_dict(
            self,
            v: dict
    ):
        super(Parameter, self).from_dict(v)
        self._value = v['value']
        self._lb, self._ub = v['lb'], v['ub']
        self._bounds_on = v['bounds_on']


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


