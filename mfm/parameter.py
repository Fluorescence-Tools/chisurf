from __future__ import annotations
from typing import List, TypeVar, Tuple

import weakref
import numpy as np

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
        s = self.value.__repr__()
        return s

    def to_dict(self) -> dict:
        d = super(Parameter, self).to_dict()
        if self.link is not None:
            d['_link'] = self.link.unique_identifier
        return d

    def from_dict(
            self,
            v: dict
    ) -> None:
        if v['_link'] is not None:
            unique_identifier = v['_link']
            for o in self.get_instances():
                if unique_identifier == o.unique_identifier:
                    v['_link'] = o
            super(Parameter, self).from_dict(v)
            if isinstance(v['_link'], str):
                raise ValueError(
                    "The linked parameter %s is not instantiated." % unique_identifier
                )

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
        super(Parameter, self).__init__(
            *args,
            **kwargs
        )
        self._bounds_on = bounds_on
        self._instances.add(weakref.ref(self))
        self._link = link
        self._value = value
        self._lb = lb
        self._ub = ub


class ParameterGroup(mfm.base.Base):
    """

    """

    def __init__(
            self,
            parameters: List[Parameter] = None,
            *args,
            **kwargs
    ):
        """

        :param args:
        :param kwargs:
        """
        super(ParameterGroup, self).__init__(
            *args,
            **kwargs
        )
        if parameters is None:
            parameters = list()
        self._parameter = parameters

    def append(
            self,
            parameter: Parameter
    ):
        self._parameter.append(parameter)

    def clear(self):
        self._parameter = list()

    @property
    def parameters(
            self
    ) -> List[Parameter]:
        return self._parameter

    @property
    def parameter_names(
            self
    ) -> List[str]:
        return [p.name for p in self.parameters]

    @property
    def values(self) -> np.array:
        return [p.value for p in self.parameters]

    # def save_txt(
    #         self,
    #         filename: str,
    #         sep: str = '\t'
    # ):
    #     with open(filename, 'w') as fp:
    #         s = ""
    #         for ph in self.parameter_names:
    #             s += ph + sep
    #         s += "\n"
    #         for l in self.values:
    #             for p in l:
    #                 s += "%.5f%s" % (p, sep)
    #             s += "\n"
    #         fp.write(s)
    #

