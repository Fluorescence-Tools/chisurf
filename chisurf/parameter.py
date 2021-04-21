from __future__ import annotations
from chisurf import typing

import numpy as np
import chinet

import chisurf.base
import chisurf.decorators

T = typing.TypeVar('T', bound='Parameter')


@chisurf.decorators.register
class Parameter(
    chisurf.base.Base
):

    @property
    def name(self) -> str:
        return self._port.name

    @name.setter
    def name(
            self,
            v: str
    ):
        self._port.name = v

    @property
    def bounds(
            self
    ) -> typing.Tuple[float, float]:
        """A tuple containing the values for the lower (first value) and
        the upper (second value) of the bound.
        """
        return self._port.bounds

    @bounds.setter
    def bounds(
            self,
            b: typing.Tuple[float, float]
    ):
        self._port.bounds = np.array(b, dtype=np.float)

    @property
    def bounds_on(self):
        return self._port.bounded

    @bounds_on.setter
    def bounds_on(self, v):
        self._port.bounded = bool(v)

    @property
    def value(
            self
    ) -> float:
        """The value of the parameter.

        This value of the parameter considers links and
        bounds, i.e., if a parameter is linked to another
        parameter the value of the linked parameter is returned.
        First, links are considered, then bounds are considered.

        :return:
        """
        return self._port.value.item(0)

    @value.setter
    def value(
            self,
            value: float
    ):
        self._port.value = np.array([value], dtype=np.double)

    @property
    def link(self) -> chisurf.parameter.Parameter:
        return self._link

    @link.setter
    def link(
            self,
            link: Parameter
    ):
        if isinstance(link, Parameter):
            self._link = link
            if self.controller is not None:
                self.controller.set_linked(link is not None)
            self._port.link = link._port
        elif link is None:
            self._link = None
            self._port.unlink()

    @property
    def is_linked(
            self
    ) -> bool:
        return self._port.is_linked

    def __add__(
            self,
            other: T
    ) -> T:
        a = self.value
        b = other.value if isinstance(other, Parameter) else other
        return self.__class__(
            value=(a + b)
        )

    def __mul__(
            self,
            other: T
    ) -> T:
        a = self.value
        b = other.value if isinstance(other, Parameter) else other
        return self.__class__(
            value=(a * b)
        )

    def __truediv__(
            self,
            other: T
    ) -> T:
        a = self.value
        b = other.value if isinstance(other, Parameter) else other
        return self.__class__(
            value=(a / b)
        )

    def __floordiv__(
            self,
            other: T
    ) -> T:
        a = self.value
        b = other.value if isinstance(other, Parameter) else other
        return self.__class__(
            value=(a // b)
        )

    def __sub__(
            self,
            other: T
    ) -> T:
        a = self.value
        b = other.value if isinstance(other, Parameter) else other
        return self.__class__(
            value=(a - b)
        )

    def __mod__(
            self,
            other: T
    ) -> T:
        a = self.value
        b = other.value if isinstance(other, Parameter) else other
        return self.__class__(
            value=(a % b)
        )

    def __pow__(
            self,
            other: T
    ) -> T:
        a = self.value
        b = other.value if isinstance(other, Parameter) else other
        return self.__class__(
            value=(a ** b)
        )

    def __invert__(
            self
    ) -> T:
        a = self.value
        return self.__class__(
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
        return super().__hash__()

    def __repr__(self):
        s = self.value.__repr__()
        return s

    def __abs__(self):
        return self.__class__(
            value=self.value.__abs__()
        )

    def __round__(self, n=None):
        return self.__class__(
            value=self.value.__round__()
        )

    # def to_dict(self) -> typing.Dict:
    #     d = super().to_dict()
    #     if self.link is not None:
    #         d['_link'] = self.link.unique_identifier
    #     return d
    #
    # def from_dict(
    #         self,
    #         v: dict
    # ) -> None:
    #     if v['_link'] is not None:
    #         unique_identifier = v['_link']
    #         for o in self.get_instances():
    #             if unique_identifier == o.unique_identifier:
    #                 v['_link'] = o
    #         super().from_dict(v)
    #         if isinstance(v['_link'], str):
    #             raise ValueError(
    #                 "The linked parameter %s is not instantiated." % unique_identifier
    #             )

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
        """
        :param value: the value of the parameter (default 1.0)
        :param link: the (optional) parameter to which the new instance is linked to
        :param lb: the lower bound of the parameter value
        :param ub: the upper bound of the parameter value
        :param bounds_on: if this is True the parameter value is bounded between
        the upper and the lower bound as specified by ub and lb.
        :param args:
        :param kwargs:
        """
        super().__init__(
            *args,
            **kwargs
        )
        name = kwargs.pop('name', '')
        if callable(value):
            self._callable = value
            self._port = chinet.Port(
                value=np.array([0.0], dtype=np.double),  # the value is not actually used
                name=name,
                lb=lb,
                ub=ub,
                is_bounded=bounds_on
            )
        else:
            self._callable = None
            self._port = chinet.Port(
                value=np.atleast_1d(value),
                name=name,
                lb=lb,
                ub=ub,
                is_bounded=bounds_on
            )
        self._link = link
        if isinstance(link, Parameter):
            self._port.link = link._port
        self.controller = None


class ParameterGroup(
    chisurf.base.Base
):

    def __init__(
            self,
            parameters: typing.List[Parameter] = None,
            *args,
            **kwargs
    ):
        """

        :param args:
        :param kwargs:
        """
        super().__init__(
            *args,
            **kwargs
        )
        if parameters is None:
            parameters = list()
        self._parameter = parameters

    def __setattr__(
            self,
            k: str,
            v: object
    ):
        try:
            propobj = getattr(self, k, None)
            if isinstance(propobj, property):
                if propobj.fset is None:
                    raise AttributeError("can't set attribute")
                propobj.fset(self, v)
            elif isinstance(propobj, chisurf.parameter.Parameter):
                propobj.value = v
            else:
                super().__setattr__(k, v)
        except KeyError:
            super().__setattr__(k, v)

    def __getattr__(
            self,
            key: str
    ):
        v = super().__getattr__(key=key)
        if isinstance(v, chisurf.parameter.Parameter):
            return v.value
        return v

    def append(
            self,
            parameter: Parameter,
            **kwargs
    ):
        self._parameter.append(parameter)

    def clear(self):
        self._parameter = list()

    @property
    def parameters(
            self
    ) -> typing.List[Parameter]:
        return self._parameter

    @property
    def parameter_names(
            self
    ) -> typing.List[str]:
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

