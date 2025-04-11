from __future__ import annotations
from chisurf import typing

import abc
import json

import numpy as np
import chinet

import chisurf.base
import chisurf.decorators

T = typing.TypeVar('T', bound='Parameter')


@chisurf.decorators.register
class Parameter(chisurf.base.Base):

    @staticmethod
    def check_recursive_link(current, target):
        if id(current) == id(target):
            return True
        if current.link is not None:
            return Parameter.check_recursive_link(current.link, target)
        return False

    @property
    def fit_idx(self):
        import chisurf.fitting
        idxs = chisurf.fitting.find_fit_idx_of_parameter(self)
        if len(idxs) > 1:
            chisurf.logging.warning("Ambiguous link call. Fitting parameter used in multiple fits")
        fit_idx_self = idxs[0]
        return fit_idx_self

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, v: str):
        self._port.name = v
        self._name = v

    @property
    def bounds(self) -> typing.Tuple[float, float]:
        """A tuple containing the values for the lower (first value) and
        the upper (second value) of the bound.
        """
        return self._port.bounds

    @bounds.setter
    def bounds(self, b: typing.Tuple[float, float]):
        self._port.bounds = np.array(b, dtype=np.float64)

    @property
    def bounds_on(self):
        return self._port.bounded

    @bounds_on.setter
    def bounds_on(self, v):
        self._port.bounded = bool(v)

    @property
    def value(self) -> float:
        """The value of the parameter.

        This value of the parameter considers links and
        bounds, i.e., if a parameter is linked to another
        parameter the value of the linked parameter is returned.
        First, links are considered, then bounds are considered.

        :return:
        """
        return self._port.value[0]

    @value.setter
    def value(self, value: float):
        f = self._port.fixed
        self._port.fixed = False
        self._port.value = value
        self._port.fixed = f

    @property
    def link(self) -> chisurf.parameter.Parameter:
        return self._link

    @link.setter
    def link(self, link: Parameter):
        if isinstance(link, Parameter):
            if Parameter.check_recursive_link(link, self):
                raise ValueError("Cannot create a recursive link between parameters.")
            self._link = link
            if self.controller is not None:
                self.controller.set_linked(link is not None)
            self._port.link = link._port
        elif link is None:
            self._link = None
            self._port.unlink()

    @property
    def is_linked(self) -> bool:
        return self._port.is_linked

    @property
    def fixed(self):
        return self._port.fixed

    @fixed.setter
    def fixed(self, v: bool):
        self._port.fixed = bool(v)

    def __add__(self, other: T) -> T:
        a = self.value
        b = other.value if isinstance(other, Parameter) else other
        return self.__class__(
            value=(a + b)
        )

    def __mul__(self, other: T) -> T:
        a = self.value
        b = other.value if isinstance(other, Parameter) else other
        return self.__class__(
            value=(a * b)
        )

    def __truediv__(self, other: T) -> T:
        a = self.value
        b = other.value if isinstance(other, Parameter) else other
        return self.__class__(
            value=(a / b)
        )

    def __floordiv__(self, other: T) -> T:
        a = self.value
        b = other.value if isinstance(other, Parameter) else other
        return self.__class__(
            value=(a // b)
        )

    def __sub__(self, other: T) -> T:
        a = self.value
        b = other.value if isinstance(other, Parameter) else other
        return self.__class__(
            value=(a - b)
        )

    def __mod__(self, other: T) -> T:
        a = self.value
        b = other.value if isinstance(other, Parameter) else other
        return self.__class__(
            value=(a % b)
        )

    def __pow__(self, other: T) -> T:
        a = self.value
        b = other.value if isinstance(other, Parameter) else other
        return self.__class__(
            value=(a ** b)
        )

    def __invert__(self) -> T:
        a = self.value
        return self.__class__(
            value=(1./a)
        )

    def __float__(self):
        return float(self.value)

    def __eq__(self, other: Parameter) -> bool:
        if isinstance(other, Parameter):
            return self.value == other.value
        return NotImplemented

    def __ne__(self, other: Parameter):
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

    def __getstate__(self):
        d = json.loads(self._port.get_json())
        return {
            'port': d
        }

    def __setstate__(self, state):
        s = json.dumps(state['port'])
        self._port.read_json(s)
        fixed = self._port.fixed
        self._port.fixed = False
        self._port.value = state['port']['value']
        self._port.fixed = fixed

    def __round__(self, n=None):
        return self.__class__(
            value=self.value.__round__()
        )

    @abc.abstractmethod
    def update(self):
        pass

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
        super().__init__(*args, **kwargs)
        self._name = kwargs.pop('name', '')
        port = kwargs.pop('port', None)
        if port is not None:
            self._port = port
        else:
            if callable(value):
                self._callable = value
                self._port = chinet.Port(
                    value=np.atleast_1d(0.0).astype(dtype=np.float64),
                    name=self._name,
                    lb=lb,
                    ub=ub,
                    is_bounded=bounds_on
                )
            else:
                self._callable = None
                self._port = chinet.Port(
                    value=np.atleast_1d(value).astype(dtype=np.float64),
                    name=self._name,
                    lb=lb,
                    ub=ub,
                    is_bounded=bounds_on
                )
        self._link = link
        if isinstance(link, Parameter):
            self._port.link = link._port
        self.controller = None


class ParameterGroup(chisurf.base.Base):

    def __init__(
            self,
            parameters: typing.List[Parameter] = None,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
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

    def __getattr__(self, key: str):
        v = super().__getattr__(key=key)
        if isinstance(v, chisurf.parameter.Parameter):
            return v.value
        return v

    def append(self,
            parameter: Parameter,
            **kwargs
    ):
        self._parameter.append(parameter)

    def clear(self):
        self._parameter = list()

    @property
    def parameters(self) -> typing.List[Parameter]:
        return self._parameter

    @property
    def parameter_names(self) -> typing.List[str]:
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

