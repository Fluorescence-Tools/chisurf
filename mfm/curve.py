from __future__ import annotations
from typing import TypeVar, Tuple, Optional, Type

import numbers
from copy import copy

import numpy as np

import mfm
import mfm.io
import mfm.decorators
from mfm.base import Base
from mfm.math.signal import calculate_fwhm

T = TypeVar('T', bound='Curve')


class Curve(Base):

    @property
    def fwhm(self) -> float:
        return calculate_fwhm(self)[0]

    @property
    def cdf(self) -> Type[Curve]:
        """Cumulative sum of function
        """
        return self.__class__(
            x=self.x,
            y=np.cumsum(self.y)
        )

    @property
    def dx(self) -> np.array:
        """
        The derivative of the x-axis
        """
        return np.diff(self.x)

    @property
    def x(self) -> np.array:
        return self._x

    @x.setter
    def x(self, v) -> None:
        self._x = v

    @property
    def y(self) -> np.array:
        return self._y

    @y.setter
    def y(self, v) -> None:
        self._y = v

    def to_dict(self) -> dict:
        d = dict()
        d.update(super(Curve, self).to_dict())
        d['_x'] = d.pop('_x').tolist()
        d['_y'] = d.pop('_y').tolist()
        return d

    def to_json(
            self,
            indent: int = 4,
            sort_keys: bool = True
    ) -> str:
        return super(Curve, self).to_json(
            indent,
            sort_keys
        )

    def to_yaml(self) -> str:
        return super(Curve, self).to_yaml()

    def from_dict(
            self,
            v: dict
    ):
        v['_y'] = np.array(v['_y'], dtype=np.float64)
        v['_x'] = np.array(v['_x'], dtype=np.float64)
        super(Curve, self).from_dict(v)

    def __init__(
            self,
            x: np.array = None,
            y: np.array = None,
            *args,
            **kwargs
    ):
        if x is None:
            x = np.array(list(), dtype=np.float64)
        if y is None:
            y = np.array(list(), dtype=np.float64)
        if len(y) != len(x):
            raise ValueError(
                "length of x (%s) and y (%s) differ" % (len(self._x), len(self._y))
            )
        self._x = np.copy(x)
        self._y = np.copy(y)
        super(Curve, self).__init__(
            *args,
            **kwargs
        )

    def normalize(
            self,
            mode: str = "max",
            curve: mfm.curve.Curve = None,
            inplace: bool = True
    ) -> float:
        """Calculates a scaling parameter for the Curve object and (optionally) scales the Curve object.

        :param mode: either 'max' to normalize the maximum to one, or 'sum' to normalize to sum to one
        :param curve:
        :param inplace: if True the Curve object is modified in place. Otherwise, only the scaling parameter
        is returned
        :return: the parameter that scales the Curve object
        """
        factor = 1.0
        if not isinstance(curve, Curve):
            if mode == "sum":
                factor = sum(self.y)
            elif mode == "max":
                factor = max(self.y)
        else:
            if mode == "sum":
                factor = sum(self.y) * sum(curve.y)
            elif mode == "max":
                if max(self.y) != 0:
                    factor = max(self.y) * max(curve.y)
        if inplace:
            self.y /= factor
        return factor

    def __add__(
            self,
            c: T
    ) -> Type[Curve]:
        if isinstance(c, Curve):
            if not np.array_equal(self.x, c.x):
                raise ValueError("The x-axis differ")
            c = c.y
        return self.__class__(
            x=copy(self.x),
            y=copy(self.y).__add__(c)
        )

    def __sub__(
            self,
            c: T
    ) -> Type[Curve]:
        if isinstance(c, Curve):
            if not np.array_equal(self.x, c.x):
                raise ValueError("The x-axis differ")
            c = c.y
        return self.__class__(
            x=copy(self.x),
            y=copy(self.y).__sub__(c)
        )

    def __mul__(
            self,
            c: T
    ) -> Type[Curve]:
        if isinstance(c, Curve):
            if not np.array_equal(self.x, c.x):
                raise ValueError("The x-axis differ")
            c = c.y
        return self.__class__(
            x=copy(self.x),
            y=copy(self.y).__mul__(c)
        )

    def __truediv__(
            self,
            c: T
    ) -> Type[Curve]:
        if isinstance(c, Curve):
            if not np.array_equal(self.x, c.x):
                raise ValueError("The x-axis differ")
            c = c.y
        return self.__class__(
            x=copy(self.x),
            y=copy(self.y).__truediv__(c)
        )

    def __lshift__(
            self,
            c: float
    ) -> Type[Curve]:
        ts = -c
        tsi = int(np.floor(ts))
        tsf = c - tsi
        ysh = np.roll(self.y, tsi) * (1 - tsf) + np.roll(self.y, tsi + 1) * tsf
        if ts > 0:
            ysh[:tsi] = 0.0
        elif ts < 0:
            ysh[tsi:] = 0.0
        return self.__class__(
            x=self.x,
            y=ysh
        )

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, key) -> Tuple[float, float]:
        x = self.x.__getitem__(key)
        y = self.y.__getitem__(key)
        return x, y

