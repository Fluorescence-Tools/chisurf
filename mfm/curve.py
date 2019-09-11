from __future__ import annotations
from typing import TypeVar, Tuple

import numbers
from copy import copy

import numpy as np

import mfm
import mfm.io
import mfm.decorators
from mfm.base import Base
from mfm.math.signal import get_fwhm

T = TypeVar('T', bound='Curve')


class Curve(Base):

    @property
    def fwhm(self) -> float:
        return get_fwhm(self)[0]

    @property
    def cdf(self) -> Curve:
        """Cumulative sum of function
        """
        return self.__class__(
            x=self.x,
            y=np.cumsum(self.y)
        )

    @property
    def dt(self) -> np.array:
        """
        The derivative of the x-axis
        """
        return np.diff(self.x)

    def to_dict(self) -> dict:
        d = Base.to_dict(self)
        d['x'] = list(self.x)
        d['y'] = list(self.y)
        return d

    def from_dict(
            self,
            v: dict
    ):
        v['y'] = np.array(v['y'], dtype=np.float64)
        v['x'] = np.array(v['x'], dtype=np.float64)
        Base.from_dict(self, v)

    def __init__(self, *args, **kwargs):
        try:
            x, y = args[0], args[1]
        except IndexError:
            x = np.array([], dtype=np.float64)
            y = np.array([], dtype=np.float64)
        if len(y) != len(x):
            raise ValueError("length of x (%s) and y (%s) differ" % (len(self._x), len(self._y)))
        kwargs['x'] = np.copy(kwargs.get('x', x))
        kwargs['y'] = np.copy(kwargs.get('y', y))
        super(Curve, self).__init__(*args, **kwargs)

    def normalize(
            self,
            mode: str = "max",
            curve: mfm.curve.Curve = None
    ):
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
        self.y /= factor
        return factor

    def __add__(
            self,
            c: T
    ) -> Curve:
        y = copy(np.array(self.y, dtype=np.float64))
        if isinstance(c, numbers.Real):
            y += c
        elif isinstance(c, Curve):
            y += np.array(c.y, dtype=np.float64)
        x = copy(self.x)
        return self.__class__(x=x, y=y)

    def __sub__(
            self,
            c: T
    ) -> Curve:
        y = copy(np.array(self.y, dtype=np.float64))
        if isinstance(c, numbers.Real):
            y -= c
        elif isinstance(c, Curve):
            y -= np.array(c.y, dtype=np.float64)
        x = copy(self.x)
        return self.__class__(x=x, y=y)

    def __mul__(
            self,
            c: T
    ) -> Curve:
        y = copy(np.array(self.y, dtype=np.float64))
        if isinstance(c, numbers.Real):
            y *= c
        elif isinstance(c, Curve):
            y *= np.array(c.y, dtype=np.float64)
        x = copy(self.x)
        return self.__class__(x=x, y=y)

    def __div__(
            self,
            c: T
    ) -> Curve:
        y = copy(np.array(self.y, dtype=np.float64))
        if isinstance(c, numbers.Real):
            y /= c
        elif isinstance(c, Curve):
            y /= np.array(c.y, dtype=np.float64)
        x = copy(self.x)
        return self.__class__(x=x, y=y)

    def __lshift__(
            self,
            c: T
    ) -> Curve:
        if isinstance(c, numbers.Real):
            ts = -c
            tsi = int(np.floor(ts))
            tsf = c - tsi
            ysh = np.roll(self.y, tsi) * (1 - tsf) + np.roll(self.y, tsi + 1) * tsf
            if ts > 0:
                ysh[:tsi] = 0.0
            elif ts < 0:
                ysh[tsi:] = 0.0
            c = self.__class__(x=self.x, y=ysh)
            return c
        else:
            return self

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, key) -> Tuple[float, float]:
        x = self.x.__getitem__(key)
        y = self.y.__getitem__(key)
        return x, y

