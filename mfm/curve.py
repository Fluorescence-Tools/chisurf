from __future__ import annotations

import numbers
import os.path
import zlib
from copy import copy

import numpy as np

import mfm
import mfm.io
import mfm.decorators
from mfm.base import Base
from mfm.math.signal import get_fwhm


class Data(Base):

    def __init__(self, *args, **kwargs):
        Base.__init__(self, *args, **kwargs)
        self._filename = kwargs.get('filename', None)
        self._data = kwargs.get('data', None)
        self._embed_data = mfm.cs_settings['database']['embed_data']
        self._max_file_size = mfm.cs_settings['database']['read_file_size_limit']

    @property
    def data(self) -> mfm.curve.Data:
        return self._data

    @data.setter
    def data(
            self,
            v: mfm.curve.Data
    ):
        self._data = v

    @property
    def name(self) -> str:
        try:
            return self._kw['name']
        except KeyError:
            return self.filename

    @name.setter
    def name(
            self,
            v: str
    ):
        self._kw['name'] = v

    @property
    def filename(self) -> str:
        try:
            return os.path.normpath(self._filename)
        except AttributeError:
            return 'No file'

    @filename.setter
    def filename(
            self,
            v: str
    ):
        self._filename = os.path.normpath(v)
        file_size = os.path.getsize(self._filename)
        if file_size < self._max_file_size and self._embed_data:
            data = open(self._filename).read()
            if len(data) > mfm.cs_settings['database']['compression_data_limit']:
                data = zlib.compress(data)
            if len(data) < mfm.cs_settings['database']['embed_data_limit']:
                self._data = data
            else:
                self._data = None
        else:
            self._data = None
        if mfm.verbose:
            print("Filename: %s" % self._filename)
            print("File size [byte]: %s" % file_size)

    def __str__(self):
        s = Base.__str__(self)
        s += "filename: %s\n" % self.filename
        return s


class Curve(Base):

    @property
    def fwhm(self) -> float:
        return get_fwhm(self)[0]

    @property
    def cdf(self) -> mfm.curve.Curve:
        """Cumulative sum of function
        """
        y = self.y
        ys = np.cumsum(y)
        x = self.x
        c = self.__class__(x=x, y=ys)
        return c

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
        #Base.__init__(self, *args, **kwargs)
        super(Curve, self).__init__(*args, **kwargs)

    def norm(
            self,
            mode: str = "max",
            c: mfm.curve.Curve = None
    ):
        factor = 1.0
        if not isinstance(c, Curve):
            if mode == "sum":
                factor = sum(self.y)
            elif mode == "max":
                factor = max(self.y)
        else:
            if mode == "sum":
                factor = sum(self.y) * sum(c.y)
            elif mode == "max":
                if max(self.y) != 0:
                    factor = max(self.y) * max(c.y)
        self.y /= factor
        return factor

    def __add__(self, c):
        y = copy(np.array(self.y, dtype=np.float64))
        if isinstance(c, numbers.Real):
            y += c
        elif isinstance(c, Curve):
            y += np.array(c.y, dtype=np.float64)
        x = copy(self.x)
        return self.__class__(x=x, y=y)

    def __sub__(self, c):
        y = copy(np.array(self.y, dtype=np.float64))
        if isinstance(c, numbers.Real):
            y -= c
        elif isinstance(c, Curve):
            y -= np.array(c.y, dtype=np.float64)
        x = copy(self.x)
        return self.__class__(x=x, y=y)

    def __mul__(self, c):
        y = copy(np.array(self.y, dtype=np.float64))
        if isinstance(c, numbers.Real):
            y *= c
        elif isinstance(c, Curve):
            y *= np.array(c.y, dtype=np.float64)
        x = copy(self.x)
        return self.__class__(x=x, y=y)

    def __div__(self, c):
        y = copy(np.array(self.y, dtype=np.float64))
        if isinstance(c, numbers.Real):
            y /= c
        elif isinstance(c, Curve):
            y /= np.array(c.y, dtype=np.float64)
        x = copy(self.x)
        return self.__class__(x=x, y=y)

    def __lshift__(self, c):
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

    def __len__(self):
        return len(self.y)

    def __getitem__(self, key):
        x = self.x.__getitem__(key)
        y = self.y.__getitem__(key)
        return x, y


