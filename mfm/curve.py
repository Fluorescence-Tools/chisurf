import numbers
import os.path
import zlib
from copy import copy
from typing import List

import numpy as np

import mfm
import mfm.decorators
from mfm.base import Base
import mfm.fitting.fit
from mfm.math.signal import get_fwhm


class Data(Base):

    def __init__(self, *args, **kwargs):
        Base.__init__(self, *args, **kwargs)
        self._filename = kwargs.get('filename', None)
        self._data = kwargs.get('data', None)
        self._embed_data = mfm.cs_settings['database']['embed_data']
        self._max_file_size = mfm.cs_settings['database']['read_file_size_limit']

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, v):
        self._data = v

    @property
    def name(self):
        try:
            return self._kw['name']
        except KeyError:
            return self.filename

    @name.setter
    def name(self, v):
        self._kw['name'] = v

    @property
    def filename(self):
        try:
            return os.path.normpath(self._filename)
        except AttributeError:
            return 'No file'

    @filename.setter
    def filename(self, v):
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


class ExperimentalData(Data):

    def __init__(self, *args, **kwargs):
        super(ExperimentalData, self).__init__(*args, **kwargs)
        self.setup = kwargs.get('setup', None)
        self._experiment = kwargs.get('experiment', None)

    @property
    def experiment(self):
        if self._experiment is None:
            if self.setup is None:
                return None
            else:
                return self.setup.experiment
        else:
            return self._experiment

    @experiment.setter
    def experiment(self, v):
        self._experiment = v

    def to_dict(self):
        d = Data.to_dict(self)
        d['setup'] = self.setup.to_dict()
        return d


class Curve(Base):

    @property
    def fwhm(self):
        return get_fwhm(self)[0]

    @property
    def cdf(self):
        """Cumulative sum of function
        """
        y = self.y
        ys = np.cumsum(y)
        x = self.x
        c = self.__class__(x=x, y=ys)
        return c

    def to_dict(self):
        d = Base.to_dict(self)
        d['x'] = list(self.x)
        d['y'] = list(self.y)
        return d

    def from_dict(self, v):
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

    def norm(self, mode="max", c=None):
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


class DataCurve(Curve, ExperimentalData):

    def __init__(self, *args, **kwargs):
        try:
            ex, ey = args[2], args[3]
        except IndexError:
            ex = kwargs.get('ex', np.ones_like(kwargs.get('x', None)))
            ey = kwargs.get('ey', np.ones_like(kwargs.get('y', None)))
        kwargs['ex'] = ex
        kwargs['ey'] = ey
        super(DataCurve, self).__init__(*args, **kwargs)
        filename = kwargs.pop('filename', '')

        if os.path.isfile(filename):
            self.load(filename, **kwargs)

    def __str__(self):
        s = "Dataset:\n"
        try:
            s += "filename: " + self.filename + "\n"
            s += "length  : %s\n" % len(self)
            s += "x\ty\terror-x\terror-y\n"

            lx = self.x[:3]
            ly = self.y[:3]
            lex = self.ex[:3]
            ley = self.ey[:3]

            ux = self.x[-3:]
            uy = self.y[-3:]
            uex = self.ex[-3:]
            uey = self.ey[-3:]

            for i in range(2):
                x, y, ex, ey = lx[i], ly[i], lex[i], ley[i]
                s += "{0:<12.3e}\t".format(x)
                s += "{0:<12.3e}\t".format(y)
                s += "{0:<12.3e}\t".format(ex)
                s += "{0:<12.3e}\t".format(ey)
                s += "\n"
            s += "....\n"
            for i in range(1):
                x, y, ex, ey = ux[i], uy[i], uex[i], uey[i]
                s += "{0:<12.3e}\t".format(x)
                s += "{0:<12.3e}\t".format(y)
                s += "{0:<12.3e}\t".format(ex)
                s += "{0:<12.3e}\t".format(ey)
                s += "\n"
        except:
            s += "This cuve does not 'own' data..."
        return s

    def to_dict(self):
        d = Curve.to_dict(self)
        d.update(ExperimentalData.to_dict(self))
        d['ex'] = list(self.ex)
        d['ey'] = list(self.ey)
        d['weights'] = list(self.weights)
        d['experiment'] = self.experiment.to_dict()
        return d

    def from_dict(self, v):
        Curve.from_dict(self, v)
        self._kw['ex'] = np.array(v['ex'], dtype=np.float64)
        self._kw['ey'] = np.array(v['ey'], dtype=np.float64)

    def save(self, filename=None, mode='json'):
        if filename is None:
            filename = os.path.join(self.name + '_data.txt')
        if mode == 'txt':
            mfm.io.ascii.Csv().save(self, filename)
        else:
            with open(filename, 'w') as fp:
                fp.write(self.to_json())

    def load(
            self,
            filename: str,
            **kwargs
    ):
        mode = kwargs.get('mode', 'txt')
        skiprows = kwargs.get('skiprows', 9)

        if mode == 'txt':
            csv = mfm.io.ascii.Csv()
            csv.load(filename, skiprows=skiprows)
            self._kw['x'] = csv.data[0]
            self._kw['y'] = csv.data[1]
            self._kw['ey'] = csv.data[2]
            self._kw['ex'] = csv.data[3]

    def set_data(
            self,
            filename: str,
            x: np.array,
            y: np.array,
            **kwargs
    ):
        """test docs
        
        :param filename: 
        :param x: 
        :param y: 
        :param kwargs: 
        :return: 
        """
        self._kw['ex'] = kwargs.get('ex', np.zeros_like(x))
        self._kw['ey'] = kwargs.get('ey', np.ones_like(y))
        self.filename = filename
        self._kw['x'] = x
        self._kw['y'] = y

    def set_weights(
            self,
            w: np.array
    ):
        self._kw['weights'] = w

    def __getitem__(
            self,
            key: str
    ):
        x, y = Curve.__getitem__(self, key)
        return x, y, self.ey[key]

    @property
    def dt(self) -> np.array:
        """
        The derivative of the x-axis
        """
        return np.diff(self.x)


class DataGroup(list, Base):

    @property
    def names(self) -> List[str]:
        return [d.name for d in self]

    @property
    def current_dataset(self):
        return self[self._current_dataset]

    @current_dataset.setter
    def current_dataset(self, i):
        self._current_dataset = i

    @property
    def name(self) -> str:
        try:
            return self._kw['name']
        except KeyError:
            return self.names[0]

    @name.setter
    def name(
            self,
            v: str
    ):
        self._name = v

    def append(
            self,
            dataset: Data
    ):
        if isinstance(dataset, ExperimentalData):
            list.append(self, dataset)
        if isinstance(dataset, list):
            for d in dataset:
                if isinstance(d, ExperimentalData):
                    list.append(self, d)

    def __init__(self, *args, **kwargs):
        if isinstance(args[0], list):
            list.__init__(self, args[0])
        else:
            list.__init__(self, [args[0]])
        Base.__init__(self, *args[1:], **kwargs)
        self._current_dataset = 0


class DataCurveGroup(DataGroup):

    @property
    def x(self):
        return self.current_dataset.x

    @x.setter
    def x(self, v):
        self.current_dataset.x = v

    @property
    def y(self):
        return self.current_dataset.y

    @y.setter
    def y(self, v):
        self.current_dataset.y = v

    @property
    def ex(self):
        return self.current_dataset.ex

    @ex.setter
    def ex(self, v):
        self.current_dataset.ex = v

    @property
    def ey(self):
        return self.current_dataset.ey

    @ey.setter
    def ey(self, v):
        self.current_dataset.ey = v

    def __str__(self):
        return [str(d) + "\n------\n" for d in self]

    def __init__(self, *args, **kwargs):
        DataGroup.__init__(self, *args, **kwargs)


class ExperimentDataGroup(DataGroup):

    @property
    def setup(self):
        return self[0].setup

    @setup.setter
    def setup(self, v):
        pass

    @property
    def experiment(self):
        return self.setup.experiment

    @experiment.setter
    def experiment(self, v):
        pass

    def __init__(self, *args, **kwargs):
        DataGroup.__init__(self, *args, **kwargs)


class ExperimentDataCurveGroup(ExperimentDataGroup, DataCurveGroup):

    @property
    def setup(self):
        return self[0].setup

    @setup.setter
    def setup(self, v):
        pass

    def __init__(self, *args, **kwargs):
        ExperimentDataGroup.__init__(self, *args, **kwargs)
        DataCurveGroup.__init__(self, *args, **kwargs)

