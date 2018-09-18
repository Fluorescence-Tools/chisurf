import os.path
import zlib
from copy import copy

import numpy as np

import mfm
from mfm import Base
from mfm.math.signal import get_fwhm


class Data(Base, mfm.SQLBase):

    __tablename__ = 'data'
    id = mfm.Column(mfm.Integer, primary_key=True)
    _filename = mfm.Column(mfm.String(255), unique=False, nullable=True)
    _data = mfm.Column(mfm.LargeBinary, nullable=True)

    def __init__(self, *args, **kwargs):
        Base.__init__(self, *args, **kwargs)
        self._filename = kwargs.get('filename', None)
        self._data = kwargs.get('data', None)

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, v):
        self._data = v

    @property
    def name(self):
        if self._name is None:
            return self.filename
        else:
            return self._name

    @name.setter
    def name(self, v):
        self._name = v

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

        if file_size < mfm.settings['database']['read_file_size_limit']:
            data = open(self._filename).read()
            if len(data) > mfm.settings['database']['compression_data_limit']:
                data = zlib.compress(data)
            if len(data) < mfm.settings['database']['embed_data_limit']:
                self._data = data
            else:
                self._data = None
        else:
            self._data = None
        if mfm.verbose:
            print "Filename: %s" % self._filename
            print "File size [byte]: %s" % file_size
            #print self._data

    def __str__(self):
        s = Base.__str__(self)
        s += "filename: %s\n" % self.filename
        return s


class ExperimentalData(Data):

    def __init__(self, *args, **kwargs):
        Data.__init__(self, *args, **kwargs)
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
    def x(self):
        return self._x

    @x.setter
    def x(self, v):
        self._x = v

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, v):
        self._y = v

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
        Base.from_dict(self, v)
        try:
            self._x = np.array(v['x'], dtype=np.float64)
            self._y = np.array(v['y'], dtype=np.float64)
        except AttributeError:
            print "Values in dictionary missing"

    def __init__(self, *args, **kwargs):
        Base.__init__(self, **kwargs)
        try:
            x = args[0]
            y = args[1]
        except IndexError:
            x = np.array([], dtype=np.float64)
            y = np.array([], dtype=np.float64)
        self._x = np.copy(kwargs.get('x', x))
        self._y = np.copy(kwargs.get('y', y))
        if len(self._y) != len(self._x):
            raise ValueError("length of x (%s) and y (%s) differ" % (len(self._x), len(self._y)))

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
        if isinstance(c, float):
            y += c
        elif isinstance(c, Curve):
            y += np.array(c.y, dtype=np.float64)
        x = copy(self.x)
        return Curve(x, y)

    def __sub__(self, c):
        y = copy(np.array(self.y, dtype=np.float64))
        if isinstance(c, float):
            y -= c
        elif isinstance(c, Curve):
            y -= np.array(c.y, dtype=np.float64)
        x = copy(self.x)
        return Curve(x, y)

    def __mul__(self, c):
        y = copy(np.array(self.y, dtype=np.float64))
        if isinstance(c, float):
            y *= c
        elif isinstance(c, Curve):
            y *= np.array(c.y, dtype=np.float64)
        x = copy(self.x)
        return Curve(x, y)

    def __div__(self, c):
        y = copy(np.array(self.y, dtype=np.float64))
        if isinstance(c, float):
            y /= c
        elif isinstance(c, Curve):
            y /= np.array(c.y, dtype=np.float64)
        x = copy(self.x)
        return Curve(x, y)

    def __lshift__(self, tsn):
        if not np.isnan(tsn):
            ts = -tsn
            tsi = int(np.floor(ts))
            tsf = ts - tsi
            ysh = np.roll(self.y, tsi) * (1 - tsf) + np.roll(self.y, tsi + 1) * tsf
            if ts > 0:
                ysh[:tsi] = 0.0
            elif ts < 0:
                ysh[tsi:] = 0.0
            c = copy(self)
            c.y = ysh
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
        Curve.__init__(self, *args, **kwargs)
        ExperimentalData.__init__(self, **kwargs)
        filename = kwargs.pop('filename', '')

        try:
            ex = args[2]
            ey = args[3]
        except IndexError:
            ex = np.zeros(self._x.shape)
            ey = np.zeros(self._y.shape)
        self._ex = kwargs.get('ex', ex)
        self._ey = kwargs.get('ey', ey)
        self._weights = kwargs.get('weights', None)

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
        self.ex = np.array(v['ex'], dtype=np.float64)
        self.ey = np.array(v['ey'], dtype=np.float64)
        self.weights = np.array(v['weights'], dtype=np.float64)

    def save(self, filename=None, mode='json'):
        if filename is None:
            filename = os.path.join(self.name + '_data.txt')
        if mode == 'txt':
            mfm.io.ascii.Csv().save(self, filename)
        else:
            with open(filename, 'w') as fp:
                fp.write(self.to_json())

    def load(self, filename, **kwargs):
        mode = kwargs.get('mode', 'txt')
        skiprows = kwargs.get('skiprows', 9)

        if mode == 'txt':
            csv = mfm.io.ascii.Csv()
            csv.load(filename, unpack=True, skiprows=skiprows)
            self.x = csv.data_x
            self.y = csv.data_y
            self._weights = csv.error_y

    def set_data(self, filename, x, y, **kwargs):
        self._ex = kwargs.get('ex', np.zeros_like(x))
        self._ey = kwargs.get('ey', np.ones_like(y))
        self._weights = kwargs.get('weights', 1./self._ey)
        self.filename = filename
        self.x = x
        self.y = y

    def set_weights(self, w):
        self._weights = w

    def __getitem__(self, key):
        x, y = Curve.__getitem__(self, key)
        w = self.weights.__getitem__(key)
        return x, y, w

    def __sub__(self, v):
        y = np.array(self.y, dtype=np.float64)
        if isinstance(v, Curve):
            y -= v.y
        else:
            y -= v
        y = np.array(y, dtype=np.float64)
        c = copy(self)
        c.y = y
        return c

    @property
    def dt(self):
        """
        The derivative of the x-axis
        """
        return np.diff(self.x)

    @property
    def ex(self):
        if isinstance(self._ex, np.ndarray):
            return self._ex
        else:
            return np.zeros_like(self.x)

    @ex.setter
    def ex(self, v):
        self._ex = v

    @property
    def ey(self):
        if isinstance(self._ey, np.ndarray):
            return self._ey
        else:
            return np.ones_like(self.y)

    @ey.setter
    def ey(self, v):
        self._ey = v

    @property
    def weights(self):
        if self._weights is None:
            er = np.copy(self.ey)
            er[er == 0] = 1
            self._weights = er
            return self._weights
        else:
            return self._weights

    @weights.setter
    def weights(self, v):
        self._weights = v


class DataGroup(list, Base):

    @property
    def names(self):
        return [d.name for d in self]

    @property
    def current_dataset(self):
        return self[self._current_dataset]

    @current_dataset.setter
    def current_dataset(self, i):
        self._current_dataset = i

    @property
    def name(self):
        if self._name is None:
            return self.names[0]
        else:
            return self._name

    @name.setter
    def name(self, v):
        self._name = v

    def append(self, dataset):
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


class ParameterGroup(Base):

    def __init__(self, fit, **kwargs):
        Base.__init__(self, **kwargs)
        self.fit = fit
        self._activeRuns = list()
        self._chi2 = list()
        self._parameter = list()
        self.parameter_names = list()

    def clear(self):
        self._chi2 = list()
        self._parameter = list()

    def save_txt(self, filename, sep='\t'):
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
    def values(self):
        try:
            re = np.vstack(self._parameter)
            re = np.column_stack((re, self.chi2s))
            return re.T
        except ValueError:
            return np.array([[0], [0]]).T

    @property
    def chi2s(self):
        return np.hstack(self._chi2)


