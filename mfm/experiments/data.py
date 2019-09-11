from __future__ import annotations

import os.path
import zlib
from typing import List
import numpy as np

import mfm
from mfm.base import Base
from mfm.curve import Curve
import mfm.experiments.experiment


class Data(Base):

    def __init__(
            self,
            *args,
            filename: str = None,
            data: mfm.experiments.data.Data = None,
            embed_data: bool = None,
            read_file_size_limit: int = None,
            **kwargs
    ):
        super(Data, self).__init__(*args, **kwargs)
        self._filename = filename
        self._data = data

        if embed_data is None:
            embed_data = mfm.settings.cs_settings['database']['embed_data']
        if read_file_size_limit is None:
            read_file_size_limit = mfm.settings.cs_settings['database']['read_file_size_limit']

        self._embed_data = embed_data
        self._max_file_size = read_file_size_limit

    @property
    def data(self) -> mfm.experiments.data.Data:
        return self._data

    @data.setter
    def data(
            self,
            v: mfm.experiments.data.Data
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
        except (AttributeError, TypeError):
            return 'No file'

    @filename.setter
    def filename(
            self,
            v: str
    ) -> None:
        self._filename = os.path.normpath(v)
        file_size = os.path.getsize(self._filename)
        if file_size < self._max_file_size and self._embed_data:
            data = open(self._filename).read()
            if len(data) > mfm.settings.cs_settings['database']['compression_data_limit']:
                data = zlib.compress(data)
            if len(data) < mfm.settings.cs_settings['database']['embed_data_limit']:
                self._data = data
            else:
                self._data = None
        else:
            self._data = None
        if mfm.verbose:
            print("Filename: %s" % self._filename)
            print("File size [byte]: %s" % file_size)

    def __str__(self):
        s = super(Data, self).__str__()
        s += "filename: %s\n" % self.filename
        return s


class ExperimentalData(Data):

    def __init__(
            self,
            *args,
            setup=None,
            experiment=None,
            **kwargs
    ):
        super(ExperimentalData, self).__init__(*args, **kwargs)
        self.setup = setup
        self._experiment = experiment

    @property
    def experiment(self) -> mfm.experiments.experiment.Experiment:
        if self._experiment is None:
            if self.setup is None:
                return None
            else:
                return self.setup.experiment
        else:
            return self._experiment

    @experiment.setter
    def experiment(
            self,
            v: mfm.experiments.experiment.Experiment
    ) -> None:
        self._experiment = v

    def to_dict(self):
        d = super(ExperimentalData, self).to_dict()
        d['setup'] = self.setup.to_dict()
        return d


class DataCurve(Curve, ExperimentalData):

    def __init__(
            self,
            *args,
            filename: str = '',
            **kwargs
    ):
        super(DataCurve, self).__init__(*args, **kwargs)
        if os.path.isfile(filename):
            self.load(filename, **kwargs)
        try:
            ex, ey = args[2], args[3]
        except IndexError:
            ex = kwargs.get('ex', np.ones_like(kwargs.get('x', None)))
            ey = kwargs.get('ey', np.ones_like(kwargs.get('y', None)))
        kwargs['ex'] = ex
        kwargs['ey'] = ey

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
        except (AttributeError, KeyError):
            s += "This curve does not 'own' data..."
        return s

    def to_dict(
            self
    ) -> dict:
        d = super(DataCurve, self).to_dict()
        d.update(ExperimentalData.to_dict(self))
        d['ex'] = list(self.ex)
        d['ey'] = list(self.ey)
        d['weights'] = list(self.weights)
        d['experiment'] = self.experiment.to_dict()
        return d

    def from_dict(
            self,
            v: dict
    ) -> None:
        super(DataCurve, self).from_dict(v)
        self._kw['ex'] = np.array(v['ex'], dtype=np.float64)
        self._kw['ey'] = np.array(v['ey'], dtype=np.float64)

    def save(
            self,
            filename: str = None,
            file_type: str = 'json',
            **kwargs
    ) -> None:
        if filename is None:
            filename = os.path.join(
                self.name,
                '_data.txt'
            )
        if file_type == 'txt':
            mfm.io.ascii.Csv().save(
                self,
                filename
            )
        else:
            with open(filename, 'w') as fp:
                fp.write(self.to_json())

    def load(
            self,
            filename: str,
            skiprows: int = 9,
            file_type: str = 'txt',
            **kwargs
    ) -> None:
        if file_type == 'txt':
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
            ex: np.array = None,
            ey: np.array = None,
    ) -> None:
        self.filename = filename
        self._kw['x'] = x
        self._kw['y'] = y

        if ex is None:
            ex = np.ones_like(x)
        if ey is None:
            ey = np.ones_like(y)
        self._kw['ex'] = ex
        self._kw['ey'] = ey

    def set_weights(
            self,
            w: np.array
    ):
        self._kw['weights'] = w

    def __getitem__(
            self,
            key: str
    ):
        x, y = super(DataCurve, self).__getitem__(key)
        return x, y, self.ey[key]


class DataGroup(list, Base):

    @property
    def names(self) -> List[str]:
        return [d.name for d in self]

    @property
    def current_dataset(self):
        return self[self._current_dataset]

    @current_dataset.setter
    def current_dataset(
            self,
            i: int
    ):
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
    def x(self) -> np.array:
        return self.current_dataset.x

    @x.setter
    def x(self,
          v: np.array):
        self.current_dataset.x = v

    @property
    def y(self) -> np.array:
        return self.current_dataset.y

    @y.setter
    def y(self,
          v: np.array):
        self.current_dataset.y = v

    @property
    def ex(self) -> np.array:
        return self.current_dataset.ex

    @ex.setter
    def ex(self,
           v: np.array):
        self.current_dataset.ex = v

    @property
    def ey(self) -> np.array:
        return self.current_dataset.ey

    @ey.setter
    def ey(self,
           v: np.array):
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