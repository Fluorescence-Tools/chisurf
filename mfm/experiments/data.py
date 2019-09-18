"""

"""
from __future__ import annotations

import os.path
from typing import List
import numpy as np

import mfm
import mfm.base
from mfm.base import Base
from mfm.curve import Curve
import mfm.experiments.experiment


class ExperimentalData(mfm.base.Data):
    """

    """

    def __init__(
            self,
            *args,
            setup=None,
            experiment=None,
            **kwargs
    ):
        """

        :param args:
        :param setup:
        :param experiment:
        :param kwargs:
        """
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
            x: np.array = None,
            y: np.array = None,
            ex: np.array = None,
            ey: np.array = None,
            filename: str = '',
            **kwargs
    ):
        super(DataCurve, self).__init__(
            *args,
            x=x,
            y=y,
            **kwargs
        )
        if os.path.isfile(filename):
            self.load(
                filename,
                **kwargs
            )
        if ex is None:
            ex = np.ones_like(self.x)
        if ey is None:
            ey = np.ones_like(self.y)
        self.ex = ex
        self.ey = ey

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
        self.ex = np.array(v['ex'], dtype=np.float64)
        self.__dict__['ey'] = np.array(v['ey'], dtype=np.float64)

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
            self.x = csv.data[0]
            self.y = csv.data[1]
            self.ex = csv.data[2]
            self.ex = csv.data[3]

    def set_data(
            self,
            filename: str,
            x: np.array,
            y: np.array,
            ex: np.array = None,
            ey: np.array = None,
    ) -> None:
        self.filename = filename
        self.x = x
        self.y = y

        if ex is None:
            ex = np.ones_like(x)
        if ey is None:
            ey = np.ones_like(y)
        self.ex = ex
        self.ey = ey

    def set_weights(
            self,
            w: np.array
    ):
        self.ey = 1. / w

    def __getitem__(
            self,
            key: str
    ):
        x, y = super(DataCurve, self).__getitem__(key)
        return x, y, self.ey[key]


class DataGroup(list, Base):

    @property
    def names(
            self
    ) -> List[str]:
        return [d.name for d in self]

    @property
    def current_dataset(
            self
    ) -> mfm.base.Data:
        return self[self._current_dataset]

    @current_dataset.setter
    def current_dataset(
            self,
            i: int
    ):
        self._current_dataset = i

    @property
    def name(
            self
    ) -> str:
        try:
            return self.__dict__['name']
        except KeyError:
            return self.names[0]

    @name.setter
    def name(
            self,
            v: str
    ) -> None:
        self._name = v

    def append(
            self,
            dataset: mfm.base.Data
    ):
        if isinstance(dataset, ExperimentalData):
            list.append(self, dataset)
        if isinstance(dataset, list):
            for d in dataset:
                if isinstance(d, ExperimentalData):
                    list.append(self, d)

    def __init__(
            self,
            seq: List,
            *args,
            **kwargs
    ):
        super(DataGroup, self).__init__(seq)
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
        super(ExperimentDataGroup, self).__init__(*args, **kwargs)


class ExperimentDataCurveGroup(ExperimentDataGroup, DataCurveGroup):

    @property
    def setup(self):
        return self[0].setup

    @setup.setter
    def setup(self, v):
        pass

    def __init__(self, *args, **kwargs):
        super(ExperimentDataCurveGroup, self).__init__(
            *args, **kwargs
        )
