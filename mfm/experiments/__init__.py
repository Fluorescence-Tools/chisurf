"""
This module is responsible for all experiments/fits

The :py:mod:`experiments` module contains the fitting models and the setups (assembled reading routines) for
different experimental setups. Furthermore, it contains a set of plotting libraries.


"""
from __future__ import annotations

import os.path
from typing import List

import numpy as np

import mfm
import mfm.base
from mfm.base import Base
from mfm.curve import Data, Curve


class Experiment(mfm.base.Base):
    """
    All information contained within `ChiSurf` is associated to an experiment. Each experiment
    is associated with a list of models and a list of setups. The list of models and the list
    of setups determine the applicable models and loadable data-types respectively.
    """

    @property
    def setups(self):
        return self.get_setups()

    @property
    def setup_names(self):
        return self.get_setup_names()

    @property
    def models(self):
        return self.model_classes

    @property
    def model_names(self):
        return self.get_model_names()

    def __init__(self, name, *args, **kwargs):
        super(Experiment, self).__init__(name, *args, **kwargs)
        self.name = name
        self.model_classes = list()
        self._setups = list()

    def add_model(self, model):
        self.model_classes.append(model)

    def add_models(self, models):
        for model in models:
            self.model_classes.append(model)

    def add_setup(self, setup):
        self.setups.append(setup)

    def add_setups(self, setups):
        self._setups += setups
        for s in setups:
            s.experiment = self

    def get_setups(self):
        return self._setups

    def get_setup_names(self):
        names = list()
        for s in self.setups:
            names.append(s.name)
        return names

    def get_model_names(self):
        names = list()
        for s in self.model_classes:
            names.append(str(s.name))
        return names


class Reader(mfm.base.Base):

    def __init__(self, *args, **kwargs):
        super(Reader, self).__init__(self, *args, **kwargs)

    @staticmethod
    def autofitrange(data, **kwargs):
        return 0, len(data.y) - 1

    def read(self, **kwargs):
        pass

    def get_data(self, **kwargs):
        data = self.read(**kwargs)
        if isinstance(data, ExperimentalData):
            data = ExperimentDataGroup([data])
        for d in data:
            d.experiment = self.experiment
            d.setup = self
        return data


from . import fcs
from . import tcspc
from . import globalfit
from . import modelling

tcspc_setups = [
    tcspc.TCSPCSetupWidget(name="CSV/PQ/IBH", **mfm.cs_settings['tcspc_csv']),
    tcspc.TCSPCSetupSDTWidget(),
    tcspc.TCSPCSetupDummyWidget()
]

fcs_setups = [
    fcs.FCSKristine(experiment=fcs),
    fcs.FCSCsv(experiment=fcs)
]

structure_setups = [
    modelling.PDBLoad()
        #.FCSKristine(experiment=fcs)
]


class ExperimentalData(Data):

    def __init__(self, *args, **kwargs):
        super(ExperimentalData, self).__init__(*args, **kwargs)
        self.setup = kwargs.get('setup', None)
        self._experiment = kwargs.get('experiment', None)

    @property
    def experiment(self) -> mfm.experiments.Experiment:
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
            v: mfm.experiments.Experiment
    ):
        self._experiment = v

    def to_dict(self):
        d = Data.to_dict(self)
        d['setup'] = self.setup.to_dict()
        return d


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

    def to_dict(self) -> dict:
        d = Curve.to_dict(self)
        d.update(ExperimentalData.to_dict(self))
        d['ex'] = list(self.ex)
        d['ey'] = list(self.ey)
        d['weights'] = list(self.weights)
        d['experiment'] = self.experiment.to_dict()
        return d

    def from_dict(
            self,
            v: dict
    ):
        Curve.from_dict(self, v)
        self._kw['ex'] = np.array(v['ex'], dtype=np.float64)
        self._kw['ey'] = np.array(v['ey'], dtype=np.float64)

    def save(
            self,
            filename: str = None,
            mode: str = 'json'
    ):
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