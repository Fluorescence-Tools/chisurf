"""
This module is responsible for all experiments/fits

The :py:mod:`experiments` module contains the fitting models and the setups (assembled reading routines) for
different experimental setups. Furthermore, it contains a set of plotting libraries.


"""
import json
from mfm import Base


class Experiment(Base):
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

    def __init__(self, name):
        Base.__init__(self)
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
            names.append(s.name)
        return names


class Setup(Base):

    def __init__(self, *args, **kwargs):
        Base.__init__(self, *args, **kwargs)
        self.experiment = kwargs.get('experiment', None)
        self._settings = kwargs

    @staticmethod
    def autofitrange(data, **kwargs):
        return 0, len(data.y) - 1

    def load_data(self, **kwargs):
        pass

    def get_data(self, **kwargs):
        data = self.load_data(**kwargs)
        if isinstance(data, mfm.curve.ExperimentalData):
            data = mfm.curve.ExperimentDataGroup([data])
        for d in data:
            d.experiment = self.experiment
            d.setup = self
        return data

    def to_dict(self):
        d = Base.to_dict(self)
        d.update(self._settings)
        return d

    def save_settings(self, filename):
        with open(filename, 'w') as fp:
            fp.write(self.to_json())

    def load_settings(self, filename):
        with open(filename, 'r') as fp:
            self._settings = json.load(fp)

    def __getattr__(self, item):
        if item in self._settings.keys():
            return self._settings[item]
        return self.__getattribute__(item)

    def __str__(self):
        s = 'Setup:\n'
        s += 'Name: %s \n' % self.name
        return s

from .fcs import *
from .tcspc import *
from .globalfit import *
from . import modelling
