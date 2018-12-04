"""
This module is responsible for all experiments/fits

The :py:mod:`experiments` module contains the fitting models and the setups (assembled reading routines) for
different experimental setups. Furthermore, it contains a set of plotting libraries.


"""
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


class Reader(Base):

    def __init__(self, *args, **kwargs):
        super(Reader, self).__init__(self, *args, **kwargs)

    @staticmethod
    def autofitrange(data, **kwargs):
        return 0, len(data.y) - 1

    def read(self, **kwargs):
        pass

    def get_data(self, **kwargs):
        data = self.read(**kwargs)
        if isinstance(data, mfm.curve.ExperimentalData):
            data = mfm.curve.ExperimentDataGroup([data])
        for d in data:
            d.experiment = self.experiment
            d.setup = self
        return data


from .fcs import *
import tcspc
from .globalfit import *
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
