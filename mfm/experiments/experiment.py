"""

"""
from __future__ import annotations
from typing import List, Type

import mfm.base


class Experiment(mfm.base.Base):
    """
    All information contained within `ChiSurf` is associated to an experiment. Each experiment
    is associated with a list of models and a list of setups. The list of models and the list
    of setups determine the applicable models and loadable data-types respectively.
    """

    @property
    def setups(
            self
    ) -> List[mfm.experiments.reader.ExperimentReader]:
        return self.get_setups()

    @property
    def setup_names(
            self
    ) -> List[str]:
        return self.get_setup_names()

    @property
    def models(
            self
    ) -> List:
        return self.model_classes

    @property
    def model_names(
            self
    ) -> List[str]:
        return self.get_model_names()

    def __init__(
            self,
            name: str,
            *args,
            **kwargs
    ):
        """

        :param name:
        :param args:
        :param kwargs:
        """
        super(Experiment, self).__init__(name, *args, **kwargs)
        self.name = name
        self.model_classes = list()
        self._setups = list()

    def add_model(
            self,
            model: mfm.models.model.Model
    ):
        self.model_classes.append(model)

    def add_models(
            self,
            models: List[Type[mfm.models.model.Model]]
    ):
        for model in models:
            self.model_classes.append(model)

    def add_setup(
            self,
            setup: mfm.experiments.reader.ExperimentReader
    ):
        self.setups.append(setup)

    def add_setups(
            self,
            setups: List
    ):
        self._setups += setups
        for s in setups:
            s.experiment = self

    def get_setups(self) -> List:
        return self._setups

    def get_setup_names(self) -> List[str]:
        names = list()
        for s in self.setups:
            names.append(s.name)
        return names

    def get_model_names(self) -> List[str]:
        names = list()
        for s in self.model_classes:
            names.append(str(s.name))
        return names
