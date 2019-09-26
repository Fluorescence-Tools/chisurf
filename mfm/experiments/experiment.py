"""

"""
from __future__ import annotations
from typing import List, Type

import mfm.base


class Experiment(
    mfm.base.Base
):
    """
    All information contained within `ChiSurf` is associated to an experiment. Each experiment
    is associated with a list of models and a list of setups. The list of models and the list
    of setups determine the applicable models and loadable data-types respectively.
    """

    @property
    def readers(
            self
    ) -> List[mfm.experiments.reader.ExperimentReader]:
        return self.get_readers()

    @property
    def reader_names(
            self
    ) -> List[str]:
        return self.get_setup_names()

    @property
    def model_classes(
            self
    ) -> List[Type[mfm.models.model.Model]]:
        return list(self._model_classes)

    @property
    def model_names(
            self
    ) -> List[str]:
        return self.get_model_names()

    def add_model_class(
            self,
            model: Type[mfm.models.model.Model]
    ):
        if model not in self._model_classes:
            self._model_classes.append(model)

    def add_model_classes(
            self,
            models: List[Type[mfm.models.model.Model]]
    ):
        for model in models:
            if model not in self._model_classes:
                self._model_classes.append(model)

    def add_reader(
            self,
            reader: mfm.experiments.reader.ExperimentReader
    ):
        if reader not in self._readers:
            self._readers.append(reader)

    def add_readers(
            self,
            setups: List[mfm.experiments.reader.ExperimentReader]
    ):
        for s in setups:
            if s not in self._readers:
                self._readers.append(s)
                s.experiment = self

    def get_readers(
            self
    ) -> List[mfm.experiments.reader.ExperimentReader]:
        return list(self._readers)

    def get_setup_names(self) -> List[str]:
        names = list()
        for s in self.readers:
            names.append(s.name)
        return names

    def get_model_names(
            self
    ) -> List[str]:
        names = list()
        for s in self.model_classes:
            names.append(str(s.name))
        return names

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
        super().__init__(
            name=name,
            *args,
            **kwargs
        )
        self._model_classes = list()
        self._readers = list()
