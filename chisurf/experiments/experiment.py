from __future__ import annotations
from chisurf import typing

import chisurf.base
import chisurf.models
import chisurf.experiments.reader


class Experiment(chisurf.base.Base):
    """
    All information contained within `ChiSurf` is associated to an experiment.
    Each experiment is associated with a list of models and a list of setups.
    The list of models and the list of setups determine the applicable models
    and loadable data-types respectively.
    """

    hidden: bool = False

    @property
    def readers(self) -> typing.List[
        chisurf.experiments.reader.ExperimentReader
    ]:
        return self.get_readers()

    @property
    def reader_names(self) -> typing.List[str]:
        return self.get_reader_names()

    @property
    def model_classes(self) -> typing.List[typing.Type[chisurf.models.Model]]:
        return list(self._model_classes)

    @property
    def model_names(self) -> typing.List[str]:
        return self.get_model_names()

    def add_model_class(self, model: typing.Type[chisurf.models.Model]):
        if model not in self.model_classes:
            self._model_classes.append(model)

    def add_model_classes(
            self,
            models: typing.List[
                typing.Type[chisurf.models.Model]
            ]
    ):
        for model in models:
            self.add_model_class(model)

    def add_reader(
            self,
            reader: chisurf.experiments.reader.ExperimentReader,
            controller: chisurf.experiments.reader.ExperimentReaderController = None
    ):
        if reader not in self.readers:
            reader.controller = controller
            self._readers.append(reader)

    def add_readers(
            self,
            readers: typing.List[
                typing.Tuple[
                    chisurf.experiments.reader.ExperimentReader,
                    chisurf.experiments.reader.ExperimentReaderController
                ]
            ]
    ):
        for reader, controller in readers:
            self.add_reader(
                reader,
                controller
            )

    def get_readers(self) -> typing.List[
        chisurf.experiments.reader.ExperimentReader
    ]:
        readers = list()
        for v in self._readers:
            if isinstance(
                    v,
                    chisurf.experiments.reader.ExperimentReader
            ):
                readers.append(v)
            elif isinstance(
                    v,
                    chisurf.experiments.reader.ExperimentReaderController
            ):
                readers.append(v.experiment_reader)
        return readers

    def get_reader_names(self) -> typing.List[str]:
        names = list()
        for s in self.readers:
            names.append(s.name)
        return names

    def get_model_names(self) -> typing.List[str]:
        names = list()
        for s in self.model_classes:
            names.append(str(s.name))
        return names

    def __getstate__(self):
        state = super().__getstate__()
        state['_model_classes'] = self._model_classes
        state['_readers'] = self._readers
        state['name'] = self.__dict__['name']
        state['hidden'] = self.hidden
        return state

    def __str__(self):
        return self.__class__.__name__ + "(" + self.name + ")"

    def __init__(
            self,
            name: str = '',
            hidden: bool = False,
            *args,
            **kwargs
    ):
        super().__init__(*args, name=name, **kwargs)
        self.hidden = hidden
        self._model_classes = list()
        self._readers = list()
