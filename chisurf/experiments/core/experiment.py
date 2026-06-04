from __future__ import annotations
from chisurf import typing

import chisurf.base
import chisurf.models

from chisurf.experiments.core.reader import ExperimentReader, ExperimentReaderController


class Experiment(chisurf.base.Base):
    """Lightweight registry of models and readers for a ChiSurf experiment.

    An :class:`Experiment` keeps track of which model classes and
    :class:`chisurf.experiments.core.reader.ExperimentReader` instances belong to
    a conceptual experiment type. Higher level GUIs use this information to
    decide which data can be loaded and which models are applicable.

    Attributes
    ----------
    model_classes : list of type[chisurf.models.Model]
        Registered model classes associated with the experiment.
    readers : list of chisurf.experiments.core.reader.ExperimentReader
        Registered readers (possibly attached via controllers).
    hidden : bool
        If *True*, the experiment is hidden from interactive UIs.

    Examples
    --------
    Create a minimal experiment without models or readers:

    >>> from chisurf.experiments.core.experiment import Experiment
    >>> exp = Experiment(name="Test")
    >>> exp.name
    'Test'
    >>> exp.model_classes
    []
    >>> exp.readers
    []
    """

    hidden: bool = False

    @property
    def readers(self) -> typing.List[ExperimentReader]:
        """List of :class:`ExperimentReader` instances registered for this experiment."""
        return self.get_readers()

    @property
    def reader_names(self) -> typing.List[str]:
        """Human-readable names of all registered readers."""
        return self.get_reader_names()

    @property
    def model_classes(self) -> typing.List[typing.Type[chisurf.models.Model]]:
        """List of registered model classes."""
        return list(self._model_classes)

    @property
    def model_names(self) -> typing.List[str]:
        """Human-readable names of all registered model classes."""
        return self.get_model_names()

    def add_model_class(self, model: typing.Type[chisurf.models.Model]):
        """Register a single model class with this experiment.

        Parameters
        ----------
        model : type
            A :class:`chisurf.models.Model` subclass to register.
        """
        if model not in self.model_classes:
            self._model_classes.append(model)

    def add_model_classes(
            self,
            models: typing.List[
                typing.Type[chisurf.models.Model]
            ]
    ):
        """Register multiple model classes at once.

        Parameters
        ----------
        models : list of type
            List of :class:`chisurf.models.Model` subclasses.
        """
        for model in models:
            self.add_model_class(model)

    def add_reader(
            self,
            reader: ExperimentReader,
            controller: ExperimentReaderController = None
    ):
        """Register a single reader (optionally with a controller).

        Parameters
        ----------
        reader : ExperimentReader
            The reader instance to register.
        controller : ExperimentReaderController, optional
            Associated controller.
        """
        if reader not in self.readers:
            reader.controller = controller
            self._readers.append(reader)

    def add_readers(
            self,
            readers: typing.List[
                typing.Tuple[
                    ExperimentReader,
                    ExperimentReaderController
                ]
            ]
    ):
        """Register multiple reader/controller pairs.

        Parameters
        ----------
        readers : list of tuple
            Each element is ``(reader, controller)``.
        """
        for reader, controller in readers:
            self.add_reader(
                reader,
                controller
            )

    def get_readers(self) -> typing.List[ExperimentReader]:
        """Return all :class:`ExperimentReader` instances for this experiment.

        Internally, ``_readers`` may hold either readers directly or
        :class:`ExperimentReaderController` objects; in the latter case the
        underlying :attr:`experiment_reader` is returned.
        """
        readers = list()
        for v in self._readers:
            if isinstance(
                    v,
                    ExperimentReader
            ):
                readers.append(v)
            elif isinstance(
                    v,
                    ExperimentReaderController
            ):
                readers.append(v.experiment_reader)
        return readers

    def get_reader_names(self) -> typing.List[str]:
        """Return the names of all registered readers."""
        names = list()
        for s in self.readers:
            names.append(s.name)
        return names

    def get_model_names(self) -> typing.List[str]:
        """Return the names of all registered model classes."""
        names = list()
        for s in self.model_classes:
            names.append(str(s.name))
        return names

    def __getstate__(self):
        """Serialize the experiment state for pickling.

        Returns
        -------
        dict
            Pickle-friendly state dictionary.
        """
        state = super().__getstate__()
        state['_model_classes'] = self._model_classes
        state['_readers'] = self._readers
        state['name'] = self.__dict__['name']
        state['hidden'] = self.hidden
        return state

    def __str__(self):
        """Human-readable string representation."""
        return self.__class__.__name__ + "(" + self.name + ")"

    def __init__(
            self,
            name: str = '',
            hidden: bool = False,
            *args,
            **kwargs
    ):
        """Initialize an experiment registry entry.

        Parameters
        ----------
        name : str
            Human-readable experiment name.
        hidden : bool
            If *True*, the experiment is hidden from user interfaces.
        """
        super().__init__(*args, name=name, **kwargs)
        self.hidden = hidden
        self._model_classes = list()
        self._readers = list()
