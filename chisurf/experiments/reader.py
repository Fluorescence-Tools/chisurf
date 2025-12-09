"""Experiment readers and their optional GUI controllers.

This module defines abstract base classes that connect low-level I/O
(:class:`ExperimentReader`) with higher-level controllers
(:class:`ExperimentReaderController`). Concrete readers are responsible
for turning files into :class:`chisurf.data.ExperimentalData` objects.
"""
from __future__ import annotations

import abc
import pathlib

from chisurf import typing

import chisurf.base
import chisurf.curve
import chisurf.data
import chisurf.experiments
class ExperimentReader(chisurf.base.Base):
    """Abstract base class for loading experimental data.

    Subclasses implement :meth:`read` (file I/O) and
    :meth:`autofitrange`. The helper :meth:`get_data` wraps a single
    dataset into an :class:`chisurf.data.ExperimentDataGroup` and
    attaches experiment and setup references.

    Examples
    --------
    Create a tiny reader that returns a synthetic curve and expose it as
    a one-element :class:`chisurf.data.ExperimentDataGroup`:

    >>> import numpy as np
    >>> class _DummyReader(ExperimentReader):
    ...     def __init__(self, *args, **kwargs):
    ...         super().__init__(*args, **kwargs)
    ...         self.experiment = None
    ...     def autofitrange(self, data, **kwargs):
    ...         return 0, len(data)
    ...     def read(self, *args, **kwargs):
    ...         return chisurf.data.DataCurve(x=np.array([0.0, 1.0]),
    ...                                      y=np.array([1.0, 2.0]))
    >>> r = _DummyReader(name="dummy")
    >>> group = r.get_data()
    >>> len(group)
    1
    """

    controller: ExperimentReaderController = None

    def __init__(
            self,
            *args,
            controller: ExperimentReaderController = None,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.controller = controller

    @abc.abstractmethod
    def autofitrange(self, data: chisurf.base.Data, **kwargs) -> typing.Tuple[int, int]:
        if isinstance(data, chisurf.data.DataCurve):
            return 0, len(data)
        else:
            return 0, 0

    def __getstate__(self):
        state = super().__getstate__()
        state.update(
            chisurf.base.to_elementary(self.__dict__.copy(), skip_qt_widgets=True)
        )
        return state

    @abc.abstractmethod
    def read(
            self,
            filename: str = None,
            *args,
            **kwargs
    ) -> chisurf.base.Data:
        pass

    def get_data(self, **kwargs) -> chisurf.data.ExperimentDataGroup:
        data = self.read(**kwargs)
        if isinstance(data, chisurf.data.ExperimentalData):
            data = chisurf.data.ExperimentDataGroup([data])
        if isinstance(data, chisurf.data.ExperimentDataGroup):
            for d in data:
                d.experiment = self.experiment
                d.setup = self
                try:
                    d.data_reader = self
                except Exception:
                    pass
            try:
                if not hasattr(data, "data_reader"):
                    data.data_reader = self
            except Exception:
                pass
        return data


class ExperimentReaderController(chisurf.base.Base):
    """Base class that couples a reader to higher-level UIs.

    The controller can manage file selection and other GUI-related
    aspects while delegating computational work to the underlying
    :class:`ExperimentReader` instance.
    """

    experiment_reader: ExperimentReader = None

    def __init__(self, experiment_reader: ExperimentReader = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.experiment_reader = experiment_reader
        self._call_dict = dict()
        if isinstance(experiment_reader, ExperimentReader):
            experiment_reader.controller = self

    def __getattr__(self, item: str):
        """Delegate attribute access to the underlying experiment reader."""
        return getattr(self.experiment_reader, item)

    @property
    @abc.abstractmethod
    def filename(self) -> str:
        pass

    @abc.abstractmethod
    def get_filename(self) -> pathlib.Path:
        pass
