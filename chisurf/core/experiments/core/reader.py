"""Experiment readers and their optional GUI controllers.

This module defines abstract base classes that connect low-level I/O
(:class:`ExperimentReader`) with higher-level controllers
(:class:`ExperimentReaderController`). Concrete readers are responsible
for turning files into :class:`chisurf.core.data.ExperimentalData` objects.
"""
from __future__ import annotations

import abc
import pathlib

from chisurf import typing

import chisurf.core.base
import chisurf.core.curve
import chisurf.core.data


class ExperimentReader(chisurf.core.base.Base):
    """Abstract base class for loading experimental data.

    Subclasses implement :meth:`read` (file I/O) and
    :meth:`autofitrange`. The helper :meth:`get_data` wraps a single
    dataset into an :class:`chisurf.core.data.ExperimentDataGroup` and
    attaches experiment and setup references.

    Examples
    --------
    Create a tiny reader that returns a synthetic curve and expose it as
    a one-element :class:`chisurf.core.data.ExperimentDataGroup`:

    >>> import numpy as np
    >>> class _DummyReader(ExperimentReader):
    ...     def __init__(self, *args, **kwargs):
    ...         super().__init__(*args, **kwargs)
    ...         self.experiment = None
    ...     def autofitrange(self, data, **kwargs):
    ...         return 0, len(data)
    ...     def read(self, *args, **kwargs):
    ...         return chisurf.core.data.DataCurve(x=np.array([0.0, 1.0]),
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
        """Initialize the experiment reader.

        Parameters
        ----------
        controller : ExperimentReaderController, optional
            Associated controller instance.
        """
        super().__init__(*args, **kwargs)
        self.controller = controller

    @abc.abstractmethod
    def autofitrange(self, data: chisurf.core.base.Data, **kwargs) -> typing.Tuple[int, int]:
        """Determine the default fit range for a dataset.

        Parameters
        ----------
        data : chisurf.core.base.Data
            The experimental data to estimate a fit range for.

        Returns
        -------
        tuple of int
            ``(start, stop)`` indices defining the fit interval.
        """
        if isinstance(data, chisurf.core.data.DataCurve):
            return 0, len(data)
        else:
            return 0, 0

    def __getstate__(self):
        """Serialize the reader state, skipping Qt widgets.

        Returns
        -------
        dict
            Pickle-friendly state dictionary.
        """
        state = super().__getstate__()
        state.update(
            chisurf.core.base.to_elementary(self.__dict__.copy(), skip_qt_widgets=True)
        )
        return state

    @abc.abstractmethod
    def read(
            self,
            filename: str = None,
            *args,
            **kwargs
    ) -> chisurf.core.base.Data:
        """Read experimental data from a file.

        Parameters
        ----------
        filename : str, optional
            Path to the data file.

        Returns
        -------
        chisurf.core.base.Data
            The loaded experimental data.
        """
        pass

    def get_data(self, **kwargs) -> chisurf.core.data.ExperimentDataGroup:
        """Read data and wrap it in an :class:`ExperimentDataGroup`.

        Attaches the experiment, setup, and data_reader references
        to each dataset in the group.

        Parameters
        ----------
        **kwargs
            Forwarded to :meth:`read`.

        Returns
        -------
        chisurf.core.data.ExperimentDataGroup
            Group containing the loaded data with metadata attached.
        """
        data = self.read(**kwargs)
        if isinstance(data, chisurf.core.data.ExperimentalData):
            data = chisurf.core.data.ExperimentDataGroup([data])
        if isinstance(data, chisurf.core.data.ExperimentDataGroup):
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


class ExperimentReaderController(chisurf.core.base.Base):
    """Base class that couples a reader to higher-level UIs.

    The controller can manage file selection and other GUI-related
    aspects while delegating computational work to the underlying
    :class:`ExperimentReader` instance.
    """

    experiment_reader: ExperimentReader = None

    def __init__(self, experiment_reader: ExperimentReader = None, *args, **kwargs):
        """Initialize the reader controller.

        Parameters
        ----------
        experiment_reader : ExperimentReader, optional
            The reader instance the controller manages.
        """
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
        """The filename of the currently selected data file."""
        pass

    @abc.abstractmethod
    def get_filename(self) -> pathlib.Path:
        """Return the path to the currently selected data file.

        Returns
        -------
        pathlib.Path
            Path object for the selected file.
        """
        pass
