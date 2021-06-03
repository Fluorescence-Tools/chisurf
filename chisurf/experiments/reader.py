"""

"""
from __future__ import annotations

import abc
from chisurf import typing

import chisurf.base
import chisurf.curve
import chisurf.data
import chisurf.experiments


class ExperimentReader(
    chisurf.base.Base
):

    controller: ExperimentReaderController = None

    def __init__(
            self,
            *args,
            controller: ExperimentReaderController = None,
            **kwargs
    ):
        """

        :param args:
        :param kwargs:
        """
        super().__init__(
            *args,
            **kwargs
        )
        self.controller = controller

    @abc.abstractmethod
    def autofitrange(
            self,
            data: chisurf.base.Data,
            **kwargs
    ) -> typing.Tuple[int, int]:
        return 0, 0

    @abc.abstractmethod
    def read(
            self,
            filename: str = None,
            *args,
            **kwargs
    ) -> chisurf.base.Data:
        """

        :param filename:
        :param kwargs:
        :return:
        """
        pass

    def get_data(
            self,
            **kwargs
    ) -> chisurf.data.ExperimentDataGroup:
        data = self.read(
            **kwargs
        )
        if isinstance(
                data,
                chisurf.data.ExperimentalData
        ):
            data = chisurf.data.ExperimentDataGroup([data])
        if isinstance(
                data,
                chisurf.data.ExperimentDataGroup
        ):
            for d in data:
                d.experiment = self.experiment
                d.setup = self
        return data


class ExperimentReaderController(
    chisurf.base.Base
):
    experiment_reader: ExperimentReader = None

    def __init__(
            self,
            experiment_reader: ExperimentReader = None,
            *args,
            **kwargs
    ):
        super().__init__(
            *args,
            **kwargs
        )
        self.experiment_reader = experiment_reader
        self._call_dict = dict()
        if isinstance(
                experiment_reader,
                ExperimentReader
        ):
            experiment_reader.controller = self

    def __getattr__(
            self,
            item: str
    ):
        return self._experiment_reader.__getattribute__(
            item
        )

    @property
    @abc.abstractmethod
    def filename(
            self
    ) -> str:
        pass

    @abc.abstractmethod
    def get_filename(
            self
    ) -> str:
        pass
