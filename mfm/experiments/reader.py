from __future__ import annotations
from typing import Tuple

from abc import abstractmethod

import mfm.base
import mfm.curve
import mfm.experiments.data


class ExperimentReader(mfm.base.Base):

    def __init__(self, *args, **kwargs):

        super(ExperimentReader, self).__init__(
            self,
            *args,
            **kwargs
        )

    @staticmethod
    @abstractmethod
    def autofitrange(
            data: mfm.base.Data,
            **kwargs
    ) -> Tuple[float, float]:
        return 0, len(data.y) - 1

    @abstractmethod
    def read(
            self,
            name: str = None,
            **kwargs
    ) -> mfm.base.Data:
        """

        :param name: A name that will be associated to the data set that is read.
        :param kwargs:
        :return:
        """
        pass

    def get_data(
            self,
            **kwargs
    ) -> mfm.experiments.data.ExperimentDataGroup:
        data = self.read(**kwargs)
        if isinstance(data, mfm.experiments.data.ExperimentalData):
            data = mfm.experiments.data.ExperimentDataGroup([data])
        if isinstance(data, mfm.experiments.data.ExperimentDataGroup):
            for d in data:
                d.experiment = self.experiment
                d.setup = self
        return data

