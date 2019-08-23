from __future__ import annotations
from typing import Tuple

import mfm.base
import mfm.curve
from mfm.experiments.data import ExperimentalData, ExperimentDataGroup


class ExperimentReader(mfm.base.Base):

    def __init__(self, *args, **kwargs):
        super(ExperimentReader, self).__init__(self, *args, **kwargs)

    @staticmethod
    def autofitrange(
            data: mfm.curve.Curve,
            **kwargs
    ) -> Tuple[float, float]:
        return 0, len(data.y) - 1

    def read(self, **kwargs):
        pass

    def get_data(
            self,
            **kwargs
    ) -> ExperimentDataGroup:
        data = self.read(**kwargs)
        if isinstance(data, ExperimentalData):
            data = ExperimentDataGroup([data])
        if isinstance(data, ExperimentDataGroup):
            for d in data:
                d.experiment = self.experiment
                d.setup = self
        return data
