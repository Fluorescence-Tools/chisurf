from __future__ import annotations

import mfm.base
from mfm.experiments.data import ExperimentalData, ExperimentDataGroup


class Reader(mfm.base.Base):

    def __init__(self, *args, **kwargs):
        super(Reader, self).__init__(self, *args, **kwargs)

    @staticmethod
    def autofitrange(data, **kwargs):
        return 0, len(data.y) - 1

    def read(self, **kwargs):
        pass

    def get_data(self, **kwargs):
        data = self.read(**kwargs)
        if isinstance(data, ExperimentalData):
            data = ExperimentDataGroup([data])
        for d in data:
            d.experiment = self.experiment
            d.setup = self
        return data