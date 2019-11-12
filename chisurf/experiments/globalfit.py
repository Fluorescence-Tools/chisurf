"""

"""
from __future__ import annotations
import typing

import chisurf.experiments
import chisurf.experiments.data
from . reader import ExperimentReader


class GlobalFitSetup(
    ExperimentReader
):
    """

    """

    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(
            *args,
            **kwargs
        )

    @staticmethod
    def autofitrange(
            *args, **kwargs
    ) -> typing.Tuple[int, int]:
        return 0, 0

    def read(
            self,
            name: str = "Global-fit",
            *args,
            **kwargs
    ):
        return chisurf.experiments.data.DataCurve(
            setup=self,
            name=name
        )

    def __str__(self):
        s = 'Global-Fit\n'
        s += 'Name: \t%s \n' % self.name
        return s


