"""

"""
from __future__ import annotations
from typing import Tuple

from collections import OrderedDict
from qtpy import QtWidgets

import chisurf.settings as mfm
import chisurf.experiments
import chisurf.experiments.data
from . reader import ExperimentReader


class GlobalFitSetup(
    ExperimentReader,
    QtWidgets.QWidget
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
        self.hide()
        self.parameterWidgets = list()
        self.parameters = OrderedDict([])

    @staticmethod
    def autofitrange(
            *args, **kwargs
    ) -> Tuple[int, int]:
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

