from __future__ import annotations

import numpy as np
import scipy.stats

import chisurf.data
import chisurf.experiments
import chisurf.macros
import chisurf.math
import chisurf.fluorescence
from chisurf.curve import Curve
from chisurf.fitting.parameter import (
    FittingParameterGroup, FittingParameter
)


class Background(FittingParameterGroup):

    @property
    def bg0(self) -> float:
        return self._bg0.value

    @bg0.setter
    def bg0(self, v: float):
        self._bg0.value = v

    @property
    def bg1(self) -> float:
        return self._bg1.value

    @bg1.setter
    def bg1(self, v: float):
        self._bg1.value = v

    def __init__(self, name: str = 'Background', **kwargs):
        super().__init__(name=name, **kwargs)
        self._bg0 = FittingParameter(
            value=0.0,
            name='bg0'
        )
        self._bg1 = FittingParameter(
            value=0.0,
            name='bg1'
        )

