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


class PdaFretNuisance(FittingParameterGroup):

    @property
    def alpha(self) -> float:
        return self._alpha.value

    @alpha.setter
    def alpha(self, v: float):
        self._alpha.value = v

    @property
    def BG(self) -> float:
        return self._bgG.value

    @BG.setter
    def BG(self, v: float):
        self._bgG.value = v

    @property
    def BR(self) -> float:
        return self._bgR.value

    @BR.setter
    def BR(self, v: float):
        self._bgR.value = v

    @property
    def gG(self) -> float:
        return self._gG.value

    @gG.setter
    def gG(self, v: float):
        self._gG.value = v

    @property
    def gR(self) -> float:
        return self._gR.value

    @gR.setter
    def gR(self, v: float):
        self._gR.value = v

    @property
    def QYD(self) -> float:
        return self._QYD.value

    @QYD.setter
    def QYD(self, v: float):
        self._QYD.value = v

    @property
    def QYA(self) -> float:
        return self._QYA.value

    @QYA.setter
    def QYA(self, v: float):
        self._QYA.value = v

    def __init__(self, name: str = 'PDA-FRET-nuisance', **kwargs):
        super().__init__(name=name, **kwargs)
        self._alpha = FittingParameter(
            value=0.0,
            name='alpha'
        )
        self._bgG = FittingParameter(
            value=0.0,
            name='BG'
        )
        self._bgR = FittingParameter(
            value=0.0,
            name='BR'
        )
        self._gG = FittingParameter(
            value=1.0,
            name='gG',
            fixed=True
        )
        self._gR = FittingParameter(
            value=1.0,
            name='gR',
            fixed=True
        )
        self._QYD = FittingParameter(
            value=0.8,
            name='QYD',
            fixed=True
        )
        self._QYA = FittingParameter(
            value=0.3,
            name='QYA',
            fixed=True
        )
