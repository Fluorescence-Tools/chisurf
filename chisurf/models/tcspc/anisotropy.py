from __future__ import annotations

import numpy as np

import chisurf.fluorescence
import chisurf.fitting
from chisurf.fitting.parameter import FittingParameterGroup
import chisurf.math.datatools


class Anisotropy(
    FittingParameterGroup
):
    """

    """

    @property
    def r0(self) -> float:
        return self._r0.value

    @r0.setter
    def r0(
            self,
            v: chisurf.fitting.parameter.FittingParameter
    ):
        self._r0.value = v

    @property
    def l1(self) -> float:
        return self._l1.value

    @l1.setter
    def l1(
            self,
            v: chisurf.fitting.parameter.FittingParameter
    ):
        self._r0.value = v

    @property
    def l2(self) -> float:
        return self._l2.value

    @l2.setter
    def l2(
            self,
            v: chisurf.fitting.parameter.FittingParameter
    ):
        self._l2.value = v

    @property
    def g(self) -> float:
        return self._g.value

    @g.setter
    def g(
            self,
            v: chisurf.fitting.parameter.FittingParameter
    ):
        self._g.value = v

    @property
    def rho(self) -> np.array:
        r = np.array([rho.value for rho in self._rhos], dtype=np.float64)
        r = np.sqrt(r**2)
        for i, v in enumerate(r):
            self._rhos[i].value = v
        return r

    @property
    def b(self) -> np.array:
        a = np.sqrt(np.array([g.value for g in self._bs]) ** 2)
        a /= a.sum()
        a *= self.r0
        for i, g in enumerate(self._bs):
            g.value = a[i]
        return a

    @property
    def rotation_spectrum(self) -> np.array:
        rot = np.empty(2 * len(self), dtype=np.float64)
        rot[0::2] = self.b
        rot[1::2] = self.rho
        return rot

    @property
    def polarization_type(self) -> str:
        return self._polarization_type

    @polarization_type.setter
    def polarization_type(
            self,
            v: str
    ):
        self._polarization_type = v

    def get_decay(
            self,
            lifetime_spectrum: np.ndarray
    ):
        return chisurf.fluorescence.anisotropy.decay.calculcate_spectrum(
            lifetime_spectrum=lifetime_spectrum,
            anisotropy_spectrum=self.rotation_spectrum,
            polarization_type=self.polarization_type,
            g_factor=self.g,
            l1=self.l1,
            l2=self.l2
        )

    def __len__(self):
        return len(self._bs)

    def add_rotation(
            self,
            b: float = 0.2,
            rho: float = 1.0,
            lb: float = 0.0,
            ub: float = 10000.0,
            fixed: bool = False,
            bound_on: bool = False,
            **kwargs
    ):
        b_value = b
        rho_value = rho

        b = chisurf.fitting.parameter.FittingParameter(
            value=b_value,
            lb=lb,
            ub=ub,
            name='b(%i)' % (len(self) + 1),
            fixed=fixed,
            bounds_on=bound_on
        )
        rho = chisurf.fitting.parameter.FittingParameter(
            value=rho_value,
            lb=lb,
            ub=ub,
            name='rho(%i)' % (len(self) + 1),
            fixed=fixed,
            bounds_on=bound_on
        )
        self._rhos.append(rho)
        self._bs.append(b)

    def remove_rotation(
            self
    ) -> None:
        self._rhos.pop().close()
        self._bs.pop().close()

    def __init__(
            self,
            polarization: str = None,
            name: str = 'Anisotropy',
            **kwargs
    ):
        super(Anisotropy, self).__init__(
            name=name,
            **kwargs
        )

        self._rhos = list()
        self._bs = list()

        if polarization is None:
            polarization = chisurf.settings.cs_settings['tcspc']['polarization']
        self._polarization_type = polarization

        self._r0 = chisurf.fitting.parameter.FittingParameter(
            name='r0',
            value=0.38,
            fixed=True
        )
        self._g = chisurf.fitting.parameter.FittingParameter(
            name='g',
            value=1.00,
            fixed=True
        )
        self._l1 = chisurf.fitting.parameter.FittingParameter(
            name='l1',
            value=0.0308,
            fixed=True
        )
        self._l2 = chisurf.fitting.parameter.FittingParameter(
            name='l2',
            value=0.0368,
            fixed=True
        )


