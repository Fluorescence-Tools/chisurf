from __future__ import annotations

"""Nuisance and background parameter groups for PDA FRET analysis.

This module defines small :class:`FittingParameterGroup` subclasses used by
PDA models to keep track of background count rates and experiment-specific
"nuisance" parameters (crosstalk, detection efficiencies, quantum yields,
and photon-number selection).

The classes are lightweight wrappers around
``chisurf.fitting.parameter.FittingParameterGroup`` and are safe to
instantiate without any external data files. They are intended to be
combined with higher-level PDA models such as
``chisurf.models.pda.simple.PdaGaussianDistanceModel``.
"""

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

    """Background count-rate parameters for two detection channels.

    The group exposes two scalar parameters, ``bg0`` and ``bg1``, which
    represent background contributions in the green and red detection
    channels, respectively. They are stored internally as
    :class:`chisurf.fitting.parameter.FittingParameter` instances but
    exposed as simple floats via properties.

    Examples
    --------
    >>> from chisurf.models.pda.nusiance import Background
    >>> bg = Background()
    >>> (bg.bg0, bg.bg1)
    (0.0, 0.0)
    >>> bg.bg0 = 5.0
    >>> bg.bg1 = 10.0
    >>> (bg.bg0, bg.bg1)
    (5.0, 10.0)
    """

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
            name='bg0',
            lb=0.0,
            ub=100.0,
            bounds_on=True
        )
        self._bg1 = FittingParameter(
            value=0.0,
            name='bg1',
            lb=0.0,
            ub=100.0,
            bounds_on=True
        )


class PdaFretNuisance(FittingParameterGroup):

    """Nuisance parameters for discrete PDA/FRET models.

    This parameter group collects experimental factors that influence the
    shape of the 2D PDA histogram but are not part of the distance
    distribution itself. The main parameters are background count rates,
    quantum yields, an optional excitation/emission description, and the
    photon-number range used for PDA scoring.

    When a ``fit`` object with PDA metadata is attached, the constructor
    attempts to initialize ``nPh_min`` and ``nPh_max`` from the dataset's
    ``minimum_number_of_photons`` and ``maximum_number_of_photons``
    entries. Without such metadata the defaults are both zero.
    """

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

    # --- Detector efficiencies per channel (green/red) -------------------

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

    # --- Excitation probabilities (absolute, user-supplied) --------------

    @property
    def ExDG(self) -> float:
        return self._ExDG.value

    @ExDG.setter
    def ExDG(self, v: float):
        self._ExDG.value = v

    @property
    def ExAG(self) -> float:
        return self._ExAG.value

    @ExAG.setter
    def ExAG(self, v: float):
        self._ExAG.value = v

    # --- Emission / detection crosstalk matrix (c_{channel|species}) ----

    @property
    def cGD(self) -> float:
        return self._cGD.value

    @cGD.setter
    def cGD(self, v: float):
        self._cGD.value = v

    @property
    def cGA(self) -> float:
        return self._cGA.value

    @cGA.setter
    def cGA(self, v: float):
        self._cGA.value = v

    @property
    def cRD(self) -> float:
        return self._cRD.value

    @cRD.setter
    def cRD(self, v: float):
        self._cRD.value = v

    @property
    def cRA(self) -> float:
        return self._cRA.value

    @cRA.setter
    def cRA(self, v: float):
        self._cRA.value = v

    # --- Derived legacy-style crosstalk parameters (read-only) -----------

    @property
    def alpha(self) -> float:
        """Donor bleed-through fraction into the red channel.

        Defined for a donor-only sample as the fraction of donor photons
        detected in the red channel,

            alpha = R_D0 / (G_D0 + R_D0)

        with G_D0 and R_D0 computed from gG/gR and cGD/cRD. The value is
        populated by PDA models that use this nuisance group and is not
        varied during fitting.
        """
        try:
            return float(self._alpha.value)
        except Exception:
            return float("nan")

    @property
    def alpha_A(self) -> float:
        """Acceptor bleed-through fraction into the green channel.

        Defined for an acceptor-only sample as the fraction of acceptor
        photons detected in the green channel,

            alpha_A = G_A0 / (G_A0 + R_A0)

        with G_A0 and R_A0 computed from gG/gR and cGA/cRA. The value is
        populated by PDA models that use this nuisance group and is not
        varied during fitting.
        """
        try:
            return float(self._alpha_A.value)
        except Exception:
            return float("nan")

    @property
    def nPh_min(self) -> float:
        return self._nPh_min.value

    @nPh_min.setter
    def nPh_min(self, v: float):
        self._nPh_min.value = v

    @property
    def nPh_max(self) -> float:
        return self._nPh_max.value

    @nPh_max.setter
    def nPh_max(self, v: float):
        self._nPh_max.value = v

    def __init__(self, name: str = 'PDA-FRET-nuisance', **kwargs):
        super().__init__(name=name, **kwargs)
        self._bgG = FittingParameter(
            value=0.0,
            name='BG',
            lb=0.0,
            ub=100.0,
            bounds_on=True
        )
        self._bgR = FittingParameter(
            value=0.0,
            name='BR',
            lb=0.0,
            ub=100.0,
            bounds_on=True
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
        # Per-channel detector efficiencies (green/red).
        self._gG = FittingParameter(
            value=1.0,
            name='gG',
            fixed=True,
        )
        self._gR = FittingParameter(
            value=1.0,
            name='gR',
            fixed=True,
        )
        # Absolute excitation probabilities (e.g. extinction coefficients at
        # the donor excitation wavelength). By default they are zero so that
        # legacy alpha/gamma-based behaviour is preserved until explicitly
        # configured by the user.
        self._ExDG = FittingParameter(
            value=1.0,
            name='ExDG',
            fixed=True,
        )
        self._ExAG = FittingParameter(
            value=0.0,
            name='ExAG',
            fixed=True,
        )
        # Full emission / detection crosstalk matrix elements. These default
        # to zero so that existing projects using alpha/gamma are unaffected
        # unless the user explicitly supplies matrix entries.
        self._cGD = FittingParameter(
            value=1.0,
            name='cGD',
            fixed=True,
        )
        self._cGA = FittingParameter(
            value=0.0,
            name='cGA',
            fixed=True,
        )
        self._cRD = FittingParameter(
            value=0.02,
            name='cRD',
            fixed=True,
        )
        self._cRA = FittingParameter(
            value=1.0,
            name='cRA',
            fixed=True,
        )
        # Derived legacy-style crosstalk parameters. These are populated by
        # PDA models that use this nuisance group and are shown as fixed,
        # read-only parameters (analogous to CPM in FCS widgets).
        self._alpha = FittingParameter(
            value=float("nan"),
            name='alpha',
            fixed=True,
            is_output=True,
        )
        self._alpha_A = FittingParameter(
            value=float("nan"),
            name='alpha_A',
            fixed=True,
            is_output=True,
        )
        # Photon-number range for PDA scoring (nPh_min, nPh_max).
        # Defaults are taken from the attached dataset's PDA metadata if
        # available; otherwise they fall back to zero.
        default_nmin = 30.0
        default_nmax = 0.0
        try:
            fit = getattr(self, 'fit', None)
            data = getattr(fit, 'data', None) if fit is not None else None
            pda_meta = getattr(data, 'pda', None)
            if isinstance(pda_meta, dict):
                default_nmin = float(pda_meta.get('minimum_number_of_photons', default_nmin))
                default_nmax = float(pda_meta.get('maximum_number_of_photons', default_nmax))
        except Exception:
            default_nmin = 30.0
            default_nmax = 0.0
        self._nPh_min = FittingParameter(
            value=default_nmin,
            name='nPh_min',
            fixed=True,
        )
        self._nPh_max = FittingParameter(
            value=default_nmax,
            name='nPh_max',
            fixed=True,
        )


class PdaPhotonRange(FittingParameterGroup):

    @property
    def nPh_min(self) -> float:
        return self._nPh_min.value

    @nPh_min.setter
    def nPh_min(self, v: float):
        self._nPh_min.value = v

    @property
    def nPh_max(self) -> float:
        return self._nPh_max.value

    @nPh_max.setter
    def nPh_max(self, v: float):
        self._nPh_max.value = v

    def __init__(self, name: str = 'PDA-photon-range', **kwargs):
        super().__init__(name=name, **kwargs)
        default_nmin = 30.0
        default_nmax = 0.0
        try:
            fit = getattr(self, 'fit', None)
            data = getattr(fit, 'data', None) if fit is not None else None
            pda_meta = getattr(data, 'pda', None)
            if isinstance(pda_meta, dict):
                default_nmin = float(pda_meta.get('minimum_number_of_photons', default_nmin))
                default_nmax = float(pda_meta.get('maximum_number_of_photons', default_nmax))
        except Exception:
            default_nmin = 30.0
            default_nmax = 0.0
        self._nPh_min = FittingParameter(
            value=default_nmin,
            name='nPh_min',
            fixed=True,
        )
        self._nPh_max = FittingParameter(
            value=default_nmax,
            name='nPh_max',
            fixed=True,
        )
