"""

"""
from __future__ import annotations
from chisurf import typing

import numpy as np
import numba as nb


@nb.jit(nopython=True)
def vm_rt_to_vv_vh(
        times: np.array,
        vm: np.array,
        anisotropy_spectrum: np.ndarray,
        g_factor: float = 1.0,
        l1: float = 0.0,
        l2: float = 0.0
) -> typing.Tuple[
    np.array,
    np.array
]:
    """Computes the VV, and VH decay from an VM decay given an anisotropy
    spectrum

    Parameters
    ----------
    times : numpy-array
        time-axis of the decay
    vm : numpy-array
        The magic angle decay
    anisotropy_spectrum : numpy-array
        The interleaved anisotropy spectrum (amplitude, correlation time)
        of the anisotropy decay
    g_factor : float
        A factor that accounts for different detection sensitivities of
        the parallel and perpendicular detector.
    l1 : float
        A factor that accounts for anisotropy mixing of parallel and
        perpendicular decay.
    l2 : float
        A factor that accounts for anisotropy mixing of parallel and
        perpendicular decay.

    Returns
    -------
    tuple
        A tuple of the parallel (vv) and perpendicular (vh) decay (vv, vh)

    Examples
    --------
    >>> import numpy as np
    >>> import chisurf.fluorescence
    >>> times = np.linspace(0, 50, 32)
    >>> lifetime_spectrum = np.array([1., 4], dtype=np.float)
    >>> times, vm = chisurf.fluorescence.general.calculate_fluorescence_decay(
    ...     lifetime_spectrum=lifetime_spectrum,
    ...     time_axis=times
    ... )
    >>> anisotropy_spectrum = np.array([0.1, 0.6, 0.38 - 0.1, 10.0])
    >>> vv, vh = chisurf.fluorescence.anisotropy.decay.vm_rt_to_vv_vh(
    ...     times,
    ...     vm,
    ...     anisotropy_spectrum
    ... )
    >>> vv
    array([1.20000000e+00, 6.77248886e-01, 4.46852328e-01, 2.98312250e-01,
           1.99308989e-01, 1.33170004e-01, 8.89790065e-02, 5.94523193e-02,
           3.97237334e-02, 2.65418577e-02, 1.77342397e-02, 1.18493310e-02,
           7.91726329e-03, 5.29000820e-03, 3.53457826e-03, 2.36166808e-03,
           1.57797499e-03, 1.05434168e-03, 7.04470208e-04, 4.70699664e-04,
           3.14503256e-04, 2.10138875e-04, 1.40406645e-04, 9.38142731e-05,
           6.26830580e-05, 4.18823877e-05, 2.79841867e-05, 1.86979480e-05,
           1.24932435e-05, 8.34750065e-06, 5.57747611e-06, 3.72665317e-06])
    >>> vh
    array([9.00000000e-01, 6.63617368e-01, 4.46232934e-01, 2.98284106e-01,
           1.99307711e-01, 1.33169946e-01, 8.89790039e-02, 5.94523192e-02,
           3.97237334e-02, 2.65418577e-02, 1.77342397e-02, 1.18493310e-02,
           7.91726329e-03, 5.29000820e-03, 3.53457826e-03, 2.36166808e-03,
           1.57797499e-03, 1.05434168e-03, 7.04470208e-04, 4.70699664e-04,
           3.14503256e-04, 2.10138875e-04, 1.40406645e-04, 9.38142731e-05,
           6.26830580e-05, 4.18823877e-05, 2.79841867e-05, 1.86979480e-05,
           1.24932435e-05, 8.34750065e-06, 5.57747611e-06, 3.72665317e-06])

    Notes
    -----

    The parallel (VV) and perpendicular decay are calculated

    .. math::

    f_{VV}(t) = f_{VM}(t)\cdot(1 + 2 \cdot r(t))
    f_{VH}(t) = f_{VM}(t)\cdot(1 - g \cdot r(t))

    where :math:`g` is the g-factor and :math:`r(t)=\sum_i b_i \cdot exp(-t/\rho_i)`
    is defined by the anisotropy spectrum, :math:`(b_i,\rho_i)_i`.

    The returned parallel, :math:`f_{VV,m}`, and perpendicular :math:`f_{VH,m}`
    decays account for mixing by the factors :math:`l_1` and :math:`l_2`

    .. math::

    f_{VV,m}(t) = (1 - l_1) \cdot f_{VV}(t) + l_1 \cdot f_{VH}(t)
    f_{VH,m}(t) = l_2 \cdot f_{VV}(t) + (1-l_2) \cdot f_{VH}(t)

    """
    rt = np.zeros_like(vm)
    n_anisotropies = int(anisotropy_spectrum.shape[0] // 2)
    for i in range(0, n_anisotropies, 2):
        b = anisotropy_spectrum[i]
        rho = anisotropy_spectrum[i + 1]
        rt += b * np.exp(-times / rho)
    vv = vm * (1 + 2.0 * rt)
    vh = vm * (1. - g_factor * rt)
    vv_j = vv * (1. - l1) + vh * l1
    vh_j = vv * l2 + vh * (1. - l2)
    return vv_j, vh_j

