"""

"""
from __future__ import annotations
from typing import Tuple

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
) -> Tuple[
    np.array,
    np.array
]:
    """Get the VV, VH decay from an VM decay given an anisotropy spectrum

    :param times: time-axis
    :param vm: magic angle decay
    :param anisotropy_spectrum: anisotropy spectrum
    :param g_factor: g-factor
    :param l1:
    :param l2:
    :return: vv, vm

    Example
    -------

    >>> import numpy as np
    >>> import mfm.fluorescence
    >>> import pylab as p
    >>> times = np.linspace(0, 50, 4096)
    >>> lifetime_spectrum = np.array([1., 4], dtype=np.float)
    >>> times, vm = mfm.fluorescence.general.calculate_fluorescence_decay(lifetime_spectrum=lifetime_spectrum, time_axis=times)
    >>> anisotropy_spectrum = np.array([0.1, 8.0, 0.38-0.1, 10.0])
    >>> vv, vh = vm_rt_to_vv_vh(times, vm, anisotropy_spectrum)
    >>> p.plot(times, vv)
    >>> p.plot(times, vh)
    >>> p.plot(times, vm)
    >>> p.show()

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

