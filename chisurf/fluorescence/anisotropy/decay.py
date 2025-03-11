from __future__ import annotations
from chisurf import typing

import numpy as np
import numba as nb
import chisurf.math


@nb.jit(nopython=True)
def vm_rt_to_vv_vh(
        times: np.array,
        vm: np.array,
        anisotropy_spectrum: np.ndarray,
        g_factor: float = 1.0,
        l1: float = 0.0,
        l2: float = 0.0
) -> typing.Tuple[np.array, np.array]:
    """
    Computes the VV and VH decays from a VM decay given an anisotropy spectrum.

    The parallel (VV) and perpendicular (VH) decays are computed as

        f_VV(t) = f_VM(t) * (1 + 2 * r(t))
        f_VH(t) = f_VM(t) * (1 - g * r(t))

    where g is the g-factor and r(t) is calculated from the anisotropy spectrum:

        r(t) = sum_i (b_i * exp(-t / rho_i))

    The mixing parameters l1 and l2 account for cross-talk between the
    polarization channels according to:

        f_VV,m(t) = (1 - l1) * f_VV(t) + l1 * f_VH(t)
        f_VH,m(t) = l2 * f_VV(t) + (1 - l2) * f_VH(t)

    Parameters
    ----------
    times : numpy.array
        Time-axis of the decay.
    vm : numpy.array
        The magic angle (VM) decay.
    anisotropy_spectrum : numpy.array
        An interleaved array containing anisotropy parameters: amplitude and
        correlation time pairs (b_i, rho_i).
    g_factor : float, optional
        Correction factor for different detection sensitivities (default is 1.0).
    l1 : float, optional
        Mixing factor for the parallel (VV) channel (default is 0.0).
    l2 : float, optional
        Mixing factor for the perpendicular (VH) channel (default is 0.0).

    Returns
    -------
    tuple of numpy.array
        A tuple (vv_j, vh_j) where vv_j is the mixed VV decay and vh_j is the
        mixed VH decay.

    Examples
    --------
    >>> import numpy as np
    >>> import chisurf.fluorescence.general
    >>> import chisurf.fluorescence.anisotropy
    >>> times = np.linspace(0, 50, 32)
    >>> lifetime_spectrum = np.array([1.0, 4.0], dtype=np.float64)
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
    >>> vv[0]  # doctest: +ELLIPSIS
    1.2...
    >>> vh[0]  # doctest: +ELLIPSIS
    0.9...

    Notes
    -----
    The returned decays account for anisotropy mixing as described in [1]_.

    References
    ----------
    .. [1] Masanori Koshioka, Keiji Sasaki, Hiroshi Masuhara, "Time-Dependent
           Fluorescence Depolarization Analysis in Three-Dimensional
           Microspectroscopy", Applied Spectroscopy, 1995, vol. 49, pp. 224-228.
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


def calculcate_spectrum(
        lifetime_spectrum: np.ndarray,
        anisotropy_spectrum: np.ndarray,
        polarization_type: str,
        g_factor: float = 1.0,
        l1: float = 0.0,
        l2: float = 0.0
) -> np.ndarray:
    """
    Generates a joint spectrum from a lifetime and an anisotropy spectrum for a specified polarization.

    This function converts a lifetime spectrum and an anisotropy spectrum into a
    joint spectrum for either the 'VV' or 'VH' detection channels. The relative
    sensitivity of the channels is adjusted via the g_factor, while l1 and l2
    describe the mixing between the VV and VH channels.

    The unmixed decays for VV and VH are given by:

        f_VV(t) = f_VM(t) * (1 + 2 * r(t))
        f_VH(t) = f_VM(t) * (1 - g * r(t))

    with anisotropy:

        r(t) = (I_VV - G * I_VH) / (I_VV + 2 * G * I_VH)

    The mixed decays are then computed as:

        f_VV,m(t) = (1 - l1) * f_VV(t) + l1 * f_VH(t)
        f_VH,m(t) = l2 * f_VV(t) + (1 - l2) * f_VH(t)

    Parameters
    ----------
    lifetime_spectrum : numpy.array
        Interleaved amplitudes and fluorescence lifetimes
        (amplitude 1, lifetime 1, amplitude 2, lifetime 2, ...).
    anisotropy_spectrum : numpy.array
        Interleaved amplitudes and depolarization times
        (amplitude 1, rho 1, amplitude 2, rho 2, ...).
    polarization_type : str
        'VV' or 'VH'. If neither, the lifetime spectrum is returned unmodified.
    g_factor : float, optional
        Correction factor for the detection channel sensitivity (default is 1.0).
    l1 : float, optional
        Fraction of VH contributing to the VV channel (default is 0.0).
    l2 : float, optional
        Fraction of VV contributing to the VH channel (default is 0.0).

    Returns
    -------
    numpy.array
        The combined spectrum for the specified detection channel.

    Examples
    --------
    >>> import numpy as np
    >>> from chisurf.fluorescence.anisotropy.decay import calculcate_spectrum
    >>> lifetime_spectrum = np.array([1.0, 4.0])
    >>> anisotropy_spectrum = np.array([1.0, 1.0])
    >>> g_factor = 1.5
    >>> calculcate_spectrum(
    ...     lifetime_spectrum=lifetime_spectrum,
    ...     anisotropy_spectrum=anisotropy_spectrum,
    ...     polarization_type='VV',
    ...     g_factor=g_factor,
    ...     l1=0.0,
    ...     l2=0.0
    ... )
    array([ 1. ,  4. ,  2. ,  0.8,  0. ,  4. , -0. ,  0.8])
    >>> calculcate_spectrum(
    ...     lifetime_spectrum=lifetime_spectrum,
    ...     anisotropy_spectrum=anisotropy_spectrum,
    ...     polarization_type='VV',
    ...     g_factor=g_factor,
    ...     l1=0.1,
    ...     l2=0.0
    ... )
    array([ 0.9 ,  4.  ,  1.8 ,  0.8 ,  0.15,  4.  , -0.3 ,  0.8 ])
    >>> calculcate_spectrum(
    ...     lifetime_spectrum=lifetime_spectrum,
    ...     anisotropy_spectrum=anisotropy_spectrum,
    ...     polarization_type='VH',
    ...     g_factor=g_factor,
    ...     l1=0.0,
    ...     l2=0.0
    ... )
    array([ 0. ,  4. ,  0. ,  0.8,  1.5,  4. , -3. ,  0.8])
    >>> calculcate_spectrum(
    ...     lifetime_spectrum=lifetime_spectrum,
    ...     anisotropy_spectrum=anisotropy_spectrum,
    ...     polarization_type='VH',
    ...     g_factor=g_factor,
    ...     l1=0.0,
    ...     l2=0.1
    ... )
    array([ 0.1 ,  4.  ,  0.2 ,  0.8,  1.35,  4.  , -2.7 ,  0.8 ])

    Notes
    -----
    If the polarization_type is neither 'VV' nor 'VH', the function simply
    returns the input lifetime spectrum without modifications.

    References
    ----------
    .. [1] Masanori Koshioka, Keiji Sasaki, Hiroshi Masuhara, "Time-Dependent
           Fluorescence Depolarization Analysis in Three-Dimensional
           Microspectroscopy", Applied Spectroscopy, 1995, vol. 49, pp. 224-228.
    .. [2] Same as [1].
    """
    polarization_type = polarization_type.upper()
    f = lifetime_spectrum
    a = anisotropy_spectrum
    if (polarization_type == "VV") or (polarization_type == "VH"):
        d = chisurf.math.datatools.elte2(a, f)
        vv = np.hstack([f, chisurf.math.datatools.e1tn(d, 2)])
        vh = chisurf.math.datatools.e1tn(
            np.hstack([f, chisurf.math.datatools.e1tn(d, -1)]),
            g_factor
        )
        if polarization_type == 'VH':
            return np.hstack(
                [chisurf.math.datatools.e1tn(vv, l2),
                 chisurf.math.datatools.e1tn(vh, 1 - l2)]
            )
        elif polarization_type == 'VV':
            r = np.hstack(
                [chisurf.math.datatools.e1tn(vv, 1 - l1),
                 chisurf.math.datatools.e1tn(vh, l1)]
            )
            return r
    else:
        return f
