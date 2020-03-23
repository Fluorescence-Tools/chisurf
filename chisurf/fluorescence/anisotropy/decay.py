"""

"""
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
    >>> import chisurf.fluorescence.general
    >>> import chisurf.fluorescence.anisotropy
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
    decays account for mixing by the factors :math:`l_1` and :math:`l_2` [1]_

    .. math::

        f_{VV,m}(t) = (1 - l_1) \cdot f_{VV}(t) + l_1 \cdot f_{VH}(t)
        f_{VH,m}(t) = l_2 \cdot f_{VV}(t) + (1-l_2) \cdot f_{VH}(t)

    References
    ----------

    .. [1] Masanori Koshioka, Keiji Sasaki, Hiroshi Masuhara, "Time-Dependent
    Fluorescence Depolarization Analysis in Three-Dimensional
    Microspectroscopy" vol. 49, pp. 224-228, Applied Spectroscopy, 1995

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
        lifetime_spectrum: np.narray,
        anisotropy_spectrum: np.ndarray,
        polarization_type: str,
        g_factor: float = 1.0,
        l1: float = 0.0,
        l2: float = 0.0
) -> np.ndarray:
    """Unites a lifetime and an anisotropy spectrum for a specified polarization
    considering detection sensitivity and mixing of polarization channels.

    This function converts a lifetime spectrum and an anisotropy spectrum into
    a joint spectrum for a detection channel. The detection channel is specified
    by the parameter *polarization_type*. The relative sensitivity of the VV
    and the VH detection channel is considered by the parameter *g_factor*.
    Here, VH stands for vertical excitation and horizontal detection, whereas
    VV stands for vertical excitation and vertical detection. The sensitivity
    for detecting polarized light is considered by the g-factor, :math:`G`

    .. math::

        g = \frac{\int I_{HV}(t)dt}{\int I_{HH}(t)dt}
        r = \frac{I_{VV} - GI_{VH}}{I_{VV} + 2 G I_{VH}}

    Above, :math:`r` is the anisotropy [1]_. The anisotropy is displayed here
    for clarity, as often a G-factor is defined as the inverse of :math:`G`.


    Notes
    -----

    The factor :math:`l_1` and :math:`l_2` describe the mixing of the parallel,
    VV, and the vertical, VH, detection channel [2]_ and are defined by the
    parameters *l1* and *l2*. By default *l1* and *l2* are set to zero. This
    results in pure decays in VV and VH. In case a non-zero value is used the
    returned spectrum corresponds to the following decays:

    .. math::

        f_{VV,m}(t) = (1 - l_1) \cdot f_{VV}(t) + l_1 \cdot f_{VH}(t)
        f_{VH,m}(t) = l_2 \cdot f_{VV}(t) + (1-l_2) \cdot f_{VH}(t)


    Parameters
    ----------
    lifetime_spectrum : numpy-array
        An interleaved array containing a set of amplitudes and fluorescence
        lifetimes (amplitude 1, lifetime 1, amplitude 2, lifetime 2, ...)
    anisotropy_spectrum : numpy-array
        An interleaved array containing a set of amplitudes and depolarization
        times (amplitude 1, rho 1, amplitude 2, rho 2, ...)
    polarization_type : str
        Either 'VV' or 'VH' if the value is neither VV nor VH the lifetime
        spectrum is returned as it is.
    g_factor : float
        Is a factor that corrects the relative sensitivity of the detection
        channels.
    l1 : float
        Is the fraction of VH in the VV detection channel.
    l2 : float
        Is the fraction of VV in the VH detection channel.

    Returns
    -------

    References
    ----------

    .. [1] Masanori Koshioka, Keiji Sasaki, Hiroshi Masuhara, "Time-Dependent
    Fluorescence Depolarization Analysis in Three-Dimensional
    Microspectroscopy" vol. 49, pp. 224-228, Applied Spectroscopy, 1995

    .. [2] Masanori Koshioka, Keiji Sasaki, Hiroshi Masuhara, "Time-Dependent
    Fluorescence Depolarization Analysis in Three-Dimensional
    Microspectroscopy" vol. 49, pp. 224-228, Applied Spectroscopy, 1995


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
    array([ 0.1 ,  4.  ,  0.2 ,  0.8 ,  1.35,  4.  , -2.7 ,  0.8 ])

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
