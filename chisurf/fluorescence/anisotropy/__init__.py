"""

"""
from chisurf.fluorescence.intensity import nusiance
import chisurf.fluorescence.anisotropy.kappa2
import chisurf.fluorescence.anisotropy.decay


@nusiance
def r_scatter(
        signal_vertical,
        signal_parallel,
        **kwargs
):
    Fp = (signal_parallel - Bp) * Gfactor
    Fs = signal_vertical - Bs
    return (Fp-Fs)/(Fp*(1. - 3. * l2) + Fs*(2. - 3. * l1))


@nusiance
def r_exp(
        signal_parallel,
        signal_vertical,
        **kwargs
) -> float:
    """Experimental anisotropy

        .. math::

        r_{exp} = (F_p-S_s) / (F_p * (1-3*l_2) + S_s * (2-3*l_1))
        F_p = S_p * Gfactor

    :param signal_parallel: Signal in the parallel detection channel
    :param signal_vertical: Signal in the perpendicular detection channel
    :param kwargs:
    :return:
    """
    Fp = signal_parallel * Gfactor
    return (Fp - signal_vertical) / (Fp * (1. - 3. * l2) + signal_vertical * (2. - 3. * l1))
