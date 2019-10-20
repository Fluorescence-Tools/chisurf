"""

"""
from chisurf.fluorescence.intensity import nusiance
from . import kappa2
from . import decay


@nusiance
def r_scatter(
        Ss,
        Sp,
        **kwargs
):
    Fp = (Sp - Bp) * Gfactor
    Fs = Ss - Bs
    return (Fp-Fs)/(Fp*(1. - 3. * l2) + Fs*(2. - 3. * l1))


@nusiance
def r_exp(
        Sp,
        Ss,
        **kwargs
) -> float:
    """Experimental anisotropy

        .. math::

        r_{exp} = (F_p-S_s) / (F_p * (1-3*l_2) + S_s * (2-3*l_1))
        F_p = S_p * Gfactor

    :param Sp: Signal in the parallel detection channel
    :param Ss: Signal in the perpendicular detection channel
    :param kwargs:
    :return:
    """
    Fp = Sp * Gfactor
    return (Fp-Ss)/(Fp*(1.-3.*l2) + Ss*(2.-3.*l1))
