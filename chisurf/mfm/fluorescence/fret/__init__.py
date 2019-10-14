"""

"""
import numpy as np

from mfm.fluorescence.intensity import nusiance
from . import acceptor


@nusiance
def sg_sr(
        Sg,
        Sr,
        **kwargs
) -> float:
    """Ratio of the green signal Sg and the red signal Sr

    :param Sg: green signal intensity
    :param Sr: red signal intensity
    :param kwargs:
    :return:
    """
    if Sr != 0:
        return float(Sg) / float(Sr)
    else:
        return float('nan')


@nusiance
def fg_fr(
        Sg,
        Sr,
        **kwargs
) -> float:
    """Ratio of the green fluorescence intensity and the red fluorescence
    intensity. The fluorescence intensities are corrected for the background
    and the cross-talk from the green dye to the red detection channel.

    :param Sg:
    :param Sr:
    :param kwargs:
    :return:
    """
    Fg = Sg - Bg
    Fr = Sr - Br - Fg * crosstalk
    return Fg / Fr


@nusiance
def proximity_ratio(
        Sg,
        Sr,
        **kwargs
) -> float:
    """Proximity ratio of the green and red detection channels. The
    proximity ratio is calculated by Sr / (Sg + Sr) where Sr is the
    uncorrected red signal, and Sg is the uncorrected green signal.

    :param Sg:
    :param Sr:
    :param kwargs:
    :return:
    """
    return Sr / (Sg + Sr)


@nusiance
def apparent_fret_efficiency(
        Sg,
        Sr,
        **kwargs
) -> float:
    """Apparent FRET efficiency (Fr / (Fg + Fr)) that is calculated using the
    fluorescence intensities of the green Fg and red Fr dye. The
    apparent FRET efficiency does not correct for instrumental
    spectral sensitivity of the instrument for the dyes and the
    fluorescence quantum yields of the dye.

    :param Sg:
    :param Sr:
    :param kwargs:
    :return:
    """
    Fg = Sg - Bg
    Fr = Sr - Br - Fg * crosstalk
    return Fr / (Fg + Fr)


@nusiance
def fret_efficiency(
        Sg,
        Sr,
        **kwargs
) -> float:
    """

    :param Sg:
    :param Sr:
    :param kwargs:
    :return:
    """
    Fg = Sg - Bg
    Fa = (Sr - Br - Fg*crosstalk) / phiA

    return Fa / (Fg / Gfactor / phiD + Fa)


@nusiance
def fluorescence_weighted_distance(
        Sg,
        Sr,
        **kwargs
) -> float:
    """The fluorescence weighted distance RDA_E

    :param Sg:
    :param Sr:
    :param kwargs:
    :return:
    """
    Fg = Sg - Bg
    Fa = (Sr - Br - Fg*crosstalk) / phiA
    return R0 * np.exp(1./6.*np.log(Fg/Gfactor/phiD/Fa))


@nusiance
def fret_efficency_to_fdfa(
        E,
        **kwargs
):
    """This function converts the transfer-efficency E to the donor-acceptor intensity ration FD/FA to

    :param E: float
        The transfer-efficency
    :param phiD: float
        donor quantum yield
    :param phiA: float
        acceptor quantum yield
    :return: float, the FRET transfer efficency
    """
    fdfa = phiA / phiD * (1.0 / E - 1.0)
    return fdfa


@nusiance
def fdfa2transfer_efficency(fdfa):
    """This function converts the donor-acceptor intensity ration FD/FA to the transfer-efficency E

    :param fdfa: float
        donor acceptor intensity ratio
    :param phiD: float
        donor quantum yield
    :param phiA: float
        acceptor quantum yield
    :return: float, the FRET transfer efficency
    """
    r = 1.0 + fdfa * phiA / phiD
    trans = 1.0/r
    return trans