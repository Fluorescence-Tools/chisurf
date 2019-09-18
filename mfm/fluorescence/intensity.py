import numpy as np


def nusiance(
        f,
        *args,
        **kwargs
):
    """Scatter-corrected anisotropy

    :param Ss: perpenticular signal
    :param Sp: parallel signal
    :param Bs: perpenticular background
    :param Bp: parallel background
    :param Gfactor: g-factor
    :param l1: Objective mixing correction factor
    :param l2: Objective mixing correction factor
    :return: Anisotropy
    """

    def m(*args, **kwargs):

        # Anisotropy
        f.func_globals['Gfactor'] = kwargs.get('Gfactor', 1.0)
        f.func_globals['Bp'] = kwargs.get('Bp', 0.0)
        f.func_globals['Bs'] = kwargs.get('Bs', 0.0)
        f.func_globals['l1'] = kwargs.get('l1', 0.0)
        f.func_globals['l2'] = kwargs.get('l2', 0.0)
        # Intensities
        f.func_globals['Bg'] = kwargs.get('Bg', 0.0)
        f.func_globals['Br'] = kwargs.get('Br', 0.0)
        f.func_globals['crosstalk'] = kwargs.get('crosstalk', 0.0)

        f.func_globals['phiA'] = kwargs.get('phiA', 1.0)
        f.func_globals['phiD'] = kwargs.get('phiD', 1.0)

        f.func_globals['R0'] = kwargs.get('R0', 52.0)

        return f(*args, **kwargs)

    return m


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
