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
def r_exp(Sp, Ss, **kwargs):
    Fp = Sp * Gfactor
    return (Fp-Ss)/(Fp*(1.-3.*l2) + Ss*(2.-3.*l1))


@nusiance
def sg_sr(Sg, Sr, **kwargs):
    if Sr != 0:
        return float(Sg) / float(Sr)
    else:
        return float('nan')


@nusiance
def fg_fr(Sg, Sr, **kwargs):
    Fg = Sg - Bg
    Fr = Sr - Br - Fg * crosstalk
    return Fg / Fr


@nusiance
def proximity_ratio(Sg, Sr, **kwargs):
    return Sr/(Sg + Sr)


@nusiance
def apparent_efficiency(Sg, Sr, **kwargs):
    Fg = Sg - Bg
    Fr = Sr - Br - Fg * crosstalk
    return Fr/(Fg + Fr)


@nusiance
def fret_efficiency(Sg, Sr, **kwargs):
    Fg = Sg - Bg
    Fa = (Sr - Br - Fg*crosstalk) / phiA

    return Fa/(Fg/Gfactor/phiD + Fa)


@nusiance
def distance(Sg, Sr, **kwargs):
    Fg = Sg - Bg
    Fa = (Sr - Br - Fg*crosstalk) / phiA
    return R0 * np.exp(1./6.*np.log(Fg/Gfactor/phiD/Fa))


def SrSg_ratio(Sg, Sr, **kwargs):
    return Sr/Sg


@nusiance
def transfer_efficency2fdfa(E, **kwargs):
    """This function converts the transfer-efficency E to the donor-acceptor intensity ration FD/FA to

    :param E: float
        The transfer-efficency
    :param phiD: float
        donor quantum yield
    :param phiA: float
        acceptor quantum yield
    :return: float, the FRET transfer efficency
    """
    fdfa = phiA/phiD*(1./E-1)
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
    r = 1 + fdfa * phiA/phiD
    trans = 1./r
    return trans