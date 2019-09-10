import numpy as np


def da_a0_to_ad(t, da, ac_s, transfer_efficency=0.5):
    """Convolves the donor decay in presence of FRET directly with the acceptor only decay to give the
     FRET-sensitized decay ad

    :param da:
    :param ac_s: acceptor lifetime spectrum
    :return:
    """
    a0 = np.zeros_like(da)
    for i in range(len(ac_s) // 2):
        a = ac_s[i]
        tau = ac_s[i + 1]
        a0 += a * np.exp(-t / tau)
    ad = np.convolve(da, a0, mode='full')[:len(da)]
    #ds = da.sum()
    return ad


def scale_acceptor(donor, acceptor, transfer_efficency):
    s_d = sum(donor)
    s_a = sum(acceptor)
    scaling_factor = 1. / ((s_a / transfer_efficency - s_a) / s_d)
    scaled_acceptor = acceptor * scaling_factor
    return donor, scaled_acceptor

