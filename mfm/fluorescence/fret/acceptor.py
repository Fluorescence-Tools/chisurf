from __future__ import annotations

import numpy as np
import mfm.fluorescence.general

def da_a0_to_ad(
        times: np.array,
        decay_DA: np.array,
        acceptor_lifetime_spectrum: np.array,
        transfer_efficency: float = 0.5
) -> np.array:
    """Convolves the donor decay in presence of FRET directly with the acceptor only decay to give the
     FRET-sensitized decay ad

    :param times:
    :param decay_DA:
    :param decay_A0:
    :param transfer_efficency:
    :return:
    """
    a0 = np.zeros_like(decay_DA)
    for i in range(len(acceptor_lifetime_spectrum) // 2):
        a = acceptor_lifetime_spectrum[i]
        tau = acceptor_lifetime_spectrum[i + 1]
        a0 += a * np.exp(-times / tau)

    ad = np.convolve(
        decay_DA,
        a0,
        mode='full'
    )[:len(da)]
    #ds = da.sum()
    return ad


def scale_acceptor(
        donor,
        acceptor,
        transfer_efficiency
):
    """

    :param donor:
    :param acceptor:
    :param transfer_efficiency:
    :return:
    """
    s_d = sum(donor)
    s_a = sum(acceptor)
    scaling_factor = 1. / ((s_a / transfer_efficiency - s_a) / s_d)
    scaled_acceptor = acceptor * scaling_factor
    return donor, scaled_acceptor

