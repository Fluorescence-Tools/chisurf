from __future__ import annotations
from typing import Tuple

from math import sqrt
import numpy as np
import numba as nb

from . import kappa2


def scale_acceptor(donor, acceptor, transfer_efficency):
    s_d = sum(donor)
    s_a = sum(acceptor)
    scaling_factor = 1. / ((s_a / transfer_efficency - s_a) / s_d)
    scaled_acceptor = acceptor * scaling_factor
    return donor, scaled_acceptor


def vm_rt_to_vv_vh(t, vm, rs, g_factor=1.0, l1=0.0, l2=0.0):
    """Get the VV, VH decay from an VM decay given an anisotropy spectrum

    :param t: time-axis
    :param vm: magic angle decay
    :param rs: anisotropy spectrum
    :param g_factor: g-factor
    :param l1:
    :param l2:
    :return: vv, vm
    """
    rt = np.zeros_like(vm)
    for i in range(0, len(rs), 2):
        b = rs[i]
        rho = rs[i+1]
        rt += b * np.exp(-t/rho)
    vv = vm * (1 + 2.0 * rt)
    vh = vm * (1. - g_factor * rt)
    vv_j = vv * (1. - l1) + vh * l1
    vh_j = vv * l2 + vh * (1. - l2)
    return vv_j, vh_j


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
    ds = da.sum()

    return ad


def s2delta(r_0, s2donor, s2acceptor, r_inf_AD):
    """calculate delta given residual anisotropies

    :param r_0:
    :param s2donor: -np.sqrt(self.r_Dinf/self.r_0)
    :param s2acceptor: np.sqrt(self.r_Ainf/self.r_0)
    :param r_inf_DA:

    Accurate Distance Determination of Nucleic Acids via Foerster Resonance Energy Transfer:
    Implications of Dye Linker Length and Rigidity
    http://pubs.acs.org/doi/full/10.1021/ja105725e
    """
    delta = r_inf_AD/(r_0*s2donor*s2acceptor)
    return delta


@nb.jit()
def kappa2_distance(
        d1: np.array,
        d2: np.array,
        a1: np.array,
        a2: np.array
) -> Tuple[float, float]:
    """Calculates the orientation-factor kappa

    Calculates for the vectors d1 and d2 pointing to the donors and the vecotrs
    a1 and a2 pointing to the ends of the acceptor dipole the orientation factor kappa

    :param d1:
    :param d2:
    :param a1:
    :param a2:
    :return:
    """
    # coordinates of the dipole
    d11 = d1[0]
    d12 = d1[1]
    d13 = d1[2]

    d21 = d2[0]
    d22 = d2[1]
    d23 = d2[2]

    # distance between the two end points of the donor
    dD21 = sqrt((d11 - d21) * (d11 - d21) +
                (d12 - d22) * (d12 - d22) +
                (d13 - d23) * (d13 - d23)
                )

    # normal vector of the donor-dipole
    muD1 = (d21 - d11) / dD21
    muD2 = (d22 - d12) / dD21
    muD3 = (d23 - d13) / dD21

    # vector to the middle of the donor-dipole
    dM1 = d11 + dD21 * muD1 / 2.0
    dM2 = d12 + dD21 * muD2 / 2.0
    dM3 = d13 + dD21 * muD3 / 2.0

    ### Acceptor ###
    # cartesian coordinates of the acceptor
    a11 = a1[0]
    a12 = a1[1]
    a13 = a1[2]

    a21 = a2[0]
    a22 = a2[1]
    a23 = a2[2]

    # distance between the two end points of the acceptor
    dA21 = sqrt((a11 - a21) * (a11 - a21) +
                (a12 - a22) * (a12 - a22) +
                (a13 - a23) * (a13 - a23)
                )

    # normal vector of the acceptor-dipole
    muA1 = (a21 - a11) / dA21
    muA2 = (a22 - a12) / dA21
    muA3 = (a23 - a13) / dA21

    # vector to the middle of the acceptor-dipole
    aM1 = a11 + dA21 * muA1 / 2.0
    aM2 = a12 + dA21 * muA2 / 2.0
    aM3 = a13 + dA21 * muA3 / 2.0

    # vector connecting the middle of the dipoles
    RDA1 = dM1 - aM1
    RDA2 = dM2 - aM2
    RDA3 = dM3 - aM3

    # Length of the dipole-dipole vector (distance)
    dRDA = sqrt(RDA1 * RDA1 + RDA2 * RDA2 + RDA3 * RDA3)

    # Normalized dipole-diple vector
    nRDA1 = RDA1 / dRDA
    nRDA2 = RDA2 / dRDA
    nRDA3 = RDA3 / dRDA

    # Orientation factor kappa2
    kappa = muA1 * muD1 + muA2 * muD2 + muA3 * muD3 - 3.0 * (muD1 * nRDA1 + muD2 * nRDA2 + muD3 * nRDA3) * (
            muA1 * nRDA1 + muA2 * nRDA2 + muA3 * nRDA3)
    return dRDA, kappa


def kappa(
        donor_dipole: np.array,
        acceptor_dipole: np.array
):
    """Calculates the orientation-factor kappa

    :param donor_dipole: 2x3 vector of the donor-dipole
    :param acceptor_dipole: 2x3 vector of the acceptor-dipole
    :return: distance, kappa

    Example
    -------

    >>> import numpy as np
    >>> donor_dipole = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
    >>> acceptor_dipole = np.array([[0.0, 0.5, 0.0], [0.0, 0.5, 1.0]], dtype=np.float64)
    >>> kappa(donor_dipole, acceptor_dipole)
    """
    return kappa2_distance(
        donor_dipole[0], donor_dipole[1],
        acceptor_dipole[0], acceptor_dipole[1]
    )


def calculate_kappa_distance(xyz, aid1, aid2, aia1, aia2):
    """Calculates the orientation factor kappa2 and the distance of a trajectory given the atom-indices of the
    donor and the acceptor.

    :param xyz: numpy-array (frame, atom, xyz)
    :param aid1: int, atom-index of d-dipole 1
    :param aid2: int, atom-index of d-dipole 2
    :param aia1: int, atom-index of a-dipole 1
    :param aia2: int, atom-index of a-dipole 2

    :return: distances, kappa2
    """
    n_frames = xyz.shape[0]
    ks = np.empty(n_frames, dtype=np.float32)
    ds = np.empty(n_frames, dtype=np.float32)

    for i_frame in range(n_frames):
        try:
            d, k = kappa2_distance(
                xyz[i_frame, aid1], xyz[i_frame, aid2],
                xyz[i_frame, aia1], xyz[i_frame, aia2]
            )
            ks[i_frame] = k
            ds[i_frame] = d
        except:
            print("Frame ", i_frame, "skipped, calculation error")

    return ds, ks


