import numpy as np
cimport numpy as np
cimport cython
from cython.parallel import parallel, prange
from libc.stdlib cimport malloc, free
from libc.math cimport sin, cos, sqrt, fabs


def kappa2_distance(d1, d2, a1, a2):
    """Calculates the orientation-factor kappa

    :param donor_dipole: 3x2 vector of the donor-dipole
    :param acceptor_dipole: 3x2 vector of the acceptor-dipole
    :return:
    """

    # coordinates of the dipole
    cdef float d11, d12, d13, d21, d22, d23
    cdef float a11, a12, a13, a21, a22, a23

    # length of the dipole
    cdef float dD21, dA21

    # normal vector of the dipoles
    cdef float muD1, muD2, muD3
    cdef float muA1, muA2, muA3

    # connection vector of the dipole
    cdef float RDA1, RDA2, RDA3

    # vector to the middle of the dipoles
    cdef float dM1, dM2, dM3
    cdef float aM1, aM2, aM3

    # normalized DA-connection vector
    cdef float nRDA1, nRDA2, nRDA3

    d11 = d1[0]
    d12 = d1[1]
    d13 = d1[2]

    d21 = d2[0]
    d22 = d2[1]
    d23 = d2[2]

    # distance between the two end points of the donor
    dD21 = sqrt( (d11 - d21)*(d11 - d21) +
                 (d12 - d22)*(d12 - d22) +
                 (d13 - d23)*(d13 - d23)
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
    dA21 = sqrt( (a11 - a21)*(a11 - a21) +
                 (a12 - a22)*(a12 - a22) +
                 (a13 - a23)*(a13 - a23)
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
    dRDA = sqrt(RDA1*RDA1 + RDA2*RDA2 + RDA3*RDA3)

    # Normalized dipole-diple vector
    nRDA1 = RDA1 / dRDA
    nRDA2 = RDA2 / dRDA
    nRDA3 = RDA3 / dRDA

    # Orientation factor kappa2
    kappa = muA1*muD1 + muA2*muD2 + muA3*muD3 - 3.0 * (muD1*nRDA1+muD2*nRDA2+muD3*nRDA3) * (muA1*nRDA1+muA2*nRDA2+muA3*nRDA3)
    return dRDA, kappa
