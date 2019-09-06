import numpy as np
cimport numpy as np
from libc.math cimport sqrt, exp, cos, sin
from libc.stdint cimport uint32_t, int32_t, uint8_t
cimport cython
from libc.stdlib cimport abort, malloc, free
from cython.parallel import prange


cdef inline int int_max(int a, int b) nogil: return a if a >= b else b
cdef inline int int_min(int a, int b) nogil: return a if a <= b else b

cdef inline double double_max(double a, double b) nogil: return a if a >= b else b
cdef inline double double_min(double a, double b) nogil: return a if a <= b else b

cdef inline double dist3c(double r1x, double r1y, double r1z, double r2x, double r2y, double r2z) nogil:
    return sqrt((r1x - r2x)*(r1x - r2x) + (r1y - r2y)*(r1y - r2y) + (r1z - r2z)*(r1z - r2z))
cdef inline double dist3c2(double r1x, double r1y, double r1z, double r2x, double r2y, double r2z) nogil:
    return (r1x - r2x)*(r1x - r2x) + (r1y - r2y)*(r1y - r2y) + (r1z - r2z)*(r1z - r2z)


@cython.boundscheck(False)
def mj(int[:, :] resLookUp, int[:] res_types, double[:, :] ca_dist, double[:, :] xyz, double[:, :] mjPot,
       double cutoff=6.5, int c_beta_pos=4):
    cdef int nRes = resLookUp.shape[0]
    cdef int i, j, res_i, res_j, nCbi, nCbj
    cdef double r1x, r1y, r1z, r2x, r2y, r2z, dij
    cdef double cut2 = cutoff * cutoff
    cdef double cut_ca2 = cutoff * cutoff
    cdef double E = 0.0
    cdef int nCont = 0
    for i in prange(nRes, nogil=True):
        res_i = res_types[i]
        nCbi = resLookUp[i, c_beta_pos]
        r1x = xyz[nCbi, 0]
        r1y = xyz[nCbi, 1]
        r1z = xyz[nCbi, 2]
        for j in range(i + 1, nRes):
            res_j = res_types[j]
            nCbj = resLookUp[j, c_beta_pos]
            if ca_dist[i, j] < (cutoff + 2.0 * 1.6) ** 2:
                r2x = xyz[nCbj, 0]
                r2y = xyz[nCbj, 1]
                r2z = xyz[nCbj, 2]
                dij = dist3c2(r1x, r1y, r1z, r2x, r2y, r2z)
                if dij < cut2:
                    E += mjPot[res_i, res_j]
                    nCont += 1
    return nCont, E


@cython.boundscheck(False)
@cython.cdivision(True)
def hBondLookUpAll(int[:, :] resLookUp, double[:, :] caDist, double[:, :] xyz, double[:, :] hPot, double cutoffCA2,
                double cutoffH2, double binWidth=0.01):
    cdef int binNbr, nCi, nCj, nNi, nNj, nOi, nOj, nHi, nHj
    cdef double dONij, dONji, dCNij, dCNji, dOHij, dOHji, dCHij, dCHji
    cdef int nRes = resLookUp.shape[0]
    cdef int ri, rj
    cdef int maxBin = hPot.shape[1]
    cdef int nHBond = 0

    cdef double E = 0.0
    for ri in prange(nRes, nogil=True):
        nNi = resLookUp[ri, 0]
        nCi = resLookUp[ri, 2]
        nOi = resLookUp[ri, 3]
        nHi = resLookUp[ri, 5]
        for rj in range(ri + 1, nRes):
            if caDist[ri, rj] < cutoffCA2:
                nNj = resLookUp[rj, 0]
                nCj = resLookUp[rj, 2]
                nOj = resLookUp[rj, 3]
                nHj = resLookUp[rj, 5]

                if nHi > 0 and nHj > 0: # Both aa no Pro
                    # i -> j
                    dOHij = dist3c2(xyz[nHj, 0], xyz[nHj, 1], xyz[nHj, 2], xyz[nOi, 0], xyz[nOi, 1], xyz[nOi, 2])
                    if dOHij < cutoffH2:
                        nHBond += 1
                        binNbr = <int>(sqrt(dOHij) / binWidth)
                        E += hPot[2, binNbr]
                        dONij = dist3c(xyz[nNj, 0], xyz[nNj, 1], xyz[nNj, 2], xyz[nOi, 0], xyz[nOi, 1], xyz[nOi, 2])
                        binNbr = <int>(dONij / binWidth)
                        E += hPot[1, binNbr]
                        dCHij = dist3c(xyz[nCi, 0], xyz[nCi, 1], xyz[nCi, 2], xyz[nHj, 0], xyz[nHj, 1], xyz[nHj, 2])
                        binNbr = <int>(dCHij / binWidth)
                        E += hPot[0, binNbr]
                        dCNij = dist3c(xyz[nNj, 0], xyz[nNj, 1], xyz[nNj, 2], xyz[nCi, 0], xyz[nCi, 1], xyz[nCi, 2])
                        binNbr = <int>(dCNij / binWidth)
                        E += hPot[3, binNbr]
                    # j -> i
                    dOHji = dist3c2(xyz[nHi, 0], xyz[nHi, 1], xyz[nHi, 2],
                                   xyz[nOj, 0], xyz[nOj, 1], xyz[nOj, 2])
                    if dOHji < cutoffH2:
                        nHBond += 1
                        binNbr = <int>(sqrt(dOHji) / binWidth)
                        E += hPot[2, binNbr]
                        dONji = dist3c(xyz[nNi, 0], xyz[nNi, 1], xyz[nNi, 2], xyz[nOj, 0], xyz[nOj, 1], xyz[nOj, 2])
                        E += hPot[1, <int>(dONji / binWidth)]
                        dCHji = dist3c(xyz[nCj, 0], xyz[nCj, 1], xyz[nCj, 2], xyz[nHi, 0], xyz[nHi, 1], xyz[nHi, 2])
                        E += hPot[0, <int>(dCHji / binWidth)]
                        dCNji = dist3c(xyz[nNi, 0], xyz[nNi, 1], xyz[nNi, 2], xyz[nCj, 0], xyz[nCj, 1], xyz[nCj, 2])
                        E += hPot[3, <int>(dCNji / binWidth)]
                elif (nHj > 0) and (nHi < 0):
                    # j -> i
                    dOHij = dist3c2(xyz[nHj, 0], xyz[nHj, 1], xyz[nHj, 2], xyz[nOi, 0], xyz[nOi, 1], xyz[nOi, 2])
                    if dOHij < cutoffH2:
                        nHBond += 1
                        E += hPot[2, <int>(sqrt(dOHij) / binWidth)]
                        dONij = dist3c(xyz[nNj, 0], xyz[nNj, 1], xyz[nNj, 2], xyz[nOi, 0], xyz[nOi, 1], xyz[nOi, 2])
                        E += hPot[1, <int>(dONij / binWidth)]
                        dCHij = dist3c(xyz[nCi, 0], xyz[nCi, 1], xyz[nCi, 2], xyz[nHj, 0], xyz[nHj, 1], xyz[nHj, 2])
                        E += hPot[0, <int>(dCHij / binWidth)]
                        dCNij = dist3c(xyz[nNj, 0], xyz[nNj, 1], xyz[nNj, 2], xyz[nCi, 0], xyz[nCi, 1], xyz[nCi, 2])
                        E += hPot[3, <int>(dCNij / binWidth)]
                        # j -> i
                elif (nHj < 0) and (nHi > 0):
                    # i -> j
                    dOHji = dist3c2(xyz[nHi, 0], xyz[nHi, 1], xyz[nHi, 2], xyz[nOj, 0], xyz[nOj, 1], xyz[nOj, 2])
                    if dOHji < cutoffH2:
                        nHBond += 1
                        E += hPot[2, <int>(sqrt(dOHji) / binWidth)]
                        dONji = dist3c(xyz[nNi, 0], xyz[nNi, 1], xyz[nNi, 2], xyz[nOj, 0], xyz[nOj, 1], xyz[nOj, 2])
                        E += hPot[1, <int>(dONji / binWidth)]
                        dCHji = dist3c(xyz[nCj, 0], xyz[nCj, 1], xyz[nCj, 2], xyz[nHi, 0], xyz[nHi, 1], xyz[nHi, 2])
                        E += hPot[0, <int>(dCHji / binWidth)]
                        dCNji = dist3c(xyz[nNi, 0], xyz[nNi, 1], xyz[nNi, 2], xyz[nCj, 0], xyz[nCj, 1], xyz[nCj, 2])
                        E += hPot[3, <int>(dCNji / binWidth)]
    return nHBond, E


@cython.cdivision(True)
@cython.boundscheck(False)
def gb(np.ndarray rAtoms, double epsilon=4.0, double epsilon0 = 80.1, double cutoff=12.0):
    """
    Generalized Born http://en.wikipedia.org/wiki/Implicit_solvation
    with cutoff-potential potential shifted by value at cutoff to smooth the energy
    :param rAtoms: atom-list
    :param epsilon: dielectric constant of proteins: 2-40
    :param epsilon0: dielectric constant of water 80.1
    :param cutoff: cutoff-distance in Angstrom
    :return:
    """

    cdef int nAtoms = rAtoms.shape[0]
    cdef double cutoff2 = cutoff ** 2
    cdef double[:, :] rs = rAtoms['coord']
    cdef double[:] aRadii = rAtoms['radius']
    cdef double[:] aCharges = rAtoms['charge']
    cdef double pre = 1. / (8 * 3.14159265359) * (1. / epsilon0 - 1. / epsilon)
    cdef double E, fgbr, D, fgbc
    cdef double r1x, r1y, r1z, r2x, r2y, r2z, rij, rij2
    cdef double ai, aj, qi, qj, qij, aij, aij2
    cdef int i, j
    E = 0.0
    for i in range(nAtoms):
        qi = aCharges[i]
        if qi == 0.0:
            continue
        r1x = rs[i, 0]
        r1y = rs[i, 1]
        r1z = rs[i, 2]
        ai = aRadii[i]
        for j in range(i, nAtoms):
            qj = aCharges[j]
            if qj == 0.0:
                continue
            r2x = rs[j, 0]
            r2y = rs[j, 1]
            r2z = rs[j, 2]
            aj = aRadii[j]
            rij2 = (r1x - r2x)*(r1x - r2x) + (r1y - r2y)*(r1y - r2y) + (r1z - r2z)*(r1z - r2z)
            if rij2 > cutoff2:
                continue
            else:
                aij2 = ai * aj
                qij = qi * qj
                D = rij2 / (4.0 * aij2)
                fgbr = sqrt(rij2 + aij2 * exp(-D))
                fgbc = sqrt(cutoff + aij2 * exp(-D))
                E = E + qij / fgbr - qij/fgbc
    E *= pre
    return E


@cython.cdivision(True)
@cython.boundscheck(False)
def clash_potential(double[:, :] xyz, double[:] vdw, double clash_tolerance=1.0, double covalent_radius=1.0):
    """
    Clash potential
    with cutoff-potential potential shifted by value at cutoff to smooth the energy
    :param xyz: Cartesian coordinates of the atoms
    :param vdw: van der Waals radii of the atoms
    :param clash_tolerance: clash tolerance
    :return:
    """

    cdef int nAtoms = xyz.shape[0]
    cdef double E
    cdef double r1x, r1y, r1z, r2x, r2y, r2z, rij, rij2
    cdef double vdwi, vdwj, vdwij
    cdef int i, j
    E = 0.0
    for i in range(nAtoms):
        r1x = xyz[i, 0]
        r1y = xyz[i, 1]
        r1z = xyz[i, 2]
        vdwi = vdw[i]
        for j in range(i + 1, nAtoms):
            vdwj = vdw[j]
            r2x = xyz[j, 0]
            r2y = xyz[j, 1]
            r2z = xyz[j, 2]
            rij = sqrt((r1x - r2x)*(r1x - r2x) + (r1y - r2y)*(r1y - r2y) + (r1z - r2z)*(r1z - r2z))
            vdwij = (vdwi + vdwj)
            if covalent_radius < rij < vdwij:
                E += ((vdwi + vdwj - rij) / clash_tolerance)**2.0
    return E


@cython.boundscheck(False)
def go_init(double[:, :] caDist, double epsilon, double nnEFactor=0.1, double cutoff=6.5):
    """
    The go-interactions are implemented as truncated LJ-(6,12) potential. The function looks for each
    residue within a radius of nativeCut for neighbors if the residues are within the cutoff. If the
    residues are within the cutoff the value in the returned interaction matrix is set to epsilon
    otherwise the value is set to epsilon*nnFraction (non-native interaction).
    """
    cdef int i, j
    cdef double rm, intE
    cdef int nRes = caDist.shape[0]

    cdef np.ndarray[np.float64_t, ndim=2] eMatrix = np.zeros((nRes, nRes), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=2] rmMatrix = np.zeros((nRes, nRes), dtype=np.float64)

    for i in range(nRes):
        for j in range(i + 1, nRes):
            rm = caDist[i, j]
            intE = (17./55. + 1.) * rm  # (2.5 sigma)
            rmMatrix[i, j] = rm
            rmMatrix[j, i] = rm
            if caDist[i, j] < cutoff:
                eMatrix[i, j] = epsilon / intE
                eMatrix[j, i] = epsilon / intE
            else:
                eMatrix[i, j] = nnEFactor * epsilon / intE
                eMatrix[j, i] = nnEFactor * epsilon / intE
    return eMatrix, rmMatrix

@cython.cdivision(True)
@cython.boundscheck(False)
def go(double[:, :] caDist, double[:, :] eMatrix, double[:, :] rmMatrix, int cutoff=1):
    """
    If cutoff is True the LJ-Potential is cutoff and shifted at 2.5 sigma

    :param resLookUp:
    :param caDist:
    :param eMatrix:
    :param rmMatrix:
    :param cutoff:
    :return:
    """
    cdef double Etemp, Etot, Enn, Ena, r, sr
    cdef int nRes
    cdef int i, j, n
    cdef double epsilon, rm

    Etemp = 0.0
    Etot = 0.0  # total energy
    nRes = caDist.shape[0]
    for i in range(nRes):
        for j in range(i + 1, nRes):
            epsilon = eMatrix[i, j]
            rm = rmMatrix[i, j]
            r = caDist[i, j]
            if rm < r < rm * 2.5:
                sr = 2*(rm / r) ** 6
                Etemp -= sr
                Etemp += sr ** 2 / 4.0
                Etemp *= epsilon
                Etemp += 0.00818 * epsilon
            else:
                Etemp = - epsilon
            Etot += Etemp
    return Etot


def spherePoints(int nSphere):
    # generate sphere points Returns list of 3d coordinates of points on a sphere using the
    # Golden Section Spiral algorithm.
    cdef double offset, rd, y, phi
    cdef double[:, :] sphere_points = np.zeros((nSphere,3), dtype=np.float64)
    cdef double inc = 3.14159265359 * (3 - sqrt(5))
    cdef int32_t k
    offset = 2.0 / (<double> nSphere)
    for k in range(nSphere):
        y = k * offset - 1.0 + (offset / 2.0)
        rd = sqrt(1 - y * y)
        phi = k * inc
        sphere_points[k, 0] = cos(phi) * rd
        sphere_points[k, 1] = y
        sphere_points[k, 2] = sin(phi) * rd
    return sphere_points



@cython.cdivision(True)
@cython.boundscheck(False)
def asa(double[:, :] xyz, int[:, :] resLookUp, double[:, :] caDist, double[:, :] sphere_points,
        double probe=1.0, double radius = 2.5, char sum=1):
    """
    Returns list of accessible surface areas of the atoms, using the probe
    and atom radius to define the surface.

    Routines to calculate the Accessible Surface Area of a set of atoms.
    The algorithm is adapted from the Rose lab's chasa.py, which uses

    :param xyz:
    :param resLookUp:
    :param caDist:
    :param sphere_points:
    :param probe:
    :param radius:
    :param sum:
    the dot density technique found in:

    Shrake, A., and J. A. Rupley. "Environment and Exposure to Solvent
    of Protein Atoms. Lysozyme and Insulin." JMB (1973) 79:351-371.
    """
    cdef char is_accessible
    cdef int32_t n_accessible_point, n_neighbor
    cdef int32_t i, j, k, nSphere, nCi, nRes

    cdef double area
    cdef double aX, aY, aZ, bX, bY, bZ

    nSphere = sphere_points.shape[0]
    nRes = caDist.shape[0]

    cdef int* neighbor_indices = <int*> malloc(sizeof(int) * nRes)

    # calcuate asa
    area = 0.0
    cdef double c = 4.0 * 3.14159265359 / (<double> nSphere)
    for i in range(nRes):
        nCi = resLookUp[i, 3 + 3]
        # find neighbors!
        n_neighbor = 0
        for j in range(nRes):
            if i != j:
                if caDist[i, j] < 2 * (radius + probe):
                    neighbor_indices[n_neighbor] = resLookUp[j, 3 + 3]
                    n_neighbor += 1
        # accessible?
        n_accessible_point = 0
        for j in range(nSphere):
            is_accessible = 1
            aX = sphere_points[j, 0] * radius + xyz[nCi, 0]
            aY = sphere_points[j, 1] * radius + xyz[nCi, 1]
            aZ = sphere_points[j, 2] * radius + xyz[nCi, 2]
            for k in range(n_neighbor):
                bX = xyz[neighbor_indices[k], 0]
                bY = xyz[neighbor_indices[k], 1]
                bZ = xyz[neighbor_indices[k], 2]
                if dist3c2(aX, aY, aZ, bX, bY, bZ) < (radius + probe) * (radius + probe):
                    is_accessible = 0
                    break
            if is_accessible > 0:
                n_accessible_point += 1
        area += c * n_accessible_point * radius ** 2
    free(neighbor_indices)
    return area

