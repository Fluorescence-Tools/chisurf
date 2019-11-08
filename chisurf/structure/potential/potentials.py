from __future__ import annotations

import os
import json
import math
import copy

import numpy as np
import numba as nb
import scipy.stats

import chisurf.fluorescence
import chisurf.fluorescence.av
import chisurf.structure
import chisurf.structure.potential.cPotentials


@nb.njit
def centroid2(
        atom_lookup,
        res_types,
        ca_dist,
        r,
        potential,
        cutoff: float = 6.5,
        centroid_pos: int = 4,
        min_dist: float = 3.5,
        max_dist: float = 19.0,
        bin_width: float = 0.05,
        repulsion: float = 100.0
):
    """
    Calculate the potential energy given the UNRES GBV-Sidechain potential and isotropic conditions.

    Parameters
    ----------
    atom_lookup : integer array
            A two dimensional lookup array to find for each residue and atom-type the corresponding
            position within the coordinate array `r`.
    centroid_pos : integer
            This variable specifies which position in the atom_lookup contains the number of the C-beta
            atom.
    res_types : integer array
            An array containing the residue types of the residues as a one dimensional array. The residue types
            are in the order as in :py:const:mfm.structure.residue_atoms_internal:
    ca_dist : double array
            The array `ca_dist` should contain the pair-wise distances of all C-alpha atoms of the protein
    r : double array
            The two-dimensional array `r` contains all Cartesian coordinates of all atoms.
    potential : double array
            This 3-dimensional array contains pre-calculated potentials of all possible combinations of amino-
            acids. potential[1,2,10] has to contain the potential of the amino-acid (1) and (2) at a distance
            of 10 whereas the actual distance is dependent on the range in which the potential was actually
            calculated. The potential can be calculated with the small program located in:
            mfm.structure.potential.database.make_unres_lookup.py
    min_dist : double
            The minimum distance for which the parameter `potential` was calculated. This parameter has to be
            supplied and depends on the provided pre-calculated potential file.
    max_dist :double
            The minimum distance for which the parameter `potential` was calculated (see min_dist)
    bin_width : double
            The spacing between the bin of the distance points in the pre-calculated potential
    cutoff : double
            Distances between the centroids bigger than the cut-off distance will be neglected

    Citation
    --------
    J. Comp. Chem. Vol. 18, 7, 849-873, 1997
    """

    nRes = atom_lookup.shape[0]

    E, nCont = 0.0, 0
    for residue_i in range(nRes):
        c_beta_atom_nbr_i = atom_lookup[residue_i, centroid_pos]
        if c_beta_atom_nbr_i < 0: # no side-chain -> next
            continue

        residue_type_i = res_types[residue_i]

        xi = r[c_beta_atom_nbr_i, 0]
        yi = r[c_beta_atom_nbr_i, 1]
        zi = r[c_beta_atom_nbr_i, 2]

        for residue_j in range(residue_i + 1, nRes - 1):
            c_beta_atom_nbr_j = atom_lookup[residue_j, centroid_pos]
            if c_beta_atom_nbr_j < 0 or ca_dist[residue_i, residue_j] > cutoff:
                continue
            nCont += 1

            residue_type_j = res_types[residue_j]

            xj = r[c_beta_atom_nbr_j, 0]
            yj = r[c_beta_atom_nbr_j, 1]
            zj = r[c_beta_atom_nbr_j, 2]

            dij = math.sqrt((xi-xj)**2 + (yi-yj)**2 + (zi-zj)**2)

            if dij < min_dist:
                E += repulsion
            else:
                dij = min(dij, max_dist)
                bin_nbr = int(dij / bin_width)
                E += potential[residue_type_i, residue_type_j, bin_nbr]

    return nCont, E


@nb.jit(nopython=True)
def internal_potential(
        internal_coordinates,
        equilibrium_internal=None,
        k_bond: float = 0.5,
        k_angle: float = 0.1,
        k_dihedral: float = 0.05
) -> float:
    """ Simple potential based internal coordinates for C-alpha bead model
    """
    if equilibrium_internal is None:
        return 0.0
    e = 0.0
    if k_bond > 0:
        bond = k_bond * (internal_coordinates['b'] - equilibrium_internal['b'])**2
        e += bond.sum()
    if k_angle > 0:
        angle = k_angle * (internal_coordinates['a'] - equilibrium_internal['a'])**2
        e += angle.sum()
    if k_dihedral > 0:
        dihedral = k_dihedral * (internal_coordinates['d'] - equilibrium_internal['d'])**2
        e += dihedral.sum()
    return e


def internal_potential_calpha(
        structure: chisurf.structure.Structure,
        **kwargs
) -> float:
    """Calculates a *internal* potential based on the similarity of the bonds length, angles, and dihedrals
    to a reference structure.

    :param structure:
    :param kwargs:
    :return:
    """
    eq = kwargs.get('equilibrium_internal', None)
    kb = kwargs.get('k_bonds', 0.5)
    ka = kwargs.get('k_angle', 0.1)
    kd = kwargs.get('k_dihedral', 0.05)
    return internal_potential(structure.internal_coordinates, eq, kb, ka, kd)


@nb.jit(nopython=True)
def lj_calpha(
        ca_coordinates: np.array,
        rm: float = 3.8208650279
) -> float:
    """Truncated Lennard Jones

    :param ca_coordinates:
    :param epsilon:
    :param rm:
    :return:
    """
    energy = 0.0  # total energy
    n_atoms = ca_coordinates.shape[0]
    rm2 = rm**2
    for i in range(n_atoms):
        x_i, y_i, z_i = ca_coordinates[i]
        for j in range(i + 1, n_atoms):
            x_j, y_j, z_j = ca_coordinates[j]
            r2 = (x_i - x_j)**2 + (y_i - y_j)**2 + (z_i - z_j)**2
            sr = (rm2 / r2) ** 3
            energy += sr * (sr - 2.)
    return energy


def lennard_jones_calpha(
        structure: chisurf.structure.Structure,
        rm: float = 3.8208650279
):
    """

    :param structure: a structure object
    :param rm: is the C-alpha equilibrium distance
    :return:
    """
    sel = chisurf.structure.get_atom_index_by_name(
        structure.atoms, ['CA']
    )
    ca_atoms = structure.atoms.take(sel)
    return lj_calpha(ca_atoms['xyz'][0], rm)


@nb.njit
def gb(
        xyz: np.array,
        epsilon: float = 4.0,
        epsilon0: float = 80.1,
        cutoff: float = 12.0
) -> float:
    """
    Generalized Born http://en.wikipedia.org/wiki/Implicit_solvation
    with cutoff-potential potential shifted by value at cutoff to smooth the energy
    :param xyz: atom-list
    :param epsilon: dielectric constant of proteins: 2-40
    :param epsilon0: dielectric constant of water 80.1
    :param cutoff: cutoff-distance in Angstrom
    :return:
    """

    nAtoms = xyz.shape[0]
    cutoff2 = cutoff ** 2
    rs = xyz['xyz']
    a_radii = xyz['radius']
    a_charge = xyz['charge']
    pre = 1. / (8 * 3.14159265359) * (1. / epsilon0 - 1. / epsilon)

    energy = 0.0
    for i in range(nAtoms):
        qi = a_charge[i]
        if qi == 0.0:
            continue
        r1x = rs[i, 0]
        r1y = rs[i, 1]
        r1z = rs[i, 2]
        ai = a_radii[i]
        for j in range(i, nAtoms):
            qj = a_charge[j]
            if qj == 0.0:
                continue
            r2x = rs[j, 0]
            r2y = rs[j, 1]
            r2z = rs[j, 2]
            aj = a_radii[j]
            rij2 = (r1x - r2x)*(r1x - r2x) + (r1y - r2y)*(r1y - r2y) + (r1z - r2z)*(r1z - r2z)
            if rij2 > cutoff2:
                continue
            else:
                aij2 = ai * aj
                qij = qi * qj
                D = rij2 / (4.0 * aij2)
                fgbr = math.sqrt(rij2 + aij2 * math.exp(-D))
                fgbc = math.sqrt(cutoff + aij2 * math.exp(-D))
                energy = energy + qij / fgbr - qij/fgbc
    return energy * pre


@nb.jit
def go(
        ca_dist: np.array,
        energy_matrix: np.array,
        rm_matrix: np.array
):
    """
    If cutoff is True the LJ-Potential is cutoff and shifted at 2.5 sigma

    :param resLookUp:
    :param ca_dist:
    :param energy_matrix:
    :param rm_matrix:
    :param cutoff:
    :return:
    """

    tmp = 0.0
    energy = 0.0  # total energy
    n_residues = ca_dist.shape[0]
    for i in range(n_residues):
        for j in range(i + 1, n_residues):
            epsilon = energy_matrix[i, j]
            rm = rm_matrix[i, j]
            r = ca_dist[i, j]
            if rm < r < rm * 2.5:
                sr = 2*(rm / r) ** 6
                tmp -= sr
                tmp += sr ** 2 / 4.0
                tmp *= epsilon
                tmp += 0.00818 * epsilon
            else:
                tmp = - epsilon
            energy += tmp
    return energy


class Ramachandran(object):

    def __init__(
            self,
            structure: chisurf.structure.Structure,
            filename: str = None
    ):
        """
        :param filename:
        :return:
        """
        if filename is None:
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)), 'database/rama_ala_pro_gly.npy'
            )
        self.structure = structure
        self.name = 'rama'
        self.filename = filename
        self.ramaPot = np.load(self.filename)

    def getEnergy(self) -> float:
        c = self.structure
        Erama = chisurf.structure.potential.cPotentials.ramaEnergy(
            c.residue_lookup_i,
            c.iAtoms,
            self.ramaPot
        )
        self.E = Erama
        return Erama


class Electrostatics(object):

    def __init__(
            self,
            structure,
            type: str = 'gb'):
        """
        :param type:
        :return:
        """
        self.structure = structure
        self.name = 'ele'
        if type == 'gb':
            self.p = chisurf.structure.potential.cPotentials.gb

    def getEnergy(self) -> float:
        structure = self.structure
        #Eel = mfm.structure.potential.cPotentials.gb(structure.xyz)
        Eel = gb(structure.xyz)
        self.E = Eel
        return Eel


class LJ_Bead(object):

    def __init__(
            self,
            structure: chisurf.structure.Structure
    ):
        self.structure = structure
        self.name = 'LJ_bead'

    def getEnergy(self) -> float:
        structure = self.structure
        self.E = lennard_jones_calpha(structure.atoms['xyz'])
        return self.E


class HPotential(object):

    name = 'H-Bond'

    def __init__(
            self,
            structure: chisurf.structure.Structure,
            cutoff_ca: float = 8.0,
            cutoff_hbond: float = 3.0,
            **kwargs
    ):
        self.structure = structure
        self.cutoffH = cutoff_hbond
        self.cutoffCA = cutoff_ca
        self.potential = kwargs.get(
            'potential',
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)), 'database/hb.npy'
            )
        )
        self.oh = kwargs.get('oh', 1.0)
        self.on = kwargs.get('on', 1.0)
        self.cn = kwargs.get('cn', 1.0)
        self.ch = kwargs.get('ch', 1.0)
        self.updateParameter()

    def getEnergy(self):
        s1 = self.structure
        cca2 = self.cutoffCA ** 2
        ch2 = self.cutoffH ** 2
        nHbond, Ehbond = chisurf.structure.potential.cPotentials.hBondLookUpAll(
            s1.l_res, s1.dist_ca, s1.xyz, self._hPot, cca2, ch2
        )
        self.E = Ehbond
        self.nHbond = nHbond
        return self.E

    def getNbrBonds(self):
        """
        :return:
        """
        if self.nHbond is None:
            return 0
        return self.nHbond

    def updateParameter(self):
        hPot = copy.deepcopy(self._hPot)
        if not self.oh:
            hPot[2, :] *= 0.0
        if not self.on:
            hPot[1, :] *= 0.0
        if not self.cn:
            hPot[3, :] *= 0.0
        if not self.ch:
            hPot[0, :] *= 0.0
        self.hPot = hPot

    @property
    def potential(self):
        return self.hPot

    @potential.setter
    def potential(self, v):
        self._hPot = np.load(v) #np.loadtxt(v, skiprows=1, dtype=np.float64).T[1:, :]
        self.hPot = self._hPot


class GoPotential(object):

    def __init__(
            self,
            structure: chisurf.structure.Structure
    ):
        self.structure = structure
        self.name = 'go'

    def setGo(self):
        c = self.structure
        nnEFactor = self.nnEFactor if self.non_native_contact_on else 0.0
        cutoff = self.cutoff if self.native_cutoff_on else 1e6
        self.eMatrix, self.sMatrix = chisurf.structure.potential.cPotentials.go_init(
            c.residue_lookup_r, c.dist_ca,
            self.epsilon, nnEFactor, cutoff
        )

    def getEnergy(self):
        c = self.structure
        Etot, nNa, Ena, nNN, Enn = chisurf.structure.potential.cPotentials.go(
            c.residue_lookup_r, c.dist_ca, self.eMatrix, self.sMatrix
        )
        #Etot = go(c., c.dist_ca, self.eMatrix, self.sMatrix)
        self.E = Etot
        return Etot

    def getNbrNonNative(self):
        return self.nNN

    def getNbrNative(self):
        return self.nNa

    def set_sMatrix(self, sMatrix):
        self.sMatrix = sMatrix

    def set_eMatrix(self, eMatrix):
        self.eMatrix = eMatrix

    def set_nMatrix(self, nMatrix):
        self.nMatrix = nMatrix


class MJPotential(object):

    name = 'Miyazawa-Jernigan'

    def __init__(
            self,
            structure: chisurf.structure.Structure,
            filename: str = None,
            ca_cutcoff: float = 6.5
    ):
        if filename is None:
            filename = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), '.database/mj.npy'
            )
        self.filename = filename
        self.structure = structure
        self.potential = filename
        self.ca_cutoff = ca_cutcoff

    @property
    def potential(self):
        return self.mjPot

    @potential.setter
    def potential(self, v):
        self.mjPot = np.load(v)

    def getEnergy(self):
        c = self.structure
        nCont, Emj = chisurf.structure.potential.cPotentials.mj(
            c.l_res, c.residue_types, c.dist_ca, c.xyz, self.mjPot, cutoff=self.ca_cutoff
        )
        self.E = Emj
        self.nCont = nCont
        return Emj

    def getNbrContacts(self):
        return self.nCont


class CEPotential(object):
    """
    Examples
    --------

    >>> import chisurf.structure
    >>> import chisurf.structure.potential

    >>> s = mfm..structure.structure.Structure('./test/data/model/hgbp1/hGBP1_closed.pdb', verbose=True, make_coarse=True)
    >>> pce = mfm.structure.potential.potentials.CEPotential(s, ca_cutoff=64.0)
    >>> pce.getEnergy()
    -0.15896629131635745
    """

    name = 'Iso-UNRES'

    def __init__(
            self,
            structure: chisurf.structure.Structure,
            potential: str = None,
            **kwargs
    ):
        """
        scaling_factor : factor to scale energies from kCal/mol to kT=1.0 at 298K
        """
        scaling_factor = 0.593
        self.structure = structure
        self.ca_cutoff = kwargs.get('ca_cutoff', 15.0)
        self._potential = None

        if potential is None:
            potential = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), 'database/unres.npy'
            )

        self.potential = potential
        self.scaling_factor = scaling_factor
        # the number of the atom in the lookup table
        # (4=C-beta, 1=C-alpha, see: mfm.structure.protein internal_atom_numbers)
        self.centroid_number = kwargs.get('centroid_number', 4)
        self.repulsion = kwargs.get('repulsion', 100.0)
        self.min_dist = kwargs.get('min_dist', 3.5)
        self._max_dist = kwargs.get('max_dist', 19.0)
        self._bin_width = kwargs.get('bin_width', 0.05)

    @property
    def potential(self):
        return self._potential

    @potential.setter
    def potential(
            self,
            v
    ):
        self._potential = np.load(v)

    def getEnergy(
            self,
            cutoff=None,
            **kwargs
    ) -> float:
        cutoff = cutoff if cutoff is not None else self.ca_cutoff
        c = self.structure
        coord = np.ascontiguousarray(c.xyz)
        dist_ca = np.ascontiguousarray(c.dist_ca)
        residue_types = np.ascontiguousarray(c.residue_types)
        l_res = np.ascontiguousarray(c.l_res)
        centroid_pos = kwargs.get('centroid_number', self.centroid_number)
        repulsion = kwargs.get('repulsion', self.repulsion)
        min_dist = kwargs.get('min_dist', self.min_dist)
        nCont, E = centroid2(
            l_res, residue_types, dist_ca, coord,
            self.potential, cutoff=cutoff,
            centroid_pos=centroid_pos, min_dist=min_dist,
            max_dist=self._max_dist, bin_width=self._bin_width,
            repulsion=repulsion
        )

        self.nCont = nCont
        return float(E * self.scaling_factor)

    def getNbrContacts(self) -> int:
        return self.nCont


class ASA(object):

    name = 'Asa-Ca'

    def __init__(
            self,
            structure: chisurf.structure.Structure,
            probe: float = 1.0,
            n_sphere_point: int = 590,
            radius: float = 2.5
    ):
        super(ASA, self).__init__()
        self.structure = structure
        self.probe = probe
        self.n_sphere_point = n_sphere_point
        self.sphere_points = chisurf.structure.potential.cPotentials.spherePoints(
            n_sphere_point
        )
        self.radius = radius

    def getEnergy(self) -> float:
        c = self.structure
        #def asa(double[:, :] xyz, int[:, :] resLookUp, double[:, :] caDist, double[:, :] sphere_points,
        #double probe=1.0, double radius = 2.5, char sum=1)
        asa = chisurf.structure.potential.cPotentials.asa(
            c.xyz,
            c.l_res,
            c.dist_ca,
            self.sphere_points,
            self.probe,
            self.radius
        )
        return asa


class ClashPotential(object):

    name = 'Clash-Potential'

    def __init__(
            self,
            structure: chisurf.structure.Structure = None,
            clash_tolerance: float = 2.0,
            covalent_radius: float = 1.5,
            **kwargs
    ):
        """
        :param kwargs:
        :return:

        Examples
        --------

        >>> import chisurf.structure
        >>> import chisurf.structure.potential

        >>> s = mfm.structure.structure.Structure('./test/data/model/hgbp1/hGBP1_closed.pdb', verbose=True, make_coarse=True)
        >>> pce = mfm.structure.potential.potentials.ClashPotential(structure=s, clash_tolerance=6.0)
        >>> pce.getEnergy()

        """
        self.structure = structure
        self.clash_tolerance = clash_tolerance
        self.covalent_radius = covalent_radius

    def getEnergy(self) -> float:
        c = self.structure
        return chisurf.structure.potential.cPotentials.clash_potential(
            c.xyz,
            c.vdw,
            self.clash_tolerance,
            self.covalent_radius
        )


class AvPotential(object):
    """
    The AvPotential class provides the possibility to calculate the reduced or unreduced chi2 given a set of
    labeling positions and experimental distances. Here the labeling positions and distances are provided as
    dictionaries.

    Examples
    --------

    >>> import json
    >>> labeling_file = './test/data/model/labeling.json'
    >>> labeling = json.load(open(labeling_file, 'r'))
    >>> distances = labeling['Distances']
    >>> positions = labeling['Positions']
    >>> import chisurf
    >>> av_potential = chisurf.structure.potential.potentials.AvPotential(distances=distances, positions=positions)
    >>> structure = chisurf.structure.Structure('./test/data/model/HM_1FN5_Naming.pdb')
    >>> av_potential.getChi2(structure)

    """
    name = 'Av'

    def __init__(
            self,
            labeling_file: str = None,
            structure: chisurf.structure.Structure = None,
            verbose: bool = False,
            rda_axis: np.array = None,
            av_samples: int = None,
            min_av: int = 150,
            **kwargs
    ):
        self._labeling_file = labeling_file
        self._structure = structure
        self.verbose = verbose

        if rda_axis is None:
            rda_axis = chisurf.fluorescence.rda_axis
        self.rda_axis = rda_axis

        self.distances = kwargs.get("Distances", None)
        self.positions = kwargs.get("Positions", None)

        if av_samples is None:
            av_samples = chisurf.settings["fps"]["distance_samples"]
        self.n_av_samples = av_samples
        self.min_av = min_av

        self.avs = dict()

    @property
    def labeling_file(self):
        return self._labeling_file

    @labeling_file.setter
    def labeling_file(self, v):
        self._labeling_file = v
        p = json.load(open(v))
        self.distances = p["Distances"]
        self.positions = p["Positions"]

    @property
    def structure(
            self
    ) -> chisurf.structure.Structure:
        """
        The Structure object used for the calculation of the accessible volumes
        """
        return self._structure

    @structure.setter
    def structure(
            self,
            structure: chisurf.structure.Structure
    ):
        self._structure = structure
        self.calc_avs()

    @property
    def chi2(self) -> float:
        """
        The current unreduced chi2 (recalculated at each call)
        """
        return self.getChi2()

    def calc_avs(self):
        """
        Calculates/recalculates the accessible volumes.
        """
        if self.structure is None:
            raise ValueError("The structure is not set")
        if self.positions is None:
            raise ValueError("Positions not set unable to calculate AVs")
        arguments = [
            dict(
                {'structure': self.structure, 'verbose': self.verbose,},
                **self.positions[position_key]
            )
            for position_key in self.positions
        ]
        avs = map(
            lambda x: chisurf.fluorescence.av.BasicAV(**x),
            arguments
        )
        for i, position_key in enumerate(self.positions):
            self.avs[position_key] = avs[i]

    def calc_distances(
            self,
            structure: chisurf.structure.Structure = None,
            verbose: bool = False
    ):
        """

        :param structure: Structure
            If this object is provided the attributes regarding dye-attachment are kept constant
            and the structure is changed prior calculation of the distances.
        :param verbose: bool
            If this is True output to stdout is generated
        """
        verbose = verbose or self.verbose
        if isinstance(
                structure,
                chisurf.structure.Structure
        ):
            self.structure = structure
        for distance_key in self.distances:
            distance = self.distances[distance_key]
            av1 = self.avs[distance['position1_name']]
            av2 = self.avs[distance['position2_name']]
            distance_type = distance['distance_type']
            R0 = distance['Forster_radius']
            if distance_type == 'RDAMean':
                d12 = av1.dRDA(av2)
            elif distance_type == 'Rmp':
                d12 = av1.dRmp(av1, av2)
            elif distance_type == 'RDAMeanE':
                d12 = av1.RDAE(av1, av2, R0)
            #elif distance_type == 'pRDA':
            #    rda = np.array(distance['rda'])
            #    d12 = functions.histogram_rda(av1, av2, rda_axis=rda)[0]
            distance['model_distance'] = d12
            if verbose:
                print("-------------")
                print("Distance: %s" % distance_key)
                print("Forster-Radius %.1f" % distance['Forster_radius'])
                print("Distance type: %s" % distance_type)
                print("Model distance: %.1f" % d12)
                print(
                    "Experimental distance: %.1f (-%.1f, +%.1f)" % (
                        distance['distance'],
                        distance['error_neg'], distance['error_pos']
                    )
                )

    def getChi2(
            self,
            structure: chisurf.structure.Structure = None,
            reduced: bool = False,
            verbose: bool = False
    ):
        """

        :param structure: Structure
            A Structure object if provided the attributes regarding dye-attachment are kept constant
            and the structure is changed prior calculation of the distances.
        :param reduced: bool
            If True the reduced chi2 is calculated (by default False)
        :param verbose: bool
            Output to stdout
        :return: A float containig the chi2 (reduced or unreduced) of the current or provided structure.
        """
        verbose = self.verbose or verbose
        if isinstance(
                structure,
                chisurf.structure.Structure
        ):
            self.structure = structure

        chi2 = 0.0
        self.calc_distances(verbose=verbose)
        for distance in list(self.distances.values()):
            dm = distance['model_distance']
            de = distance['distance']
            if distance['distance_type'] == 'pRDA':
                prda = np.array(distance['prda'])
                prda /= sum(prda)
                chi2 += sum((dm - prda)**2)
            else:
                error_neg = distance['error_neg']
                error_pos = distance['error_pos']
                d = dm - de
                chi2 += (d / error_neg) ** 2 if d < 0 else (d / error_pos) ** 2
        if reduced:
            return chi2 / (len(list(self.distances.keys())) - 1.0)
        else:
            return chi2

    def getEnergy(
            self,
            structure: chisurf.structure.Structure = None,
            gauss_bond: bool = True
    ):
        if isinstance(structure, chisurf.structure.Structure):
            self.structure = structure
        if gauss_bond:
            energy = 0.0
            self.calc_distances()
            for distance in list(self.distances.values()):
                dm = distance['model_distance']
                de = distance['distance']
                error_neg = distance['error_neg']
                error_pos = distance['error_pos']
                err = error_neg if (dm - de) < 0 else error_pos
                energy -= scipy.stats.norm.pdf(de, dm, err)
            return energy
        else:
            return self.getChi2(
                structure=self.structure
            )

