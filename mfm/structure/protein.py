from __future__ import annotations

import math
from collections import OrderedDict
from copy import copy, deepcopy
import numpy as np
import numba as nb

import mfm.math.linalg as la
from mfm.structure.structure import Structure


internal_formats = ['i4', 'i4', 'i4', 'i4', 'f8', 'f8', 'f8']

internal_atom_numbers = [
    ('N', 0),
    ('CA', 1),
    ('C', 2),
    ('O', 3),
    ('CB', 4),
    ('H', 5),
    ('CG', 6),
    ('CD', 7),
]
residue_atoms_internal = OrderedDict([
    ('CYS', ['N', 'C', 'CA', 'CB', 'O', 'H']),
    ('MET', ['N', 'C', 'CA', 'CB', 'O', 'H']),
    ('PHE', ['N', 'C', 'CA', 'CB', 'O', 'H']),
    ('ILE', ['N', 'C', 'CA', 'CB', 'O', 'H']),
    ('LEU', ['N', 'C', 'CA', 'CB', 'O', 'H']),
    ('VAL', ['N', 'C', 'CA', 'CB', 'O', 'H']),
    ('TRP', ['N', 'C', 'CA', 'CB', 'O', 'H']),
    ('TYR', ['N', 'C', 'CA', 'CB', 'O', 'H']),
    ('ALA', ['N', 'C', 'CA', 'CB', 'O', 'H']),
    ('GLY', ['N', 'C', 'CA', 'O', 'H']),
    ('THR', ['N', 'C', 'CA', 'CB', 'O', 'H']),
    ('SER', ['N', 'C', 'CA', 'CB', 'O', 'H']),
    ('GLN', ['N', 'C', 'CA', 'CB', 'O', 'H']),
    ('ASN', ['N', 'C', 'CA', 'CB', 'O', 'H']),
    ('GLU', ['N', 'C', 'CA', 'CB', 'O', 'H']),
    ('ASP', ['N', 'C', 'CA', 'CB', 'O', 'H']),
    ('HIS', ['N', 'C', 'CA', 'CB', 'O', 'H']),
    ('ARG', ['N', 'C', 'CA', 'CB', 'O', 'H']),
    ('LYS', ['N', 'C', 'CA', 'CB', 'O', 'H']),
    ('PRO', ['N', 'C', 'CA', 'CB', 'O', 'H', 'CG', 'CD']),
]
)
internal_keys = ['i', 'ib', 'ia', 'id', 'b', 'a', 'd']
a2id = dict(internal_atom_numbers)
id2a = dict([(a[1], a[0]) for a in internal_atom_numbers])
res2id = dict([(aa, i) for i, aa in enumerate(residue_atoms_internal)])


def r2i(coord_i, a1, a2, a3, a4, ai):
    """Cartesian coordinates to internal-coordinates

    :param coord_i: a numpy array which contains the number of the atoms a1, a2, a3, a4 and the bond-length, angle,
    and dihedral angle between the four atoms
    :param a1: first atoms
    :param a2: second atom
    :param a3: third atom
    :param a4: fourth atom (the next atom)
    :param ai: The index of the internal coordinate which is defined by the four atoms
    :return: the index of the next atom (ai+1)
    """
    vn = a4['xyz']
    v1 = a1['xyz']
    v2 = a2['xyz']
    v3 = a3['xyz']
    b = la.norm3(v3 - vn)
    a = la.angle(v2, v3, vn)
    d = la.dihedral(v1, v2, v3, vn)
    coord_i[ai] = a4['i'], a3['i'], a2['i'], a1['i'], b, a, d
    return ai + 1


@nb.jit('float64[:,:](float64[:,:], int32[:,:], float64[:,:], int32)', nogil=True, nopython=True)
def atom_dist(aDist, resLookUp, xyz, aID):
    n_residues = resLookUp.shape[0]
    for i in range(n_residues):
        ia1 = resLookUp[i, aID]
        if ia1 < 0:
            continue
        a1 = xyz[ia1, 0]
        a2 = xyz[ia1, 1]
        a3 = xyz[ia1, 2]
        for j in range(i, n_residues):
            ia2 = resLookUp[j, aID]
            if ia2 < 0:
                continue
            b1 = xyz[ia2, 0] - a1
            b2 = xyz[ia2, 1] - a2
            b3 = xyz[ia2, 2] - a3

            d12 = math.sqrt(b1*b1 + b2*b2 + b3*b3)

            aDist[i, j] = d12
            aDist[j, i] = d12
    return aDist


def move_center_of_mass(
        structure: Structure,
        all_atoms
):
    for i, res in enumerate(structure.residue_ids):
        at_nbr = np.where(all_atoms['res_id'] == res)[0]
        cb_nbr = structure.l_cb[i]
        if cb_nbr > 0:
            cb = structure.atoms[cb_nbr]
            cb['xyz'] *= cb['mass']
            for at in at_nbr:
                atom = all_atoms[at]
                residue_name = atom['res_name']
                if atom['atom_name'] not in residue_atoms_internal[residue_name]:
                    cb['xyz'] += atom['xyz'] * atom['mass']
                    cb['mass'] += atom['mass']
            cb['xyz'] /= cb['mass']
            structure.atoms[cb_nbr] = cb


def make_residue_lookup_table(
        structure: Structure
):
    l_residue = np.zeros((structure.n_residues, structure.max_atom_residue), dtype=np.int32) - 1
    n = 0
    res_dict = structure.residue_dict
    for residue in list(res_dict.values()):
        for atom in list(residue.values()):
            atom_name = atom['atom_name']
            if atom_name in list(a2id.keys()):
                l_residue[n, a2id[atom_name]] = atom['i']
        n += 1
    l_ca = l_residue[:, a2id['CA']]
    l_cb = l_residue[:, a2id['CB']]
    l_c = l_residue[:, a2id['C']]
    l_n = l_residue[:, a2id['N']]
    l_h = l_residue[:, a2id['H']]
    return l_residue, l_ca, l_cb, l_c, l_n, l_h


@nb.jit(nogil=True, nopython=True)
def internal_to_cartesian(
        bond: np.array,
        angle: np.array,
        dihedral: np.array,
        ans,
        i_b: np.array,
        i_a: np.array,
        i_d: np.array,
        n_atoms: int,
        r,
        p,
        startPoint: int
):
    """

    :param bond: bond-length
    :param angle: angle
    :param dihedral: dihedral
    :param ans:
    :param i_b:
    :param i_a:
    :param i_d:
    :param n_atoms:
    :param r:
    :param startPoint:
    :return:
    """

    for i in range(n_atoms):
        sin_theta = math.sin(angle[i])
        cos_theta = math.cos(angle[i])
        sin_phi = math.sin(dihedral[i])
        cos_phi = math.cos(dihedral[i])

        p[i*3 + 0] = bond[i] * sin_theta * sin_phi
        p[i*3 + 1] = bond[i] * sin_theta * cos_phi
        p[i*3 + 2] = bond[i] * cos_theta

    for i in range(n_atoms):
        if i_b[i] != 0 and i_a[i] != 0 and i_d[i] != 0:
            ab1 = (r[i_b[i], 0] - r[i_a[i], 0])
            ab2 = (r[i_b[i], 1] - r[i_a[i], 1])
            ab3 = (r[i_b[i], 2] - r[i_a[i], 2])
            nab = math.sqrt(ab1*ab1+ab2*ab2+ab3*ab3)
            ab1 /= nab
            ab2 /= nab
            ab3 /= nab
            bc1 = (r[i_a[i], 0] - r[i_d[i], 0])
            bc2 = (r[i_a[i], 1] - r[i_d[i], 1])
            bc3 = (r[i_a[i], 2] - r[i_d[i], 2])
            nbc = math.sqrt(bc1*bc1+bc2*bc2+bc3*bc3)
            bc1 /= nbc
            bc2 /= nbc
            bc3 /= nbc
            v1 = ab3 * bc2 - ab2 * bc3
            v2 = ab1 * bc3 - ab3 * bc1
            v3 = ab2 * bc1 - ab1 * bc2
            cos_alpha = ab1*bc1 + ab2*bc2 + ab3*bc3
            sin_alpha = math.sqrt(1.0 - cos_alpha * cos_alpha)
            v1 /= sin_alpha
            v2 /= sin_alpha
            v3 /= sin_alpha
            u1 = v2 * ab3 - v3 * ab2
            u2 = v3 * ab1 - v1 * ab3
            u3 = v1 * ab2 - v2 * ab1

            r[ans[i], 0] = r[i_b[i], 0] + v1 * p[i * 3 + 0] + u1 * p[i * 3 + 1] - ab1 * p[i * 3 + 2]
            r[ans[i], 1] = r[i_b[i], 1] + v2 * p[i * 3 + 0] + u2 * p[i * 3 + 1] - ab2 * p[i * 3 + 2]
            r[ans[i], 2] = r[i_b[i], 2] + v3 * p[i * 3 + 0] + u3 * p[i * 3 + 1] - ab3 * p[i * 3 + 2]


def calc_internal_coordinates_bb(
        structure: Structure,
        verbose: bool = None,
        **kwargs
):
    if verbose is None:
        verbose = mfm.verbose

    structure.coord_i = np.zeros(
        structure.atoms.shape[0],
        dtype={'names': internal_keys, 'formats': internal_formats}
    )
    rp, ai = None, 0
    res_nr = 0
    for rn in list(structure.residue_dict.values()):
        res_nr += 1
        # BACKBONE
        if rp is None:
            structure.coord_i[ai] = rn['N']['i'], 0, 0, 0, 0.0, 0.0, 0.0
            ai += 1
            structure.coord_i[ai] = rn['CA']['i'], rn['N']['i'], 0, 0, \
                                    la.norm3(rn['N']['xyz'] - rn['CA']['xyz']), 0.0, 0.0
            ai += 1
            structure.coord_i[ai] = rn['C']['i'], rn['CA']['i'], rn['N']['i'], 0, \
                                    la.norm3(rn['CA']['xyz'] - rn['C']['xyz']), \
                                    la.angle(rn['C']['xyz'], rn['CA']['xyz'], rn['N']['xyz']), \
                                    0.0
            ai += 1
        else:
            ai = r2i(structure.coord_i, rp['N'], rp['CA'], rp['C'], rn['N'], ai)
            ai = r2i(structure.coord_i, rp['CA'], rp['C'], rn['N'], rn['CA'], ai)
            ai = r2i(structure.coord_i, rp['C'], rn['N'], rn['CA'], rn['C'], ai)
        ai = r2i(structure.coord_i, rn['N'], rn['CA'], rn['C'], rn['O'], ai)  # O
        # SIDECHAIN
        resName = rn['CA']['res_name']
        if resName != 'GLY':
            ai = r2i(structure.coord_i, rn['O'], rn['C'], rn['CA'], rn['CB'], ai)  # CB
        if resName != 'PRO':
            ai = r2i(structure.coord_i, rn['N'], rn['CA'], rn['C'], rn['H'], ai)  # H
        else:
            ai = r2i(structure.coord_i, rn['N'], rn['CA'], rn['CB'], rn['CG'], ai)  # CG
            ai = r2i(structure.coord_i, rn['CA'], rn['CB'], rn['CG'], rn['CD'], ai)  # CD
        rp = rn
    if verbose:
        print("Atoms internal: %s" % (ai + 1))
        print("--------------------------------------")
    structure.coord_i = structure.coord_i[:ai]
    structure._phi_indices = [list(structure.coord_i['i']).index(x) for x in structure.l_c]
    structure._omega_indices = [list(structure.coord_i['i']).index(x) for x in structure.l_ca]
    structure._psi_indices = [list(structure.coord_i['i']).index(x) for x in structure.l_n]
    structure._chi_indices = [list(structure.coord_i['i']).index(x) for x in structure.l_cb if x >= 0]


class ProteinCentroid(Structure):
    """

    Examples
    --------

    >>> import mfm
    >>> sp = mfm.structure.protein.ProteinCentroid('../test/data/structure/HM_1FN5_Naming.pdb', verbose=True, make_coarse=True)
    ======================================
    Filename: /data/structure/HM_1FN5_Naming.pdb
    Path: /data/structure
    Number of atoms: 9316
    --------------------------------------
    Atoms internal: 3457
    --------------------------------------
    >>> print(sp)
    ATOM   2274    O ALA   386      52.927  10.468 -17.263  0.00  0.00             O
    ATOM   2275   CB ALA   386      53.143  12.198 -14.414  0.00  0.00             C
    >>> sp.omega *= 0.0
    >>> sp.omega += 3.14
    >>> print(sp)
    ATOM   2273    C ALA   386      47.799  59.970  21.123  0.00  0.00             C
    ATOM   2274    O ALA   386      47.600  59.096  20.280  0.00  0.00             O
    >>> sp.write('test_out.pdb')
    >>> s_aa = mfm.structure.protein.ProteinCentroid('./test/data/structure/HM_1FN5_Naming.pdb', verbose=True, make_coarse=False)
    >>> print(s_aa)
    ATOM   9312    H MET   583      40.848  10.075  17.847  0.00  0.00             H
    ATOM   9313   HA MET   583      40.666   8.204  15.667  0.00  0.00             H
    ATOM   9314  HB3 MET   583      38.898   7.206  16.889  0.00  0.00             H
    ATOM   9315  HB2 MET   583      38.796   8.525  17.846  0.00  0.00             H
    >>> print(s_aa.omega)
    array([], dtype=float64)
    >>> s_aa.to_coarse()
    >>> print(s_aa)
    ATOM   3451   CA MET   583      40.059   8.800  16.208  0.00  0.00             C
    ATOM   3452    C MET   583      38.993   9.376  15.256  0.00  0.00             C
    ATOM   3453    O MET   583      38.405  10.421  15.616  0.00  0.00             O
    ATOM   3454   CB MET   583      39.408   7.952  17.308  0.00  0.00             C
    ATOM   3455    H MET   583      40.848  10.075  17.847  0.00  0.00             H
    >>> print(s_aa.omega)
    array([ 0.        ,  3.09665806, -3.08322105,  3.13562203,  3.09102453,...])
    """

    max_atom_residue = 16

    @property
    def internal_coordinates(self):
        return self.coord_i

    @property
    def phi(self):
        return self.internal_coordinates[self._phi_indices]['d']

    @phi.setter
    def phi(self, v):
        self.internal_coordinates['d'][self._phi_indices] = v
        self.update_coordinates()

    @property
    def omega(self):
        return self.internal_coordinates[self._omega_indices]['d']

    @omega.setter
    def omega(self, v):
        self.internal_coordinates['d'][self._omega_indices] = v
        self.update_coordinates()

    @property
    def chi(self):
        return self.internal_coordinates[self._chi_indices]['d']

    @chi.setter
    def chi(self, v):
        self.internal_coordinates['d'][self._chi_indices] = v
        self.update_coordinates()

    @property
    def psi(self):
        return self.internal_coordinates[self._psi_indices]['d']

    @psi.setter
    def psi(self, v):
        self.internal_coordinates['d'][self._psi_indices] = v
        self.update_coordinates()

    def __deepcopy__(self, memo):
        new = copy(self)
        new.atoms = np.copy(self.atoms)
        new.dist_ca = np.copy(self.dist_ca)
        new.filename = copy(self.filename)
        new.coord_i = np.copy(self.internal_coordinates)
        new.io = self.io

        new.l_ca = deepcopy(self.l_ca)
        new.l_cb = deepcopy(self.l_cb)
        new.l_c = deepcopy(self.l_c)
        new.l_n = deepcopy(self.l_n)
        new.l_h = deepcopy(self.l_h)
        new.l_res = deepcopy(self.l_res)

        return new

    def __init__(
            self,
            *args,
            make_lookup: bool = True,
            **kwargs
    ):
        super(ProteinCentroid).__init__(
            *args,
            **kwargs)
        self.coord_i = np.zeros(
            self.atoms.shape[0],
            dtype={'names': internal_keys, 'formats': internal_formats}
        )
        self.dist_ca = np.zeros(
            (self.n_residues, self.n_residues),
            dtype=np.float64
        )
        self._phi_indices = list()
        self._omega_indices = list()
        self._psi_indices = list()
        self._chi_indices = list()
        self._temp = np.empty(
            self.n_atoms * 3,
            dtype=np.float64
        )  # used to convert internal to cartesian coordinates

        self.to_coarse()
        ####################################################
        #              LOOKUP TABLES                       #
        ####################################################
        if make_lookup:
            self.l_res, self.l_ca, self.l_cb, self.l_c, self.l_n, self.l_h = make_residue_lookup_table(self)
            self.residue_types = np.array([res2id[res] for res in list(self.atoms['res_name'])
                                           if res in list(res2id.keys())], dtype=np.int32)

    def update_dist(self):
        atom_dist(self.dist_ca, self.l_res, self.xyz, a2id['CA'])

    def update(
            self,
            start_point: int = 0
    ):
        ic = self.internal_coordinates
        n_atoms = ic.shape[0]

        bond = ic['b']
        angle = ic['a']
        dihedral = ic['d']
        ans = ic['i']

        ib = ic['ib']
        ia = ic['ia']
        id = ic['id']

        r = self.xyz
        p = self._temp
        internal_to_cartesian(bond, angle, dihedral, ans, ib, ia, id, n_atoms, r, p, start_point)
        self.update_dist()

    def to_coarse(self):
        """
        Converts the structure-instance into a coarse structure.

        Examples
        --------

        >>> s_aa = mfm.structure.ProteinCentroid('./test/data/modelling/pdb_files/hGBP1_closed.pdb', verbose=True, make_coarse=False)
        >>> print(s_aa)
        ATOM   9312    H MET   583      40.848  10.075  17.847  0.00  0.00             H
        ATOM   9313   HA MET   583      40.666   8.204  15.667  0.00  0.00             H
        ATOM   9314  HB3 MET   583      38.898   7.206  16.889  0.00  0.00             H
        ATOM   9315  HB2 MET   583      38.796   8.525  17.846  0.00  0.00             H
        >>> s_aa.to_coarse()
        >>> print(s_aa)
        ATOM   3451   CA MET   583      40.059   8.800  16.208  0.00  0.00             C
        ATOM   3452    C MET   583      38.993   9.376  15.256  0.00  0.00             C
        ATOM   3453    O MET   583      38.405  10.421  15.616  0.00  0.00             O
        ATOM   3454   CB MET   583      39.408   7.952  17.308  0.00  0.00             C
        ATOM   3455    H MET   583      40.848  10.075  17.847  0.00  0.00             H
        print(s_aa.omega)
        array([ 0.        ,  3.09665806, -3.08322105,  3.13562203,  3.09102453,...])
        """
        self.is_coarse = True

        ####################################################
        ######       TAKE ONLY INTERNAL ATOMS         ######
        ####################################################
        all_atoms = np.copy(self.atoms)
        tmp = [atom for atom in all_atoms if atom['atom_name'] in residue_atoms_internal[atom['res_name']]]
        atoms = np.empty(len(tmp), dtype={'names': mfm.io.pdb.keys, 'formats': mfm.io.pdb.formats})
        atoms[:] = tmp
        atoms['i'] = np.arange(atoms.shape[0])
        atoms['atom_id'] = np.arange(atoms.shape[0])
        self.atoms = atoms

        ####################################################
        ######         LOOKUP TABLES                  ######
        ####################################################
        self.l_res, self.l_ca, self.l_cb, self.l_c, self.l_n, self.l_h = make_residue_lookup_table(self)
        tmp = [res2id[res] for res in list(self.atoms['res_name']) if res in list(res2id.keys())]
        self.residue_types = np.array(tmp, dtype=np.int32)
        self.dist_ca = np.zeros((self.n_residues, self.n_residues), dtype=np.float64)

        ####################################################
        ######         REASSIGN COORDINATES           ######
        ####################################################
        move_center_of_mass(self, all_atoms)

        ####################################################
        ######         INTERNAL  COORDINATES          ######
        ####################################################
        #coord_i = np.zeros(self.atoms.shape[0], dtype={'names': internal_keys, 'formats': internal_formats})
        calc_internal_coordinates_bb(self)
        self.update_coordinates()


class ProteinBead(Structure):

    """
    >>> s_aa = mfm.structure.protein.ProteinBead('./test/data/modelling/pdb_files/hGBP1_closed.pdb', verbose=True)
    """

    max_atom_residue = 2

    @property
    def internal_coordinates(self):
        return self.coord_i

    def __init__(
            self,
            *args,
            **kwargs
    ):
        super(ProteinBead, self).__init__(*args, **kwargs)
        self.coord_i = np.zeros(
            self.atoms.shape[0],
            dtype={
                'names': internal_keys,
                'formats': internal_formats
            }
        )

        self.dist_ca = np.zeros(
            (self.n_residues, self.n_residues),
            dtype=np.float64
        )
        self._temp = np.empty(
            self.n_atoms * 3,
            dtype=np.float64
        )  # used to convert internal to cartesian coordinates
        self.to_coarse()
        self.coord_i_initial = np.copy(self.coord_i)

    def update(
            self,
            start_point: int = 0
    ):
        internal_coordinates = self.internal_coordinates
        n_atoms = internal_coordinates.shape[0]

        bond = internal_coordinates['b']
        angle = internal_coordinates['a']
        dihedral = internal_coordinates['d']
        ans = internal_coordinates['i']

        i_bonds = internal_coordinates['ib']
        i_angles = internal_coordinates['ia']
        i_dihedrals = internal_coordinates['id']

        r = self.xyz
        p = self._temp
        internal_to_cartesian(
            bond,
            angle,
            dihedral,
            ans,
            i_bonds,
            i_angles,
            i_dihedrals,
            n_atoms,
            r,
            p,
            start_point
        )

    def calc_internal_coordinates(self):

        def internal_coordinates_bead(structure):
            s = structure
            coord_i = np.zeros(
                s.atoms.shape[0],
                dtype={'names': internal_keys, 'formats': internal_formats}
            )
            r_1, r_2, r_3, r_4, ai = None, None, None, None, 0
            rd = s.residue_dict.values()

            coord_i[0] = rd[0]['CA']['i'], 0, 0, 0, 0.0, 0.0, 0.0
            coord_i[1] = rd[0]['CA']['i'], rd[1]['CA']['i'], 0, 0, la.norm3(
                rd[0]['CA']['xyz'] - rd[1]['CA']['xyz']), 0.0, 0.0
            coord_i[2] = rd[0]['CA']['i'], rd[1]['CA']['i'], rd[2]['CA']['i'], 0, la.norm3(
                rd[2]['CA']['xyz'] - rd[1]['CA']['xyz']), \
                         la.angle(rd[0]['CA']['xyz'], rd[1]['CA']['xyz'], rd[2]['CA']['xyz']), \
                         0.0
            for i in range(3, len(rd)):
                ai = r2i(coord_i, rd[i - 3]['CA'], rd[i - 2]['CA'], rd[i - 1]['CA'], rd[i]['CA'], i)

            return coord_i[:ai]

        self.coord_i = internal_coordinates_bead(self)

    def to_coarse(self):
        self.is_coarse = True

        ####################################################
        #            TAKE ONLY INTERNAL ATOMS              #
        ####################################################
        tmp = [atom for atom in self.atoms if atom['atom_name'] in ['CA']]
        atoms = np.empty(len(tmp), dtype={'names': mfm.io.pdb.keys, 'formats': mfm.io.pdb.formats})
        atoms[:] = tmp
        atoms['i'] = np.arange(atoms.shape[0])
        atoms['atom_id'] = np.arange(atoms.shape[0])
        self.atoms = atoms

        ####################################################
        #              LOOKUP TABLES                       #
        ####################################################
        tmp = [res2id[res] for res in list(self.atoms['res_name']) if res in list(res2id.keys())]
        self.residue_types = np.array(tmp, dtype=np.int32)
        self.dist_ca = np.zeros((self.n_residues, self.n_residues), dtype=np.float64)

        ####################################################
        #         INTERNAL  COORDINATES                    #
        ####################################################
        self.coord_i = np.zeros(self.atoms.shape[0], dtype={'names': internal_keys, 'formats': internal_formats})
        self.calc_internal_coordinates()
        self.update_coordinates()

