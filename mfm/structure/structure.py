import copy
import os
import tempfile
from collections import OrderedDict
from copy import deepcopy
import mdtraj as md
import numpy as np
import mfm.io.pdb


cartesian_keys = ['i', 'chain', 'res_id', 'res_name', 'atom_id', 'atom_name', 'element', 'coord',
                  'charge', 'radius', 'bfactor']

cartesian_formats = ['i4', '|S1', 'i4', '|S5', 'i4', '|S5', '|S1', '3f8', 'f4int', 'f4', 'f4']

clusterCriteria = ['maxclust', 'inconsistent', 'distance']


def onRMSF(structures, selectedNbrs, atomName=None, **kwargs):
    """Calculates the root mean square deviation with respect to the average structure
    for a given set of structures. The structures do not have to be aligned.

    :param structures:
    :param selectedNbrs:
    :param atomName:
    :param weights:
    :return:
    """
    weights = kwargs.get('weights', np.ones(len(selectedNbrs), dtype=np.float32))
    weights /= sum(weights)

    candidateStructures = [deepcopy(structures[i]) for i in selectedNbrs]
    print("calculating average structure as a reference")
    reference = average(candidateStructures, weights=weights)
    print("aligning selected structures with respect to reference")
    for s in candidateStructures:
        super_impose(reference, s)
    print("Getting %s-atoms of reference" % atomName)
    ar = reference.getAtoms(atomName=atomName)
    cr = ar['coord']
    msf = np.zeros(len(ar), dtype=np.float32)
    for i, s in enumerate(candidateStructures):
        a = s.getAtoms(atomName=atomName)
        ca = a['coord']
        msf += weights[i] * np.sum((cr - ca) ** 2, axis=1)
    return np.sqrt(msf)


def rmsd(structure_a, structure_b, atom_indices=None):
    """Calculates the root-mean-squared deviation between two structures. In case the indices of the atoms are
    provided only the, respective atoms are used to calculate the RMSD.

    :param structure_a:
    :param structure_b:
    :param atom_indices:

    :return: float, root-mean-squared deviation

    Examples
    --------

    >>> import mfm
    >>> t = mfm.TrajectoryFile('./sample_data/structure/2807_8_9_b.h5', mode='r', stride=1)
    >>> s1 = t[10]
    >>> s1
    <mfm.structure.structure.Structure at 0x135f3ad0>
    >>> s2 = t[0]
    >>> s2
    <mfm.structure.structure.Structure at 0x1348fbb0>
    >>> rmsd(s1, s2)
    6.960082250440536

    """
    if atom_indices is not None:
        a = structure_a.xyz[atom_indices]
        b = structure_b.xyz[atom_indices]
    else:
        a = structure_a.xyz
        b = structure_b.xyz
    return float(np.sqrt(1. / a.shape[0] * ((a - b) ** 2).sum()))


def super_impose(structure_ref, structure_align, atom_indices=None):
    """Superimpose two structures

    :param structure_ref:
    :param structure_align:
    :param atom_indices:
    :return:
    """
    if atom_indices is not None:
        a_atoms = structure_align.xyz[atom_indices]
        r_atoms = structure_ref.xyz[atom_indices]
    else:
        a_atoms = structure_align.xyz
        r_atoms = structure_ref.xyz

    # Center coordinates
    n = r_atoms.shape[0]
    av1 = a_atoms.sum(axis=0) / n
    av2 = r_atoms.sum(axis=0) / n
    re = structure_ref.xyz - av2
    al = structure_align.xyz - av1

    # Calculate rotation matrix
    a = np.dot(np.transpose(al), re)
    u, d, vt = np.linalg.svd(a)

    rot = np.transpose(np.dot(np.transpose(vt), np.transpose(u)))
    if np.linalg.det(rot) < 0:
        vt[2] = -vt[2]
        rot = np.transpose(np.dot(np.transpose(vt), np.transpose(u)))

    # Rotate structure
    structure_align.xyz = np.dot(al, rot)


def find_best(target, reference, atom_indices=None):
    """
    target and reference are both of type mdtraj.Trajectory
    reference is of length 1, target of arbitrary length

    returns a Structure object and the index within the trajectory

    Examples
    --------

    >>> import mfm
    >>> t = t = mfm.TrajectoryFile('./sample_data/structure/2807_8_9_b.h5', mode='r', stride=1)
    >>> find_best(t.mdtraj, t.mdtraj[2])
    (2, <mdtraj.Trajectory with 1 frames, 2495 atoms, 164 residues, without unitcells at 0x13570b30>)
    """
    rmsds = md.rmsd(target, reference, atom_indices=atom_indices)
    iMin = np.argmin(rmsds)
    return iMin, target[iMin]


def make_dictionary_of_atoms(atoms):
    """Organizes atoms residue-wise in an OrderedDict

    :param atoms:
    :return: OrderedDict
    """
    residue_dict = OrderedDict()
    for res in list(set(atoms['res_id'])):
        at_nbr = np.where(atoms['res_id'] == res)
        residue_dict[res] = OrderedDict()
        for atom in atoms[at_nbr]:
            residue_dict[res][atom['atom_name']] = atom
    return residue_dict


def get_coordinates_of_residues(atoms, quencher, verbose=False):
    """
    Returns coordinates and indices of all atoms of certain residue type.

    :param pdb: numpy-array
        numpy-array as returned by PDB.read
    :param quencher: dict
    :param verbose: bool
    :return:

    Examples
    --------

    >>> import mfm
    >>> pdb_file = './sample_data/model/hgbp1/hGBP1_closed.pdb'
    >>> pdb = mfm.io.pdb_file.read(pdb_file, verbose=True)
    Opening PDB-file: ./sample_data/model/hgbp1/hGBP1_closed.pdb
    ======================================
    Filename: ./sample_data/model/hgbp1/hGBP1_closed.pdb
    Path: ./sample_data/model/hgbp1
    Number of atoms: 9316
    --------------------------------------
    >>> mfm.tools.dye_diffusion.dye_diffusion.get_quencher_coordinates(pdb, {'TRP': ['CA']})
    OrderedDict([('TRP', array([[ 74.783,  -7.884,  18.04 ],
        [ 71.558, -13.346,  15.126],
       [ 70.873,   4.983,  23.373],
       [ 61.507,  -8.977,  33.237]]))])
    """
    if verbose:
        print("Finding quenchers")
        print(quencher)
    atom_idx = get_atom_index_of_residue_types(atoms, quencher)
    coordinates = OrderedDict()
    for res_key in quencher:
        coordinates[res_key] = atoms['coord'][atom_idx[res_key]]
    if verbose:
        print("Quencher atom-indeces: \n %s" % atom_idx)
        print("Quencher coordinates: \n %s" % coordinates)
    return coordinates


def get_atom_index_of_residue_types(pdb, res_types, verbose=False):
    """
    Returns atom-indices given a selection of residue types and atom names as OrderedDict.
    The selection is based on dictionaries. For each residue-type only the atoms within a
    list are selected (see example)

    :param pdb:
    :param res_types: dict
    :param verbose: bool
    :return: OrderedDict

    Examples
    --------

    >>> import mfm
    >>> pdb_file = './sample_data/model/hgbp1/hGBP1_closed.pdb'
    >>> pdb = mfm.io.PDB.read(pdb_file, verbose=True)
    Opening PDB-file: ./sample_data/model/hgbp1/hGBP1_closed.pdb
    ======================================
    Filename: ./sample_data/model/hgbp1/hGBP1_closed.pdb
    Path: ./sample_data/model/hgbp1
    Number of atoms: 9316
    --------------------------

    Get all atom-indices of all C-alphas in the TRY

    >>> mfm.tools.dye_diffusion.dye_diffusion.selection2atom_idx(pdb, {'TRP': ['CA']})
    OrderedDict([('TRP', array([1114, 1155, 1651, 2690], dtype=uint32))])
    >>> mfm.tools.dye_diffusion.dye_diffusion.selection2atom_idx(pdb, {'TRP': ['CA'], 'TYR': ['CA']})
    OrderedDict([('TRP', array([1114, 1155, 1651, 2690], dtype=uint32)), ('TYR', array([ 608,  707, 1879,
    2107, 2128, 3013, 3067, 4231, 4667, 5169, 6651, 6733, 7005, 7026, 7289, 8272], dtype=uint32))])
    """
    atom_idx = OrderedDict()
    for residue_key in res_types:
        atoms = []
        for atom_name in res_types[residue_key]:
            atoms.append(np.where((pdb['res_name'] == residue_key) & (pdb['atom_name'] == atom_name))[0])
        if len(atoms) > 0:
            atom_idx[residue_key] = np.array(np.hstack(atoms), dtype=np.uint32)
        else:
            atom_idx[residue_key] = np.array([], dtype=np.uint32)
    return atom_idx


def get_atom_index_by_name(pdb, atom_names, verbose=False):
    """
    Returns atom-indices given a an atomic name as list.

    :param pdb:
    :param atom_names: list of strings
    :param verbose: bool
    :return: list

    Examples
    --------

    >>> import mfm
    >>> pdb_file = './sample_data/model/hgbp1/hGBP1_closed.pdb'
    >>> pdb = mfm.io.PDB.read(pdb_file, verbose=True)
    Opening PDB-file: ./sample_data/model/hgbp1/hGBP1_closed.pdb
    ======================================
    Filename: ./sample_data/model/hgbp1/hGBP1_closed.pdb
    Path: ./sample_data/model/hgbp1
    Number of atoms: 9316
    --------------------------

    Get all atom-indices of all C-alphas in the TRY

    >>> mfm.tools.dye_diffusion.dye_diffusion.selection2atom_idx(pdb, {'TRP': ['CA']})
    OrderedDict([('TRP', array([1114, 1155, 1651, 2690], dtype=uint32))])
    >>> mfm.tools.dye_diffusion.dye_diffusion.selection2atom_idx(pdb, {'TRP': ['CA'], 'TYR': ['CA']})
    OrderedDict([('TRP', array([1114, 1155, 1651, 2690], dtype=uint32)), ('TYR', array([ 608,  707, 1879,
    2107, 2128, 3013, 3067, 4231, 4667, 5169, 6651, 6733, 7005, 7026, 7289, 8272], dtype=uint32))])
    """
    atoms = []
    for atom_name in atom_names:
        atoms.append(np.where((pdb['atom_name'] == atom_name))[0])
    return atoms


def sequence(structure_obj):
    """Return dictionary of sequences keyed to chain and type of sequence used.

    :param structure_obj: Structure
    """
    # Check to see if there are multiple models.  If there are, only look
    # at the first model.
    p = structure_obj.atoms
    atoms = [a for a in p if a['atom_name'] == "CA "]
    chain_dict = dict([(l[21], []) for l in atoms])
    for c in list(chain_dict.keys()):
        chain_dict[c] = [l[17:20] for l in atoms if l[21] == c]
    return chain_dict


def count_atoms(topology_dict):
    """Count the number of atoms in a topology dict

    :param topology_dict: dict
        a topology of a structure as python dict (as described by MDTraj)
    :return: int
    """
    n_atoms = 0
    for chain in topology_dict['chains']:
        for residue in chain['residues']:
            n_atoms += len(residue['atoms'])
    return n_atoms


class Structure(object):#mfm.curve.ExperimentalData):
    """

    Attributes
    ----------
    max_atom_residue : int
        Maximum atoms per residue. The maximum atoms per residue have to be limited, because of the
        residue/atom lookup tables. By default maximum 16 atoms per residue are allowed.

    xyz : numpy-array
        The xyz-attribute is an array of the cartesian coordinates represented by the attribute atom

    residue_names: list
        residue names is a list of the residue-names. Here the residues are represented by the same
        name as the initial PDB-file, usually 3-letter amino-acid code. The residue_names attribute
        may look like: ['TYR', 'HIS', 'ARG']

    n_atoms: int
        The number of atoms

    b_factos: array
        An array of the b-factors.

    radius_gyration: float
        The attribute radius_gyration returns the radius of gyration of the structure. The radius of gyration
        is given by: rG = (np.sqrt((coord - rM) ** 2).sum(axis=1)).mean()
        Here rM are the mean coordinates of the structure.


    :param filename: str
        Path to the pdb file on disk
    :param make_coarse: bool
        Conversion to coarse representation using internal coordinates. Side-chains are
        not considered. The Cbeta-atoms is moved to center of mass of the side-chain.
    :param verbose: bool
        print output to stdout
    :param auto_update: bool
        update cartesian-coordiantes automatically after change of internal coordinates.
        This only applies for coarse-grained coordinates

    Examples
    --------

    >>> import mfm.structure
    >>> structure = mfm.structure.Structure('1dg3', verbose=True)
    ======================================
    Filename: ../sample_data/structure/HM_1FN5_Naming.pdb
    Path: ../sample_data/structure
    Number of atoms: 9316
    >>> print(str(structure)[:324])
    ATOM      1    N HIS A   6     -18.863  20.262  33.465  0.00  0.00             N
    ATOM      2   CA HIS A   6     -17.949  21.401  33.768  0.00  0.00             C
    ATOM      3    C HIS A   6     -18.327  22.599  32.897  0.00  0.00             C
    ATOM      4    O HIS A   6     -19.507  22.812  32.614  0.00  0.00             O

    Write structure to disk

    >>> structure.write('test_out.pdb')

    Radius of gyration

    >>> structure.radius_gyration
    50.31556120901161

    """

    def __init__(self, p_object, **kwargs):
        self.auto_update = kwargs.get('auto_update', False)
        self._filename = kwargs.get('filename', None)
        self._atoms = None
        self._residue_dict = None
        self._potentials = list()
        self._sequence = None
        self.verbose = kwargs.get('verbose', False)

        self.io = mfm.io.pdb
        self.pdbid = None
        ####################################################
        #              Load ATOM COORDINATES               #
        ####################################################
        if isinstance(p_object, Structure):
            self.atoms = np.copy(p_object.atoms)
            self.filename = copy.copy(p_object.filename)
        elif p_object is None:
            self._atoms = None
            self._filename = None
        else:
            try:
                self._atoms = self.io.read(p_object, verbose=self.verbose)
                self.filename = p_object
            except:
                self._atoms = self.io.fetch_pdb(p_object)
                self.pdbid = p_object

    @property
    def sequence(self):
        if self._sequence is None:
            self._sequence = sequence(self)
        return self._sequence

    @property
    def internal_coordinates(self):
        return None

    @property
    def energy(self):
        energies = [e(self, **kwargs) for e, kwargs in self._potentials]
        return np.sum(energies)

    @property
    def atoms(self):
        if isinstance(self._atoms, np.ndarray):
            return self._atoms
        else:
            return np.zeros(1, dtype={'names': mfm.io.pdb.keys, 'formats': mfm.io.pdb.formats})

    @atoms.setter
    def atoms(self, v):
        if isinstance(v, np.ndarray):
            self._atoms = v

    @property
    def xyz(self):
        """Cartesian coordinates of all atoms
        """
        return self.atoms['coord']

    @xyz.setter
    def xyz(self, v):
        self.atoms['coord'] = v

    @property
    def vdw(self):
        """Van der Waals radii of all atoms
        """
        return self.atoms['radius']

    @vdw.setter
    def vdw(self, v):
        self.atoms['radius'] = v

    @property
    def residue_names(self):
        res_name = list(set(self.atoms['res_name']))
        res_name.sort()
        return res_name

    @property
    def residue_dict(self):
        if self._residue_dict is None:
            residue_dict = make_dictionary_of_atoms(self.atoms)
            self._residue_dict = residue_dict
        return self._residue_dict

    @property  # OK
    def n_atoms(self):
        return len(self.atoms)

    @property  # OK
    def n_residues(self):
        return len(self.residue_ids)

    @property  # OK
    def atom_types(self):
        return set(self.atoms['atom_name'])

    @property  # OK
    def residue_ids(self):
        residue_ids = list(set(self.atoms['res_id']))
        return residue_ids

    def get_atom_index(self, atom_names):
        """Returns a list of the indeces with a given atom name

        :param atom_names: list of string of the atom-names
        :return: list of integers the indices with a given atom name
        """
        get_atom_index_by_name(self.atoms, atom_names)

    @property
    def b_factors(self):
        """B-factors of the C-alpha atoms
        """
        sel = get_atom_index_by_name(self.atoms, ['CA'])[0]
        bfac = self.atoms[sel]['bfactor']
        return bfac

    @b_factors.setter
    def b_factors(self, v):
        s = self
        sel = get_atom_index_by_name(s.atoms, ['CA'])[0]
        for ai, bi in zip(sel, v):
            s._atoms[ai]['bfactor'] = bi

    @property
    def radius_gyration(self):
        coord = self.xyz
        rM = coord[:, :].distance(axis=0)
        rG = (np.sqrt((coord - rM) ** 2).sum(axis=1)).distance()
        return float(rG)

    def append_potential(self, function, kwargs=dict()):
        """

        :param function:
        :param kwargs:
        :return:

        Examples
        --------

        >>> import mfm.structure
        >>> structure = mfm.structure.Structure('./sample_data/modelling/pdb_files/hGBP1_closed.pdb', verbose=True)
        >>> structure.append_potential(mfm.structure.potential.lennard_jones_calpha)
        >>> structure.energy
        -948.0396693387753

        >>> structure.append_potential(mfm.structure.potential.internal_potential_calpha, kwargs={})
        """
        self._potentials.append([function, kwargs])

    def update_coordinates(self):
        if self.auto_update:
            self.update()

    def write(self, *args, **kwargs):
        """
        Write the structure to a filename. By default it uses PDB files

        :param filename: The filename
        :param append_model: bool
            If True the structure is appended to the filename as a new model otherwise the file is overwritten
        """
        filename = kwargs.get('filename', self._filename)
        append_model = kwargs.get('append_model', False)
        append_coordinates = kwargs.get('append_coordinates', False)
        if len(args) > 0:
            filename = args[0]
        aw = np.copy(self.atoms)
        aw['coord'] = self.xyz
        self.io.write(filename, aw, append_model=append_model, append_coordinates=append_coordinates)

    def update(self):
        pass

    def __str__(self):
        s = ""
        if self.atoms is not None:
            s = ""
            for at in self.atoms:
                s += "%-6s%5d %4s%1s%3s %1s%4d%1s   %8.3f%8.3f%8.3f%6.2f%6.2f          %2s%2s\n" % \
                     ("ATOM ",
                      at['atom_id'], at['atom_name'], " ", at['res_name'], at['chain'],
                      at['res_id'], " ",
                      at['coord'][0], at['coord'][1], at['coord'][2],
                      0.0, 0.0, "  ", at['element'])
        return s + "END\n"

    def __deepcopy__(self, memo):
        new = copy.copy(self)
        new._atoms = np.copy(self._atoms)
        new.filename = copy.copy(self.filename)
        new.io = self.io
        return new


def average(structures, weights=None, write=True, filename=None):
    """
    Calculates weighted average of a list of structures.
    saves to filename if write is True
    if filename not provided makes new "average.pdb" file in temp-folder
    of the system

    Examples
    --------

    >>> import mfm
    >>> t = mfm.TrajectoryFile('./sample_data/structure/2807_8_9_b.h5', mode='r', stride=1)
    >>> avg = t.average
    >>> avg
    <mfm.structure.structure.Structure at 0x117ff770>
    >>> avg.labeling_file
    'c:\\users\\peulen\\appdata\\local\\temp\\average.pdb'
    """
    if weights is None:
        weights = np.ones(len(structures), dtype=np.float64)
        weights /= weights.sum()
    else:
        weights = np.array(weights)
    avg = Structure()
    avg.atoms = np.copy(structures[0].atoms)
    avg.xyz *= 0.0
    for i, s in enumerate(structures):
        avg.xyz += weights[i] * s.xyz
    filename = os.path.join(tempfile.tempdir, "average.pdb") if filename is None else filename
    if write:
        avg.filename = filename
        avg.write()
    return avg