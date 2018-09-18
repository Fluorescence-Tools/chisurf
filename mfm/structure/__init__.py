"""
The structure module contains most functions and classes to handle structural models of proteins. It contains basically
two submodules:

#. :py:mod:`.mfm.structure.structure`
#. :py:mod:`.mfm.structure.trajectory`
#. :py:mod:`.mfm.potential`

The module :py:mod:`.mfm.potential` provides a set of potentials. The module :py:mod:`.mfm.structure.structure`
provides a set of functions and classes to work with structures and trajectories.

"""
import mfm


def sequence(structure):
    """Determines the amino acid sequence of a structure

    :param structure:
    :return: list of the residue names
    """
    try:
        a = structure.atoms
        s = list()
        for i in structure.residue_ids:
            s += list(set(a[np.where(a['res_id'] == i)[0]]['res_name']))
        return s
    except:
        ro = None
        se = list()
        for a in structure.atoms:
            if a['res_id'] != ro:
                se.append(a['res_name'])
                ro = a['res_id']
        return se


class Structure(mfm.curve.ExperimentalData):
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

    >>> import mfm
    >>> structure = mfm.structure.Structure('./sample_data/modelling/pdb_files/hGBP1_closed.pdb', verbose=True)
    ======================================
    Filename: ../sample_data/structure/HM_1FN5_Naming.pdb
    Path: ../sample_data/structure
    Number of atoms: 9316
    >>> print str(structure)[:324]
    ATOM      1    N MET     7      72.739 -17.501   8.879  0.00  0.00             N
    ATOM      2   CA MET     7      73.841 -17.042   9.747  0.00  0.00             C
    ATOM      3    C MET     7      74.361 -18.178  10.643  0.00  0.00             C
    ATOM      4    O MET     7      73.642 -18.708  11.489  0.00  0.00             O

    Write structure to disk

    >>> structure.write('test_out.pdb')

    Radius of gyration

    >>> structure.radius_gyration
    50.31556120901161

    """

    def __init__(self, p_object, **kwargs):
        mfm.curve.ExperimentalData.__init__(self, **kwargs)
        self.auto_update = kwargs.get('auto_update', False)
        self._filename = None
        self._atoms = None
        self._residue_dict = None
        self._potentials = list()
        self._sequence = None

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
        rM = coord[:, :].mean(axis=0)
        rG = (np.sqrt((coord - rM) ** 2).sum(axis=1)).mean()
        return float(rG)

    def append_potential(self, function, kwargs=dict()):
        """

        :param function:
        :param kwargs:
        :return:

        Examples
        --------

        >>> import mfm
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
        filename = kwargs.get('filename', self.filename)
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

from .structure import *
from .protein import *
from .trajectory import *
from .labeled_structure import *
from . import potential
