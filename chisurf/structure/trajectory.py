from __future__ import annotations
from chisurf import typing

import copy
import os
import tempfile

import mdtraj
import numba as nb
import numpy as np

import chisurf.base
import chisurf.structure


class Universe(object):

    def __init__(
            self,
            structure: chisurf.structure.Structure = None
    ):
        self.structures = [] if structure is None else [structure]
        self.potentials = list()
        self.scaling = list()

    def addPotential(
            self,
            potential,
            scale: float = 1.0
    ) -> None:
        self.potentials.append(potential)
        self.scaling.append(scale)

    def removePotential(
            self,
            potentialNbr: int = None
    ) -> None:
        if potentialNbr == -1:
            self.potentials.pop()
            self.scaling.pop()
        else:
            self.potentials.pop(potentialNbr)
            self.scaling.pop(potentialNbr)

    def clearPotentials(self) -> None:
        self.potentials = list()
        self.scaling = list()

    def getEnergy(
            self,
            structure: chisurf.structure.Structure = None
    ) -> float:
        for p in self.potentials:
            p.structure = structure
        Es = self.getEnergies()
        E = Es.sum()
        return E

    def getEnergies(
            self,
            structure: chisurf.structure.Structure = None
    ) -> np.ndarray:
        for p in self.potentials:
            p.structure = structure
        scales = np.array(self.scaling)
        Es = np.array([pot.getEnergy() for pot in self.potentials])
        return scales * Es


class TrajectoryFile(
    mdtraj.Trajectory,
    chisurf.base.Base
):

    """
    Creates an Trajectory of mfm.structure.Structures given a HDF5-File using
    mdtraj.HDF5TrajectoryFile

    Parameters
    ----------

    structure : string / mfm.structure.Structure
        determines the topology
        is either a string containing the filename of a PDB-File or an instance
        of mfm.structure.Structure() Obligatory in write reading_routine, not
        needed in reading reading_routine

    filename_hdf : string
        the filename of the HDF5-file

    Other Parameters
    ----------------
    verbose : bool

    stride : int, default=None
        Only read every stride-th frame.

    frame : integer or None
        If frame is an integer only the frame number provided by the integer is
        loaded otherwise the whole trajectory is loaded.

    See Also
    --------

    mfm.mfm.structure.Structure

    Examples
    --------

    Making new h5-Trajectory file

    >>> import chisurf.structure
    >>> from chisurf.structure import TrajectoryFile
    >>> s = chisurf.structure.Structure('./test/data/modelling/trajectory/h5-file/T4L_Topology.pdb', verbose=True, make_coarse=False)
    >>> traj = chisurf.structure.TrajectoryFile('./test/data/modelling/trajectory/h5-file/hgbp1_transition.h5', s, reading_routine='w')
    >>> traj[0]
    <mfm.structure.structure.mfm.structure.Structure at 0x11f34e10>
    >>> print(traj[0])
    ATOM      1    N MET     1     -10.750  14.401  -5.002  0.00  0.00             N
    ATOM      2   H1 MET     1     -11.310  14.080  -4.226  0.00  0.00             H
    ATOM      3   H2 MET     1     -11.333  14.344  -5.825  0.00  0.00             H
    ATOM      4   H3 MET     1     -10.387  15.316  -4.776  0.00  0.00             H
    ATOM      5   CA MET     1      -9.555  13.486  -5.166  0.00  0.00             C
    ....

    Opening h5-Trajectory file

    >>> import chisurf.structure
    >>> from chisurf.structure import TrajectoryFile
    >>> traj = TrajectoryFile('./test/data/modelling/trajectory/h5-file/hgbp1_transition.h5', reading_routine='r', stride=1)
    >>> print(traj[0:3])
    [<mfm.structure.structure.mfm.structure.Structure at 0x1345d5d0>,
    <mfm.structure.structure.mfm.structure.Structure at 0x1345d610>,
    <mfm.structure.structure.mfm.structure.Structure at 0x132d2230>]

    Name of the trajectory

    >>> print(traj.name)
    '/ data/ structure/ data/ structure/ T4L_Trajectory.h5'

    initialize with mdtraj.Trajectory

    >>> import chisurf.structure
    >>> from chisurf.structure import TrajectoryFile
    >>> traj = TrajectoryFile('./test/data/modelling/trajectory/h5-file/hgbp1_transition.h5', reading_routine='r', stride=1)
    >>> t2 = TrajectoryFile(traj.mdtraj, filename='test.h5')

    Attributes:
    -----------
    rmsd : array/list containing the rmsd vs the reference structure of -new- / added structures upon addition
    of the strucutre

    """

    parameterNames = [
        'rmsd',
        'drmsd',
        'energy',
        'chi2'
    ]

    def __init__(
            self,
            p_object,
            filename: str = None,
            rmsd_ref_state: int = 0,
            stride: int = 1,
            inverse_trajectory: bool = False,
            center: bool = False,
            verbose: bool = False,
            atom_indices: typing.List[int] = None,
            mode: str = 'r'
    ):
        """

        :param p_object: either a string pointing to an .hdf or .pdb file; a mdtraj.Trajectory object;
            or a chisurf.structure.Structure object;
        :param filename:
        :param rmsd_ref_state:
        :param stride:
        :param inverse_trajectory:
        :param center:
        :param verbose:
        :param atom_indices:
        :param mode:
        :param args:
        :param kwargs:
        """
        self.mode = mode
        self.atom_indices = atom_indices
        self.stride = stride
        self._rmsd_ref_state = rmsd_ref_state
        self.verbose = verbose
        self.center = center
        self._invert = inverse_trajectory
        self._structure = None

        _, h5_tmp = tempfile.mkstemp(
            suffix=".h5"
        )
        _, pdb_tmp = tempfile.mkstemp(
            suffix=".pdb"
        )
        if filename is None:
            self._filename = h5_tmp
        else:
            self._filename = filename

        if isinstance(
                p_object, str
        ):
            if p_object.endswith('.pdb'):
                structure = chisurf.structure.Structure(p_object)
                self._mdtraj = mdtraj.Trajectory.load(p_object)
                self._mdtraj.save_hdf5(self._filename)
            elif p_object.endswith('.h5'):
                self._filename = p_object
                self._mdtraj = mdtraj.Trajectory.load(
                    p_object,
                    stride=self.stride
                )
                self._mdtraj[0].save_pdb(pdb_tmp)
                structure = chisurf.structure.Structure(pdb_tmp)
            else:
                structure = chisurf.structure.Structure(pdb_tmp)
        elif isinstance(
                p_object,
                mdtraj.Trajectory
        ):
            self._mdtraj = p_object
            self._mdtraj[0].save_pdb(pdb_tmp)
            structure = chisurf.structure.Structure(pdb_tmp)
        elif isinstance(
                p_object,
                chisurf.structure.Structure
        ):
            p_object.write(pdb_tmp)
            self._mdtraj = mdtraj.Trajectory.load(pdb_tmp)
            self._mdtraj.save_hdf5(filename=self._filename)
            structure = p_object

        self.structure = structure
        super().__init__(
            xyz=self._mdtraj.xyz,
            topology=self._mdtraj.topology
        )

        if self.center:
            self._mdtraj.center_coordinates()

        self.rmsd_ref_state = rmsd_ref_state
        self.rmsd = list()
        self.drmsd = list()
        self.energy = list()
        self.chi2r = list()
        self.offset = 0

    def clear(self):
        self.rmsd = list()
        self.drmsd = list()
        self.energy = list()
        self.chi2r = list()
        self.rmsd_ref_state = 0

    @property
    def xyz(self):
        """Cartesian coordinates of each atom in each simulation frame

        If the attribute :py:attribute:`.TrajectoryFile.invert` is True the 
        oder of the trajectory is inverted
        """
        if self.invert:
            return self._mdtraj.xyz[::-1]
        else:
            return self._mdtraj.xyz

    @xyz.setter
    def xyz(self, v):
        self._xyz = v

    @property
    def structure(
            self
    ) -> chisurf.structure.Structure:
        return self._structure

    @structure.setter
    def structure(
            self,
            v: chisurf.structure.Structure
    ):
        self._structure = copy.copy(v)

    @property
    def invert(
            self
    ) -> bool:
        """If True the oder of the trajectory is inverted (by default False)
        """
        return self._invert

    @invert.setter
    def invert(
            self,
            v: bool
    ):
        self._invert = bool(v)

    @property
    def filename(
            self
    ) -> str:
        """The filename of the trajectory
        """
        return self._filename

    @filename.setter
    def filename(
            self,
            v: str
    ):
        if isinstance(v, str):
            self._filename = v
            mdtraj.Trajectory.save(self, filename=v)

    @property
    def mdtraj(
            self
    ) -> mdtraj.Trajectory:
        return self._mdtraj

    @property
    def name(
            self
    ) -> str:
        """The name of the trajectory composed of the directory and the filename
        """
        try:
            fn = copy.copy(self.directory + self.filename)
            return fn.replace('/', '/ ')
        except AttributeError:
            return "None"

    @property
    def rmsd_ref_state(
            self
    ) -> int:
        """The index (frame number) of the reference state used for the RMSD
        calculation
        """
        return self._rmsd_ref_state

    @rmsd_ref_state.setter
    def rmsd_ref_state(
            self,
            ref_frame: int
    ):
        self._rmsd_ref_state = ref_frame
        self.rmsd = mdtraj.rmsd(self, self, ref_frame)

    @property
    def directory(
            self
    ) -> str:
        """Directory in which the filename of the trajectory is located in
        """
        return os.path.dirname(self.filename)

    @property
    def reference(
            self
    ) -> chisurf.structure.Structure:
        """The reference structure used for RMSD-calculation. This cannot be
        set directly but has to be set via the number of the reference state
         :py:attribute`.rmsd_ref_state`
        """
        if self.rmsd_ref_state == 'average':
            return self.average
        else:
            return self[int(self.rmsd_ref_state)]

    @property
    def average(
            self
    ) -> chisurf.structure.Structure:
        """
        The average structure (:py:class:`~mfm.structure.mfm.structure.Structure`)
        of the trajectory
        """
        return chisurf.structure.average(self[:len(self)])

    @property
    def values(
            self
    ) -> np.array:
        """A 2D-numpy array containing the RMSD, dRMSD, energy and the chi2
        values of the trajectory

        Examples
        --------

        >>> import chisurf.settings as mfm
        >>> from chisurf.structure import TrajectoryFile
        >>> traj = TrajectoryFile('./test/data/structure/2807_8_9_b.h5', reading_routine='r', stride=1)
        >>> traj
        <mdtraj.Trajectory with 92 frames, 2495 atoms, 164 residues, without unitcells at 0x117f3b70>
        >>> traj.values
        array([], shape=(4, 0), dtype=float64)
        >>> traj.append(times[0])
        inf     inf     0.0000  0.6678
        >>> traj.append(times[0])
        inf     inf     0.0000  0.0000
        >>> traj.values
        array([ [  5.96507968e-09,   5.96508192e-09],
                [  6.67834069e-01,   5.96507944e-09],
                [             inf,              inf],
                [             inf,              inf]])
        """
        rmsd = np.array(self.rmsd)
        drmsd = np.array(self.drmsd)
        energy = np.array(self.energy)
        chi2 = np.array(self.chi2r)
        return np.vstack([rmsd, drmsd, energy, chi2])

    def append(
            self,
            xyz,
            update_rmsd: bool = True,
            energy: float = np.inf,
            energy_fret: float = np.inf,
            verbose: bool = False
    ):
        """Append a structure of type :py::class`mfm.mfm.structure.Structure`
        to the trajectory

        :param structure: mfm.structure.Structure
        :param update_rmsd: bool
        :param energy: float
            Energy of the system
        :param energy_fret: float
            Energy of the FRET-potential
        :param verbose: bool
            By default True. If True energy, energy_fret, RMSD and dRMSD are
            printed to std-out.

        Examples
        --------

        >>> import chisurf.settings as mfm
        >>> from chisurf.structure import TrajectoryFile
        >>> traj = TrajectoryFile('./test/data/structure/2807_8_9_b.h5', reading_routine='r', stride=1)
        >>> traj
        <mdtraj.Trajectory with 92 frames, 2495 atoms, 164 residues, without unitcells at 0x11762b70>
        >>> t.append(traj[0])
        <mdtraj.Trajectory with 93 frames, 2495 atoms, 164 residues, without unitcells at 0x11762b70>
        """
        verbose = verbose or self.verbose
        if isinstance(xyz, chisurf.structure.Structure):
            xyz = xyz.xyz

        xyz = xyz.reshape((1, xyz.shape[0], 3)) / 10.0
        # write to trajectory file
        mode = 'a' if os.path.isfile(self.filename) else 'w'
        t = mdtraj.formats.hdf5.HDF5TrajectoryFile(self.filename, mode=mode)
        t.write(xyz, time=len(t))
        t.close()

        if update_rmsd:
            self._xyz = np.append(self._xyz, xyz, axis=0)
            self._time = np.arange(len(self._xyz))

            self.mdtraj._xyz = self._xyz
            self.mdtraj._time = self.time
            new = self.mdtraj[-1]
            previous = self.mdtraj[-2]
            next_drmsd = mdtraj.rmsd(new, previous) * 10.0
            next_rmsd = mdtraj.rmsd(new, self.mdtraj[self.rmsd_ref_state]) * 10.0
        else:
            next_drmsd = [0.0]
            next_rmsd = [0.0]
        self.drmsd.append(float(next_drmsd[0]))
        self.rmsd.append(float(next_rmsd[0]))
        self.energy.append(energy)
        self.chi2r.append(energy_fret)
        if verbose:
            print("%.3f\t%.3f\t%.4f\t%.4f" % (energy, energy_fret, next_rmsd[0], next_drmsd[0]))

    def __iter__(self):
        """
        Implements iterator
        >>> import chisurf.structure
        >>> from chisurf.structure import TrajectoryFile
        >>> traj = TrajectoryFile('./test/data/structure/2807_8_9_b.h5', reading_routine='r', stride=1)
        >>> for s in traj:
        >>>     print(s)
        [<mfm.structure.structure.mfm.structure.Structure object at 0x12FAE330>, <mfm.structure.structure.mfm.structure.Structure object at 0x12FAE3B0>, <li
        b.structure.mfm.structure.Structure.mfm.structure.Structure object at 0x11852070>, <mfm.structure.structure.mfm.structure.Structure object at 0x131052D0>, <mfm.st
        ructure.mfm.structure.Structure.mfm.structure.Structure object at 0x13195270>, <mfm.structure.structure.mfm.structure.Structure object at 0x13228210>]
        """
        for i in range(len(self)):
            yield self[i]

    def __next__(self):
        """
        Iterate trough the trajectory. The current frame is stored in the
        trajectory property ``offset``

        Returns
        -------
        next : mfm.structure.Structure
            Returns the next structure in the trajectory

        Example
        -------

        >>> import chisurf.structure
        >>> traj = chisurf.structure.TrajectoryFile('./test/data/modelling/trajectory/h5-file/hgbp1_transition.h5', reading_routine='r', stride=1)
        >>> s = str(traj.next())
        >>> print(s[:500])
        ATOM      1    N MET A   1       7.332 -10.706 -15.034  0.00  0.00             N
        ATOM      2    H MET A   1       7.280 -10.088 -15.830  0.00  0.00             H
        ATOM      3   H2 MET A   1       7.007 -11.615 -15.330  0.00  0.00             H
        ATOM      4   H3 MET A   1       8.267 -10.697 -14.653  0.00  0.00             H
        ATOM      5   CA MET A   1       6.341 -10.257 -14.033  0.00  0.00             C
        ATOM      6   HA MET A   1       5.441  -9.927 -14.551  0.00  0.00             H
        >>> s = str(traj.next())
        >>> print(s[:500])
        ATOM      1    N MET A   1      12.234  -5.443 -11.675  0.00  0.00             N
        ATOM      2    H MET A   1      12.560  -5.462 -10.719  0.00  0.00             H
        ATOM      3   H2 MET A   1      12.359  -4.507 -12.036  0.00  0.00             H
        ATOM      4   H3 MET A   1      12.767  -6.064 -12.265  0.00  0.00             H
        ATOM      5   CA MET A   1      10.824  -5.798 -11.763  0.00  0.00             C
        ATOM      6   HA MET A   1      10.490  -5.577 -12.777  0.00  0.00             H
        ATOM      7
        """
        if self.offset == len(self):
            raise StopIteration
        element = self[self.offset]
        self.offset += 1
        return element

    def slice(self, key, copy=True):
        s = self.mdtraj.slice(key, copy=True)
        return TrajectoryFile(p_object=s)

    def __getitem__(self, key):
        # TODO: do sth. about the evaluation speed (maybe lazy evaluation)
        # http://code.activestate.com/recipes/576410-lazy-lists/
        if isinstance(key, int):
            s = copy.copy(self.structure)
            s.xyz = self.mdtraj[key].xyz * 10.0
            s.update()
            return s

        elif isinstance(key, slice):
            start, stop, step = key.start, key.stop, key.step
            # set start and step to their integer defaults if they are None.
            if start is None:
                start = 0
            if step is None:
                step = 1

            def make_structure(i):
                s = copy.copy(self.structure)
                s._filename = self.structure.labeling_file
                s.xyz = self.mdtraj[i].xyz * 10.0
                s.update()
                return s

            # make a generator instead of a list
            return [make_structure(i) for i in range(start, stop, step)]


@nb.jit
def translate(
        xyz: np.ndarray,
        vector: np.ndarray
) -> None:
    """ Translate a trajectory by an vector

    :param xyz: numpy array
        (frame fit_index, atom_number, coord)
    :param vector:
    :return:
    """
    n_frames = xyz.shape[0]
    n_atoms = xyz.shape[1]

    for i_frame in range(n_frames):
        for i_atom in range(n_atoms):
            for i_dim in range(3):
                xyz[i_frame, i_atom, i_dim] += vector[i_dim]


@nb.jit
def rotate(
        xyz: np.ndarray,
        rm: np.ndarray
) -> None:
    """ Rotates a trajectory (frame, atom, coord)

    :param xyz: numpy array
        The coordinates (frame fit_index, atom fit_index, coord)

    :param rm: numpy array 3x3 dtpye np.float32 - the rotation matrix
    :return:

    Examples
    --------

    >>> from chisurf.structure.trajectory import TrajectoryFile
    >>> import numpy as np
    >>> traj = TrajectoryFile('stride_100.h5')
    >>> xyz = traj.xyz
    >>> b = np.array([[-0.856274009, 0.513258278, -0.057972118], [0.513934493, 0.835381866, -0.194957629], [-0.051634759, -0.196731016, -0.979096889]], dtype=np.float32)
    >>> rotate(xyz, b)

    """
    n_frames = xyz.shape[0]
    n_atoms = xyz.shape[1]
    for i_frame in range(n_frames):
        for i_atom in range(n_atoms):
            # matrix vector product
            x = xyz[i_frame, i_atom, 0]
            y = xyz[i_frame, i_atom, 1]
            z = xyz[i_frame, i_atom, 2]

            t1 = rm[0, 0] * x + rm[0, 1] * y + rm[0, 2] * z
            t2 = rm[1, 0] * x + rm[1, 1] * y + rm[1, 2] * z
            t3 = rm[2, 0] * x + rm[2, 1] * y + rm[2, 2] * z

            xyz[i_frame, i_atom, 0] = t1
            xyz[i_frame, i_atom, 1] = t2
            xyz[i_frame, i_atom, 2] = t3