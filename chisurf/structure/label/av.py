from __future__ import annotations
from scikit_fluorescence import typing

import warnings

import numba as nb
import numpy as np
import LabelLib as ll
import IMP.core
import IMP.atom

import scikit_fluorescence as skf
import scikit_fluorescence.io
import scikit_fluorescence.math
import scikit_fluorescence.decay
import chisurf.structure.label
import chisurf.structure.label.distribution
from . import distribution


class LabelDistributionAV(distribution.LabelDistribution):
    """Simulates the accessible volume of a label

    Attributes
    ----------
    _av : LabelLib.Grid3D
        The X coordinate
    _density : numpy.array
        The Y coordinate
    origin : numpy.array
        For accessible volumes typically the attachment position of the label.
    verbose : bool
        The Y coordinate
    _atoms : numpy.array
        The Y coordinate

    Examples
    --------
    Create AV from coordinate files

    >>> import scikit_fluorescence as skf
    >>> import scikit_fluorescence.modeling
    >>> import scikit_fluorescence.io
    >>> import scikit_fluorescence.data
    >>> pdb_filename = skf.data.get_pdb_filename()
    >>> atoms = skf.io.structure.read_coordinates(pdb_filename)
    >>> av1 = skf.modeling.label.av.LabelDistributionAV(atoms, residue_seq_number=18, chain='A', atom_name='CB')
    >>> av2 = skf.modeling.label.av.LabelDistributionAV(atoms, residue_seq_number=577, chain='A', atom_name='CB')
    >>> av1.average_distance(av2)
    59.7337863075394

    Create AV from IMP Hierarchy
    >>> import scikit_fluorescence as skf
    >>> import scikit_fluorescence.modeling
    >>> import IMP.atom
    >>> import scikit_fluorescence.data
    >>> pdb_filename = skf.data.get_pdb_filename()
    >>> model = IMP.Model()
    >>> mp = IMP.atom.read_pdb(pdb_filename, model, IMP.atom.NonWaterPDBSelector())
    >>> av1 = skf.modeling.label.av.LabelDistributionAV(atoms=None, hierarchy=mp, residue_seq_number=18, atom_name='CB')
    >>> av2 = skf.modeling.label.av.LabelDistributionAV(atoms=None, hierarchy=mp, residue_seq_number=577, atom_name='CB')
    >>> av1.average_distance(av2)
    59.74012554144173

    """

    @staticmethod
    @nb.jit(nopython=True)
    def __atoms_in_reach(
            xyz: np.ndarray,
            vdw: np.ndarray,
            dmaxsq: float,
            atom_i: int,
            return_subset: bool = True
    ) -> (np.ndarray, np.ndarray, typing.List[int]):
        """Finds atoms that are within a squared distance of an atom id

        :param xyz: coordinates of atoms
        :param vdw: van der Waals radii of atoms
        :param dmaxsq: maximum squared distance
        :param atom_i: search atom idx
        :param return_subset: if set to True the coordinates and vdw radii are returned
        :return:

        """
        # copy all atoms in proximity to the label into a smaller array and
        # move coordinate frame to attachment point
        n_atoms = xyz.shape[0]
        atomindex = np.empty(n_atoms, dtype=np.uint32)
        r0 = xyz[atom_i]
        natomsgrid = 0
        for i in range(0, n_atoms):
            dsq = ((xyz[i] - r0) ** 2.0).sum()
            if dsq < dmaxsq:
                atomindex[natomsgrid] = i
                natomsgrid += 1
        if not return_subset:
            natomsgrid = 0
        re_idx = np.empty(natomsgrid, dtype=np.uint32)
        ra = np.empty((natomsgrid, 3), dtype=np.float64)
        vdwr = np.empty(natomsgrid, dtype=np.float64)
        for i in range(natomsgrid):
            n = atomindex[i]
            re_idx[i] = n
            ra[i] = xyz[n]
            vdwr[i] = vdw[n]
        return ra, vdwr, re_idx

    _av: ll.Grid3D = None
    _density: np.ndarray = None
    _atoms: np.ndarray = None
    simulation_grid_resolution: float

    def __init__(
            self,
            atoms: np.ndarray,
            simulation_grid_resolution: float = 0.5,
            allowed_sphere_radius: float = 1.5,
            radius1: float = 3.5,
            radius2: float = 4.5,
            radius3: float = 1.5,
            linker_width: float = 0.5,
            linker_length: float = 20.5,
            simulation_type: str = "AV1",
            hierarchy: IMP.core.Hierarchy = None,
            origin: np.ndarray = None,
            chain_identifier: str = None,
            residue_name: str = None,
            residue_seq_number: int = None,
            atom_name: str = None,
            position_name: str = None,
            contact_volume_thickness: float = None,
            contact_volume_trapped_fraction: float = None,
            min_sphere_volume_fraction: float = None,
            strip_mask: str = None,
            chain_weighting: bool = None,
            vdw_max: float = 2.5,
            **kwargs
    ):
        """Initialize a LabelDistributionAV from atom coordinates and AV params."""
        super(LabelDistributionAV, self).__init__(
            simulation_type=simulation_type,
            origin=origin,
            simulation_grid_resolution=simulation_grid_resolution,
            position_name=position_name,
            **kwargs
        )
        # Raise warnings for optional options that are not implemented
        if isinstance(contact_volume_thickness, float):
            warnings.warn("Contact_volume_thickness is not implemented")
        if isinstance(contact_volume_trapped_fraction, float):
            warnings.warn("contact_volume_trapped_fraction is not implemented")
        if isinstance(contact_volume_trapped_fraction, float):
            warnings.warn("contact_volume_trapped_fraction is not implemented")
        if isinstance(min_sphere_volume_fraction, float):
            warnings.warn("min_sphere_volume_fraction is not implemented")
        if isinstance(strip_mask, str):
            warnings.warn("strip_mask is not implemented")
        if isinstance(chain_weighting, bool):
            warnings.warn("chain_weighting is not implemented")
        if isinstance(hierarchy, IMP.core.Hierarchy):
            atoms = scikit_fluorescence.io.structure.convert_atoms(
                IMP.atom.get_leaves(hierarchy)
            )

        self.allowed_sphere_radius = max(allowed_sphere_radius, linker_width / 2)
        self.residue_name = residue_name
        self.attachment_residue = residue_seq_number
        self.attachment_atom = atom_name

        self.radius1 = radius1
        self.radius2 = radius2
        self.radius3 = radius3
        self.linker_width = linker_width
        self.linker_length = linker_length

        x, y, z = atoms['xyz'][:, 0], atoms['xyz'][:, 1], atoms['xyz'][:, 2]
        r = np.copy(atoms['radius'])
        if origin is None:
            attachment_atom_index = kwargs.get(
                'attachment_atom_index',
                skf.io.structure.find_atom_index(
                    atoms,
                    chain_identifier,
                    self.attachment_residue,
                    self.attachment_atom,
                    self.residue_name,
                    verbose=self.verbose
                )
            )
            # set attachment atom radius to zero otherwise the
            # search cannot start and yields an empty AV.
            r[attachment_atom_index] = 0.0
            # set radii of all obstacles in allow sphere radius to zero to
            # make a bit more room for the label.
            if self.allowed_sphere_radius > 0:
                xyz_a, vdw_a, indices = self.__atoms_in_reach(
                    xyz=atoms['xyz'],
                    vdw=atoms['radius'],
                    dmaxsq=(self.allowed_sphere_radius + vdw_max) ** 2,
                    atom_i=attachment_atom_index
                )
                if self.verbose:
                    print("Masking atoms: ", indices)
            else:
                indices = []
            r[indices] = 0.0
            origin = np.array(
                [
                    x[attachment_atom_index],
                    y[attachment_atom_index],
                    z[attachment_atom_index]
                ]
            )

        xyzr = np.vstack([x, y, z, r])
        if self.verbose:
            print("Computing AV")
            print("-- allowed_sphere_radius: %s" % self.allowed_sphere_radius)
            print("-- simulation_type: %s" % simulation_type)
            print("-- dye_attachment_point: %s" % origin)
            print("-- linker_length: %s" % linker_length)
            print("-- linker_width: %s" % linker_width)
            print("-- radius1: %s" % radius1)
            print("-- radius2: %s" % radius2)
            print("-- radius3: %s" % radius3)
            print("-- simulation_grid_resolution: %s" % simulation_grid_resolution)
        if self.simulation_type == 'AV3':
            av = ll.dyeDensityAV3(
                xyzr,
                origin,
                linker_length,
                linker_width / 2,
                [radius1, radius2, radius3],
                simulation_grid_resolution
            )
        else:
            av = ll.dyeDensityAV1(
                xyzr,
                origin,
                linker_length,
                linker_width / 2,
                radius1,
                simulation_grid_resolution
            )
        self._atoms = atoms
        self._density = np.array(av.grid, dtype=np.float).reshape(av.shape)
        self._av = av

    @property
    def ng(self):
        """Grid resolution (number of grid points in one dimension)."""
        return self.density.shape[0]

    @property
    def density(self):
        """3D density array of the accessible volume."""
        return self._density

    @property
    def points(self):
        """Sampled point cloud of the accessible volume."""
        return self._av.points().T

    @property
    def mean_position(
            self
    ) -> np.ndarray:
        """
        The mean position of the accessible volume (average x, y, z coordinate)

        Examples
        --------
        >>> import scikit_fluorescence as skf
        >>> import scikit_fluorescence.io
        >>> import scikit_fluorescence.modeling
        >>> atoms = skf.io.structure.read_coordinates('./test/data/1fat.cif')
        >>> av1 = skf.modeling.label.av.LabelDistributionAV(atoms, residue_seq_number=18, atom_name='CB')
        """
        weights = self.points[:, 3]
        weights /= weights.sum()
        xyz = self.points[:, [0, 1, 2]]
        return np.average(xyz, weights=weights, axis=0)
