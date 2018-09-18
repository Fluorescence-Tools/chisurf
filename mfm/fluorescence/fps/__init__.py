import os
from collections import OrderedDict
import json
import scipy.stats

import mfm
import numpy as np
from PyQt4 import QtCore, QtGui, uic
from mfm.fluorescence.fps import functions
from .static import calculate_1_radius, calculate_3_radius
from . import functions
from .functions import assign_diffusion_to_grid_1, assign_diffusion_to_grid_2, get_kQ_rC, create_quenching_map
from mfm.structure import Structure
import static
import dynamic


package_directory = os.path.dirname(__file__)
dye_file = os.path.join(mfm.package_directory, 'settings/dye_definition.json')
try:
    dye_definition = json.load(open(dye_file))
except IOError:
    dye_definition = dict()
    dye_definition['a'] = 0
dye_names = dye_definition.keys()


class BasicAV(mfm.Base):
    """Simulates the accessible volume of a dye

    Examples
    --------

    >>> import mfm
    >>> structure = mfm.structure.Structure('./sample_data/modelling/pdb_files/hGBP1_closed.pdb')
    >>> av = mfm.fluorescence.fps.BasicAV(structure, residue_seq_number=18, atom_name='CB')

    """

    def __init__(self, structure, *args, **kwargs):
        mfm.Base.__init__(self, *args, **kwargs)

        self.dg = kwargs.get('simulation_grid_resolution', mfm.settings['fps']['simulation_grid_resolution'])
        self.allowed_sphere_radius = kwargs.get('allowed_sphere_radius', mfm.settings['fps']['allowed_sphere_radius'])
        self.position_name = kwargs.get('position_name', None)
        self.residue_name = kwargs.get('residue_name', None)
        self.attachment_residue = kwargs.get('residue_seq_number', None)
        self.attachment_atom = kwargs.get('atom_name', None)

        chain_identifier = kwargs.get('chain_identifier', None)
        self.radius1 = kwargs.get('radius1', 1.5)
        self.radius2 = kwargs.get('radius2', 4.5)
        self.radius3 = kwargs.get('radius3', 3.5)
        self.linker_width = kwargs.get('linker_width', 1.5)

        self.linker_length = kwargs.get('linker_length', 20.5)
        self.simulation_type = kwargs.get('simulation_type', 'AV1')

        if isinstance(structure, Structure):
            self.structure = structure
        elif isinstance(structure, str):
            self.structure = Structure(structure, **kwargs)

        attachment_atom_index = kwargs.get('attachment_atom_index',
                                           mfm.io.pdb.get_atom_index(self.atoms,
                                                                     chain_identifier,
                                                                     self.attachment_residue,
                                                                     self.attachment_atom,
                                                                     self.residue_name)
                                           )

        x, y, z = self.atoms['coord'][:, 0], self.atoms['coord'][:, 1], self.atoms['coord'][:, 2]
        vdw = self.atoms['radius']

        if self.simulation_type == 'AV3':
            density, ng, x0 = calculate_3_radius(self.linker_length,
                                                 self.linker_width,
                                                 self.radius1,
                                                 self.radius2,
                                                 self.radius3,
                                                 attachment_atom_index,
                                                 x, y, z, vdw,
                                                 linkersphere=self.allowed_sphere_radius,
                                                 dg=self.dg, **kwargs)
        else:
            density, ng, x0 = calculate_1_radius(self.linker_length,
                                                 self.linker_width,
                                                 self.radius1,
                                                 attachment_atom_index,
                                                 x, y, z, vdw,
                                                 linkersphere=self.allowed_sphere_radius,
                                                 dg=self.dg, **kwargs)

        self.x0 = x0
        self._bounds = density.astype(dtype=np.uint8)
        density /= density.sum()
        self._density = density
        self._points = None

    @property
    def bounds(self):
        return self._bounds

    @property
    def ng(self):
        return self.density.shape[0]

    @property
    def density(self):
        return self._density

    @property
    def points(self):
        if self._points is None:
            self.update_points()
        return self._points

    @property
    def atoms(self):
        return self.structure.atoms

    def update_points(self):
        ng = self.ng
        density = self.density
        x0, dg = self.x0, self.dg
        self._points = functions.density2points(ng, dg, density, x0)

    def update(self):
        self.update_points()

    def save(self, filename, mode='xyz', **kwargs):
        """Saves the accessible volume as xyz-file or open-dx density file

        Examples
        --------

        >>> import mfm
        >>> structure = mfm.structure.Structure('./sample_data/modelling/pdb_files/hGBP1_closed.pdb')
        >>> av = mfm.fluorescence.fps.BasicAV(structure, residue_seq_number=18, atom_name='CB')
        >>> av.save('c:/temp/test', mode='xyz')
        >>> av.save('c:/temp/test', mode='dx')

        """
        if mode == 'dx':
            density = kwargs.get('density', self.density)
            d = density / density.max() * 0.5
            ng = self.ng
            dg = self.dg
            offset = (ng - 1) / 2 * dg
            mfm.io.pdb.write_open_dx(filename,
                                     d,
                                     self.x0 - offset,
                                     ng, ng, ng,
                                     dg, dg, dg
                                     )
        else:
            p = kwargs.get('points', self.points)
            d = p[:, [3]].flatten()
            d /= max(d) * 50.0
            xyz = p[:, [0, 1, 2]]
            mfm.io.pdb.write_points(filename=filename + '.'+mode,
                                    points=xyz,
                                    mode=mode, verbose=self.verbose,
                                    density=d)

    def dRmp(self, av):
        """
        Calculate the distance between the mean positions with respect to the accessible volume `av`

        :param av: accessible volume object
        :return:

        Examples
        --------

        >>> import mfm
        >>> structure = mfm.structure.Structure('./sample_data/modelling/pdb_files/hGBP1_closed.pdb')
        >>> av1 = mfm.fluorescence.fps.BasicAV(structure, residue_seq_number=18, atom_name='CB')
        >>> av2 = mfm.fluorescence.fps.BasicAV(structure, residue_seq_number=577, atom_name='CB')
        >>> av1.dRmp(av2)
        """
        return functions.dRmp(self, av)

    def dRDA(self, av, **kwargs):
        """Calculate the mean distance to the second accessible volume

        :param av:
        :return:

        Examples
        --------

        >>> import mfm
        >>> structure = mfm.structure.Structure('./sample_data/modelling/pdb_files/hGBP1_closed.pdb')
        >>> av1 = mfm.fluorescence.fps.BasicAV(structure, residue_seq_number=18, atom_name='CB')
        >>> av2 = mfm.fluorescence.fps.BasicAV(structure, residue_seq_number=577, atom_name='CB')
        >>> av1.dRDA(av2)
        """
        return functions.RDAMean(self, av, **kwargs)

    def widthRDA(self, av):
        """Calculates the width of a DA-distance distribution

        :param av:
        :return:

        Examples
        --------

        >>> import mfm
        >>> structure = mfm.structure.Structure('./sample_data/modelling/pdb_files/hGBP1_closed.pdb')
        >>> av1 = mfm.fluorescence.fps.BasicAV(structure, residue_seq_number=18, atom_name='CB')
        >>> av2 = mfm.fluorescence.fps.BasicAV(structure, residue_seq_number=577, atom_name='CB')
        >>> av1.widthRDA(av2)

        """
        return functions.widthRDA(self, av)

    def dRDAE(self, av):
        """Calculate the FRET-averaged mean distance to the second accessible volume

        :param av: Accessible volume
        :return:

        Examples
        --------

        >>> import mfm
        >>> structure = mfm.structure.Structure('./sample_data/modelling/pdb_files/hGBP1_closed.pdb')
        >>> av1 = mfm.fluorescence.fps.BasicAV(structure, residue_seq_number=18, atom_name='CB')
        >>> av2 = mfm.fluorescence.fps.BasicAV(structure, residue_seq_number=577, atom_name='CB')
        >>> av1.dRDAE(av2)
        """
        return functions.RDAMeanE(self, av)

    def pRDA(self, av, **kwargs):
        """Calculates the distance distribution with respect to a second accessible volume and returns the
        distance axis and the probability of the respective distance. By default the distance-axis "mfm.rda_axis"
        is taken to generate the histogram.

        :param av: Accessible volume
        :param kwargs:
        :return:

        Examples
        --------

        >>> import mfm
        >>> structure = mfm.structure.Structure('./sample_data/modelling/pdb_files/hGBP1_closed.pdb')
        >>> av1 = mfm.fluorescence.fps.BasicAV(structure, residue_seq_number=18, atom_name='CB')
        >>> av2 = mfm.fluorescence.fps.BasicAV(structure, residue_seq_number=577, atom_name='CB')
        >>> y, x = av1.pRDA(av2)

        """
        return functions.histogram_rda(self, av, **kwargs)

    @property
    def Rmp(self):
        """
        The mean position of the accessible volume (average x, y, z coordinate)
        """
        weights = self.points[:, 3]
        weights /= weights.sum()
        xyz = self.points[:, [0, 1, 2]]
        return np.average(xyz, weights=weights, axis=0)


class ACV(BasicAV):
    """
    Example
    -------

    >>> import mfm
    >>> structure = mfm.structure.Structure('./sample_data/modelling/pdb_files/hGBP1_closed.pdb')
    >>> trapped_fraction = 0.5
    >>> av1 = mfm.fluorescence.fps.ACV(structure, residue_seq_number=18, atom_name='CB', contact_volume_trapped_fraction=trapped_fraction)
    >>> av2 = mfm.fluorescence.fps.ACV(structure, residue_seq_number=577, atom_name='CB', contact_volume_trapped_fraction=trapped_fraction)
    >>> av1.save('c:/temp/test_05', mode='dx')
    >>> y1, x1 = av1.pRDA(av2)

    >>> import mfm
    >>> structure = mfm.structure.Structure('./sample_data/modelling/pdb_files/hGBP1_closed.pdb')
    >>> trapped_fraction = 0.9
    >>> av1 = mfm.fluorescence.fps.ACV(structure, residue_seq_number=18, atom_name='CB', contact_volume_trapped_fraction=trapped_fraction)
    >>> av2 = mfm.fluorescence.fps.ACV(structure, residue_seq_number=577, atom_name='CB', contact_volume_trapped_fraction=trapped_fraction)
    >>> av1.save('c:/temp/test_09', mode='dx')
    >>> y2, x2 = av1.pRDA(av2)

    """

    @property
    def contact_volume_trapped_fraction(self):
        return self._contact_volume_trapped_fraction

    @contact_volume_trapped_fraction.setter
    def contact_volume_trapped_fraction(self, v):
        self._contact_volume_trapped_fraction = v
        self.update()

    @property
    def slow_centers(self):
        return self._slow_centers

    @slow_centers.setter
    def slow_centers(self, v):
        atoms = self.atoms
        if isinstance(v, str):
            if v == 'all':
                slow_centers = atoms['coord']
            else:
                a = np.where(atoms['atom_name'] == v)[0]
                slow_centers = atoms['coord'][a]
        self._slow_centers = slow_centers

    @property
    def slow_radius(self):
        return self._slow_radius

    @slow_radius.setter
    def slow_radius(self, v):
        slow_centers = self.slow_centers
        if isinstance(v, (int, long, float)):
            slow_radii = np.ones(slow_centers.shape[0]) * v
        else:
            slow_radii = np.array(v)
            if slow_radii.shape[0] != slow_centers.shape[0]:
                raise ValueError("The size of the slow_radius doesnt match the number of slow_centers")
        self._slow_radius = slow_radii

    @property
    def contact_volume_trapped_fraction(self):
        return self._contact_volume_trapped_fraction

    @contact_volume_trapped_fraction.setter
    def contact_volume_trapped_fraction(self, v):
        self._contact_volume_trapped_fraction = v

    @property
    def contact_density(self):
        return self._contact_density

    def update_density(self):
        av = self
        contact_volume_trapped_fraction = av.contact_volume_trapped_fraction
        dg, x0 = av.dg, av.x0
        slow_radius = av.slow_radius
        slow_centers = av.slow_centers
        density = av.density
        nc, nn, cd, nd = functions.split_av_acv(density, dg, slow_radius, slow_centers, x0)

        cd *= contact_volume_trapped_fraction / nc
        self._contact_density = cd

        nd *= (1. - contact_volume_trapped_fraction) / nn
        density = np.zeros(self.density.shape, dtype=np.float64)
        density += cd
        density += nd

        self._density = density

    def update(self):
        self.update_density()
        BasicAV.update(self)

    def __init__(self, *args, **kwargs):
        BasicAV.__init__(self, *args, **kwargs)
        self._slow_centers = None
        self.slow_centers = kwargs.get('slow_centers', 'CB')

        self._slow_radius = None
        self.slow_radius = kwargs.get('slow_radius', 10.0)

        self._contact_volume_trapped_fraction = None
        self.contact_volume_trapped_fraction = kwargs.get('contact_volume_trapped_fraction', 0.8)

        self._contact_density = None
        self.update_density()


class DynamicAV(BasicAV):

    @property
    def diffusion_map(self):
        return self._diffusion_coefficient_map

    @property
    def rate_map(self):
        if self._fret_rate_map is not None:
            return self._quenching_rate_map + self._fret_rate_map
        else:
            return self._quenching_rate_map

    @property
    def fret_rate_map(self):
        return self._fret_rate_map

    @property
    def quenching_rate_map(self):
        return self._quenching_rate_map

    @property
    def fluorescence_lifetime(self):
        return self._tau0

    @fluorescence_lifetime.setter
    def fluorescence_lifetime(self, v):
        self._tau0 = float(v)

    @property
    def contact_distance(self):
        return self._contact_distance

    @contact_distance.setter
    def contact_distance(self, v):
        av = self
        self._contact_distance = float(v) + max(av.radius1, av.radius2, av.radius3)

    @property
    def slow_factor(self):
        return self._slow_factor

    @slow_factor.setter
    def slow_factor(self, v):
        self._slow_factor = v

    @property
    def donor_only_fluorescence(self):
        return self._d0_time, self._d0_fluorescence

    @property
    def excited_state_map(self):
        return self._ex_state

    def update_diffusion_map(self, **kwargs):
        """Updates the diffusion coefficient map.

        Example
        -------
        >>> import mfm
        >>> structure = mfm.structure.Structure('./sample_data/modelling/pdb_files/hGBP1_closed.pdb')
        >>> free_diffusion = 8.0
        >>> av = mfm.fluorescence.fps.DynamicAV(structure, residue_seq_number=18, atom_name='CB', slow_factor=0.9, contact_distance=1.5, diffusion_coefficients=free_diffusion)
        >>> p.imshow(av.bounds[:,:,20])
        >>> p.show()
        >>> p.imshow(av.diffusion_map[:,:,20])
        >>> p.show()
        >>> p.imshow(av.density[:,:,20])
        >>> p.show()
        >>> p.hist(av.diffusion_map.flatten(), bins=np.arange(0.01, free_diffusion, 0.5))
        """
        diffusion_coefficient = kwargs.get('diffusion_coefficient', self._diffusion_coefficient)
        slow_factor = kwargs.get('slow_factor', self.slow_factor)
        stick_distance = kwargs.get('stick_distance', self.contact_distance)

        av = self
        coordinates = av.atoms['coord']
        ds_sq = stick_distance**2.0
        density = av._density
        r0 = av.x0
        dg = av.dg

        diffusion_mode = kwargs.get('diffusion_mode', self.diffusion_mode)
        if diffusion_mode == 'two_state':
            d_map = assign_diffusion_to_grid_1(density, r0, dg, diffusion_coefficient, coordinates, ds_sq, slow_factor)
        else:
            d_map = assign_diffusion_to_grid_2(density, r0, dg, diffusion_coefficient, coordinates, stick_distance, slow_factor)
        self._diffusion_coefficient_map = d_map

    def update_equilibrium(self, **kwargs):
        """ Updates the equilibrium probabilities of the dye
        Example
        -------
        >>> import mfm
        >>> structure = mfm.structure.Structure('./sample_data/modelling/pdb_files/hGBP1_closed.pdb')
        >>> free_diffusion = 8.0
        >>> av = mfm.fluorescence.fps.DynamicAV(structure, residue_seq_number=18, atom_name='CB', slow_factor=0.985, contact_distance=3.5, diffusion_coefficients=free_diffusion)
        >>> p.imshow(av.diffusion_map[:,:,20])
        >>> p.show()
        >>> p.imshow(av.density[:,:,20])
        >>> p.show()
        >>> av.update_equilibrium(t_max=100.)
        >>> p.imshow(av.density[:,:,20])
        >>> p.show()
        >>> y, x = np.histogram(av.diffusion_map.flatten(), range=(0.01, 10), bins=20, weights=av.density.flatten())
        >>> p.plot(x[1:], y)

        """
        t_step = kwargs.get('t_step', self.t_step_eq)
        t_max = kwargs.get('t_max', 50.0)
        max_it = kwargs.get('max_it', 1e6)
        n_it = min(int(t_max / t_step), max_it)
        n_out = kwargs.get('n_out', n_it + 1)
        self._di.build_program(t_step=t_step)
        t, n, c = self._di.execute(n_it=n_it, n_out=n_out, **kwargs)
        c /= c.sum()
        self._density = c

    def update_quenching_map(self, **kwargs):
        """
        Example
        -------

        >>> import mfm
        >>> structure = mfm.structure.Structure('./sample_data/modelling/pdb_files/hGBP1_closed.pdb')
        >>> av = mfm.fluorescence.fps.DynamicAV(structure, residue_seq_number=18, atom_name='CB')
        >>> p.imshow(av.quenching_rate_map[:,:,20])
        >>> p.show()

        """
        av = self
        atoms = av.atoms
        r0 = av.x0
        dg = av.dg
        density = av.density

        quencher = self.quencher = kwargs.get('quencher', self.quencher)
        kQ, rC = get_kQ_rC(atoms, quencher=quencher)
        tau0 = self.fluorescence_lifetime
        dye_radius = min(self.radius1, self.radius2, self.radius3)
        self._quenching_rate_map = create_quenching_map(density, r0, dg, atoms['coord'], tau0, kQ, rC, dye_radius)

    def update_fret_map(self, acceptor, **kwargs):
        """
        In the FRET-map the average time of FRET to occur is used as FRET rate constant for each grid point
        of the donor. In an excat solution it should be considered, that the FRET-rate constants of the donor
        at each grid points follow a distribution the function create_fret_rate_map approximates this
        distribution by the average time of FRET.

        Example
        -------

        >>> import mfm
        >>> structure = mfm.structure.Structure('./sample_data/modelling/pdb_files/hGBP1_closed.pdb')
        >>> av_d = mfm.fluorescence.fps.DynamicAV(structure, residue_seq_number=18, atom_name='CB')
        >>> av_a = mfm.fluorescence.fps.DynamicAV(structure, residue_seq_number=577, atom_name='CB')
        >>> av_d.update_fret_map(av_a)
        #
        >>> rda = (av_d._fret_rate_map.flatten() * av_d.fluorescence_lifetime) ** (-1./6.) * 52.
        >>> y, x = np.histogram(rda, range=(0.0, 100), bins=100, weights=av_d.density.flatten())
        >>> p.plot(x[1:], y)
        >>> y, x = av_d.pRDA(av=av_a)
        >>> p.plot(x, y)
        >>> p.show()
        >>> average_rate = np.dot(av_d._fret_rate_map.flatten(), av_d.density.flatten())
        >>> (average_rate * av_d.fluorescence_lifetime) ** (-1./6.) * 52.
        >>> np.dot(x, y)

        >>> p.imshow(av_d.fret_rate_map[:,:,20])
        >>> p.show()
        >>> p.imshow(av_d.quenching_rate_map[:,:,20])
        >>> p.show()
        >>> p.imshow(av_d.rate_map[:,:,20])
        >>> p.show()

        """
        density_donor = self.density
        density_acceptor = acceptor.density
        x0_donor = self.x0
        x0_acceptor = acceptor.x0
        dg_donor = self.dg
        dg_acceptor = acceptor.dg
        foerster_radius = kwargs.get('foerster_radius', 52.0)
        kf = kwargs.get('foerster_radius', 1. / self.fluorescence_lifetime)

        self._fret_rate_map = functions.create_fret_rate_map(
            density_donor,
            density_acceptor,
            x0_donor,
            x0_acceptor,
            dg_donor,
            dg_acceptor,
            foerster_radius,
            kf
        )

    def update_donor(self, **kwargs):
        """

        :param kwargs:
        :return:

        Example
        -------
        >>> import mfm
        >>> structure = mfm.structure.Structure('./sample_data/modelling/pdb_files/hGBP1_closed.pdb')
        >>> av = mfm.fluorescence.fps.DynamicAV(structure, residue_seq_number=18, atom_name='CB')
        >>> p.imshow(av.density[:,:,20])
        >>> p.show()

        # Integrate step-wise the first 2 nano-seconds
        #>>> d_0t2 = av.update_donor(n_it=500, t_step=0.001)
        #>>> av.save(filename='c:/temp/0t2', density=d_0t2[1], mode='dx')
        #>>> d_2t4 = av.update_donor(n_it=500, t_step=0.001, reset_density=False)

        >>> r = [av.update_donor(n_it=500, t_step=0.001, reset_density=False) for i in range(8)]
        >>> av.save(filename='c:/temp/2t4', density=d_0t2[1], mode='dx')
        >>> p.semilogy(d_0t2[0], d_0t2[2])
        >>> p.semilogy(d_2t4[0], d_2t4[2])

        """
        t_step = kwargs.get('t_step', self.t_step_fl)
        t_max = kwargs.get('t_max', 50.0)
        max_it = kwargs.get('max_it', 1e6)
        n_out = kwargs.get('n_out', 10)
        reset_density = kwargs.get('reset_density', True)
        n_it = kwargs.get('n_it', min(int(t_max / t_step), max_it))

        self._di.build_program(t_step=t_step)
        if reset_density or self.excited_state_map is None:
            self._di.to_device(k=self.quenching_rate_map, p=self.density, it=0)
        else:
            self._di.to_device(k=self.quenching_rate_map, p=self.excited_state_map, it=0)
        t, n, c = self._di.execute(n_it=n_it, n_out=n_out)
        self._d0_time = t
        self._d0_fluorescence = n
        self._ex_state = c
        return t, c, n

    def __init__(self, *args, **kwargs):
        BasicAV.__init__(self, *args, **kwargs)
        self._tau0 = None
        self._contact_distance = None
        self._quenching_rate_map = None
        self._fret_rate_map = None
        self._di = None
        self._slow_factor = None
        self._diffusion_coefficient_map = None
        self._d0_time, self._d0_fluorescence = None, None
        self._ex_state = None

        self.fluorescence_lifetime = kwargs.get('tau0', 4.2)
        self._diffusion_coefficient = kwargs.get('diffusion_coefficient', 8.0)

        # These values were "manually" optimized that the diffusion coefficient distribution
        # matches more or less what is expected by MD-simulations on nucleic acids (Stas-paper)
        # the contact distance is deliberately big, so that the dye does not stick in very
        # tinny pockets.
        self.slow_factor = kwargs.get('slow_factor', 0.985)
        self.contact_distance = kwargs.get('contact_distance', 3.5) # th

        self.quencher = kwargs.get('quencher', mfm.common.quencher)
        self.diffusion_mode = kwargs.get('diffusion_mode', 'two_state')
        self.t_step_fl = kwargs.get('t_step_fl', 0.001)  # integration time step of fluorescence decay
        self.t_step_eq = kwargs.get('t_step_eq', 0.015)  # integration time step of equilibration

        self.update_diffusion_map()
        self.update_quenching_map()

        # Iterator for equilibration and calculation of fluorescence decay
        self._di = functions.DiffusionIterator(self.diffusion_map, self.bounds, self.density,
                                               dg=self.dg, t_step=self.t_step_eq)

        self.update_equilibrium()


class AvPotential(object):
    """
    The AvPotential class provides the possibility to calculate the reduced or unreduced chi2 given a set of
    labeling positions and experimental distances. Here the labeling positions and distances are provided as
    dictionaries.

    Examples
    --------

    >>> import json
    >>> labeling_file = './sample_data/model/labeling.json'
    >>> labeling = json.load(open(labeling_file, 'r'))
    >>> distances = labeling['Distances']
    >>> positions = labeling['Positions']
    >>> import mfm
    >>> av_potential = mfm.fps.AvPotential(distances=distances, positions=positions)
    >>> structure = mfm.Structure('/sample_data/model/HM_1FN5_Naming.pdb')
    >>> av_potential.getChi2(structure)

    """
    name = 'Av'

    def __init__(self, distances=None, positions=None, av_samples=10000, min_av=150, verbose=False):
        self.verbose = verbose
        self.distances = distances
        self.positions = positions
        self.n_av_samples = av_samples
        self.min_av = min_av
        self.avs = OrderedDict()

    @property
    def structure(self):
        """
        The Structure object used for the calculation of the accessible volumes
        """
        return self._structure

    @structure.setter
    def structure(self, structure):
        self._structure = structure
        self.calc_avs()

    @property
    def chi2(self):
        """
        The current unreduced chi2 (recalculated at each call)
        """
        return self.getChi2()

    def calc_avs(self):
        """
        Calculates/recalculates the accessible volumes.
        """
        if self.positions is None:
            raise ValueError("Positions not set unable to calculate AVs")
        arguments = [
            dict(
                {'structure': self.structure,
                 'verbose': self.verbose,
                },
                **self.positions[position_key]
            )
            for position_key in self.positions
        ]
        avs = map(lambda x: ACV(**x), arguments)
        for i, position_key in enumerate(self.positions):
            self.avs[position_key] = avs[i]

    def calc_distances(self, structure=None, verbose=False):
        """

        :param structure: Structure
            If this object is provided the attributes regarding dye-attachment are kept constant
            and the structure is changed prior calculation of the distances.
        :param verbose: bool
            If this is True output to stdout is generated
        """
        verbose = verbose or self.verbose
        if isinstance(structure, Structure):
            self.structure = structure
        for distance_key in self.distances:
            distance = self.distances[distance_key]
            av1 = self.avs[distance['position1_name']]
            av2 = self.avs[distance['position2_name']]
            distance_type = distance['distance_type']
            R0 = distance['Forster_radius']
            if distance_type == 'RDAMean':
                d12 = functions.RDAMean(av1, av2)
            elif distance_type == 'Rmp':
                d12 = functions.dRmp(av1, av2)
            elif distance_type == 'RDAMeanE':
                d12 = functions.RDAMeanE(av1, av2, R0)
            distance['model_distance'] = d12
            if verbose:
                print("-------------")
                print("Distance: %s" % distance_key)
                print("Forster-Radius %.1f" % distance['Forster_radius'])
                print("Distance type: %s" % distance_type)
                print("Model distance: %.1f" % d12)
                print("Experimental distance: %.1f (-%.1f, +%.1f)" % (distance['distance'],
                                                                      distance['error_neg'], distance['error_pos']))

    def getChi2(self, structure=None, reduced=False, verbose=False):
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
        if isinstance(structure, Structure):
            self.structure = structure

        chi2 = 0.0
        self.calc_distances(verbose=verbose)
        for distance in list(self.distances.values()):
            dm = distance['model_distance']
            de = distance['distance']
            error_neg = distance['error_neg']
            error_pos = distance['error_pos']
            d = dm - de
            chi2 += (d / error_neg) ** 2 if d < 0 else (d / error_pos) ** 2
        if reduced:
            return chi2 / (len(list(self.distances.keys())) - 1.0)
        else:
            return chi2

    def getEnergy(self, structure=None, gauss_bond=True):
        if isinstance(structure, Structure):
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
            return self.getChi2(self, self.structure)


class AvWidget(AvPotential, QtGui.QWidget):
    def __init__(self, parent=None):
        QtGui.QWidget.__init__(self, parent=parent)
        uic.loadUi('./mfm/ui/fluorescence/avWidget.ui', self)
        AvPotential.__init__(self)
        self._filename = None
        self.connect(self.actionOpenLabeling, QtCore.SIGNAL("triggered()"), self.onLoadAvJSON)

    def onLoadAvJSON(self):
        self.filename = mfm.widgets.open_file('Open FPS-JSON', 'FPS-file (*.fps.json)')

    @property
    def filename(self):
        return self._filename

    @filename.setter
    def filename(self, v):
        self._filename = v
        p = json.load(open(v))
        self.distances = p["Distances"]
        self.positions = p["Positions"]
        self.lineEdit_2.setText(v)

    @property
    def n_av_samples(self):
        return int(self.spinBox_2.value())

    @n_av_samples.setter
    def n_av_samples(self, v):
        self.spinBox_2.setValue(int(v))

    @property
    def min_av(self):
        return int(self.spinBox_2.value())

    @min_av.setter
    def min_av(self, v):
        self.spinBox.setValue(int(v))
