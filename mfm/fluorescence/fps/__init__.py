import json
import os
import numpy as np

from . import dynamic
import mfm
import mfm.base
from . import static
from mfm.fluorescence.fps import functions
from mfm.structure.structure import Structure
from . import functions
from .functions import assign_diffusion_to_grid_1, assign_diffusion_to_grid_2, \
    get_kQ_rC, create_quenching_map, assign_diffusion_to_grid_3
from .static import calculate_1_radius, calculate_3_radius

package_directory = os.path.dirname(__file__)
dye_file = os.path.join(mfm.package_directory, 'settings/dye_definition.json')
try:
    dye_definition = json.load(open(dye_file))
except IOError:
    dye_definition = dict()
    dye_definition['a'] = 0
dye_names = dye_definition.keys()


class BasicAV(object):
    """Simulates the accessible volume of a dye

    Examples
    --------

    >>> import mfm
    >>> structure = mfm.structure.Structure('./sample_data/modelling/pdb_files/hGBP1_closed.pdb')
    >>> av = mfm.fluorescence.fps.BasicAV(structure, residue_seq_number=18, atom_name='CB')

    """

    def __init__(self, structure, *args, **kwargs):
        super(BasicAV, *args, **kwargs).__init__()

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
        n, p = functions.density2points(ng, dg, density, x0)
        self._points = p[:n]

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

    def dRDAE(self, av, forster_radius):
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
        return functions.RDAMeanE(self, av, forster_radius)

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

        #diffusion_mode = kwargs.get('diffusion_mode', self.diffusion_mode)

        def f(x):
            a1, a2, a3 = 10.5, 500., 37.2
            m1, m2, m3 = 20.2, 11.7, 1.40
            s1, s2, s3 = 0.47, 11.8, 1.54
            b = -11.0
            y = a1 * np.exp(-.5 * ((x - m1) / s1) ** 2) / (s1 * (2 * np.pi) ** .5) + \
                a2 * np.exp(-.5 * ((x - m2) / s2) ** 2) / (s2 * (2 * np.pi) ** .5) + \
                a3 * np.exp(-.5 * ((x - m3) / s3) ** 2) / (s3 * (2 * np.pi) ** .5) + \
                b
            return np.maximum(y, 0)
        d_map = assign_diffusion_to_grid_3(density, r0, dg, f)
        d_map = assign_diffusion_to_grid_1(d_map, density, r0, dg, coordinates, ds_sq, slow_factor)
        #d_map = assign_diffusion_to_grid_2(density, r0, dg, diffusion_coefficient, coordinates, stick_distance, slow_factor)
        self._diffusion_coefficient_map = d_map

    def update_equilibrium(self, **kwargs):
        """ Updates the equilibrium probabilities of the dye
        Example
        -------
        >>> import mfm
        >>> structure = mfm.structure.Structure('./sample_data/modelling/pdb_files/hGBP1_closed.pdb')
        >>> free_diffusion = 8.0
        >>> av = mfm.fluorescence.fps.DynamicAV(structure, residue_seq_number=577, atom_name='CB', slow_factor=0.985, contact_distance=3.5, diffusion_coefficients=free_diffusion, simulation_grid_resolution=2.0)
        >>> p.imshow(av.diffusion_map[:,:,20])
        >>> p.show()
        >>> p.imshow(av.density[:,:,20])
        >>> p.show()
        >>> av.update_diffusion_map()
        >>> av.update_equilibrium()#t_max=100.)
        >>> p.imshow(av.density[:,:,20])
        >>> p.show()
        >>> y, x = np.histogram(av.diffusion_map.flatten(), range=(0.01, 10), bins=20, weights=av.density.flatten())
        >>> p.plot(x[1:], y)

        """
        t_step = kwargs.get('t_step', self.t_step_eq)
        t_max = kwargs.get('t_max', 250.0)
        max_it = kwargs.get('max_it', 1e6)
        n_it = min(int(t_max / t_step), max_it)
        n_out = kwargs.get('n_out', n_it + 1)

        k = np.zeros_like(self.quenching_rate_map)
        d = self.diffusion_map
        b = self.bounds
        p = np.ones_like(self._density) * b
        self._di.to_device(k=k, p=p, d=d, b=b, it=0, t_step=t_step)
        t, n, c = self._di.execute(n_it=n_it, n_out=n_out, **kwargs)
        cs = c.sum()
        c /= cs
        self._density = c

    def update_quenching_map(self, **kwargs):
        """ Assigns a quenching rate constant to each grid point of the AV
        
        Example
        -------

        >>> import mfm
        >>> structure = mfm.structure.Structure('./sample_data/modelling/pdb_files/hGBP1_closed.pdb')
        >>> av = mfm.fluorescence.fps.DynamicAV(structure, residue_seq_number=18, atom_name='CB')
        >>> p.imshow(av.quenching_rate_map[:,:,20])
        >>> p.show()
        >>> p.hist(av.quenching_rate_map.flatten(), bins=np.arange(0.01, av.fluorescence_lifetime, 0.5))

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
        #self._quenching_rate_map = create_quenching_map(density, r0, dg, atoms['coord'], tau0, kQ, rC, dye_radius)
        v = np.ones_like(rC) * self.rC_electron_transfer
        self._quenching_rate_map = create_quenching_map(density, r0, dg, atoms['coord'], tau0, kQ, v, dye_radius)

    def update_fret_map(self, acceptor, **kwargs):
        """ Calculates an average FRET-rate constant for every AV grid point.
        
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

    def get_donor_only_decay(self, **kwargs):
        """

        :param kwargs:
        :return:

        Example
        -------
        >>> import mfm
        >>> structure = mfm.structure.Structure('./sample_data/modelling/pdb_files/hGBP1_closed.pdb')
        >>> av = mfm.fluorescence.fps.DynamicAV(structure, residue_seq_number=577, atom_name='CB')
        >>> p.imshow(av.density[:,:,20])
        >>> p.show()
        >>> t_step = 0.0141
        >>> times, density, counts = av.get_donor_only_decay(n_it=4095, t_step=0.0141, n_out=1)
        >>> av.save(filename='c:/temp/0t2', density=density, mode='dx')
        >>> irf = mfm.curve.DataCurve(filename='./sample_data/tcspc/ibh_sample/Prompt.txt', skiprows=9)
        >>> data = mfm.curve.DataCurve(filename='./sample_data/tcspc/ibh_sample/Decay_577D.txt', skiprows=9) 
        >>> irf.x *= t_step; data.x *= t_step
        >>> convolve = mfm.fluorescence.tcspc.convolve.Convolve(fit=None, dt=t_step, rep_rate=10, irf=irf, data=data)
        >>> decay = convolve.convolve(counts, mode='full')
        >>> p.semilogy(times, decay)
        
        """
        t_step = kwargs.get('t_step', self.t_step_fl)
        t_max = kwargs.get('t_max', 50.0)
        max_it = kwargs.get('max_it', 1e6)
        n_out = kwargs.get('n_out', 10)
        n_it = kwargs.get('n_it', min(int(t_max / t_step), max_it))

        self._di.to_device(k=self.quenching_rate_map,
                           p=self.density,
                           d=self.diffusion_map,
                           b=self.bounds,
                           it=0,
                           t_step=t_step)
        t, n, c = self._di.execute(n_it=n_it, n_out=n_out)
        self._d0_time = t
        self._d0_fluorescence = n
        self._ex_state = c
        return t, c, n

    def __init__(self, *args, **kwargs):
        BasicAV.__init__(self, *args, **kwargs)

        # Initialization of internal variables
        self._tau0 = None
        self._contact_distance = None
        self._quenching_rate_map = None
        self._fret_rate_map = None
        self._di = None
        self._slow_factor = None
        self._diffusion_coefficient_map = None
        self._d0_time, self._d0_fluorescence = None, None
        self._ex_state = None
        self.rC_electron_transfer = 1.5

        self.fluorescence_lifetime = kwargs.get('tau0', 4.2)
        self._diffusion_coefficient = kwargs.get('diffusion_coefficient', 8.0)

        # These values were "manually" optimized that the diffusion coefficient distribution
        # matches more or less what is expected by MD-simulations on nucleic acids (Stas-paper)
        # the contact distance is deliberately big, so that the dye does not stick in very
        # tinny pockets.
        self.slow_factor = kwargs.get('slow_factor', 0.99)
        self.contact_distance = kwargs.get('contact_distance', 3.5) # th

        self.quencher = kwargs.get('quencher', mfm.common.quencher)
        #self.diffusion_mode = kwargs.get('diffusion_mode', 'two_state')
        self.t_step_fl = kwargs.get('t_step_fl', 0.001)  # integration time step of fluorescence decay
        self.t_step_eq = kwargs.get('t_step_eq', 0.02)  # integration time step of equilibration

        self.update_diffusion_map()
        self.update_quenching_map()

        # Iterator for equilibration and calculation of fluorescence decay
        self._di = functions.DiffusionIterator(self.diffusion_map, self.bounds, self.density,
                                               dg=self.dg, t_step=self.t_step_eq)
        self._di.build_program()
        self.update_equilibrium()


