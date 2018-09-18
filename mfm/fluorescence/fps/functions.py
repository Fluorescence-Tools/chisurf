from mfm.fluorescence.fps import _fps
import mfm
import numpy as np
import numba as nb
from math import exp, sqrt
try:
    import pyopencl as cl
    from pyopencl import array as cl_array
except ImportError:
    cl = None
    cl_array = None
fps_settings = mfm.settings['fps']


def histogram_rda(av1, av2, **kwargs):
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
    rda_axis = kwargs.get('rda_axis', mfm.rda_axis)
    same_size = kwargs.get('same_size', True)
    n_samples = kwargs.get('distance_samples', fps_settings['distance_samples'])
    ds = random_distances(av1.points, av2.points, n_samples)
    r = ds[:, 0]
    w = ds[:, 1]
    p = np.histogram(r, bins=rda_axis, weights=w)[0]
    if same_size:
        p = np.append(p, [0])
    return p, rda_axis


def RDAMean(av1, av2, **kwargs):
    """Calculate the mean distance between two accessible volumes

    >>> import mfm
    >>> pdb_filename = '/sample_data/structure/T4L_Topology.pdb'
    >>> structure = mfm.structure.Structure(pdb_filename)
    >>> av1 = mfm.fluorescence.fps.AV(structure, residue_seq_number=72, atom_name='CB')
    >>> av2 = mfm.fluorescence.fps.AV(structure, residue_seq_number=134, atom_name='CB')
    >>> mfm.fluorescence.fps.functions.RDAMean(av1, av2)
    52.93390285282142
    """
    n_samples = kwargs.get('distance_samples', fps_settings['distance_samples'])
    d = random_distances(av1.points, av2.points, n_samples)
    return np.dot(d[:, 0], d[:, 1])


def widthRDA(av1, av2, **kwargs):
    """Calculate the width of the distance distribution between two accessible volumes

    >>> import mfm
    >>> pdb_filename = '/sample_data/structure/T4L_Topology.pdb'
    >>> structure = mfm.structure.Structure(pdb_filename)
    >>> av1 = mfm.fluorescence.fps.AV(structure, residue_seq_number=72, atom_name='CB')
    >>> av2 = mfm.fluorescence.fps.AV(structure, residue_seq_number=134, atom_name='CB')
    >>> mfm.fluorescence.fps.functions.widthRDA(av1, av2)
    52.93390285282142
    """
    n_samples = kwargs.get('distance_samples', fps_settings['distance_samples'])
    d = random_distances(av1.points, av2.points, n_samples)
    s = np.dot(d[:, 0]**2.0, d[:, 1])
    f = np.dot(d[:, 0], d[:, 1])**2.0
    v = s - f
    return np.sqrt(v)


def RDAMeanE(av1, av2, R0=52.0, **kwargs):
    """Calculate the FRET-averaged (PDA/Intensity) distance between two accessible volumes

    >>> import mfm
    >>> pdb_filename = '/sample_data/structure/T4L_Topology.pdb'
    >>> structure = mfm.Structure(pdb_filename)
    >>> av1 = mfm.fps.AV(structure, residue_seq_number=72, atom_name='CB')
    >>> av2 = mfm.fps.AV(structure, residue_seq_number=134, atom_name='CB')
    >>> mfm.fps.RDAMeanE(av1, av2)
    52.602731299544686
    """
    n_samples = kwargs.get('distance_samples', fps_settings['distance_samples'])
    d = random_distances(av1.points, av2.points, n_samples)
    r = d[:, 0]
    w = d[:, 1]
    e = (1./(1.+(r/R0)**6.0))
    mean_fret = np.dot(w, e)
    return (1./mean_fret - 1.)**(1./6.) * R0


def dRmp(av1, av2):
    """Calculate the distance between the mean position of two accessible volumes

    >>> import mfm
    >>> pdb_filename = '/sample_data/structure/T4L_Topology.pdb'
    >>> structure = mfm.Structure(pdb_filename)
    >>> av1 = mfm.fps.AV(structure, residue_seq_number=72, atom_name='CB')
    >>> av2 = mfm.fps.AV(structure, residue_seq_number=134, atom_name='CB')
    >>> mfm.fps.dRmp(av1, av2)
    49.724995634807691
    """
    return np.sqrt(((av1.Rmp-av2.Rmp)**2).sum())


@nb.jit
def density2points(ng, dg, density, r0):
    r = np.empty((ng**3, 4), dtype=np.float64, order='C')
    npm = (ng - 1) / 2
    gd = np.arange(-npm, npm, dtype=np.float64) * dg
    x0, y0, z0 = r0[0], r0[1], r0[2]

    n = 0
    for ix in range(ng):
        for iy in range(ng):
            for iz in range(ng):
                if density[ix, iy, iz] > 0:
                    r[n, 0] = gd[ix] + x0
                    r[n, 1] = gd[iy] + y0
                    r[n, 2] = gd[iz] + z0
                    r[n, 3] = density[ix, iy, iz]
                    n += 1
    return r[:n]


@nb.jit(nopython=True)
def assign_diffusion_to_grid_1(density, r0, dg, free_diffusion, atoms_coord, min_distance_sq, atomic_slow_factor):
    """Creates a grid of diffusion coefficients based on distance to the atoms in a structure. The more atoms
    are in proximity to the grid-point the slower the diffusion.

    :param density: 3D-numpy array of the denisty defined by AV-simulations
    :param r0: attachment position of the AV (center of the 3D-grid)
    :param ng: number of grid-points in each dimenstion
    :param dg: distance between grid-points
    :param free_diffusion: diffusion coefficient of the free dye
    :param atomic_slow_factor: factor by which diffusion coefficient is lowered per atom in contact
    :param min_distance_sq: squared distance between dye and atom which defines a contact ~(dye-radius + 2 * vdW-radius)
    :param atoms_coord: coordinates of the atoms
    :return:

    J. Chem. Phys. 126, 044707 2007 eq. (3)
    """
    r = np.empty_like(density)
    n_atoms = atoms_coord.shape[0]
    x0, y0, z0 = r0
    ng = density.shape[0]

    npm = int((ng - 1) / 2)
    grid_axis = np.empty(ng, dtype=np.float64)
    for i in range(ng):
        grid_axis[i] = (i - npm) * dg

    for ix in range(-npm, npm):
        x_d = grid_axis[ix + npm] + x0
        for iy in range(-npm, npm):
            y_d = grid_axis[iy + npm] + y0
            for iz in range(-npm, npm):
                z_d = grid_axis[iz + npm] + z0
                slow_factor = 1.0
                if density[ix + npm, iy + npm, iz + npm] > 0:
                    for ia in range(n_atoms):
                        x_a = atoms_coord[ia][0]
                        y_a = atoms_coord[ia][1]
                        z_a = atoms_coord[ia][2]
                        d2 = (x_a - x_d) ** 2 + (y_a - y_d) ** 2 + (z_a - z_d) ** 2
                        if d2 < min_distance_sq:
                            slow_factor *= atomic_slow_factor
                    r[ix + npm, iy + npm, iz + npm] = free_diffusion * slow_factor
                else:
                    r[ix + npm, iy + npm, iz + npm] = 0.0
    return r


@nb.jit(nopython=True)
def assign_diffusion_to_grid_2(density, r0, dg, free_diffusion, atoms_coord, radius, atomic_slow_factor):
    """Creates a grid of diffusion coefficients based on distance to the atoms in a structure. The more atoms
    are in proximity to the grid-point the slower the diffusion.

    :param density: 3D-numpy array of the denisty defined by AV-simulations
    :param r0: attachment position of the AV (center of the 3D-grid)
    :param ng: number of grid-points in each dimenstion
    :param dg: distance between grid-points
    :param free_diffusion: diffusion coefficient of the free dye
    :param atomic_slow_factor: factor by which diffusion coefficient is lowered per atom in contact
    :param atoms_coord: coordinates of the atoms
    :return:

    J. Chem. Phys. 126, 044707 2007 eq. (3)
    """
    r = np.zeros_like(density)
    n_atoms = atoms_coord.shape[0]
    x0, y0, z0 = r0
    ng = density.shape[0]

    npm = int((ng - 1) / 2)
    grid_axis = np.empty(ng, dtype=np.float64)
    for i in range(ng):
        grid_axis[i] = (i - npm) * dg

    for ix in range(-npm, npm + 1):
        x_d = grid_axis[ix + npm] + x0
        for iy in range(-npm, npm + 1):
            y_d = grid_axis[iy + npm] + y0
            for iz in range(-npm, npm + 1):
                z_d = grid_axis[iz + npm] + z0
                if density[ix + npm, iy + npm, iz + npm] > 0:
                    v = free_diffusion
                    for ia in range(n_atoms):
                        x_a = atoms_coord[ia][0]
                        y_a = atoms_coord[ia][1]
                        z_a = atoms_coord[ia][2]
                        d = sqrt((x_a - x_d) ** 2 + (y_a - y_d) ** 2 + (z_a - z_d) ** 2)
                        alpha = d / radius
                        slow_factor = 1. - atomic_slow_factor * exp(-alpha)
                        v *= slow_factor
                    r[ix + npm, iy + npm, iz + npm] = v
    return r


assign_diffusion_to_grid = assign_diffusion_to_grid_1


class DiffusionIterator:

    def __init__(self, d, b, p, **kwargs):
        self.d = d  # diffusion coefficient map
        self.b = b  # bounds map: zero where no density 1 where density
        self.p = p  # initial densities
        self.k = kwargs.get('k', np.zeros(b.shape, dtype=np.float64))  # map of rate constant (fluorescence)
        self.t_step = kwargs.get('t_step', 1.0)  # the time step of integration
        self.dg = kwargs.get('dg', 1.0)  # the grid spacing parameter (distance between grid points)
        self.idg2 = 1. / (self.dg**2)
        self.it = 0  # number of performed iterations

        self.ng = ng = self.d.shape[0]  # number of grid points in one dimension (the maps are quadratic)

        # This is for OpenCL and defines the global and local workgroup sizes
        self.global_size = (ng**3, )#, ng - 2, ng - 2,)
        self.global_size_3d = (ng - 2, ng - 2, ng - 2,) ## the edges are omitted
        #sd = int(mfm.cl_device.max_compute_units / 3)
        self.local_size = None #(64, 8, 2,)

        # This is for OpenCL and creates a OpenCL context, a command queue and builds the program (kernel)
        self.ctx = cl.create_some_context()
        self.queue = cl.CommandQueue(self.ctx)
        self.program = self.build_program()

        self.to_device(d=d, b=b, **kwargs)

    def build_program(self, filename='./opencl/iterated.c', **kwargs):
        ng = self.ng
        idg2 = self.idg2
        self.t_step = kwargs.get('t_step', self.t_step)
        defines = '''
            # define IDG2 %s
            # define NG %s
            # define NG2 %s
            # define DT %s
            ''' % (idg2, ng, ng ** 2, self.t_step)
        f = open(filename, 'r')
        kernel = defines + "".join(f.readlines())
        return cl.Program(self.ctx, kernel).build()

    def to_device(self, **kwargs):
        queue = self.queue
        t_step = self.t_step

        b = self.b = kwargs.get('b', self.b)
        p = self.p = kwargs.get('p', self.p)
        d = self.d = kwargs.get('d', self.d)
        k = self.k = kwargs.get('k', self.k)
        self.it = kwargs.get('it', self.it)

        # Client side arrays
        b_np = b.astype(np.uint8).flatten('C')
        p_np = p.astype(np.float32).flatten('C')
        d_np = (d * self.idg2 * self.t_step).astype(np.float32).flatten('C')
        k_np = (k * t_step).astype(np.float32).flatten('C')

        # create OpenCL buffers
        self._pd = cl_array.to_device(queue, p_np)
        self._dd = cl_array.to_device(queue, d_np)
        self._kd = cl_array.to_device(queue, k_np)
        self._bd = cl_array.to_device(queue, b_np)

    def execute(self, n_it=1, **kwargs):
        if len(kwargs) > 0:
            self.to_device(**kwargs)

        # this defines how often the calculations are copied back from the compute unit (GPU)
        # e.g. 10 means that every 10th iteration is copied from the computing unit (GPU) to "python"
        n_out = kwargs.get('n_out', 10)

        queue = self.queue
        prg = self.program
        local_size = self.local_size
        ng = self.ng

        # initialize the next step
        n_local = 1024
        i_out = 0
        total_out = (n_it // n_out + 1)
        time_axis = np.zeros(total_out, dtype=np.float64)

        tmp_s = cl_array.zeros(queue, (n_local * total_out,), dtype=np.float32)
        tmp_p = cl_array.empty_like(self._pd)


        prg.copy3d(queue, self.global_size, local_size, tmp_p.data, self._pd.data, self._bd.data).wait()
        for i in range(n_it):
            prg.iterate(queue, self.global_size_3d, local_size,
                        tmp_p.data,
                        self._pd.data,
                        self._bd.data,
                        self._dd.data,
                        self._kd.data
                        )
            prg.copy3d(queue, self.global_size, local_size,
                       self._pd.data,
                       tmp_p.data,
                       self._bd.data
                       )
            if i % n_out == 0:
                prg.sumGPU(queue, self.global_size, self.local_size,
                           self._pd.data,
                           tmp_s.data,
                           np.uint32(i_out),
                           np.uint32(n_local),
                           cl.LocalMemory(n_local * 32)
                           )
                time_axis[i_out] = i * self.t_step
                i_out += 1
            self.it += 1

        # transfer decay array in device to local array
        w = tmp_s.get().reshape(total_out, n_local)
        n_ex = w.sum(axis=1)
        p_np = self._pd.get()
        return time_axis, n_ex, p_np.reshape((ng, ng, ng), order='C')


@nb.jit(nopython=True, nogil=True)
def create_fret_rate_map(density_donor, density_acceptor, r0_donor, r0_acceptor, dg_donor, dg_acceptor,
                         foerster_radius, kf, acceptor_step=2):
    """ On every grid point (possible position of the donor) a distribution of FRET-rate constants
    is possible. Hence, the FRET-induced donor decay (fid) for a single donor position (i) is given by
    fid = sum_i (xi exp(-kret(i)*t)) where (i) are the (i) are the possible acceptor positions.
    Hence, a single FRET-rate constant is not sufficient to describe the fid of a fixed donor position.
    Here, the fid is approximated by the average time <tret> for FRET to occur.
    fid = sum_i (xi exp(-kret(i)*t)) approx exp(-<kret>*t)

    :param density_donor:
    :param density_acceptor:
    :param r0_donor:
    :param r0_acceptor:
    :param dg_donor:
    :param dg_acceptor:
    :param foerster_radius:
    :param acceptor_step: sets the steps for iterating over the acceptor
    :param kf:
    :return:
    """
    r = np.zeros(density_donor.shape, dtype=np.float64)
    r02 = foerster_radius ** 2.0
    dg_d, dg_a = dg_donor, dg_acceptor
    x0d, y0d, z0d = r0_donor
    x0a, y0a, z0a = r0_acceptor
    ng_d = density_donor.shape[0]
    ng_a = density_acceptor.shape[0]
    s = acceptor_step

    npm_d = int((ng_d - 1) / 2)
    grid_axis_d = np.empty(ng_d, dtype=np.float64)
    for i in range(ng_d):
        grid_axis_d[i] = (i - npm_d) * dg_d

    npm_a = int((ng_a - 1) / 2)
    grid_axis_a = np.empty(ng_a, dtype=np.float64)
    for i in range(ng_a):
        grid_axis_a[i] = (i - npm_a) * dg_a

    for ix_d in range(-npm_d, npm_d):
        x_d = grid_axis_d[ix_d + npm_d] + x0d
        for iy_d in range(-npm_d, npm_d):
            y_d = grid_axis_d[iy_d + npm_d] + y0d
            for iz_d in range(-npm_d, npm_d):
                z_d = grid_axis_d[iz_d + npm_d] + z0d
                # Here the average time of FRET is calculated by iterating over all acceptor
                # positions
                tret = 0.0  # the time of FRET
                if density_donor[ix_d + npm_d, iy_d + npm_d, iz_d + npm_d] > 0:
                    tda = 0.0
                    for ix_a in range(-npm_a, npm_a, s):
                        x_a = grid_axis_a[ix_a + npm_a] + x0a
                        sx = (x_a - x_d)**2
                        for iy_a in range(-npm_a, npm_a, s):
                            y_a = grid_axis_a[iy_a + npm_a] + y0a
                            sy = (y_a - y_d) ** 2
                            for iz_a in range(-npm_a, npm_a, s):
                                z_a = grid_axis_a[iz_a + npm_a] + z0a
                                da = density_acceptor[ix_a + npm_a, iy_a + npm_a, iz_a + npm_a]
                                if da > 0.0:
                                    rda2 = sx + sy + (z_a - z_d) ** 2
                                    r2 = r02 / rda2
                                    tret += da / ((r2 * r2 * r2) * kf)
                                    tda += da
                    tret /= tda
                    r[ix_d + npm_d, iy_d + npm_d, iz_d + npm_d] = 1. / tret
    return r


@nb.jit(nopython=True, nogil=True)
def create_quenching_map(density, r0, dg, atoms_coord, tau0, kQ, rC, dye_radius):
    """Creates a grid of diffusion coefficients based on distance to the atoms in a structure. The more atoms
    are in proximity to the grid-point the slower the diffusion.

    :param density: 3D-numpy array of the denisty defined by AV-simulations
    :param r0: attachment position of the AV (center of the 3D-grid)
    :param ng: number of grid-points in each dimenstion
    :param dg: distance between grid-points
    :param atoms_coord: coordinates of the atoms
    :param kQ: atomic quenching constants
    :param rC: characteristic distances for quenching
    :param tau0: lifetime of the dye without quenching
    :return:
    """
    r = np.zeros(density.shape, dtype=np.float64)
    n_atoms = atoms_coord.shape[0]
    x0, y0, z0 = r0
    ng = density.shape[0]

    npm = int((ng - 1) / 2)
    grid_axis = np.empty(ng, dtype=np.float64)
    for i in range(ng):
        grid_axis[i] = (i - npm) * dg

    for ix in range(-npm, npm):
        x_d = grid_axis[ix + npm] + x0
        for iy in range(-npm, npm):
            y_d = grid_axis[iy + npm] + y0
            for iz in range(-npm, npm):
                z_d = grid_axis[iz + npm] + z0
                if density[ix + npm, iy + npm, iz + npm] > 0.0:
                    v = 1. / tau0
                    for ia in range(n_atoms):
                        kQi = kQ[ia]
                        rCi = rC[ia]
                        if kQi == 0.0 or rCi == 0.0:
                            continue
                        x_a = atoms_coord[ia][0]
                        y_a = atoms_coord[ia][1]
                        z_a = atoms_coord[ia][2]
                        d = sqrt((x_a - x_d) ** 2 + (y_a - y_d) ** 2 + (z_a - z_d) ** 2) - dye_radius
                        v += kQ[ia] * exp(-d / rC[ia])
                    r[ix + npm, iy + npm, iz + npm] = v
    return r


def get_kQ_rC(atoms, **kwargs):
    """Get an array of quenching rate constant (kQ) and critical distances (rC) for all atoms

    :param atoms: an array of atoms
    :return: array of quenching rate constants and array of critical distances
    """
    quencher_dict = kwargs.get('quencher', mfm.common.quencher)
    kQ = np.zeros(atoms.shape[0], dtype=np.float64)
    rC = np.zeros(atoms.shape[0], dtype=np.float64)
    for i, a in enumerate(atoms):
        try:
            res_name = a['res_name']
            atom_name = a['atom_name']
            kQ[i], rC[i] = quencher_dict[res_name][atom_name]
        except KeyError:
            pass
    return kQ, rC


def reset_density_av(density):
    """Sets all densities in av to 1.0 if the density is bigger than 0.0

    :param density: numpy-array
    :return:
    """
    ng = density.shape[0]
    _fps.reset_density_av(density, ng)


@nb.jit
def random_distances(p1, p2, n_samples):
    """

    :param xyzw: a 4-dim vector xyz and the weight of the coordinate
    :param nSamples:
    :return:
    """

    n_p1 = p1.shape[0]
    n_p2 = p2.shape[0]

    distances = np.empty((n_samples, 2), dtype=np.float64)

    for i in range(n_samples):
        i1 = np.random.randint(0, n_p1)
        i2 = np.random.randint(0, n_p2)
        distances[i, 0] = sqrt(
            (p1[i1, 0] - p2[i2, 0]) * (p1[i1, 0] - p2[i2, 0]) +
            (p1[i1, 1] - p2[i2, 1]) * (p1[i1, 1] - p2[i2, 1]) +
            (p1[i1, 2] - p2[i2, 2]) * (p1[i1, 2] - p2[i2, 2])
        )
        distances[i, 1] = p1[i1, 3] * p2[i2, 3]
    distances[:, 1] /= distances[:, 1].sum()

    #return _fps.random_distances(av1, av2, nSamples)

    return distances


@nb.jit
def split_av_acv(density, dg, radius, rs, r0):
    """

    :param density: numpy-array
        density of the accessible volume dimension ng, ng, ng uint8 numpy array as obtained of fps
    :param ng: int
        number of grid points
    :param dg: float
        grid resolution of the density grid
    :param radius: float
        radius around the list of points. all points within a radius of r0 around the points in rs are part
        of the subav. each slow point is assiciated to one slow-radius
    :param rs: list
        list of points (x,y,z) defining the subav
    :param r0:
        is the position of the accessible volume


    """
    ng = density.shape[0]
    n_radii = rs.shape[0]

    radius = np.array(radius, dtype=np.float64)
    if len(radius) != n_radii:
        radius = np.zeros(n_radii, dtpye=np.float64) + radius[0]

    d1 = np.zeros_like(density)
    d2 = np.zeros_like(density)

    # iterate through all possible grid points in the density
    n1 = int(0)
    n2 = int(0)
    for ix in range(ng):
        for iy in range(ng):
            for iz in range(ng):
                if density[ix, iy, iz] <= 0:
                    continue
                # count the overlaps with slow-centers
                overlapped = 0

                for isa in range(n_radii):
                    ix0 = int(((rs[isa, 0]-r0[0])/dg) + (ng - 1)/2)
                    iy0 = int(((rs[isa, 1]-r0[1])/dg) + (ng - 1)/2)
                    iz0 = int(((rs[isa, 2]-r0[2])/dg) + (ng - 1)/2)
                    radius_idx = int((radius[isa] / dg))
                    if ((ix - ix0)**2 + (iy - iy0)**2 + (iz - iz0)**2) < radius_idx**2:
                        overlapped = 1
                        break

                if overlapped > 0:
                    d1[ix, iy, iz] = 1
                    n1 += 1
                else:
                    d2[ix, iy, iz] = 1
                    n2 += 1
    return n1, n2, d1, d2


def modify_av(density, dg, radius, rs, r0, factor):
    """
    Multiplies density by factor if within radius

    :param density: numpy-array
        density of the accessible volume dimension ng, ng, ng uint8 numpy array as obtained of fps
    :param dg: float
        grid resolution of the density grid
    :param radius: numpy-array/list
        radius around the list of points. all points within a radius of r0 around the points in rs are part
        of the subav. each slow point is associated to one slow-radius
    :param rs: numpy-array/list
        list of points (x,y,z) defining the subav
    :param r0: numpy-array
        is the position of the accessible volume
    :param factor: float
        factor by which density is multiplied
    """

    ng = density.shape[0]
    n_radii = rs.shape[0]

    radius = np.array(radius, dtype=np.float64)
    if len(radius) != n_radii:
        radius = np.zeros(n_radii, dtpye=np.float64) + radius[0]

    density = np.copy(density)

    _fps.modify_av(density, ng, dg, radius, rs, r0, n_radii, factor)
