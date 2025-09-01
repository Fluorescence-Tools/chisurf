from __future__ import annotations
from chisurf import typing

import chisurf.fio as io
import chisurf.fluorescence
import chisurf.structure.av.fps_
import chisurf.settings
import numpy as np
import numba as nb
import math

# Try to import pyopencl, but make it optional
try:
    import pyopencl as cl
    from pyopencl import array as cl_array
    HAS_OPENCL = True
except ImportError:
    HAS_OPENCL = False
fps_settings = chisurf.settings.cs_settings['fps']


def histogram_rda(
        av1: typing.Type[chisurf.structure.av.BasicAV],
        av2: typing.Type[chisurf.structure.av.BasicAV],
        **kwargs
) -> typing.Tuple[
    np.ndarray,
    np.ndarray
]:
    """Calculates the distance distribution with respect to a second accessible volume and returns the
    distance axis and the probability of the respective distance. By default the distance-axis "mfm.rda_axis"
    is taken to generate the histogram.

    :param av1: Accessible volume
    :param av2: Accessible volume
    :param kwargs:
    :return:

    Examples
    --------

    >>> import chisurf.structure
    >>> import chisurf.structure.av
    >>> structure = chisurf.structure.Structure('./test/data/atomic_coordinates/pdb_files/hGBP1_closed.pdb')
    >>> av1 = chisurf.structure.av.BasicAV(structure, residue_seq_number=18, atom_name='CB')
    >>> av2 = chisurf.structure.av.BasicAV(structure, residue_seq_number=577, atom_name='CB')
    >>> y, x = av1.pRDA(av2)

    """
    rda_axis = kwargs.get('rda_axis', chisurf.fluorescence.rda_axis)
    same_size = kwargs.get('same_size', True)
    n_samples = kwargs.get('distance_samples', fps_settings['distance_samples'])
    ds = random_distances(av1.points, av2.points, n_samples)
    r = ds[:, 0]
    w = ds[:, 1]
    p = np.histogram(r, bins=rda_axis, weights=w)[0]
    if same_size:
        p = np.append(p, [0])
    return p, rda_axis


def RDAMean(
        av1: chisurf.structure.av.BasicAV,
        av2: chisurf.structure.av.BasicAV,
        **kwargs
) -> float:
    """Calculate the mean distance between two accessible volumes

    >>> import chisurf
    >>> pdb_filename = './test/data/structure/T4L_Topology.pdb'
    >>> structure = chisurf.structure(pdb_filename)
    >>> av1 = chisurf.structure.av.AV(structure, residue_seq_number=72, atom_name='CB')
    >>> av2 = chisurf.structure.av.AV(structure, residue_seq_number=134, atom_name='CB')
    >>> chisurf.structure.av.functions.RDAMean(av1, av2)
    52.93390285282142
    """
    n_samples = kwargs.get('distance_samples', fps_settings['distance_samples'])
    d = random_distances(av1.points, av2.points, n_samples)
    return np.dot(d[:, 0], d[:, 1]) / d[:, 1].sum()


def widthRDA(
        av1: chisurf.structure.av.BasicAV,
        av2: chisurf.structure.av.BasicAV,
        **kwargs
):
    """Calculate the width of the distance distribution between two accessible volumes

    >>> import chisurf
    >>> pdb_filename = './test/data/structure/T4L_Topology.pdb'
    >>> structure = mfm.structure.Structure(pdb_filename)
    >>> av1 = chisurf.structure.av.AV(structure, residue_seq_number=72, atom_name='CB')
    >>> av2 = chisurf.structure.av.AV(structure, residue_seq_number=134, atom_name='CB')
    >>> chisurf.structure.av.functions.widthRDA(av1, av2)
    52.93390285282142
    """
    n_samples = kwargs.get('distance_samples', fps_settings['distance_samples'])
    d = random_distances(av1.points, av2.points, n_samples)
    s = np.dot(d[:, 0]**2.0, d[:, 1])
    f = np.dot(d[:, 0], d[:, 1])**2.0
    v = s - f
    return np.sqrt(v)


def RDAMeanE(
        av1: chisurf.structure.av.BasicAV,
        av2: chisurf.structure.av.BasicAV,
        R0: float = 52.0,
        **kwargs
) -> float:
    """Calculate the FRET-averaged (PDA/Intensity) distance between two accessible volumes

    >>> import chisurf
    >>> pdb_filename = './test/data/structure/T4L_Topology.pdb'
    >>> structure = mfm.Structure(pdb_filename)
    >>> av1 = chisurf.fps.AV(structure, residue_seq_number=72, atom_name='CB')
    >>> av2 = chisurf.fps.AV(structure, residue_seq_number=134, atom_name='CB')
    >>> mfm.fps.RDAMeanE(av1, av2)
    52.602731299544686
    """
    n_samples = kwargs.get('distance_samples', fps_settings['distance_samples'])
    d = random_distances(av1.points, av2.points, n_samples)
    r = d[:, 0]
    w = d[:, 1]
    e = (1./(1.+(r/R0)**6.0))
    mean_fret = np.dot(w, e) / w.sum()
    return (1./mean_fret - 1.)**(1./6.) * R0


def dRmp(
        av1: chisurf.structure.av.BasicAV,
        av2: chisurf.structure.av.BasicAV,
) -> float:
    """Calculate the distance between the mean position of two accessible volumes

    >>> import chisurf
    >>> pdb_filename = './test/data/structure/T4L_Topology.pdb'
    >>> structure = chisurf.Structure(pdb_filename)
    >>> av1 = chisurf.fps.AV(structure, residue_seq_number=72, atom_name='CB')
    >>> av2 = chisurf.fps.AV(structure, residue_seq_number=134, atom_name='CB')
    >>> mfm.fps.dRmp(av1, av2)
    49.724995634807691
    """
    return np.sqrt(((av1.Rmp-av2.Rmp)**2).sum())


@nb.jit(nopython=True)
def density2points(ng, dg, density, r0):
    r = np.empty((ng**3, 4), dtype=np.float64)
    npm = (ng - 1) / 2 + 1

    gd = np.empty(ng, dtype=np.float64)
    for i in range(-npm, npm):
        gd[i + npm] = i * dg

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
    return n, r


@nb.jit(nopython=True)
def assign_diffusion_to_grid_1(
        d_map,
        density,
        r0,
        dg,
        atoms_coord,
        min_distance_sq,
        atomic_slow_factor
):
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
                    r[ix + npm, iy + npm, iz + npm] = d_map[ix + npm, iy + npm, iz + npm] * slow_factor
                else:
                    r[ix + npm, iy + npm, iz + npm] = 0.0
    return r


@nb.jit(nopython=True)
def assign_diffusion_to_grid_2(
        density,
        r0,
        dg,
        free_diffusion,
        atoms_coord,
        radius,
        atomic_slow_factor
):
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
                        d = math.sqrt((x_a - x_d) ** 2 + (y_a - y_d) ** 2 + (z_a - z_d) ** 2)
                        alpha = d / radius
                        slow_factor = 1. - atomic_slow_factor * math.exp(-alpha)
                        v *= slow_factor
                    r[ix + npm, iy + npm, iz + npm] = v
    return r


def assign_diffusion_to_grid_3(
        density,
        r0,
        dg,
        f
):
    """Assigns diffusion coefficients according to the function f, which should depend on the 
    distance from the attachment point

    :param density: 3D-numpy array of the denisty defined by AV-simulations
    :param r0: attachment position of the AV (center of the 3D-grid)
    :param dg: distance between grid-points
    :return:

    """
    r = np.zeros_like(density)
    ng = density.shape[0]

    npm = int((ng - 1) / 2)
    grid_axis = np.empty(ng, dtype=np.float64)
    for i in range(ng):
        grid_axis[i] = (i - npm) * dg

    for ix in range(-npm, npm + 1):
        x_d = grid_axis[ix + npm]
        for iy in range(-npm, npm + 1):
            y_d = grid_axis[iy + npm]
            for iz in range(-npm, npm + 1):
                z_d = grid_axis[iz + npm]
                d = math.sqrt(x_d ** 2 + y_d ** 2 + z_d ** 2)
                r[ix + npm, iy + npm, iz + npm] = f(d)
    return r


assign_diffusion_to_grid = assign_diffusion_to_grid_1


@nb.jit(nopython=True, nogil=True)
def iterate_cpu(n, p, d, k, b, ng):
    """CPU implementation of the iterate kernel"""
    for ix in range(1, ng-1):
        for iy in range(1, ng-1):
            for iz in range(1, ng-1):
                i = ix + ng * iy + ng * ng * iz
                if b[i] > 0:
                    ixl = (ix - 1) + ng * iy + ng * ng * iz
                    ixr = (ix + 1) + ng * iy + ng * ng * iz
                    iyl = ix + ng * (iy - 1) + ng * ng * iz
                    iyr = ix + ng * (iy + 1) + ng * ng * iz
                    izl = ix + ng * iy + ng * ng * (iz - 1)
                    izr = ix + ng * iy + ng * ng * (iz + 1)

                    dp = d[i] * p[i]
                    xl = (dp - d[ixl] * p[ixl]) * b[ixl]
                    xr = (dp - d[ixr] * p[ixr]) * b[ixr]
                    yl = (dp - d[iyl] * p[iyl]) * b[iyl]
                    yr = (dp - d[iyr] * p[iyr]) * b[iyr]
                    zl = (dp - d[izl] * p[izl]) * b[izl]
                    zr = (dp - d[izr] * p[izr]) * b[izr]

                    ts = k[i] * p[i]

                    n[i] = p[i] - (xl + xr + yl + yr + zl + zr + ts)
    return n

@nb.jit(nopython=True, nogil=True)
def reduce_decay_cpu(p, k, time_i, ng):
    """CPU implementation of the reduce_decay kernel"""
    decay_sum = 0.0
    population_sum = 0.0
    for i in range(ng*ng*ng):
        decay_sum += p[i] * math.exp(-time_i * k[i])
        population_sum += p[i]
    return decay_sum, population_sum

class DiffusionIterator:

    def __init__(self, d, b, p, **kwargs):
        self.ng = ng = d.shape[0]  # number of grid points in one dimension (the maps are quadratic)

        self.d = d  # diffusion coefficient map
        self.b = b  # bounds map: zero where no density 1 where density
        self.p = p  # initial densities
        self.k = kwargs.get('k', np.zeros(b.shape, dtype=np.float32))  # map of rate constant (fluorescence)
        self.t_step = kwargs.get('t_step', 1.0)  # the time step of integration
        self.dg = kwargs.get('dg', 1.0)  # the grid spacing parameter (distance between grid points)
        self.idg2 = 1. / (self.dg**2)
        self.it = 0  # number of performed iterations

        if HAS_OPENCL:
            # This is for OpenCL and defines the global and local workgroup sizes
            self.global_size = (ng**3, )
            self.global_size_3d = (ng, ng, ng,)  # the edges are omitted
            self.local_size = None

            # This is for OpenCL and creates a OpenCL context, a command queue and builds the program (kernel)
            self.ctx = cl.create_some_context()
            self.queue = cl.CommandQueue(self.ctx)
            self.program = self.build_program()
        else:
            # For CPU implementation
            self.p_np = None
            self.n_np = None
            self.d_np = None
            self.k_np = None
            self.b_np = None

    def build_program(self, filename='iterated.c'):
        if not HAS_OPENCL:
            return None

        ng = self.ng
        idg2 = self.idg2
        defines = '''
            # define IDG2 %s
            # define NG %s
            # define NG2 %s
            ''' % (idg2, ng, ng ** 2)
        with io.zipped.open_maybe_zipped(
            filename, 'r'
        ) as f:
            kernel = defines + "".join(f.readlines())
            return cl.Program(self.ctx, kernel).build()
        return None

    def to_device(self, **kwargs):
        self.t_step = t_step = kwargs.get('t_step', None)
        b = self.b = kwargs.get('b', self.b)
        p = self.p = kwargs.get('p', self.p)
        d = self.d = kwargs.get('d', self.d)
        k = self.k = kwargs.get('k', self.k)
        self.it = kwargs.get('it', self.it)

        # Client side arrays
        self.b_np = b.astype(np.uint8).flatten('C')
        self.p_np = p.astype(np.float32).flatten('C')
        self.d_np = (d * self.idg2 * t_step).astype(np.float32).flatten('C')
        self.k_np = (k * t_step).astype(np.float32).flatten('C')

        if HAS_OPENCL:
            # create OpenCL buffers
            mf = cl.mem_flags
            self.p_gp = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.p_np)
            self.n_gp = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.p_np)
            self.d_gp = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.d_np)
            self.k_gp = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.k_np)
            self.b_gp = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.b_np)
        else:
            # For CPU implementation, we'll just use the numpy arrays directly
            self.n_np = self.p_np.copy()

    def execute(self, n_it=1, **kwargs):
        # this defines how often the calculations are copied back from the compute unit (GPU)
        # e.g. 10 means that every 10th iteration is copied from the computing unit (GPU) to "python"
        n_out = kwargs.get('n_out', 10)
        ng = self.ng

        # initialize the next step
        i_out = 0
        total_out = (n_it // n_out + 1)
        time_axis = np.arange(total_out, dtype=np.float32) * self.t_step
        n_excited = np.zeros(total_out, dtype=np.float32)
        n_excited[0] = 1.0

        if HAS_OPENCL:
            queue = self.queue
            prg = self.program
            local_size = self.local_size
            n_local = 512

            tmp_1 = cl_array.zeros(queue, (n_local * total_out,), dtype=np.float32)
            tmp_2 = cl_array.zeros(queue, (n_local * total_out,), dtype=np.float32)

            p = self.p_gp
            n = self.n_gp
            b = self.b_gp
            d = self.d_gp
            k = self.k_gp

            for time_i in range(n_it):
                if time_i % 2 > 0:
                    p, n = n, p
                prg.iterate(queue, self.global_size_3d, local_size,
                            n, p, d, k, b
                            )
                if time_i % n_out == 0:
                    prg.reduce_decay(queue, self.global_size, self.local_size,
                                    p, k,
                                    cl.LocalMemory(n_local * 32),
                                    cl.LocalMemory(n_local * 32),
                                    np.int32(self.global_size[0]),
                                    np.int32(n_local),
                                    np.int32(i_out),
                                    np.float32(time_i),
                                    tmp_1.data,
                                    tmp_2.data
                                    )
                    i_out += 1
                self.it += 1

            dc = (tmp_1.map_to_host()).reshape((total_out, n_local)).sum(axis=1)
            ds = (tmp_2.map_to_host()).reshape((total_out, n_local)).sum(axis=1)
            n_ex = dc / ds
            cl.enqueue_copy(queue, self.p_np, self.p_gp)
            self.p = self.p_np.reshape((ng, ng, ng), order='C')
        else:
            # CPU implementation
            dc = np.zeros(total_out, dtype=np.float32)
            ds = np.zeros(total_out, dtype=np.float32)

            p_np = self.p_np
            n_np = self.n_np

            for time_i in range(n_it):
                if time_i % 2 > 0:
                    p_np, n_np = n_np, p_np

                # Call the CPU implementation of iterate
                n_np = iterate_cpu(n_np, p_np, self.d_np, self.k_np, self.b_np, ng)

                if time_i % n_out == 0:
                    # Call the CPU implementation of reduce_decay
                    decay_sum, population_sum = reduce_decay_cpu(p_np, self.k_np, time_i, ng)
                    dc[i_out] = decay_sum
                    ds[i_out] = population_sum
                    i_out += 1
                self.it += 1

            n_ex = dc / ds
            self.p = p_np.reshape((ng, ng, ng), order='C')

        return time_axis, n_ex, self.p


@nb.jit(nopython=True, nogil=True)
def create_fret_rate_map(
        density_donor,
        density_acceptor,
        r0_donor,
        r0_acceptor,
        dg_donor,
        dg_acceptor,
        foerster_radius,
        kf,
        acceptor_step=2
):
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
def create_quenching_map(
        density,
        r0,
        dg,
        atoms_coord,
        tau0,
        kQ,
        rC,
        dye_radius
):
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
                        d = math.sqrt((x_a - x_d) ** 2 + (y_a - y_d) ** 2 + (z_a - z_d) ** 2) - dye_radius
                        v += kQ[ia] * math.exp(-d / rC[ia])
                    r[ix + npm, iy + npm, iz + npm] = v
    return r


def get_kQ_rC(atoms, **kwargs):
    """Get an array of quenching rate constant (kQ) and critical distances (rC) for all atoms

    :param atoms: an array of atoms
    :return: array of quenching rate constants and array of critical distances
    """
    quencher_dict = kwargs.get('quencher', chisurf.common.quencher)
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
    chisurf.structure.av.fps_.reset_density_av(density, ng)


@nb.jit(nopython=True)
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
        distances[i, 0] = math.sqrt(
            (p1[i1, 0] - p2[i2, 0]) ** 2.0 +
            (p1[i1, 1] - p2[i2, 1]) ** 2.0 +
            (p1[i1, 2] - p2[i2, 2]) ** 2.0
        )
        distances[i, 1] = p1[i1, 3] * p2[i2, 3]
    distances[:, 1] /= distances[:, 1].sum()

    #return _fps.random_distances(av1, av2, nSamples)

    return distances


@nb.jit(nopython=True)
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

    chisurf.structure.av.fps_.modify_av(
        density, ng, dg,
        radius, rs, r0, n_radii, factor
    )
