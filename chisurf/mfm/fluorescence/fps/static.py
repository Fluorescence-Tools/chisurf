import ctypes as C
import platform
import os
import numpy as np
import mfm
import numba as nb

b, o = platform.architecture()

package_directory = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(package_directory, './dll')

if 'Windows' in o:
    if '32' in b:
        fpslibrary = os.path.join(
            path,
            'fpsnative.win32.dll'
        )
    elif '64' in b:
        fpslibrary = os.path.join(
            path,
            'fpsnative.win64.dll'
        )
else:
    if platform.system() == 'Linux':
        fpslibrary = os.path.join(
            path,
            'liblinux_fps.so'
        )
    else:
        fpslibrary = os.path.join(
            path,
            'libav.dylib'
        )


_fps = np.ctypeslib.load_library(fpslibrary, ".")
_fps.calculate1R.restype = C.c_int
_fps.calculate1R.argtypes = [
    C.c_double, C.c_double, C.c_double,
    C.c_int, C.c_double,
    C.POINTER(C.c_double), C.POINTER(C.c_double), C.POINTER(C.c_double),
    C.POINTER(C.c_double), C.c_int, C.c_double,
    C.c_double, C.c_int,
    C.POINTER(C.c_char)
]
_fps.calculate3R.argtypes = [
    C.c_double, C.c_double, C.c_double, C.c_double, C.c_double,
    C.c_int, C.c_double,
    C.POINTER(C.c_double), C.POINTER(C.c_double), C.POINTER(C.c_double),
    C.POINTER(C.c_double), C.c_int, C.c_double,
    C.c_double, C.c_int,
    C.POINTER(C.c_char)
]


def calculate_1_radius(
        l,
        w,
        r,
        atom_i, x, y, z, vdw, **kwargs):
    """
    :param l: float
        linker length
    :param w: float
        linker width
    :param r: float
        dye-radius
    :param atom_i: int
        attachment-atom index
    :param x: array
        Cartesian coordinates of atoms (x)
    :param y: array
        Cartesian coordinates of atoms (y)
    :param z: array
        Cartesian coordinates of atoms (z)
    :param vdw:
        Van der Waals radii (same length as number of atoms)
    :param linkersphere: float
        Initial linker-sphere to start search of allowed dye positions
    :param linknodes: int
        By default 3
    :param vdwRMax: float
        Maximal Van der Waals radius
    :param dg: float
        Resolution of accessible volume in Angstrom
    :param verbose: bool
        If true informative output is printed on std-out
    :param n_mul: bool
        number by which the number of grid points in each dimension has to be dividable if this
        is not the case the av is zero padded

    Examples
    --------
    Calculating accessible volume using provided pdb-file

    >>> import mfm
    >>> pdb_filename = './test/data/structure/T4L_Topology.pdb'
    >>> residue_number = 18
    >>> atom_name = 'CB'
    >>> attachment_atom = 1
    >>> av = mfm.fps.AV(pdb_filename, attachment_atom=1, verbose=True)

    Calculating accessible volume using provided Structure object

    >>> import mfm
    >>> pdb_filename = './test/data/structure/T4L_Topology.pdb'
    >>> structure = mfm.Structure(pdb_filename)
    >>> av = mfm.fps.AV(structure, attachment_atom=1, verbose=True)
    Calculating accessible volume
    -----------------------------
    Loading PDB
    Calculating initial-AV
    Linker-length  : 20.00
    Linker-width   : 0.50
    Linker-radius  : 5.00
    Attachment-atom: 1
    AV-resolution  : 0.50
    AV: calculate1R
    Number of atoms: 2647
    Attachment atom: [ 33.28   58.678  40.397]
    Points in AV: 111911
    Points in total-AV: 111911

    Using residue_seq_number and atom_name to calculate accessible volume, this also works without
    chain_identifier. However, only if a single-chain is present.

    >>> import mfm
    >>> pdb_filename = './test/data/structure/T4L_Topology.pdb'
    >>> structure = mfm.Structure(pdb_filename)
    >>> av = mfm.fluorescence.fps.BasicAV(structure, residue_seq_number=11, atom_name='CB', verbose=True)

    If save_av is True the calculated accessible volume is save to disk. The filename of the calculated
    accessible volume is determined by output_file

    >>> import mfm.fluorescence
    >>> import mfm.structure
    >>> pdb_filename = './test/data/structure/T4L_Topology.pdb'
    >>> structure = mfm.structure.Structure(pdb_filename)
    >>> av = mfm.fluorescence.fps.AV(structure, residue_seq_number=11, atom_name='CB', verbose=True, save_av=True, output_file='test')
    Calculating accessible volume
    -----------------------------
    Loading PDB
    Calculating initial-AV
    Linker-length  : 20.00
    Linker-width   : 0.50
    Linker-radius  : 5.00
    Attachment-atom: 174
    AV-resolution  : 0.50
    AV: calculate1R
    Number of atoms: 2647
    Attachment atom: [ 41.606  44.953  36.625]
    Points in AV: 22212
    Points in total-AV: 22212

    write_xyz
    ---------
    Filename: test.xyz
    -------------------

    """
    verbose = kwargs.get('verbose', mfm.settings['verbose'])
    linkersphere = kwargs.get('linkersphere', mfm.settings['fps']['allowed_sphere_radius'])
    linknodes = kwargs.get('linknodes', mfm.settings['fps']['linknodes'])
    vdw_max = kwargs.get('vdw_max', mfm.settings['fps']['vdw_max'])
    dg = kwargs.get('dg', mfm.settings['fps']['simulation_grid_resolution'])
    n_mul = kwargs.get('n_mul', 32)

    n_atoms = len(vdw)

    npm = int(np.floor(l / dg))
    ng = 2 * npm + 1
    ng3 = ng * ng * ng
    density = np.zeros(ng3, dtype=np.uint8)
    x0, y0, z0 = x[atom_i], y[atom_i], z[atom_i]
    r0 = np.array([x0, y0, z0])

    # http://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.ctypes.html
    # Be careful using the ctypes attribute - especially on temporary arrays or arrays
    # constructed on the fly. For example, calling (a+b).ctypes.data_as(ctypes.c_void_p)
    # returns a pointer to memory that is invalid because the array created as (a+b) is
    # deallocated before the next Python statement.

    x = np.ascontiguousarray(x)
    y = np.ascontiguousarray(y)
    z = np.ascontiguousarray(z)
    vdw = np.ascontiguousarray(vdw)

    _x = x.ctypes.data_as(C.POINTER(C.c_double))
    _y = y.ctypes.data_as(C.POINTER(C.c_double))
    _z = z.ctypes.data_as(C.POINTER(C.c_double))
    _vdw = vdw.ctypes.data_as(C.POINTER(C.c_double))

    _density = density.ctypes.data_as(C.POINTER(C.c_char))
    n = _fps.calculate1R(l, w, r, atom_i, dg, _x, _y, _z, _vdw,
                         n_atoms, vdw_max, linkersphere, linknodes, _density)
    if verbose:
        print("Number of atoms: %i" % n_atoms)
        print("Attachment atom coordinates: %s" % r0)
        print("Points in AV: %i" % n)

    density = density.astype(np.float64)
    density = density.reshape([ng, ng, ng])

    ng_n = ng + n_mul - ng % n_mul
    d2 = np.zeros((ng_n, ng_n, ng_n), dtype=np.float64)
    off = (ng_n - ng) / 2
    d2[off:off+ng, off:off+ng, off:off+ng] = density
    return d2, ng_n, r0


def calculate_3_radius(l, w, r1, r2, r3, atom_i, x, y, z, vdw, **kwargs):
    """
    :param l: float
        linker length
    :param w: float
        linker width
    :param r1: float
        Dye-radius 1
    :param r2: float
        Dye-radius 2
    :param r3: float
        Dye-radius 3
    :param atom_i: int
        attachment-atom index
    :param x: array
        Cartesian coordinates of atoms (x)
    :param y: array
        Cartesian coordinates of atoms (y)
    :param z: array
        Cartesian coordinates of atoms (z)
    :param vdw:
        Van der Waals radii (same length as number of atoms)
    :param linkersphere: float
        Initial linker-sphere to start search of allowed dye positions
    :param linknodes: int
        By default 3
    :param vdw_max: float
        Maximal Van der Waals radius
    :param dg: float
        Resolution of accessible volume in Angstrom
    :param verbose: bool
        If true informative output is printed on std-out
    :return:

    """
    verbose = kwargs.get('verbose', mfm.settings['verbose'])
    linkersphere = kwargs.get('linkersphere', mfm.settings['fps']['allowed_sphere_radius'])
    linknodes = kwargs.get('linknodes', mfm.settings['fps']['linknodes'])
    vdw_max = kwargs.get('vdw_max', mfm.settings['fps']['vdw_max'])
    dg = kwargs.get('dg', mfm.settings['fps']['simulation_grid_resolution'])

    if verbose:
        print("AV: calculate3R")
    n_atoms = len(vdw)

    npm = int(np.floor(l / dg))
    ng = 2 * npm + 1
    ng3 = ng * ng * ng
    density = np.zeros(ng3, dtype=np.uint8)
    x0, y0, z0 = x[atom_i], y[atom_i], z[atom_i]
    r0 = np.array([x0, y0, z0])

    x = np.ascontiguousarray(x)
    y = np.ascontiguousarray(y)
    z = np.ascontiguousarray(z)
    vdw = np.ascontiguousarray(vdw)

    _x = x.ctypes.data_as(C.POINTER(C.c_double))
    _y = y.ctypes.data_as(C.POINTER(C.c_double))
    _z = z.ctypes.data_as(C.POINTER(C.c_double))
    _vdw = vdw.ctypes.data_as(C.POINTER(C.c_double))

    _density = density.ctypes.data_as(C.POINTER(C.c_char))
    n = _fps.calculate3R(l, w, r1, r2, r3, atom_i, dg, _x, _y, _z, _vdw,
                         n_atoms, vdw_max, linkersphere, linknodes, _density)
    if verbose:
        print("Number of atoms: %i" % n_atoms)
        print("Attachment atom: %s" % r0)
        print("Points in AV: %i" % n)

    density = density.astype(np.float32)
    density = density.reshape([ng, ng, ng])
    return density, ng, r0


@nb.jit(nopython=True)
def atoms_in_reach(xyz, vdw, dmaxsq, atom_i):
    """Return the xyz coordinates of atoms within reach (defined by dmaxsq) of a list of atoms

    :param xyz:
    :param vdw:
    :param dmaxsq:
    :param atom_i:
    :return:

    Example
    -------
    >>> pdb_filename = './test/data/modelling/pdb_files/hGBP1_open.pdb'
    >>> structure = mfm.structure.Structure(pdb_filename)
    >>> xs, vs = atoms_in_reach(structure.xyz, structure.vdw, 10.0, 1)

    """
    # copy all atoms in proximity to the dye into a smaller array and move coordinate frame to attachment point
    n_atoms = xyz.shape[0]
    atomindex = np.empty(n_atoms, dtype=np.uint32)
    r0 = xyz[atom_i]
    natomsgrid = 0
    for i in range(0, n_atoms):
        dsq = ((xyz[i] - r0)**2.0).sum()
        if (dsq < dmaxsq) and (i != atom_i):
            atomindex[natomsgrid] = i
            natomsgrid += 1

    ra = np.empty((natomsgrid, 3), dtype=np.float64)
    vdwr = np.empty(natomsgrid, dtype=np.float64)
    for i in range(natomsgrid):
        n = atomindex[i]
        ra[i] = xyz[n]
        vdwr[i] = vdw[n]
    return ra, vdwr


@nb.jit(nopython=True)
def make_grid_axis(dg, n):
    grid_x = np.empty(n, dtype=np.float64)
    npm = (n - 1) / 2
    for i in range(n):
        grid_x[i] = (i - npm) * dg
    return grid_x


@nb.jit(nopython=True)
def find_atom_clashes(xyz, vdw, density, dg, min_clash):
    """

    :param xyz: atomic coordinates
    :param vdw: van der waals radii of atoms
    :param density: initial assumed density as a 3D-grid
    :param dg: grid spacing
    :param min_clash:
    :return:
    """
    nx, ny, nz = density.shape
    grid_x = make_grid_axis(dg, nx)
    grid_y = make_grid_axis(dg, ny)
    grid_z = make_grid_axis(dg, nz)

    for i in range(xyz.shape[0]):
        min_clash_sq = (min_clash + vdw[i]) ** 2
        xa, ya, za = xyz[i, 0], xyz[i, 1], xyz[i, 2]
        for ix in range(density.shape[0]):
            dx = grid_x[ix] - xa
            for iy in range(density.shape[1]):
                dy = grid_y[iy] - ya
                for iz in range(density.shape[2]):
                    if density[ix, iy, iz] > 0.0:
                        dz = grid_z[iz] - za
                        d2 = dx * dx + dy * dy + dz * dz
                        if d2 <= min_clash_sq:
                            density[ix, iy, iz] = 0
    return density


@nb.jit(nopython=True)
def define_starting_positions(linker_sphere, w, dg, l, ng):
    """

    :param density:
    :param linker_sphere: a float (unit-less) the initial positions are a sphere around the attachment point
     of the radius linker_sphere * w, where w is the linker-width
    :param w: the linker width
    :param dg: the grid spacing
    :return:
    """
    # identify all grid points which are within a "linker-sphere" as starting point
    # for the later search
    rmaxsqint = int(linker_sphere * linker_sphere * w * w / dg / dg)
    npm = (ng - 1) / 2

    grid_sq = np.empty(ng, dtype=np.uint32)
    for i in range(ng):
        grid_sq[i] = (i - npm)**2

    newpos = np.zeros((ng * ng * ng, 3), dtype=np.uint32)  # an array keeping the positions for the search
    length_linker = np.zeros((ng, ng, ng), dtype=np.float64)  # an array with the determined linker-length
    nnew = 0  # the number of active grid-points for the search
    for ix in range(ng):
        ix2 = grid_sq[ix]
        for iy in range(ng):
            iy2 = grid_sq[iy]
            for iz in range(ng):
                iz2 = grid_sq[iz]
                di = ix2 + iy2 + iz2
                if di < rmaxsqint:
                    length_linker[ix, iy, iz] = (di ** 0.5) * dg
                    newpos[nnew] = ix, iy, iz
                    nnew += 1
                else:
                    length_linker[ix, iy, iz] = l + l
    return length_linker, newpos, nnew


@nb.jit(nopython=True)
def distance_lookup(linknodes, dg):
    sqrts_dg = np.zeros((linknodes, linknodes, linknodes), dtype=np.float64)
    for ix in range(linknodes):
        for iy in range(linknodes):
            for iz in range(linknodes):
                sqrts_dg[ix, iy, iz] = np.sqrt((ix * ix + iy * iy + iz * iz)) * dg
    return sqrts_dg


@nb.jit(nopython=True)
def calc_linker_distance(density, linker_sphere, linknodes, dg, w, l):
    ng = density.shape[0]
    linker_length, newpos, nnew = define_starting_positions(linker_sphere, w, dg, l, ng)
    visit_map = np.zeros(density.shape, dtype=np.uint8)

    # calculate distance lookup table
    sqrts_dg = distance_lookup(linknodes + 1, dg)

    ng = density.shape[0]
    while nnew > 0:
        for n in range(nnew):
            xi0, yi0, zi0 = newpos[n]
            rlink0 = linker_length[xi0, yi0, zi0]

            xmin = max(0, xi0 - linknodes)
            xmax = min(ng, xi0 + linknodes)
            ymin = max(0, yi0 - linknodes)
            ymax = min(ng, yi0 + linknodes)
            zmin = max(0, zi0 - linknodes)
            zmax = min(ng, zi0 + linknodes)

            for xi in range(xmin, xmax):
                dx = abs(xi - xi0)
                for yi in range(ymin, ymax):
                    dy = abs(yi - yi0)
                    for zi in range(zmin, zmax):
                        dz = abs(zi - zi0)
                        r = rlink0 + sqrts_dg[dx, dy, dz]
                        if (linker_length[xi, yi, zi] > r) and (r < l) and density[xi, yi, zi] > 0:
                            linker_length[xi, yi, zi] = r
                            visit_map[xi, yi, zi] |= 0x04

        # update "new" positions
        nnew = 0
        for xi in range(ng):
            for yi in range(ng):
                for zi in range(ng):
                    if visit_map[xi, yi, zi] & 0x04:
                        newpos[nnew] = xi, yi, zi
                        nnew += 1
                    visit_map[xi, yi, zi] &= 0x03

    # collect results
    for xi in range(ng):
        for yi in range(ng):
            for zi in range(ng):
                if linker_length[xi, yi, zi] > l:
                    linker_length[xi, yi, zi] = 0.0
                    density[xi, yi, zi] = 0.0
    return density, linker_length


def calc_av1_py(l, w, r, atom_i, ng, xyz, vdw, vdw_max=3.5, linker_sphere=2.0, linknodes=3):
    """

    :param l: linker-length
    :param w: linker-width
    :param r: dye-radius
    :param atom_i: attachment atom
    :param ng: number of grid points
    :param xyz: coordinates of atoms
    :param vdw: van der waals radii of atoms
    :param vdw_max: maximum van der waals radius
    :param linker_sphere:
    :param linknodes:
    :return:

    Example
    -------

    >>> import mfm
    >>> import pylab as p
    >>> pdb_filename = './test/data/modelling/pdb_files/hGBP1_open.pdb'
    >>> structure = mfm.structure.Structure(pdb_filename)
    >>> av = calc_av1_py(l=20.0, w=2.0, r=3.5, atom_i=1, ng=33, xyz=structure.xyz, vdw=structure.vdw)
    >>> l=20.0
    >>> w=2.0
    >>> r=3.5
    >>> atom_i=1
    >>> ng=33
    >>> xyz=structure.xyz
    >>> vdw=structure.vdw
    >>> p.imshow(linker_length[:, 22,:], interpolation='none')
    """
    dg = l / ((ng - 1.0) / 2.0)
    density = np.ones((ng, ng, ng), dtype=np.uint32)

    # select a subset of the atoms (only the ones in reach of the dye linker)
    xyz_a, vdw_a = atoms_in_reach(xyz=xyz, vdw=vdw, dmaxsq=(l + r + vdw_max)**2, atom_i=atom_i)
    # Move the coordinates of the atoms
    r0 = xyz[atom_i]
    xyz_a -= r0

    # identify clashes
    min_clash = max(r, (0.5 * w))
    density = find_atom_clashes(xyz_a, vdw_a, density, dg, min_clash)

    # consider only positions within the linker-length reach
    density, linker_distance = calc_linker_distance(density, linker_sphere, linknodes, dg, w, l)

    # weight positions according to linker

    return density, linker_distance, r0


def calc_weights_from_traj(traj, res_id, atom_name, chain_id,
                           ng, dg, r0_res, r0_atom_name, r0_chain):
    """

    :param density:
    :param traj:
    :param atom_i:
    :param r0: xyz coordinates of attachment
    :param dg: grid spacing
    :return:

     Example
     -------

     >>> import mdtraj as md
     >>> import pylab as p
     >>> import mfm
     >>> from mfm.fluorescence.fps.static import calc_av1_py, calc_weights_from_traj
     >>> traj = md.load('e:/simulations_free_dye/t_join.h5')

     >>> res_id = 3  # the chromophore
     >>> atom_name = "O91"
     >>> chain_id = 0

     >>> r0_atom_name = "S1"
     >>> r0_res = 1
     >>> r0_chain = 0

     >>> ng = 50
     >>> dg = 1.0
     >>> hist_3d, r0 = calc_weights_from_traj(traj, res_id, atom_name, chain_id, ng, dg, r0_res, r0_atom_name, r0_chain)
     >>> offset = (ng - 1) / 2 * dg
     >>> mfm.io.pdb.write_open_dx("c:/temp/test.dx", hist_3d, r0 - offset, ng, ng, ng, dg, dg, dg)

    >>> pdb_filename = 'e:/simulations_free_dye/peptide.pdb'
    >>> structure = mfm.structure.Structure(pdb_filename)
    >>> av = calc_av1_py(l=23.0, w=2.0, r=3.5, atom_i=10, ng=ng, xyz=structure.xyz, vdw=structure.vdw)
    >>> y, x = np.histogram(av[1].flatten(), weights=hist_3d.flatten(), bins=np.arange(0, ng * dg, 0.5))
    >>> p.plot(x[1:], y)

    """

    topology = traj.top
    hist_3d = np.zeros((ng, ng, ng), np.float64)
    atom_id = topology.select("resSeq %s and name %s and chainid %s" % (res_id, atom_name, chain_id))[0]
    r0_id = topology.select("resSeq %s and name %s and chainid %s" % (r0_res, r0_atom_name, r0_chain))[0]
    r0 = traj.xyz[:, r0_id, :] * 10.0
    coords = traj.xyz[:, atom_id, :] * 10.0 - r0
    coords_i = (coords / dg).astype(np.int32)
    npm = (ng - 1) / 2

    coords_i += npm
    for c in coords_i:
        ix, iy, iz = c
        if ix < ng and iy < ng and iz < ng:
            hist_3d[ix, iy, iz] += 0.005
    r0 = r0.mean(axis=0)
    return hist_3d, r0


def calc_distance_from_traj(traj, res_id, atom_name, chain_id, ng, dg, r0_res, r0_atom_name, r0_chain, dt):
    """

    :param density:
    :param traj:
    :param atom_i:
    :param r0: xyz coordinates of attachment
    :param dg: grid spacing
    :return:

     Example
     -------

     >>> import mdtraj as md
     >>> import pylab as p
     >>> import mfm
     >>> from mfm.fluorescence.fps.static import calc_av1_py, calc_weights_from_traj
     >>> traj = md.load('e:/simulations_free_dye/t_join.h5')

     >>> res_id = 3  # the chromophore
     >>> atom_name = "O91"
     >>> chain_id = 0

     >>> r0_atom_name = "S1"
     >>> r0_res = 1
     >>> r0_chain = 0

     >>> ng = 50
     >>> dg = 1.0
     >>> hist_3d, r0 = calc_weights_from_traj(traj, res_id, atom_name, chain_id, ng, dg, r0_res, r0_atom_name, r0_chain)
     >>> offset = (ng - 1) / 2 * dg
     >>> mfm.io.pdb.write_open_dx("c:/temp/test.dx", hist_3d, r0 - offset, ng, ng, ng, dg, dg, dg)

    >>> pdb_filename = 'e:/simulations_free_dye/peptide.pdb'
    >>> structure = mfm.structure.Structure(pdb_filename)
    >>> av = calc_av1_py(l=23.0, w=2.0, r=3.5, atom_i=10, ng=ng, xyz=structure.xyz, vdw=structure.vdw)
    >>> y, x = np.histogram(av[1].flatten(), weights=hist_3d.flatten(), bins=np.arange(0, ng * dg, 0.5))
    >>> p.plot(x[2:], y[1:])

    """

    # Calculate the distance from the attachment point for every frame
    topology = traj.top
    atom_id = topology.select("resSeq %s and name %s and chainid %s" % (res_id, atom_name, chain_id))[0]
    r0_id = topology.select("resSeq %s and name %s and chainid %s" % (r0_res, r0_atom_name, r0_chain))[0]
    r0 = traj.xyz[:, r0_id, :] * 10.0
    coords = traj.xyz[:, atom_id, :] * 10.0 - r0
    distance_attach = np.linalg.norm(coords, ord=2, axis=1)

    # calculate the speed of leaving each of the trajectory
    v_distance_attach = np.diff(distance_attach, n=1) / dt

    #
    distance_bins = np.arange(0, 25, 0.2)
    d_array = np.zeros(distance_bins.shape[0], dtype=np.float32)
    occupancy_array = np.ones_like(d_array)
    idx = np.digitize(distance_attach, distance_bins)
    for j, di in enumerate(idx[1:]):
        d_array[di] += v_distance_attach[j] ** 2. * dt
        occupancy_array[di] += 1
    avg_d = 1. / 3. * (d_array / occupancy_array)


    diffusion_coefficients = np.zeros_like(distance_bins)
    for j in range(distance_bins.shape[0]):
        diffusion_coefficients[j] = np.correlate(occupancy[:, j], occupancy[:, j]) * dt
    diffusion_coefficients /= 3.

    coords_i += npm
    for c in coords_i:
        ix, iy, iz = c
        if ix < ng and iy < ng and iz < ng:
            hist_3d[ix, iy, iz] += 0.005
    r0 = r0.mean(axis=0)
    return hist_3d, r0

