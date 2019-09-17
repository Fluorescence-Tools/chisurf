import numpy as np
cimport numpy as np
cimport cython
from libc.stdlib cimport malloc, free
from libc.math cimport sqrt, cos, sin
from libc.stdint cimport int32_t, uint32_t
from cython.parallel import prange


cdef inline double dist3c2(double r1x, double r1y, double r1z, double r2x, double r2y, double r2z) nogil:
    return (r1x - r2x)*(r1x - r2x) + (r1y - r2y)*(r1y - r2y) + (r1z - r2z)*(r1z - r2z)


cdef extern from "mtrandom.h":
    cdef cppclass MTrandoms:
        void seedMT()
        double random0i1i() nogil
        double random0i1e() nogil


cdef MTrandoms rmt
rmt.seedMT()



@cython.cdivision(True)
@cython.boundscheck(False)
cdef inline c_random_distances(
        av1,
        av2,
        double* distances, uint32_t nSamples):

    cdef uint32_t i, i1, i2
    cdef int32_t lp1, lp2

    cdef double[:, :] p1 = av1.points
    cdef double[:, :] p2 = av2.points

    lp1 = p1.shape[0]
    lp2 = p2.shape[0]

    for i in prange(nSamples, nogil=True):
        i1 = <uint32_t>(rmt.random0i1e() * lp1)
        i2 = <uint32_t>(rmt.random0i1e() * lp2)
        distances[i] = sqrt(
            (p1[i1, 0] - p2[i2, 0]) * (p1[i1, 0] - p2[i2, 0]) + \
            (p1[i1, 1] - p2[i2, 1]) * (p1[i1, 1] - p2[i2, 1]) + \
            (p1[i1, 2] - p2[i2, 2]) * (p1[i1, 2] - p2[i2, 2])
        )


def random_distances(av1, av2, uint32_t nSamples=10000):
    cdef np.ndarray[ndim=1, dtype=np.float64_t] dist = np.empty(nSamples, dtype=np.float64)
    c_random_distances(av1, av2, <double*> dist.data, nSamples)
    return dist


@cython.boundscheck(False)
cdef inline rotate_translate_vector(
        double psi,
        double theta,
        double phi,
        double[:] vec,
        double[:] trans
):
    """This rotates a vector according to the euler angles psi, theta and phi followed by a translation.
    (This function modifies the vector vec)

    Here the convention of the Euler-angles is the z, y', x convention DIN 9300. This is only valid for
    small angles as the sin and cos are approximated by first order.

    :param psi: double
    :param theta: double
    :param phi: double
    :param vec: numpy array
    :param trans: numpy array

    """
    cdef double sin_psi, sin_theta, sin_phi
    cdef double cos_psi, cos_theta, cos_phi
    cdef double m00, m01, m02, m10, m11, m12, m20, m21, m22
    cdef double v0, v1, v2

    #sin_psi, sin_theta, sin_phi = psi, theta, phi
    #cos_psi, cos_theta, cos_phi = 1-psi * psi, 1 -theta * theta, 1 - phi * phi
    sin_psi, sin_theta, sin_phi = sin(psi), sin(theta), sin(phi)
    cos_psi, cos_theta, cos_phi = cos(psi), cos(theta), cos(phi)

    m00 =  cos_theta * cos_psi
    m01 =  cos_theta * sin_psi
    m02 = -sin_theta

    m10 = sin_phi * sin_theta * cos_psi - cos_phi * sin_psi
    m11 = sin_phi * sin_theta * sin_psi + cos_phi * cos_psi
    m12 = sin_phi * cos_theta

    m20 = cos_phi * sin_theta * cos_psi + sin_phi * sin_psi
    m21 = cos_phi * sin_theta * sin_psi - sin_phi * cos_psi
    m22 = cos_phi * cos_theta

    v0 = m00 * vec[0] + m01 * vec[1] + m02 * vec[2]
    v1 = m10 * vec[0] + m11 * vec[1] + m12 * vec[2]
    v2 = m20 * vec[0] + m21 * vec[1] + m22 * vec[2]

    vec[0] = v0 + trans[0]
    vec[1] = v1 + trans[1]
    vec[2] = v2 + trans[2]



@cython.cdivision(True)
@cython.boundscheck(False)
def make_subav(float[:,:,:] density, int32_t ng, double dg, double[:] radius, double[:, :] rs, double[:] r0, int32_t n_slow_center):
    """
    density: density of the accessible volume dimension ng, ng, ng uint8 numpy array as obtained of fps
    dg: grid resolution of the density grid
    slow_radius: radius around the list of points. all points within a radius of r0 around the points in rs are part
    of the subav. each slow point is assiciated to one slow-radius
    rs: list of points (x,y,z) defining the subav
    r0: is the position of the accessible volume
    """
    cdef int32_t ix0, iy0, iz0
    cdef int32_t ix, iy, iz
    cdef int32_t radius_idx, isa
    cdef char overlapped

    # iterate through all possible grid points in the density
    for ix in prange(ng, nogil=True):
        for iy in range(ng):
            for iz in range(ng):
                if density[ix, iy, iz] <= 0:
                    continue
                # count the overlaps with slow-centers
                overlapped = 0

                for isa in range(n_slow_center):
                    ix0 = <int>((rs[isa, 0]-r0[0])/dg) + (ng - 1)/2
                    iy0 = <int>((rs[isa, 1]-r0[1])/dg) + (ng - 1)/2
                    iz0 = <int>((rs[isa, 2]-r0[2])/dg) + (ng - 1)/2
                    radius_idx = <int>(radius[isa] / dg)
                    if ((ix - ix0)**2 + (iy - iy0)**2 + (iz - iz0)**2) < radius_idx**2:
                        overlapped = 1
                        break

                if overlapped > 0:
                    density[ix, iy, iz] = 1
                else:
                    density[ix, iy, iz] = 0


@cython.cdivision(True)
@cython.boundscheck(False)
def modify_av(float[:,:,:] density, int32_t ng, double dg, double[:] radius, double[:, :] rs, double[:] r0,
               int32_t n_slow_center, float slow_factor):
    """
    Multiplies density by factor if within radius

    density: density of the accessible volume dimension ng, ng, ng uint8 numpy array as obtained of fps
    dg: grid resolution of the density grid
    slow_radius: radius around the list of points. all points within a radius of r0 around the points in rs are part
    of the subav. each slow point is assiciated to one slow-radius
    rs: list of points (x,y,z) defining the subav
    r0: is the position of the accessible volume
    """
    cdef int32_t ix0, iy0, iz0
    cdef int32_t ix, iy, iz
    cdef int32_t radius_idx, isa

    # iterate through all possible grid points in the density
    for ix in prange(ng, nogil=True):
        for iy in range(ng):
            for iz in range(ng):
                if density[ix, iy, iz] <= 0:
                    continue

                for isa in range(n_slow_center):
                    ix0 = <int>((rs[isa, 0]-r0[0])/dg) + (ng - 1)/2
                    iy0 = <int>((rs[isa, 1]-r0[1])/dg) + (ng - 1)/2
                    iz0 = <int>((rs[isa, 2]-r0[2])/dg) + (ng - 1)/2
                    radius_idx = <int>(radius[isa] / dg)
                    if ((ix - ix0)**2 + (iy - iy0)**2 + (iz - iz0)**2) < radius_idx**2:
                        density[ix, iy, iz] *= slow_factor
                        break

@cython.cdivision(True)
@cython.boundscheck(False)
def reset_density_av(float[:,:,:] density, int32_t ng):
    """Sets all densities in av to 1.0 if the density is bigger than 0.0

    :param density: numpy-array
    :param ng:
    :return:
    """
    cdef int32_t ix0, iy0, iz0
    cdef int32_t ix, iy, iz
    cdef int32_t radius_idx, isa

    # iterate through all possible grid points in the density
    for ix in prange(ng, nogil=True):
        for iy in range(ng):
            for iz in range(ng):
                density[ix, iy, iz] = 0.0 if density[ix, iy, iz] <= 0 else 1.0


@cython.cdivision(True)
@cython.boundscheck(False)
def simulate_traj_point(np.ndarray[dtype=np.float32_t, ndim=3] d, np.ndarray[dtype=np.float32_t, ndim=3] ds, double dg,
                        double t_max, double t_step, double diffusion_coefficient, double slow_fact):
    """
    `d` density of whole av in shape ng, ng, ng (as generated by fps library)
    `ds` density_slow of slow av (has to be same shape as fast av only different occupancy)
    dimensions = 2;         % two dimensional simulation
    tau = .1;               % time interval in seconds
    time = tau * 1:N;       % create a time vector for plotting

    k = sqrt(D * dimensions * tau);

    http://labs.physics.berkeley.edu/mediawiki/index.php/Simulating_Brownian_Motion
    """
    cdef uint32_t n_accepted, n_rejected, i_accepted
    cdef uint32_t i, x_idx, y_idx, z_idx
    cdef uint32_t n_samples = int(t_max/t_step)
    cdef uint32_t ng = d.shape[0]

    sigma = np.sqrt(diffusion_coefficient * 3 * t_step) / dg
    cdef np.ndarray[dtype=np.float32_t, ndim=2] pos = np.zeros([n_samples, 3], dtype=np.float32)
    cdef char[:] accepted = np.zeros(n_samples, dtype=np.uint8)
    cdef double[:, :] r = np.random.normal(loc=0, scale=sigma, size=(n_samples, 3))
    slow_fact = np.sqrt(slow_fact)

    # find random point with density > 0 in density_av
    # take random points and use lookup-table whether point is really
    # within the accessible volume
    while True:
        op = np.array(np.where(d > 0))
        rnd_idx = np.random.randint(0, op[0].shape[0], 3)
        pos[0, 0] = op[0, rnd_idx[0]]
        pos[0, 1] = op[1, rnd_idx[1]]
        pos[0, 2] = op[2, rnd_idx[2]]

        x_idx = <uint32_t>(pos[0, 0])
        y_idx = <uint32_t>(pos[0, 1])
        z_idx = <uint32_t>(pos[0, 2])

        if d[x_idx, y_idx, z_idx] > 0:
            break

    # Diffusion
    n_accepted = 1
    n_rejected = 0
    i_accepted = 0
    for i in range(n_samples):
        x_idx = <uint32_t>(pos[i_accepted, 0])
        y_idx = <uint32_t>(pos[i_accepted, 1])
        z_idx = <uint32_t>(pos[i_accepted, 2])

        if ds[x_idx, y_idx, z_idx] > 0:
            r[i, 0] *= slow_fact
            r[i, 1] *= slow_fact
            r[i, 2] *= slow_fact

        pos[i, 0] = pos[i_accepted, 0] + r[i, 0]
        pos[i, 1] = pos[i_accepted, 1] + r[i, 1]
        pos[i, 2] = pos[i_accepted, 2] + r[i, 2]

        x_idx = <uint32_t>(pos[i, 0])
        y_idx = <uint32_t>(pos[i, 1])
        z_idx = <uint32_t>(pos[i, 2])

        if 0 < x_idx < ng and 0 < y_idx < ng and 0 < z_idx < ng:
            if d[x_idx, y_idx, z_idx] > 0:
                i_accepted = i
                n_accepted += 1
                accepted[i] = True
            else:
                n_rejected += 1
                accepted[i] = False
    return (pos - (ng - 1) / 2) * dg, accepted, n_accepted, n_rejected


@cython.cdivision(True)
@cython.boundscheck(False)
def simulate_traj_points(np.ndarray[dtype=np.float32_t, ndim=3] av_d,
                         np.ndarray[dtype=np.float32_t, ndim=3] s_av_d, double dg,
                         double t_max, double t_step,
                         double d_trans, double d_rot_phi, double d_rot_theta, double d_rot_psi, double slow_fact,
                         np.ndarray[dtype=np.float64_t, ndim=1] a, np.ndarray[dtype=np.float64_t, ndim=1] b,
                         verbose=True):
    """Simulates the diffusion of a dipole within an accessible volume including the dipole rotation.

    Here the dye is approximated by three points (the volumetric aspects should be already fulfilled due to the
    accessible volumes). The geometry is defined by the points `a` and `b`. The third points is by default (0,0,0).
    The dipole is defined by two coordinates `a` and `b`. Only steps during the simulation are allowed in which
    all three-points are within the accessible volume.

    :param a: numpy-array
        definition of the dye-geometry (dipole)
    :param b: numpy-array
        definition of the the dye-geometry (dipole)
    :param av_d: numpy-array
        density of whole av in shape ng, ng, ng (as generated by fps library)
    :param s_av_d: numpy-array
        density of the slow av shape (ng, ng, ng)
    :param dg: double
        grid size-parameter
    :param t_max: double
        maximum simulation time
    :param t_step: double
        time-step
    :param d_trans: double
        translational-diffusion coefficient
    :param d_rot_phi: double
        rotational-diffusion coefficient
    :param d_rot_theta: double
        rotational-diffusion coefficient
    :param d_rot_psi: double
        rotational-diffusion coefficient
    :param slow_fact: double
        the translational diffusion is slowed down by this factor if the dye is within a slow part of the
        volume

    http://labs.physics.berkeley.edu/mediawiki/index.php/Simulating_Brownian_Motion

    Examples
    --------

    First import the libraries

    >>> import numpy as np
    >>> import mfm

    Now make a new structure
    >>> pdb_filename = './sample_data/model/hgbp1/hGBP1_closed.pdb'
    >>> structure = mfm.Structure(pdb_filename)

    Calculate an accessible volume at an given attachment point
    >>> residue_number = 18
    >>> atom_name = 'CB'
    >>> av = mfm.fps.AV(structure, residue_seq_number=residue_number, atom_name=atom_name)

    Calcualte a slow part of the accessible volume
    >>> slow_point = structure.residue_dict[18]['CB']['coord'].reshape([1,3])
    >>> av.calc_slow_av(slow_centers=slow_point, slow_radius=5)

    Define the dye geometry. Here the dye geometry is determined by three points.
    A, B and C. Only the points A and B have to be determined as C is at the origin of the
    cartesian coordinate system C=(0,0,0).
    >>> a = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    >>> b = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    >>> abc = mfm.fps.fps.simulate_traj_dipole(av.density_fast, av.density_slow, av.dg, t_max=1000.0, t_step=0.01, d_trans=50.0, d_rot_phi=0.1, d_rot_theta=0.1, d_rot_psi=0.1, slow_fact=0.05, a=a, b=b, verbose=True)

    >>> import pylab as p
    >>> p.plot(abc[0,:,1])
    >>> p.plot(abc[1,:,1])
    >>> p.plot(abc[2,:,1])
    >>> p.show()

    """

    # Internally the grid is calculated in units of dg.
    # Therefore the dye-definition is recalculated.
    a[0] /= dg
    a[1] /= dg
    a[2] /= dg

    b[0] /= dg
    b[1] /= dg
    b[2] /= dg

    # running index (the number of samples)
    cdef int32_t i

    # The displacements in x,y,z
    cdef double tx, ty, tz

    # current position of the atoms
    cdef double[:] va = np.zeros(3, dtype=np.float64)
    cdef double[:] vb = np.zeros(3, dtype=np.float64)
    cdef double[:] vc = np.zeros(3, dtype=np.float64)

    # The indices of next to the three points defining the dye
    cdef int32_t ax_idx, ay_idx, az_idx
    cdef int32_t bx_idx, by_idx, bz_idx
    cdef int32_t cx_idx, cy_idx, cz_idx

    # angles to calculate position of atoms
    cdef double phi, psi, theta

    # The number of grid-points of the av in one direction (x,y,z)
    cdef int32_t ng = av_d.shape[0]

    # The number of samples is determined by the maximum-time and the time-step
    cdef int32_t n_samples = <int32_t>(t_max/t_step)

    # The diffusion-coefficient defines the step-width of the random-vector
    sigma_trans = sqrt(2 * d_trans * 3 * t_step) * dg
    sigma_phi = sqrt(d_rot_phi * t_step)
    sigma_psi = sqrt(d_rot_psi * t_step)
    sigma_theta = sqrt(d_rot_theta * t_step)

    # coordinates of the dye the points a, b, c
    cdef np.ndarray[dtype=np.float64_t, ndim=3] pos = np.empty([3, n_samples, 3], dtype=np.float64)

    # the random displacement-vector of the translation and the rotation
    cdef double[:, :] r_trans = np.random.normal(loc=0, scale=sigma_trans, size=(n_samples, 3))
    cdef double[:] r_phi   = np.random.normal(loc=0, scale=sigma_phi, size=n_samples)
    cdef double[:] r_psi   = np.random.normal(loc=0, scale=sigma_psi, size=n_samples)
    cdef double[:] r_theta = np.random.normal(loc=0, scale=sigma_theta, size=n_samples)
    slow_fact = sqrt(slow_fact)

    # find random point with density > 0 in density_av for all three points
    if verbose:
        print("Starting search for initial point...")
    phi, psi, theta = 0.0, 0.0, 0.0
    while True:
        # take random point
        op = np.array(np.where(av_d > 0))
        rnd_idx = np.random.randint(0, op[0].shape[0], 3)

        vc[0] = op[0, rnd_idx[0]]
        vc[1] = op[1, rnd_idx[1]]
        vc[2] = op[2, rnd_idx[2]]
        cx_idx = <int32_t>(vc[0])
        cy_idx = <int32_t>(vc[1])
        cz_idx = <int32_t>(vc[2])

        if av_d[cx_idx, cy_idx, cz_idx] > 0:
            # found an initial point within the accessible volume
            # now looking for possible rotations so that the dye fits
            # give it about 500 trials than start over
            for i in range(500):
                theta = <double>(rmt.random0i1e() * 0.7853981633974483)
                phi   = <double>(rmt.random0i1e() * 6.283185307179586)
                psi   = <double>(rmt.random0i1e() * 6.283185307179586)

                va[0], va[1], va[2]= a[0], a[1], a[2]
                rotate_translate_vector(psi, theta, phi, va, vc)
                ax_idx = <int32_t>(va[0])
                ay_idx = <int32_t>(va[1])
                az_idx = <int32_t>(va[2])

                if 0 < ax_idx < ng and 0 < ay_idx < ng and 0 < az_idx < ng:
                    if av_d[ax_idx, ay_idx, az_idx] > 0:

                        vb[0], vb[1], vb[2]= b[0], b[1], b[2]
                        rotate_translate_vector(psi, theta, phi, vb, vc)
                        bx_idx = <int32_t>(vb[0])
                        by_idx = <int32_t>(vb[1])
                        bz_idx = <int32_t>(vb[2])

                        if 0 < bx_idx < ng and 0 < by_idx < ng and 0 < bz_idx < ng:
                            if av_d[bx_idx, by_idx, bz_idx] > 0:
                                break
                        else:
                            continue
                else:
                    continue

        if av_d[cx_idx, cy_idx, cz_idx] > 0 and av_d[ax_idx, ay_idx, az_idx] > 0 and av_d[bx_idx, by_idx, bz_idx] > 0:
            pos[0, 0, 0] = va[0]
            pos[0, 0, 1] = va[1]
            pos[0, 0, 2] = va[2]

            pos[1, 0, 0] = vb[0]
            pos[1, 0, 1] = vb[1]
            pos[1, 0, 2] = vb[2]

            pos[2, 0, 0] = vc[0]
            pos[2, 0, 1] = vc[1]
            pos[2, 0, 2] = vc[2]

            break
    if verbose:
        print("Initial points:")
        print("a: %.2f, %.2f, %.2f" % (va[0], va[1], va[2]))
        print("b: %.2f, %.2f, %.2f" % (vb[0], vb[1], vb[2]))
        print("c: %.2f, %.2f, %.2f" % (vc[0], vc[1], vc[2]))
        print("Starting MC trajectory...")
        print("Nsamples: %i" % n_samples)

    # Integration
    for i in range(1, n_samples):
        # By default the dye doesnt move
        pos[0, i, 0] = pos[0, i - 1, 0]
        pos[0, i, 1] = pos[0, i - 1, 1]
        pos[0, i, 2] = pos[0, i - 1, 2]

        pos[1, i, 0] = pos[1, i - 1, 0]
        pos[1, i, 1] = pos[1, i - 1, 1]
        pos[1, i, 2] = pos[1, i - 1, 2]

        pos[2, i, 0] = pos[2, i - 1, 0]
        pos[2, i, 1] = pos[2, i - 1, 1]
        pos[2, i, 2] = pos[2, i - 1, 2]

        # translational displacement
        # find the indices to adjust the displacement (is this position a slow position)
        ax_idx = <int32_t>(pos[0, i, 0])
        ay_idx = <int32_t>(pos[0, i, 1])
        az_idx = <int32_t>(pos[0, i, 2])

        bx_idx = <int32_t>(pos[1, i, 0])
        by_idx = <int32_t>(pos[1, i, 1])
        bz_idx = <int32_t>(pos[1, i, 2])

        cx_idx = <int32_t>(pos[2, i, 0])
        cy_idx = <int32_t>(pos[2, i, 1])
        cz_idx = <int32_t>(pos[2, i, 2])

        tx = r_trans[i, 0]
        ty = r_trans[i, 1]
        tz = r_trans[i, 2]

        if s_av_d[ax_idx, ay_idx, az_idx] > 0 or s_av_d[bx_idx, by_idx, bz_idx] > 0 or s_av_d[cx_idx, cy_idx, cz_idx] > 0:
            tx *= slow_fact
            ty *= slow_fact
            tz *= slow_fact

        # rotational displacement
        psi   += r_psi[i]
        theta += r_theta[i]
        phi   += r_phi[i]

        psi   %= 6.283185307179586
        phi   %= 6.283185307179586
        theta %= 0.7853981633974483

        ## Translational-displacement
        vc[0] = pos[2, i, 0] + tx
        vc[1] = pos[2, i, 1] + ty
        vc[2] = pos[2, i, 2] + tz

        va[0], va[1], va[2] = a[0], a[1], a[2]
        rotate_translate_vector(psi, theta, phi, va, vc)

        vb[0], vb[1], vb[2] = b[0], b[1], b[2]
        rotate_translate_vector(psi, theta, phi, vb, vc)

        # check if possible index
        ax_idx = <int32_t>(va[0])
        ay_idx = <int32_t>(va[1])
        az_idx = <int32_t>(va[2])
        if 0 < ax_idx < ng and 0 < ay_idx < ng and 0 < az_idx < ng:
            # check if av has some density
            if av_d[ax_idx, ay_idx, az_idx] > 0:

                # do the same for the other atoms
                bx_idx = <int32_t>(vb[0])
                by_idx = <int32_t>(vb[1])
                bz_idx = <int32_t>(vb[2])
                if 0 < bx_idx < ng and 0 < by_idx < ng and 0 < bz_idx < ng:

                    if av_d[bx_idx, by_idx, bz_idx] > 0:

                        cx_idx = <int32_t>(vc[0])
                        cy_idx = <int32_t>(vc[1])
                        cz_idx = <int32_t>(vc[2])
                        if 0 < cx_idx < ng and 0 < cy_idx < ng and 0 < cz_idx < ng:
                            if av_d[cx_idx, cy_idx, cz_idx] > 0:
                                # all atoms are within the av
                                # this is a acceptable move
                                pos[0, i, 0] = va[0]
                                pos[0, i, 1] = va[1]
                                pos[0, i, 2] = va[2]

                                pos[1, i, 0] = vb[0]
                                pos[1, i, 1] = vb[1]
                                pos[1, i, 2] = vb[2]

                                pos[2, i, 0] = vc[0]
                                pos[2, i, 1] = vc[1]
                                pos[2, i, 2] = vc[2]
                                continue

    return (pos - (ng - 1) / 2) * dg

