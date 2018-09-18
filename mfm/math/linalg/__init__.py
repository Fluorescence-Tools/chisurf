import numpy as np
import _vector
import numba as nb


def cartesian(arrays, out=None):
    """Generate a cartesian product of input arrays.

    :param arrays: list of arrays
        1-D arrays to form the cartesian product of.
    :param out: 2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.
    :return: 2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.

    Examples
    --------

    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])

    """

    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = n / arrays[0].size
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m,1:])
        for j in range(1, arrays[0].size):
            out[j*m:(j+1)*m,1:] = out[0:m,1:]
    return out


def angle(a, b, c):
    """
    The angle between three vectors

    :param a: numpy array
    :param b: numpy array
    :param c: numpy array
    :return: angle between three vectors/points in space
    """
    return _vector.angle(a, b, c)


def dihedral(v1, v2, v3, v4):
    """
    Dihedral angle between four-vectors

    :param v1:
    :param v2:
    :param v3:
    :param v4:

    :return: dihedral angle between four vectors
    """
    return _vector.dihedral(v1, v2, v3, v4)


def sub(a, b):
    """
    Subtract two 3D-vectors
    :param a:
    :param b:
    :return:
    """
    return _vector.sub3(a, b)


def add(a, b):
    """
    Add two 3D-vectors
    :param a:
    :param b:
    :return:
    """
    return _vector.add3(a, b)


def dot(a, b):
    """
    Vector product of two 3D-vectors.

    :param a: 1D numpy-array
    :param b: 1D numpy-array
    :return: dot-product of the two numpy arrays
    """
    return _vector.dot3(a, b)


def dist(u, v):
    """
    Distance between two vectors

    :param u:
    :param v:
    :return:
    """
    return _vector.dist(u, v)


def dist2(u, v):
    """
    Squared distance between two vectors
    :param u:
    :param v:
    :return:
    """
    return _vector.sq_dist(u, v)


def norm(v):
    """
    Euclidean  norm of a vector

    :param v: 1D numpy-array
    :return: normalized numpy-array
    """
    return _vector.norm3(v)


def cross(a, b):
    """
    Cross-product of two vectors

    :param a: numpy array of length 3
    :param b: numpy array of length 3
    :return: cross-product of a and b
    """
    return _vector.cross(a, b)


@nb.jit(nopython=True, nogil=True)
def grad3d(d, b=None, dg=1.0):
    """Calculates the gradient of a 3D scalar field within a certain shape defined the bounds.

    :param d: 3D scalar field
    :param bd: 3D bounds (1 within structure, 0 outside)
    :param dg: grid spacing
    :return: 3D vector field as numpy-array (first axis defines x, y, z (0, 1, 2))
    """
    if b is None:
        bd = np.ones_like(d)
    else:
        bd = b
    nx, ny, nz = d.shape
    dd = np.zeros((3, nx, ny, nz))
    i2dg = 1. / (2. * dg)
    for ix in range(1, nx - 1):
        for iy in range(1, ny - 1):
            for iz in range(1, nz - 1):
                if bd[ix, iy, iz] == 0:
                    continue
                dd[0, ix, iy, iz] = 0.5 * (d[ix - 1, iy, iz] - d[ix + 1, iy, iz]) * i2dg
                dd[1, ix, iy, iz] = 0.5 * (d[ix, iy - 1, iz] - d[ix, iy + 1, iz]) * i2dg
                dd[2, ix, iy, iz] = 0.5 * (d[ix, iy, iz - 1] - d[ix, iy, iz + 1]) * i2dg
    return dd


@nb.jit(nopython=True, nogil=True)
def laplace3d_1(c, b=None, dg=1.0):
    """Calculates the Laplacian of a 3D scalar field within a certain shape defined the bounds.

    :param c:
    :param b:
    :param dg:
    :return:
    """
    if b is None:
        b = np.ones_like(c)
    nx, ny, nz = c.shape
    l = np.zeros((3, nx, ny, nz), dtype=np.float64)
    nx, ny, nz = c.shape
    idg2 = 1. / (dg**2)
    for ix in range(1, nx - 1):
        for iy in range(1, ny - 1):
            for iz in range(1, nz - 1):
                if b[ix, iy, iz] == 0:
                    continue
                di = c[ix, iy, iz]
                l[0, ix, iy, iz] = (c[ix + 1, iy, iz] - 2 * di + c[ix - 1, iy, iz]) * idg2
                l[1, ix, iy, iz] = (c[ix, iy + 1, iz] - 2 * di + c[ix, iy - 1, iz]) * idg2
                l[2, ix, iy, iz] = (c[ix, iy, iz + 1] - 2 * di + c[ix, iy, iz - 1]) * idg2
    return l


