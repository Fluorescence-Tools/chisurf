from __future__ import annotations
from typing import List

from math import sin, cos, sqrt

import numba as nb
import numpy as np


def cartesian(
        arrays: List[np.array],
        out=None
):
    """Generate a cartesian product of input arrays.

    :param arrays: list of arrays
        1-D arrays to form the cartesian product of.
    :param out: 2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.
    :return: 2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.

    Examples
    --------

    >>> cartesian([[1, 2, 3], [4, 5], [6, 7]])
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

    m = n // arrays[0].size
    out[:, 0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m,1:])
        for j in range(1, arrays[0].size):
            out[j*m:(j+1)*m,1:] = out[0:m,1:]
    return out


@nb.jit
def angle(
        a: np.array,
        b: np.array,
        c: np.array):
    """
    The angle between three vectors

    :param a: numpy array
    :param b: numpy array
    :param c: numpy array
    :return: angle between three vectors/points in space

    Example
    -------

    >>> import numpy as np
    >>> a = np.array([0,0,0], dtype=np.float64)
    >>> b = np.array([1,0,0], dtype=np.float64)
    >>> c = np.array([0,1,0], dtype=np.float64)
    >>> angle(a, b, c) / np.pi * 360
    90.000000000000014

    """
    r12 = sub3(a, b)
    r23 = sub3(c, b)
    r12n = np.sqrt(dot3(r12, r12))
    r23n = np.sqrt(dot3(r23, r23))
    d = dot3(r12, r23) / (r12n * r23n)
    return np.arccos(d)


@nb.jit
def sq_dist3(
        u: np.array,
        v: np.array
):
    r = (u[0]-v[0])**2
    r += (u[1] - v[1]) ** 2
    r += (u[2] - v[2]) ** 2
    return r


@nb.jit
def cross3(
        a: np.array,
        b: np.array
):
    o = np.empty(3, dtype=np.float64)
    o[0] = a[1]*b[2]-a[2]*b[1]
    o[1] = a[2]*b[0]-a[0]*b[2]
    o[2] = a[0]*b[1]-a[1]*b[0]
    return o


@nb.jit
def norm3(
        a: np.array
):
    """The length of a 3D-vector

    :param a:
    :return:
    """
    return np.sqrt(a[0]**2 + a[1]**2 + a[2]**2)


@nb.jit
def dot3(
        a: np.array,
        b: np.array
):
    """Dot product of 2 3D-vectors

    :param a:
    :param b:
    :return:
    """
    s = 0.0
    s += a[0]*b[0]
    s += a[1]*b[1]
    s += a[2]*b[2]
    return s


@nb.jit()
def add3(
        a: np.array,
        b: np.array
):
    """Adds two 3D vectors

    :param a:
    :param b:
    :return:
    """
    o = np.empty(3, dtype=np.float64)
    o[0] = a[0]+b[0]
    o[1] = a[1]+b[1]
    o[2] = a[2]+b[2]
    return o


@nb.jit()
def sub3(
        a: np.array,
        b: np.array
):
    o = np.empty(3, dtype=np.float64)
    o[0] = a[0]-b[0]
    o[1] = a[1]-b[1]
    o[2] = a[2]-b[2]
    return o


@nb.jit()
def dist3(
        a: np.array,
        b: np.array
):
    d2 = (a[0] - b[0])**2
    d2 += (a[1] - b[1]) ** 2
    d2 += (a[2] - b[2]) ** 2
    return np.sqrt(d2)


@nb.jit()
def dihedral(
        v1: np.array,
        v2: np.array,
        v3: np.array,
        v4: np.array
):
    """Dihedral angle between four-vectors

    Given the coordinates of the four points, obtain the vectors b1, b2, and b3 by vector subtraction.
    Let me use the nonstandard notation <v> to denote v/|v|, the unit vector in the direction of the
    vector v. Compute n1=<b1xb2> and n2=<b2xb3>, the normal vectors to the planes containing b1 and b2,
    and b2 and b3 respectively. The angle we seek is the same as the angle between n1 and n2.

    The three vectors n1, <b2>, and m1:=n1x<b2> form an orthonormal frame. Compute the coordinates of
    n2 in this frame: x=n1*n2 and y=m1*n2. (You don't need to compute <b2>*n2 as it should always be zero.)

    The dihedral angle, with the correct sign, is atan2(y,x).

    (The reason I recommend the two-argument atan2 function to the traditional cos-1 in this case is both
    because it naturally produces an angle over a range of 2pi, and because cos-1 is poorly conditioned
    when the angle is close to 0 or +-pi.)
    :param v1:
    :param v2:
    :param v3:
    :param v4:

    :return: dihedral angle between four vectors

    Example
    -------
    >>> import numpy as np
    >>> a = np.array([-1,1,0], dtype=np.float64)
    >>> b = np.array([-1,0,0], dtype=np.float64)
    >>> c = np.array([0,0,0], dtype=np.float64)
    >>> d = np.array([0,-1,0], dtype=np.float64)
    >>> dihedral(a, b, c, d) / np.pi * 360
    -360

    """
    b1 = sub3(v1, v2)
    b2 = sub3(v2, v3)
    b3 = sub3(v3, v4)
    n1 = cross3(b1, b2)
    n2 = cross3(b2, b3)
    m1 = cross3(b2, n1)
    n1_inv = 1.0 / norm3(n1)
    n2_inv = 1.0 / norm3(n2)
    m1_inv = 1.0 / norm3(m1)

    cos_phi = dot3(n1, n2)*(n1_inv * n2_inv)
    sin_phi = dot3(m1, n2)*(m1_inv * n2_inv)
    if cos_phi < -1:
        cos_phi = -1
    elif cos_phi > 1:
        cos_phi = 1
    if sin_phi < -1:
        sin_phi = -1
    elif sin_phi > 1:
        sin_phi = 1
    phi = -np.arctan2(sin_phi, cos_phi)
    return phi


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


def solve_richardson_lucy(
        p: np.array,
        u: np.array,
        d: np.array,
        max_iter: int
) -> np.array:
    """

    :param p:
    :param u:
    :param d:
    :param max_iter:
    :return:
    """
    n_i = p.shape[0]
    n_j = p.shape[1]
    un = np.copy(u)
    c = np.zeros(n_i)

    for iteration in range(max_iter):

        for i in range(n_i):
            c[i] = 0.0
            for k in range(n_j):
                c[i] += p[i, k] * u[k]

        for j in range(n_j):
            s = 0.0
            for i in range(n_i):
                s += d[i] / c[i] * p[i, j]
            un[j] = u[j] * s
        for j in range(n_j):
            u[j] = un[j]
    return u


def euler_matrix(psi, theta, phi, approx=False):
    """Return homogeneous rotation matrix from Euler angles psi, theta and phi

    Here the Euler-angles are defined according to DIN 9300. For small angles the Trigonometric functions can be
    approximated by the first order if the parameter approx is True.

    :param psi: double
        yaw-angle
    :param theta: double
        pitch-angle
    :param phi: double
        yaw-angle
    :param approx: bool

    """

    if approx is False:
        sin_psi, sin_theta, sin_phi = sin(psi), sin(theta), sin(phi)
        cos_psi, cos_theta, cos_phi = cos(psi), cos(theta), cos(phi)
    else:
        sin_psi, sin_theta, sin_phi = psi, theta, phi
        cos_psi, cos_theta, cos_phi = 1-abs(psi), 1-abs(theta), 1-abs(phi)

    m = np.identity(3)

    m[0, 0] = cos_theta * cos_psi
    m[0, 1] = cos_theta * sin_psi
    m[0, 2] = -sin_theta

    m[1, 0] = sin_phi * sin_theta * cos_psi - cos_phi * sin_psi
    m[1, 1] = sin_phi * sin_theta * sin_psi + cos_phi * cos_psi
    m[1, 2] = sin_phi * cos_theta

    m[2, 0] = cos_phi * sin_theta * cos_psi + sin_phi * sin_psi
    m[2, 1] = cos_phi * sin_theta * sin_psi - sin_phi * cos_psi
    m[2, 2] = cos_phi * cos_theta
    return m


def vector4_norm(v):
    """Normalize a vector of length 4
    """
    # untested
    s = 0.0
    s = sqrt(v[0]**2 + v[1]**2 + v[2]**2 + v[3]**2)
    v[0] /= s
    v[1] /= s
    v[2] /= s
    v[3] /= s
    return v


def quaternion_about_axis(angle, axis):
    """Return quaternion for rotation about axis.
    """
    # untested
    q = np.array([0.0, axis[0], axis[1], axis[2]], dtype=np.float64)
    vector4_norm(q)
    q[1] *= sin(angle/2.0)
    q[2] *= sin(angle/2.0)
    q[3] *= sin(angle/2.0)
    q[0] = cos(angle/2.0)
    return q


def quaternion_multiply(quaternion0, quaternion1):
    """Multiply quaternion1 to quaternion0
    (inplace, quaternion0 is modified)
    """
    # untested
    w0, x0, y0, z0 = quaternion0
    w1, x1, y1, z1 = quaternion1

    quaternion0[0] = -x1*x0 - y1*y0 - z1*z0 + w1*w0
    quaternion0[1] = x1*w0 + y1*z0 - z1*y0 + w1*x0
    quaternion0[2] = -x1*z0 + y1*w0 + z1*x0 + w1*y0
    quaternion0[3] = x1*y0 - y1*x0 + z1*w0 + w1*z0
    return quaternion0


def rotate_point(p3, quaternion):
    # untested
    v = quaternion[1:3]
    w = quaternion[0]

    vCp3 = cross3(v, p3)
    vCvCp3 = cross3(v, vCp3) * 2

    p_new = np.zeros(3, dtype=np.float64)
    p_new[0] = p3[0] + vCp3[0]*(2*w) + vCvCp3[0]
    p_new[1] = p3[1] + vCp3[1]*(2*w) + vCvCp3[1]
    p_new[2] = p3[2] + vCp3[2]*(2*w) + vCvCp3[2]

    return p_new

