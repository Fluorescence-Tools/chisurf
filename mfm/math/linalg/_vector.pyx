import numpy as np
cimport numpy as np
from libc.math cimport exp, ceil, acos, acos, sqrt, atan2, sin, cos
cimport cython
from cython.parallel import parallel, prange


@cython.boundscheck(False)
cdef inline double norm3c(double[:] v):
    cdef double s = 0.0
    cdef int k
    for k in xrange(3):
        s += v[k]*v[k]
    return sqrt(s)

@cython.boundscheck(False)
def norm3(double[:] v):
    """
    Euclidean  norm of a vector

    :param v: 1D numpy-array
    :return: normalized numpy-array
    """
    return norm3c(v)


@cython.boundscheck(False)
cdef inline double distc(double[:] u, double[:] v):
    cdef int i
    cdef double d2
    d2 = 0.0
    for i in range(3):
        d2 += (u[i]-v[i])*(u[i]-v[i])
    return sqrt(d2)


@cython.boundscheck(False)
def dist(double[:] u, double[:] v):
    """
    Distance between two vectors

    :param u:
    :param v:
    :return:
    """
    return distc(u, v)

@cython.boundscheck(False)
cdef inline double sq_distc(double[:] u, double[:] v):
    cdef int i
    cdef float d2
    d2 = 0.0
    for i in range(u.shape[0]):
        d2 += (u[i]-v[i])*(u[i]-v[i])
    return d2

@cython.boundscheck(False)
def sq_dist(double[:] u, double[:] v):
    """
    Squared distance between two vectors
    :param u:
    :param v:
    :return:
    """
    return sq_distc(u, v)

@cython.boundscheck(False)
cdef inline double dot3c(double[:] a, double[:] b):
    """
    :param a: 1D numpy-array
    :param b: 1D numpy-array
    :return: dot-product of the two numpy arrays
    """
    cdef double s = 0.0
    s += a[0]*b[0]
    s += a[1]*b[1]
    s += a[2]*b[2]
    return s

@cython.boundscheck(False)
def dot3(double[:] a, double[:] b):
    """
    Vector product of two 3D-vectors.

    :param a: 1D numpy-array
    :param b: 1D numpy-array
    :return: dot-product of the two numpy arrays
    """
    return dot3c(a, b)

@cython.boundscheck(False)
cdef inline cross3c(double[:] a, double[:] b):
    cdef np.ndarray[np.float64_t, ndim=1] o = np.empty(3, dtype=np.float64)
    o[0] = a[1]*b[2]-a[2]*b[1]
    o[1] = a[2]*b[0]-a[0]*b[2]
    o[2] = a[0]*b[1]-a[1]*b[0]
    return o

@cython.boundscheck(False)
def cross(double[:] a, double[:] b):
    """
    Cross-product of two vectors

    :param a: numpy array of length 3
    :param b: numpy array of length 3
    :return: cross-product of a and b
    """
    return cross3c(a, b)

@cython.boundscheck(False)
cdef inline np.ndarray sub3c(double[:] a, double[:] b):
    cdef np.ndarray[np.float64_t, ndim=1] o = np.empty(a.shape[0], dtype=np.float64)
    o[0] = a[0]-b[0]
    o[1] = a[1]-b[1]
    o[2] = a[2]-b[2]
    return o

@cython.boundscheck(False)
def sub3(double[:] a, double[:] b):
    """
    Subtract two 3D vectors

    :param a:
    :param b:
    :return:
    """
    return sub3c(a, b)

@cython.boundscheck(False)
cdef inline np.ndarray add3c(double[:] a, double[:] b):
    cdef np.ndarray[np.float64_t, ndim=1] o = np.empty(a.shape[0], dtype=np.float64)
    o[0] = a[0]+b[0]
    o[1] = a[1]+b[1]
    o[2] = a[2]+b[2]
    return o

@cython.boundscheck(False)
cdef add3(double[:] a, double[:] b):
    """
    Add two 3D-vectors

    :param a:
    :param b:
    :return:
    """
    return add3c(a, b)

@cython.boundscheck(False)
@cython.cdivision(True)
def angle(double[:] a, double[:] b, double[:] c):
    """
    The angle between three vectors

    :param a: numpy array
    :param b: numpy array
    :param c: numpy array
    :return: angle between three vectors/points in space
    """
    cdef double r12n, r23n
    r12 = sub3c(a, b)
    r23 = sub3c(c, b)
    r12n = norm3c(r12)
    r23n = norm3c(r23)
    return acos(dot3c(r12,r23) / (r12n*r23n) )

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def dihedral(double[:] v1, double[:] v2,
              double[:] v3, double[:] v4):
    """
    Dihedral angle between four-vectors

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

    :param v1: numpy array
    :param v2: numpy array
    :param v3: numpy array
    :param v4: numpy array
    :return: dihedral angle between four vectors
    """
    # TODO: also calculate angle between vectors here: speed up calculations
    cdef double phi, n1_inv, n2_inv, m1_inv
    b1 = sub3c(v1, v2)
    b2 = sub3c(v2, v3)
    b3 = sub3c(v3, v4)
    n1 = cross3c(b1, b2)
    n2 = cross3c(b2, b3)
    m1 = cross3c(b2, n1)
    n1_inv = 1.0 / norm3c(n1)
    n2_inv = 1.0 / norm3c(n2)
    m1_inv = 1.0 / norm3c(m1)

    cos_phi = dot3c(n1,n2)*(n1_inv * n2_inv)
    sin_phi = dot3c(m1,n2)*(m1_inv * n2_inv)
    if cos_phi < -1: cos_phi = -1
    if cos_phi >  1: cos_phi =  1
    if sin_phi < -1: sin_phi = -1
    if sin_phi >  1: sin_phi =  1
    phi= -atan2(sin_phi,cos_phi)
    return phi


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def solve_richardson_lucy(np.ndarray[ndim=2, dtype=np.float64_t] p, double[:] u, double[:] d, int max_iter):
    cdef int n_i, n_j, i, j, k, iteration
    cdef double s

    n_i = p.shape[0]
    n_j = p.shape[1]
    cdef np.ndarray[ndim=1, dtype=np.float64_t] un = np.copy(u)
    cdef np.ndarray[ndim=1, dtype=np.float64_t] c = np.zeros(n_i)

    for iteration in range(max_iter):

        for i in prange(n_i, nogil=True):
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


@cython.boundscheck(False)
@cython.wraparound(False)
def euler_matrix(double psi, double theta, double phi, approx=False):
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
    cdef double sin_psi, sin_theta, sin_phi
    cdef double cos_psi, cos_theta, cos_phi

    if approx is False:
        sin_psi, sin_theta, sin_phi = sin(psi), sin(theta), sin(phi)
        cos_psi, cos_theta, cos_phi = cos(psi), cos(theta), cos(phi)
    else:
        sin_psi, sin_theta, sin_phi = psi, theta, phi
        cos_psi, cos_theta, cos_phi = 1-abs(psi), 1-abs(theta), 1-abs(phi)

    cdef np.ndarray[dtype=np.float64_t, ndim=2] m = np.identity(3)

    m[0, 0] =  cos_theta * cos_psi
    m[0, 1] =  cos_theta * sin_psi
    m[0, 2] = -sin_theta

    m[1, 0] = sin_phi * sin_theta * cos_psi - cos_phi * sin_psi
    m[1, 1] = sin_phi * sin_theta * sin_psi + cos_phi * cos_psi
    m[1, 2] = sin_phi * cos_theta

    m[2, 0] = cos_phi * sin_theta * cos_psi + sin_phi * sin_psi
    m[2, 1] = cos_phi * sin_theta * sin_psi - sin_phi * cos_psi
    m[2, 2] = cos_phi * cos_theta
    return m

cdef inline vector4_norm(v):
    """Normalize a vector of length 4
    (inplace operation, vector is overwritten)
    """
    cdef double s = 0.0
    s = sqrt(v[0]**2 + v[1]**2 + v[2]**2 + v[3]**2)
    v[0] /= s
    v[1] /= s
    v[2] /= s
    v[3] /= s

cdef inline quaternion_about_axis(double angle, double[:] axis):
    """Return quaternion for rotation about axis.
    """
    cdef double[:] q = np.array([0.0, axis[0], axis[1], axis[2]], dtype=np.float64)
    vector4_norm(q)
    q[1] *= sin(angle/2.0)
    q[2] *= sin(angle/2.0)
    q[3] *= sin(angle/2.0)
    q[0] = cos(angle/2.0)
    return q

cdef inline quaternion_multiply(double[:] quaternion0, double[:] quaternion1):
    """Multiply quaternion1 to quaternion0
    (inplace, quaternion0 is modified)
    """
    cdef double w0, x0, y0, z0
    cdef double w1, x1, y1, z1

    w0, x0, y0, z0 = quaternion0
    w1, x1, y1, z1 = quaternion1

    quaternion0[0] = -x1*x0 - y1*y0 - z1*z0 + w1*w0
    quaternion0[1] = x1*w0 + y1*z0 - z1*y0 + w1*x0
    quaternion0[2] = -x1*z0 + y1*w0 + z1*x0 + w1*y0
    quaternion0[3] = x1*y0 - y1*x0 + z1*w0 + w1*z0

cdef inline rotate_point(double[:] p3, double[:] quaternion):
    """
    """
    cdef double[:] v = quaternion[1:3]
    cdef double w = quaternion[0]

    cdef double[:] vCp3 = cross3c(v, p3)
    cdef double[:] vCvCp3 = cross3c(v, vCp3) * 2

    cdef double[:] p_new = np.zeros(3, dtype=np.float64)
    p_new[0] = p3[0] + vCp3[0]*(2*w) + vCvCp3[0]
    p_new[1] = p3[1] + vCp3[1]*(2*w) + vCvCp3[1]
    p_new[2] = p3[2] + vCp3[2]*(2*w) + vCvCp3[2]

    return p_new
