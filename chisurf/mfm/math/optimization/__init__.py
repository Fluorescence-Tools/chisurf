import numpy as np
import scipy.optimize

import leastsqbound
from mfm.math.optimization import fnnls
import mfm.math.linalg as la


def solve_nnls(A, b, reg=1e-5, x_max=1e12):
    """
    Solve :math:`argmin_x || Ax - b ||_2 for x>=0`. This is a wrapper for a FORTAN non-negative least squares solver (as
    in the SciPy. In addition to that the matrix A is regularized by a Thikonov-regularization. If values bigger
    than x_max are encountered in the solution they are simply replaced by zeros.

    :param A: numpy-array
        Matrix A as shown above.
    :param b: numpy-array
        Right-hand side vector.
    :param reg: float
        Regularization factor
    :param x_max: float
    :return:
    """
    right = np.dot(A, b)
    left = np.dot(A, A.T) + reg * np.diag(np.ones_like(right)) * A.T.shape[0] / A.T.shape[1]
    x = scipy.optimize.nnls(left, right)[0]
    x[x > x_max] = 0
    return x


def solve_richardson_lucy(A, d, x0=None, max_iter=10, min_value=0.001):
    """
    :param A: numpy-array
        The convolution matrix
    :param d: numpy-array
        The measured observable
    :param x0: numpy-array
        The initially assumed solution. If this value is None ones are used.
    :param max_iter: int
        The maximum number of iterations
    :param min_res_norm: float
        TODO The minimum residual norm. (not implemented so far)
    :return: a tuple containing the residual norm and the solution vector
    """
    A = A.T
    n_j, n_i = A.shape
    u = np.ones(n_i) if x0 is None else np.copy(x0)
    u = la.solve_richardson_lucy(A, u, d, max_iter)
    return u