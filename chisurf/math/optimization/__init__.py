import numpy as np

import chisurf.math.linalg
from .mem import maxent


#leastsqbound = skf.math.optimize.leastsqbound.leastsqbound
from chisurf.math.optimization.leastsqbound import leastsqbound



def solve_richardson_lucy(
        A: np.array,
        d: np.array,
        x0: np.array =None,
        max_iter: int = 10,
        min_value: float = 0.001
) -> np.array:
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
    u = chisurf.math.linalg.solve_richardson_lucy(A, u, d, max_iter)
    return u
