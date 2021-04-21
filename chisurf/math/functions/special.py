from __future__ import annotations


import numpy as np
import numba as nb


@nb.jit(nopython=True)
def i0(
        x: float
):
    """Modified Bessel-function I0(x) for any real x
    (according to numerical recipes function - `bessi0`,
    Polynomal approximation Abramowitz and Stegun )

    References
    ----------

    .. [1] Abramowitz, M and Stegun, I.A. 1964, Handbook of Mathematical
       Functions, Applied Mathematics Series, Volume 55 (Washington:
       National Bureal of Standards; reprinted 1968 by Dover Publications,
       New York), Chapter 10

    :param x:
    :return:
    """

    axi = abs(x)
    if axi < 3.75:
        yi = axi / 3.75
        yi *= yi
        ayi = 1.0 + yi * (3.5156299 + yi * (
            3.0899424 + yi * (1.2067492 + yi * (
                0.2659732 + yi * (0.360768e-1 + yi *
                                 0.45813e-2)))))
    else:
        yi = 3.75 / axi
        ayi = (np.exp(axi) / np.sqrt(axi)) * \
              (0.39894228 + yi * (0.1328592e-1 + yi * (
                  0.225319e-2 + yi * ( -0.157565e-2 + yi * (
                      0.916281e-2 + yi * (-0.2057706e-1 + yi * (
                          0.2635537e-1 + yi * (-0.1647633e-1 + yi *
                                               0.392377e-2))))))))


    return ayi


@nb.jit(nopython=True)
def i0_array(
        x: np.array
):
    """Modified Bessel-function I0(x) for any real x
    (according to numerical recipes function - `bessi0`,
    Polynomal approximation Abramowitz and Stegun )

    References
    ----------

    .. [1] Abramowitz, M and Stegun, I.A. 1964, Handbook of Mathematical
       Functions, Applied Mathematics Series, Volume 55 (Washington:
       National Bureal of Standards; reprinted 1968 by Dover Publications,
       New York), Chapter 10

    :param x:
    :return:
    """
    ax = np.abs(x)

    ay = np.zeros_like(ax)
    n_ax = len(ax)
    for i in range(n_ax):
        axi = ax[i]
        ay[i] = i0(axi)
    return ay
