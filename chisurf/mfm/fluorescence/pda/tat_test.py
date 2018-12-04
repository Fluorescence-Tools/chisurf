import math
from numba import jit
from mfm.math.functions.distributions import poisson_0toN


@jit(nopython=True)
def define_time_windows(mt, tw_min, tw_max, n_min=2, n_max=1000):
    """Follows the photon-stream and defines non-overlapping subsequent time windows with at least n_min photons and
    at most n_max photons. Returns an array which contains the start and the stop photon number, the number of
    photons between start and stop photon and the width of the time-window

    :param mt: macro-time array (array of integers)
    :param tw: time-window width (integer) number of macro-times
    :param n_min: minimum number of photons in time-window (at least two: start photon, stop photon)
    :param n_max: maximum number of photons in time-window (at least two: start photon, stop photon)
    :return: an numpy array containing start, stop, "number of photons", length of selection in number of tw

    Examples
    --------

    >>> import numpy as np
                       ________________      ______________                  __________________,_______________
    >>> mt = np.array([0, 1, 2, 3, 4, 5, 11, 20, 22, 23, 25, 27, 31, 45, 60, 71, 72, 73, 75, 76, 77, 78, 79, 80])
    >>> tw = define_time_windows(mt, 5, n_min=4)
    >>> print tw
    [[ 0  5  6  5]
     [ 7 10  4  5]
     [15 19  5  5]
     [19 23  6  4]]
    """
    n_ph = len(mt)
    start_stop = np.empty((n_ph, 4), dtype=np.int64)

    left = 0
    right = 0
    tw_number = 0
    n_ph_i = 1
    dt = 0
    while right < n_ph - 1:
        while dt < tw_max:
            right += 1
            n_ph_i += 1
            if right > n_ph - 1:
                right -= 1
                break
            dt = (mt[right] - mt[left])
        if (n_min <= n_ph_i <= n_max) and (tw_min <= dt <= tw_max):
            start_stop[tw_number, 0] = left
            start_stop[tw_number, 1] = right
            start_stop[tw_number, 2] = n_ph_i
            start_stop[tw_number, 3] = dt
            tw_number += 1
        left = right
        dt = 0
        n_ph_i = 1
    return start_stop[:tw_number]



@jit(nopython=True)
def make_sgsr_matrix(mt, rout, tw_min, tw_max, green=[0, 8], red=[1, 9], ph_max=500):
    """

    :param mt:
    :param rout:
    :param tw:
    :param green:
    :param red:
    :param ph_max:
    :param n_rout_max:
    :return:
    """
    sgsr = np.zeros(((ph_max + 1), (ph_max + 1)), dtype=np.float64)
    n_ph = len(mt)

    left = 0
    right = 0
    n_ph_i = 1
    dt = 0
    while right < n_ph - 1:
        left = right
        dt = 0
        n_ph_i = 1
        while dt < tw_max:
            right += 1
            n_ph_i += 1
            if right > n_ph - 1:
                right -= 1
                break
            dt = (mt[right] - mt[left])

        # Make the histogram
        n_green = 0
        n_red = 0
        for rt in rout[left:right]:
            for g in green:
                if rt == g:
                    n_green += 1
            for r in red:
                if rt == r:
                    n_red += 1
        if (n_green <= ph_max and n_red <= ph_max) and (tw_min <= dt <= tw_max):
            sgsr[n_green, n_red] += 1
    return sgsr


def sg_sr(Sg, Sr, **kwargs):
    """Green-red intensity ratio

    :param Sg:
    :param Sr:
    :param kwargs:
    :return:
    """
    if Sr != 0:
        return float(Sg) / float(Sr)
    else:
        return float('nan')


def fg_fr(Sg, Sr, **kwargs):
    BG = kwargs.get('BG', 0.0)
    BR = kwargs.get('BR', 0.0)
    return sg_sr(Sg - BG, Sr - BR, **kwargs)


def fd_fa(Sg, Sr, **kwargs):
    g = kwargs.get('det_eff_ratio', 1.0)
    return fg_fr(Sg, Sr, **kwargs) / g


def transfer_efficency(Sg, Sr, **kwargs):
    qyd = kwargs.get('qy_d', 1.0)
    qya = kwargs.get('qy_a', 1.0)
    return 1.0 / (1.0 + fg_fr(Sg, Sr, **kwargs) / qyd * qya)


@jit(nopython=True)
def convolve_background(FgFr, Bg, Br):
    f = FgFr
    sgr = np.zeros_like(f)
    tmp = np.zeros_like(f)
    Nmax = f.shape[0]

    bg = poisson_0toN(Bg, Nmax)
    br = poisson_0toN(Br, Nmax)

    Br_max = 2 * Br+52
    Bg_max = 2 * Bg+52

    for red in range(0, Nmax):
        i_start = red-Br_max if red > Br_max else 0
        for green in range(0, Nmax - red):
            s = 0.0
            for i in range(i_start, red + 1):
                s += f[green, i] * br[red - i]
            tmp[green, red] = s

    for green in range(0, Nmax):
        i_start = green-Bg_max if green > Bg_max else 0
        for red in range(0, Nmax - green):
            s = 0.0
            for i in range(i_start, green + 1):
                s += tmp[i, red] * bg[green - i]
            sgr[green, red] = s

    return sgr.reshape((Nmax, Nmax))


@jit(nopython=True)
def polynom2_conv(p, n, p2):
    """Returns a vector of length of p which is convoluted with the
    vector [p2 1-p2] up to the position n

    :param p: vector
    :param n: convolution limit
    :param p2: float
    :return:
    """
    r = np.zeros_like(p)
    r[0] = p[0] * p2
    for i in range(1, n):
        r[i] = p[i-1] * (1.-p2) + p[i] * p2
    r[n] = p[n - 1] * (1. - p2)
    return r


@jit(nopython=True)
def fgfr(n_max, amplitudes, pg_theor, pF=None):
    """Calculates the radial distribution function of a worm-like-chain given the multiple piece-solution
    according to:

    The radial distribution function of worm-like chain
    Eur Phys J E, 32, 53-69 (2010)

    :param n_max: maximum number of photons
    :param amplitudes: vector of amplitudes of pg_theor
    :param pg_theor: vecotr of theoretical green probabilities
    :param pF: vector of probabilities of getting a certain number of fluorescent photons
    :return:

    Examples
    --------

    >>> maximum_number_of_photons = 5
    >>> amplitude_species_1 = 0.2
    >>> amplitude_species_2 = 0.8
    >>> pg_species_1 = 0.2
    >>> pg_species_2 = 0.8
    >>> fgfr(maximum_number_of_photons, [amplitude_species_1, amplitude_species_2], [pg_species_1, pg_species_2])
    array([[ 1.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ],
       [ 0.     ,  0.32   ,  0.1536 ,  0.1024 ,  0.08704,  0.     ],
       [ 0.     ,  0.3264 ,  0.1536 ,  0.08192,  0.     ,  0.     ],
       [ 0.     ,  0.3328 ,  0.17408,  0.     ,  0.     ,  0.     ],
       [ 0.     ,  0.32896,  0.     ,  0.     ,  0.     ,  0.     ],
       [ 0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ]])

    References
    ----------

    .. [1] Separating Structural Heterogeneities from Stochastic Variations in Fluorescence Resonance
        Energy Transfer Distributions via Photon Distribution Analysis
        Matthew Antonik, Suren Felekyan, Alexander Gaiduk, and Claus A. M. Seidel
        J. Phys. Chem. B 2006, 110, 6970-6978

    """
    if pF is None:
        pF = np.ones(n_max, dtype=np.float64)
    pm = np.zeros((n_max+1, n_max+1), dtype=np.float64)
    FgFr = np.zeros((n_max+1, n_max+1), dtype=np.float64)
    pm[0, 0] = 1.0
    for aj, pj in zip(amplitudes, pg_theor):
        for i in range(1, n_max + 1):
            pm[i] = polynom2_conv(pm[i - 1], i, pj)
            for n_red in range(1, i):
                n_green = i - n_red
                FgFr[n_green, n_red] += pm[i, n_red] * pF[i - 1] * aj
    FgFr[0, 0] = 1.0
    for red in range(0, n_max):
        FgFr[0, red] = FgFr[red, 0]
    return FgFr


@jit(nopython=True)
def sgsr(n_max, pg_theor, Bg, Br):
    FgFr = fgfr(n_max, pg_theor)
    # SgSr: matrix, SgSr(i,j) = p(Sg = i, Sr = j)
    return convolve_background(FgFr, Bg, Br)


def sgsr_histogram(sgsr, eq, **kwargs):
    """Generates a histogram given a SgSr-array using the function 'eq'

    :param SgSr:
    :param eq:
    :param kwargs:
    :return:

    Examples
    --------

    >>>
    """
    n_bins = kwargs.get('n_bins', 101)
    n_min = kwargs.get('n_min', 1)
    x_max = kwargs.get('x_max', 100)
    x_min = kwargs.get('x_min', 0.1)
    exp_param = kwargs.get('exp_param', dict())
    log_x = kwargs.get('log_x', False)

    histogram_x = np.zeros(n_bins, dtype=np.float64)
    histogram_y = np.zeros(n_bins, dtype=np.float64)

    if log_x:
        xminlog = math.log(x_min)

        bin_width = (math.log(x_max) - xminlog)/(float(n_bins) - 1.)
        i_bin_width = 1./bin_width
        xmincorr = math.log(x_min) - 0.5 * bin_width

        # histogram x
        for i_bin in range(0, n_bins):
            histogram_x[i_bin] = math.exp(xminlog + bin_width*float(i_bin))
        # histogram y
        for green in range(1, n_max):
            firstred = 1 if green > n_min else n_min - green
            for red in range(firstred, n_max - green):
                x1 = eq(green, red, **exp_param)
                if np.isnan(x1):
                    continue
                if x1 > 0.0:
                    x = math.log(x1)
                    i_bin = int(math.floor((x-xmincorr)*i_bin_width))
                    if 0 < i_bin < n_bins:
                        histogram_y[i_bin] += sgsr[green*(n_max+1) + red]
    else:
        bin_width = (x_max - x_min)/(float(n_bins) - 1.)
        i_bin_width = 1./bin_width
        xmincorr = x_min - 0.5*bin_width

        # histogram x
        for i_bin in range(0, n_bins):
            histogram_x[i_bin] = x_min + bin_width * float(i_bin)
        # histogram y
        for green in range(0, n_bins):
            firstred = 0 if green > n_min else n_min - green
            for red in range(firstred, n_max - green):
                x = eq(green, red, **exp_param)
                if np.isnan(x):
                    continue
                i_bin = int(math.floor((x-xmincorr)*i_bin_width))
                if 0 < i_bin < n_bins:
                    histogram_y[i_bin] += sgsr[green*(n_max+1) + red]
    return histogram_x, histogram_y


import _sgsr
import pylab as p
import numpy as np


n_max = 50  # The maximum number of considered photons
bg = 0.0  # Green background
br = 0.0  # Red background
transfer_efficencies = np.array([0.4], np.float64)
a = np.array([1.0], dtype=np.float64)

pg = 1 - transfer_efficencies
pf = np.ones(n_max + 1, dtype=np.float64)

# Calculate the 2D-probability distirbution to find a certain combination of green and red photons
s = _sgsr.sgsr_pn_manypg(n_max, pf, bg, br, pg, a)
p.imshow(s.reshape(n_max+1, n_max+1))
p.show()

x, y = sgsr_histogram(s, sg_sr, x_min=0.1, x_max=1000., log_x=True,
                      n_min=10,
                      n_max=90,
                      exp_param={
                          "BG": 0.0,
                          "BR": 0.0,
                          "qy_d": 1.0,
                          "qy_a": 1.0,
                          "det_eff_ratio": 1.0
                      })
p.semilogx(x, y)
p.show()

