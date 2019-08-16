from __future__ import annotations

from itertools import tee
import numba as nb
import numexpr as ne
import numpy as np

import scipy.optimize

rate2lifetime = lambda rate, lifetime: ne.evaluate("1. / (1. / lifetime + rate)")
et = lambda fd0, fda: fda / fd0


def p_isotropic_orientation_factor(k2, normalize=True):
    """Calculates an the probability of a given kappa2 according to 
    an isotropic orientation factor distribution
    http://www.fretresearch.org/kappasquaredchapter.pdf
    
    :param k2: kappa squared 
    :param normalize: if True (
    :return: 
    """
    ks = np.sqrt(k2)
    s3 = np.sqrt(3.)
    r = np.zeros_like(k2)
    for i, k in enumerate(ks):
        if 0 <= k <= 1:
            r[i] = 0.5 / (s3 * k) * np.log(2 + s3)
        elif 1 <= k <= 2:
            r[i] = 0.5 / (s3 * k) * np.log((2 + s3) / (k + np.sqrt(k**2 - 1.0)))
    if normalize:
        r /= max(1.0, r.sum())
    return r


def fretrate2distance(fretrate, forster_radius, tau0, kappa2=2./3.):
    """Calculate the distance given a FRET-rate

    :param fretrate: FRET-rate
    :param forster_radius: Forster radius
    :param tau0: lifetime of the donor
    :param kappa2: orientation factor
    :return:
    """
    return forster_radius * (fretrate * tau0/kappa2 * 2./3.)**(-1./6)


def interleaved_to_two_columns(ls, sort=False):
    """
    Converts an interleaved spectrum to two column-data
    :param ls: numpy array
        The interleaved spectrum (amplitude, lifetime)
    :return: two arrays (amplitudes), (lifetimes)
    """
    lt = ls.reshape((ls.shape[0] // 2, 2))
    if sort:
        s = lt[np.argsort(lt[:, 1])]
        y = s[:, 0]
        x = s[:, 1]
        return y, x
    else:
        return lt[:, 0], lt[:, 1]


def two_column_to_interleaved(x, t):
    """
    Converts two column lifetime spectra to interleaved lifetime spectra
    :param ls: The
    :return:
    """
    c = np.vstack((x, t)).ravel([-1])
    return c


def species_averaged_lifetime(fluorescence, normalize=True, is_lifetime_spectrum=True):
    """
    Calculates the species averaged lifetime given a lifetime spectrum

    :param fluorescence: either a inter-leaved lifetime-spectrum (if is_lifetime_spectrum is True) or a 
        fluorescence decay (times, fluorescence intensity)
    :return:
    """
    if is_lifetime_spectrum:
        x, t = interleaved_to_two_columns(fluorescence)
        if normalize:
            x /= x.sum()
        tau_x = np.dot(x, t)
        return tau_x
    else:
        time_axis = fluorescence[0]
        intensity = fluorescence[1]

        dt = (time_axis[1] - time_axis[0])
        i2 = intensity / max(intensity)
        return np.sum(i2) * dt


def fluorescence_averaged_lifetime(fluorescence, taux=None, normalize=True, is_lifetime_spectrum=True):
    """

    :param fluorescence: interleaved lifetime spectrum
    :param taux: float
        The species averaged lifetime. If this value is not provided it is calculated based
        on th lifetime spectrum
    :return:
    """
    if is_lifetime_spectrum:
        taux = species_averaged_lifetime(fluorescence) if taux is None else taux
        x, t = interleaved_to_two_columns(fluorescence)
        if normalize:
            x /= x.sum()
        tau_f = np.dot(x, t**2) / taux
        return tau_f
    else:
        time_axis = fluorescence[0]
        intensity = fluorescence[1]
        return np.sum(intensity * time_axis) / np.sum(intensity)


def phasor_giw(f, n, omega, times):
    """Phasor plot gi(w)
    The phasor approach to fluorescence lifetime page 236

    :param f: array of the fluorescence intensity at the provided times
    :param n: the nth harmonics
    :param omega: the angular frequency (2*pi*frequency)
    :param times: the times of the fluorescence intensities
    :return:
    """
    y = f * np.cos(n * omega * times)
    x = times
    return np.trapz(y, x) / np.trapz(f, x)


def phasor_siw(f, n, omega, times):
    """Phasor plot gi(w)
    The phasor approach to fluorescence lifetime page 236

    :param f: array of the fluorescence intensity at the provided times
    :param n: the nth harmonics
    :param omega: the angular frequency (2*pi*frequency)
    :param times: the times of the fluorescence intensities
    :return:
    """
    y = f * np.sin(n * omega * times)
    x = times
    return np.trapz(y, x) / np.trapz(f, x)


@nb.jit
def distance_to_fret_rate_constant(
        r,
        forster_radius: float,
        tau0: float,
        kappa2: float = 2./3.
):
    """ Converts the DA-distance to a FRET-rate

    :param r: donor-acceptor distance
    :param forster_radius: Forster-radius
    :param tau0: lifetime
    :param kappa2: orientation factor
    :return:
    """
    return 3. / 2. * kappa2 * 1. / tau0 * (forster_radius / r) ** 6.0


@nb.jit
def distance_to_fret_efficiency(
        distance: float,
        forster_radius: float
):
    """

    .. math::

        E = 1.0 / (1.0 + (R_{DA} / R_0)^6)

    :param distance: DA-distance
    :param forster_radius: Forster-radius
    :return:
    """
    return 1.0 / (1.0 + (distance / forster_radius) ** 6)


@nb.jit
def lifetime_to_fret_efficiency(
        tau: float,
        tau0: float
):
    """

    .. math::

        E = 1 - tau / tau_0

    :param tau: fluorescence lifetime in the presence of FRET
    :param tau0: fluorescence lifetime in the absence of FRET
    :return:
    """
    return 1 - tau / tau0


@nb.jit
def fret_efficiency_to_distance(
        fret_efficiency: float,
        forster_radius: float
):
    """
    Converts the transfer-efficiency to a distance

    .. math::

        R_{DA} = R_0 (1 / E - 1)^{1/6}

    :param fret_efficiency: Transfer-efficency
    :param forster_radius: Forster-radius
    :return:
    """
    return (1 / fret_efficiency - 1) ** (1.0 / 6.0) * forster_radius


@nb.jit
def fret_efficiency_to_lifetime(
        fret_efficiency: float,
        tau0: float
):
    """

    .. math::

        tau_{DA} = (1 - E) * tau_0

    :param fret_efficiency:
    :param tau0:
    :return:
    """
    return (1 - fret_efficiency) * tau0


def transfer_space(
        transfer_efficiency_min: float,
        transfer_efficiency_max: float,
        n_steps: int,
        forster_radius: float = 52.0
):
    """
    Generates distances with equally spaced transfer efficiencies

    :param transfer_efficiency_min: float
        minimum transfer efficency
    :param transfer_efficiency_max: float
        maximum transfer efficency
    :param n_steps: int
        number of distances
    :param forster_radius: float
        Forster-radius
    :return:
    """
    es = np.linspace(transfer_efficiency_min, transfer_efficiency_max, n_steps)
    rdas = fret_efficiency_to_distance(es, forster_radius)
    return rdas


def kappasq(delta, sD2, sA2, beta1=None, beta2=None):
    """
    Calculates the kappa2 distribution given the order parameter sD2 and sA2

    :param delta:
    :param sD2: order parameter of donor s2D = - sqrt(r_inf_D/r0)
    :param sA2: order parameter of acceptor s2A = sqrt(r_inf_A/r0)
    :param beta1:
    :param beta2:
    """
    if beta1 is None or beta2 is None:
        beta1 = 0
        beta2 = delta

    s2delta = (3.0 * np.cos(delta) * np.cos(delta) - 1.0) / 2.0
    s2beta1 = (3.0 * np.cos(beta1) * np.cos(beta1) - 1.0) / 2.0
    s2beta2 = (3.0 * np.cos(beta2) * np.cos(beta2) - 1.0) / 2.0
    k2 = 2.0 / 3.0 * (1 + sD2 * s2beta1 + sA2 * s2beta2 +
                  sD2 * sA2 * (s2delta +
                                 6 * s2beta1 * s2beta2 +
                                 1 + 2 * s2beta1 +
                                 2 * s2beta2 -
                                 9 * np.cos(beta1) *
                                 np.cos(beta2) * np.cos(delta)))
    return k2


def calc_transfer_matrix(t, rDA_min=1.0, rDA_max=200.0, n_steps=200.0, kappa2=0.667, space='lin', **kwargs):
    """
    Calculates a matrix converting a distance distribution to an E(t)-decay

    :param t:
    :param rDA_min:
    :param rDA_max:
    :param n_steps:
    :param kappa2:
    :param log_space:
    :param kwargs:
        tau0 - lifetime,
        R0 - Forster-radius,
        n_donor_bins: int (default 10)
            the number of bins with rate 0.0 (Donor-only). The longest distances are
            replaced by zero-rates

    Examples
    --------

    >>> t = np.arange(0, 20, 0.0141)
    >>> m, r_da = calc_transfer_matrix(t)

    .. plot:: plots/e_transfer_matrix.py

    """
    R0 = kwargs.get('R0', 52.0)
    tau0 = kwargs.get('tau0', 4.0)
    r_DA = kwargs.get('r_DA', None)
    n_donor_bins = kwargs.get('n_donor_bins', 10)

    if r_DA is None:
        if space == 'lin':
            r_DA = np.linspace(rDA_min, rDA_max, n_steps)
        elif space == 'log':
            lmin = np.log10(rDA_min)
            lmax = np.log10(rDA_max)
            r_DA = np.logspace(lmin, lmax, n_steps)
        elif space == 'trans':
            e_min = distance_to_fret_efficiency(rDA_min, R0)
            e_max = distance_to_fret_efficiency(rDA_max, R0)
            r_DA = transfer_space(e_min, e_max, n_steps, R0)

    rates = distance_to_fret_rate_constant(
        r_DA, R0, tau0, kappa2
    )
    # Use the last bins for D-Only
    rates[-1:-n_donor_bins] = 0.0
    m = np.outer(rates, t)
    M = np.nan_to_num(np.exp(-m))
    return M, r_DA


def calc_decay_matrix(t, tau_min=0.01, tau_max=200.0, n_steps=200.0, space='lin'):
    """
    Calculates a fluorescence decay matrix converting probabilities of lifetimes to a time-resolved
    fluorescence intensity

    :param t:
    :param tau_min:
    :param tau_max:
    :param n_steps:
    :param space:

    Examples
    --------

    >>> t = np.arange(0, 20, 0.0141)
    >>> m, r_da = calc_decay_matrix(t)

    Now plot the decay matrix

    >>> import pylab as p
    >>> p.imshow(m)
    >>> p.show()
    """
    if space == 'lin':
        taus = np.linspace(tau_min, tau_max, n_steps)
    elif space == 'log':
        lmin = np.log10(tau_min)
        lmax = np.log10(tau_max)
        taus = np.logspace(lmin, lmax, n_steps)
    rates = 1. / taus
    m = np.outer(rates, t)
    M = np.nan_to_num(np.exp(-m))
    return M, taus


def et2pRDA(ts, et, t_matrix=None, r_DA=None, **kwargs):
    """Calculates the distance distribution given an E(t) decay
    Here the amplitudes of E(t) are passed as well as the time-axis. If no transfer-matrix is provided it will
    be calculated in a range from 5 Ang to 200 Ang assuming a lifetime of 4 ns with a Forster-radius of 52 Ang.
    These parameters can be provided by *kwargs*

    :param t:
    :param et:
    :param t_matrix:
    :param r_DA:
    :param kwargs: tau0 donor-lifetime in absence of quenching, R0 - Forster-Radius
    :return: a distance and probability array

    Examples
    --------

    First calculate an E(t)-decay

    >>> rda_mean = [45.1, 65.0]
    >>> rda_sigma = [8.0, 8.0]
    >>> amplitudes = [0.6, 0.4]
    >>> rates = gaussian2rates(rda_mean, rda_sigma, amplitudes, interleaved=False)
    >>> a = rates[:,0]
    >>> kFRET = rates[:,1]
    >>> ts = np.logspace(0.1, 3, 18000)
    >>> et = np.array([np.dot(a, np.exp(-kFRET * t)) for t in ts])

    """
    if t_matrix is None or r_DA is None:
        t_matrix, r_DA = calc_transfer_matrix(ts, 5, 200, 200, **kwargs)

    p_rDA = scipy.optimize.nnls(t_matrix.T, et)[0]
    return r_DA, p_rDA


def stack_lifetime_spectra(lifetime_spectra, fractions, normalize_fractions=True):
    """
    Takes an array of lifetime spectra and an array of fractions and returns an mixed array of lifetimes
    whereas the amplitudes are multiplied by the fractions. `normalize_fractions` is True the fractions
    are normalized to one.

    :return: numpy-array

    """
    fn = np.array(fractions, dtype=np.float64) / sum(fractions) if normalize_fractions else fractions
    re = []
    for i, ls in enumerate(lifetime_spectra):
        ls = np.copy(ls)
        ls[::2] = ls[::2] * fn[i]
        re.append(ls)
    return np.hstack(re)


def distribution2rates(distribution, tau0, kappa2, R0, remove_negative=False):
    """
    gets distribution in form: (1,2,3)
    0: number of distribution
    1: amplitude
    2: distance

    returns:
    0: number of dist
    1: amplitude
    2: rate

    :param distribution:
    :param tau0:
    :param kappa2: a interleaved list of orientation factors (orientation factor spectrum)
    :param R0:
    """

    n_dist, n_ampl, n_points = distribution.shape
    rate_dist = np.copy(distribution)
    if remove_negative:
        rate_dist = np.maximum(rate_dist, 0)

    for i in range(n_dist):
        rate_dist[i, 1] = distance_to_fret_rate_constant(
            rate_dist[i, 1],
            R0,
            tau0,
            0.66667
        )
    return rate_dist

    k2 = kappa2.reshape((len(kappa2)/2, 2))
    k2_amp = k2[:, 0]
    k2_val = k2[:, 1]
    rate_const = np.outer(k2_val, rate_dist[:, 0]).flatten()
    amplitudes = np.outer(k2_amp, rate_dist[:, 1]).flatten()

    c = np.empty((1, 2, rate_const.size), dtype=rate_const.dtype)
    c[:, 0] = amplitudes
    c[:, 1] = rate_const
    print("c-shape")
    print(c.shape)
    return c


def gaussian2rates(means, sigmas, amplitudes, tau0=4.0, kappa2=0.667, R0=52.0,
                   n_points=64, m_sigma=1.5, interleaved=True):
    """
    Calculate distribution of FRET-rates given a list of normal/Gaussian distributed
    distances.

    :param means: array
    :param sigmas: array
    :param amplitudes: array
    :param tau0: float
    :param kappa2: float
    :param R0: float
    :param n_points: int
    :param m_sigma: float

    :return: either an interleaved rate-spectrum, or a 2D-array stack

    Examples
    --------

    This generates an interleaved array here it follows the *rule*:
    amplitude, rate, amplitude, rate, ...

    >>> gaussian2rates([50], [8.0], [1.0], interleaved=True, n_points=8)
    array([ 0.06060222,  1.64238732,  0.10514622,  0.97808684,  0.1518205 ,
            0.60699836,  0.18243106,  0.39018018,  0.18243106,  0.25853181,
            0.1518205 ,  0.17589017,  0.10514622,  0.12247918,  0.06060222,
            0.08706168])

    If *interleaved* is False a 2D-numpy array is returned. The first dimension corresponds
    to the amplitudes the second to the rates.

    >>> gaussian2rates([50], [8.0], [1.0], interleaved=False, n_points=8)
    array([[ 0.06060222,  1.64238732],
           [ 0.10514622,  0.97808684],
           [ 0.1518205 ,  0.60699836],
           [ 0.18243106,  0.39018018],
           [ 0.18243106,  0.25853181],
           [ 0.1518205 ,  0.17589017],
           [ 0.10514622,  0.12247918],
           [ 0.06060222,  0.08706168]])
    """
    means = np.array(means, dtype=np.float64)
    sigmas = np.array(sigmas, dtype=np.float64)
    amplitudes = np.array(amplitudes, dtype=np.float64)
    n_gauss = means.shape[0]

    rates = np.empty((n_gauss, n_points), dtype=np.float64)
    p = np.empty_like(rates)

    for i in range(n_gauss):
        g_min = max(1e-9, means[i] - m_sigma * sigmas[i])
        g_max = means[i] + m_sigma * sigmas[i]
        bins = np.linspace(g_min, g_max, n_points)
        p[i] = np.exp(-(bins - means[i]) ** 2 / (2 * sigmas[i] ** 2))
        p[i] /= np.sum(p[i])
        p[i] *= amplitudes[i]
        rates[i] = distance_to_fret_rate_constant(
            bins,
            R0,
            tau0,
            0.66667
        )
    ls = rates.ravel()
    ps = p.ravel()

    if interleaved:
        return np.dstack((ps, ls)).ravel()
    else:
        return np.dstack((ps, ls))[0]


def rates2lifetimes_old(rates, donors, x_donly=0.0):
    """
    Converts an interleaved rate spectrum to an interleaved lifetime spectrum
    given an interleaved donor spectrum and the fraction of donor-only

    :param rates: numpy-array
    :param donors: numpy-array
    :param x_donly: float

    """
    n_donors = donors.shape[0]/2
    n_rates = rates.shape[0]/2

    pd, ld = donors.reshape((n_donors, 2)).T
    pr, r = rates.reshape((n_rates, 2)).T

    # allocated memory
    ls = np.empty((n_donors, n_rates), dtype=np.float64)
    ps = np.empty_like(ls)

    x_fret = (1.0 - x_donly)
    ## Quench donor ##
    for i in range(n_donors):
        ps[i] = pd[i] * pr * x_fret
        ls[i] = rate2lifetime(r, ld[i])
    ls = ls.ravel()
    ps = ps.ravel()

    gl = np.dstack((ps, ls)).ravel()
    if x_donly > 0.0:
        donor = np.dstack((pd * x_donly, ld)).ravel()
        return np.hstack([gl, donor])
    else:
        return gl


def rates2lifetimes_new(fret_rate_spectrum, donor_rate_spectrum, x_donly=0.0):
    """Converts an interleaved rate spectrum to an interleaved lifetime spectrum
    given an interleaved donor spectrum and the fraction of donor-only

    :param fret_rate_spectrum: numpy-array
    :param donor_rate_spectrum: numpy-array
    :param x_donly: float

    """
    rate_spectrum = ere2(fret_rate_spectrum, donor_rate_spectrum)
    scaled_fret = e1tn(rate_spectrum, 1. - x_donly)
    scaled_donor = e1tn(donor_rate_spectrum, x_donly)
    rs = np.append(scaled_fret, scaled_donor)
    return ilt(rs)

rates2lifetimes = rates2lifetimes_new


@nb.jit(nopython=True, nogil=True)
def elte2(
        e1: np.array,
        e2: np.array
) -> np.array:
    """
    Takes two interleaved spectrum of lifetimes (a11, l11, a12, l12,...) and (a21, l21, a22, l22,...)
    and return a new spectrum of lifetimes of the form (a11*a21, 1/(1/l11+1/l21), a12*a22, 1/(1/l22+1/l22), ...)

    :param e1: array-like
        Lifetime spectrum 1
    :param e2: array-like
        Lifetime spectrum 2
    :return: array-like
        Lifetime-spectrum

    Examples
    --------

    >>> import numpy as np
    >>> e1 = np.array([1,2,3,4])
    >>> e2 = np.array([5,6,7,8])
    >>> elte2(e1, e2)
    array([  5.        ,   1.5       ,   7.        ,   1.6       ,
        15.        ,   2.4       ,  21.        ,   2.66666667])
    """
    n1 = e1.shape[0]/2
    n2 = e2.shape[0]/2
    r = np.empty(n1*n2*2, dtype=np.float64)

    k = 0
    for i in range(n1):
        for j in range(n2):
            r[k * 2 + 0] = e1[i * 2 + 0] * e2[j * 2 + 0]
            r[k * 2 + 1] = 1. / (1. / e1[i * 2 + 1] + 1. / e2[j * 2 + 1])
            k += 1
    return r


@nb.jit(nopython=True, nogil=True)
def ere2(
        e1: np.array,
        e2: np.array
) -> np.array:
    """
    Takes two interleaved spectrum of rates (a11, r11, a12, r12,...) and (a21, r21, a22, r22,...)
    and return a new spectrum of lifetimes of the form (a11*a21, r11+r21), a12*a22, r22+r22), ...)

    :param e1: array-like
        Lifetime spectrum 1
    :param e2: array-like
        Lifetime spectrum 2
    :return: array-like
        Lifetime-spectrum

    Examples
    --------

    >>> import numpy as np
    >>> e1 = np.array([1,2,3,4])
    >>> e2 = np.array([5,6,7,8])
    >>> elte2(e1, e2)
    array([  5.        ,   1.5       ,   7.        ,   1.6       ,
        15.        ,   2.4       ,  21.        ,   2.66666667])
    """
    n1 = e1.shape[0]/2
    n2 = e2.shape[0]/2
    r = np.zeros(n1*n2*2, dtype=np.float64)

    k = 0
    for i in range(n1):
        for j in range(n2):
            r[k * 2 + 0] = e1[i * 2 + 0] * e2[j * 2 + 0]
            r[k * 2 + 1] = e1[i * 2 + 1] + e2[j * 2 + 1]
            k += 1
    return r


@nb.jit(nopython=True, nogil=True)
def ilt(
        e1: np.array
) -> np.array:
    """Converts interleaved lifetime to rate spectra and vice versa

    :param e1: array-like
        Lifetime/rate spectrum

    Examples
    --------

    >>> import numpy as np
    >>> e1 = np.array([1,2,3,4])
    >>> ilt(e1)
    array([ 1.  ,  0.5 ,  3.  ,  0.25])
    """
    n1 = e1.shape[0]/2
    r = np.empty(n1*2, dtype=np.float64)

    for i in range(n1):
        r[i * 2 + 0] = e1[i * 2 + 0]
        r[i * 2 + 1] = 1. / (e1[i * 2 + 1])
    return r


@nb.jit(nopython=True, nogil=True)
def e1tn(
        e1: np.array,
        n: float
) -> np.array:
    """
    Multiply amplitudes of interleaved rate/lifetime spectrum by float

    :param e1: array-like
        Rate spectrum
    :param n: float

    Examples
    --------

    >>> e1 = np.array([1,2,3,4])
    >>> e1tn(e1, 2.0)
    array([2, 2, 6, 4])
    """
    n2 = e1.shape[0]
    for i in range(0, n2, 2):
        e1[i] *= n
    return e1


@nb.jit(nopython=True, nogil=True)
def e1ti2(
        e1: np.array,
        e2: np.array
) -> np.array:
    n1 = e1.shape[0]/2
    n2 = e2.shape[0]/2
    r = np.zeros(n1 * n2 * 2, dtype=np.float64)

    k = 0
    for i in range(n1):
        for j in range(n2):
            r[k * 2 + 0] = e1[i] * e2[j]
            r[k * 2 + 1] = e1[i + 1] * e2[j + 1]
            k += 1
    return r


def pairwise(iterable):
    """Iterate in pairs of 2 over iterable

    :param iterable:
    :return:
    """
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


@nb.jit
def calculate_fluorescence_decay(
        lifetime_spectrum: np.array,
        time_axis: np.array = None,
        normalize: bool = True
) -> (np.array, np.array):
    """Converts a interleaved lifetime spectrum into a intensity decay

    :param lifetime_spectrum: interleaved lifetime spectrum
    :param time_axis: time-axis
    :return:

    Examples
    --------

    >>> time_axis = np.linspace(0, 20, num=100)
    >>> structure = mfm.structure.Structure('./sample_data/modelling/pdb_files/hGBP1_closed.pdb')
    >>> donor_description = {'residue_seq_number': 344, 'atom_name': 'CB'}
    >>> acceptor_description = {'residue_seq_number': 496, 'atom_name': 'CB'}
    >>> donor_lifetime_spectrum = np.array([1., 4.])
    >>> lifetime_spectrum = structure.av_lifetime_spectrum(donor_lifetime_spectrum, donor_description, acceptor_description)
    >>> time_axis, decay = calculate_fluorescence_decay(lifetime_spectrum, time_axis)
    """
    if time_axis is None:
        time_axis = np.linspace(0, 50, num=100)
    decay = np.zeros(time_axis.shape)
    am = lifetime_spectrum[0::2]
    if normalize:
        am /= am.sum()
    ls = lifetime_spectrum[1::2]
    for a, l in zip(am, ls):
        if l == 0:
            continue
        decay += np.exp(-time_axis / l) * a
    return time_axis, decay

