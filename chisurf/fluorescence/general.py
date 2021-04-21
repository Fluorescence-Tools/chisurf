from __future__ import annotations
from chisurf import typing

import numba as nb
import numpy as np

import scipy.optimize

import chisurf.math.datatools

import scikit_fluorescence as skf
import scikit_fluorescence.decay

# Moved to scikit_fluorescence
rate_constant_to_lifetime = skf.decay.rate_spectra.rate_constant_to_lifetime
fretrate_to_distance = skf.decay.rate_spectra.fretrate_to_distance
combine_lifetime_spectra = skf.decay.rate_spectra.combine_interleaved_spectra
fret_induced_donor_decay = skf.decay.fret_induced_donor_decay
species_averaged_lifetime = skf.decay.lifetime.species_averaged_lifetime
fluorescence_averaged_lifetime = skf.decay.lifetime.fluorescence_averaged_lifetime
distance_to_fret_rate_constant = skf.decay.rate_spectra.distance_to_fret_rate_constant


@nb.jit
def distance_to_fret_efficiency(
        distance: float,
        forster_radius: float
) -> float:
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
) -> float:
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
) -> float:
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
) -> float:
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


def calc_transfer_matrix(
        times: np.array,
        rDA_min: float = 1.0,
        rDA_max: float = 200.0,
        n_steps: int =200,
        kappa2: float = 0.667,
        space: str = 'lin',
        **kwargs
):
    """
    Calculates a matrix converting a distance distribution to an E(t)-model_decay

    :param times:
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

    >>> times = np.arange(0, 20, 0.0141)
    >>> m, r_da = calc_transfer_matrix(times)

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
    m = np.outer(rates, times)
    M = np.nan_to_num(np.exp(-m))
    return M, r_DA


def calc_decay_matrix(
        times: np.array,
        tau_min: float = 0.01,
        tau_max: float = 200.0,
        n_steps: int = 200,
        space: str = 'lin'
):
    """
    Calculates a fluorescence model_decay matrix converting probabilities of lifetimes to a time-resolved
    fluorescence intensity

    :param t:
    :param tau_min:
    :param tau_max:
    :param n_steps:
    :param space:

    Examples
    --------

    >>> times = np.arange(0, 20, 0.0141)
    >>> m, r_da = calc_decay_matrix(times)

    Now plot the model_decay matrix

    >>> import pylab as p
    >>> p.imshow(m)
    >>> p.show()
    """
    if space == 'lin':
        taus = np.linspace(tau_min, tau_max, n_steps)
    else:  #elif space == 'log':
        lmin = np.log10(tau_min)
        lmax = np.log10(tau_max)
        taus = np.logspace(lmin, lmax, n_steps)
    rates = 1. / taus
    m = np.outer(rates, times)
    M = np.nan_to_num(np.exp(-m))
    return M, taus


def et2pRDA(
        ts,
        et,
        t_matrix=None,
        r_DA=None,
        **kwargs
):
    """Calculates the distance distribution given an E(t) model_decay
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

    First calculate an E(t)-model_decay

    >>> rda_mean = [45.1, 65.0]
    >>> rda_sigma = [8.0, 8.0]
    >>> amplitudes = [0.6, 0.4]
    >>> rates = gaussian2rates(rda_mean, rda_sigma, amplitudes, interleaved=False)
    >>> a = rates[:,0]
    >>> kFRET = rates[:,1]
    >>> ts = np.logspace(0.1, 3, 18000)
    >>> et = np.array([np.dot(a, np.exp(-kFRET * times)) for times in ts])

    """
    if t_matrix is None or r_DA is None:
        t_matrix, r_DA = calc_transfer_matrix(ts, 5, 200, 200, **kwargs)
    p_rDA = scipy.optimize.nnls(t_matrix.T, et)[0]
    return r_DA, p_rDA


stack_lifetime_spectra = combine_lifetime_spectra
# def stack_lifetime_spectra(
#         lifetime_spectra,
#         fractions,
#         normalize_fractions: bool = True
# ):
#     """
#     Takes an array of lifetime spectra and an array of fractions and returns an mixed array of lifetimes
#     whereas the amplitudes are multiplied by the fractions. `normalize_fractions` is True the fractions
#     are normalized to one.
#
#     :return: numpy-array
#
#     """
#     fn = np.array(fractions, dtype=np.float64) / sum(fractions) if normalize_fractions else fractions
#     re = list()
#     for i, ls in enumerate(lifetime_spectra):
#         ls = np.copy(ls)
#         ls[::2] = ls[::2] * fn[i]
#         re.append(ls)
#     return np.hstack(re)


def distribution2rates(
        distribution,
        tau0: float,
        kappa2,
        forster_radius: float,
        remove_negative: bool = False
):
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
    :param forster_radius:
    """

    n_dist, n_ampl, n_points = distribution.shape
    rate_dist = np.copy(distribution)
    if remove_negative:
        rate_dist = np.maximum(rate_dist, 0)

    for i in range(n_dist):
        rate_dist[i, 1] = distance_to_fret_rate_constant(
            rate_dist[i, 1],
            forster_radius,
            tau0,
            0.66667
        )
    return rate_dist
    #
    # k2 = kappa2.reshape((len(kappa2)/2, 2))
    # k2_amp = k2[:, 0]
    # k2_val = k2[:, 1]
    # rate_const = np.outer(k2_val, rate_dist[:, 0]).flatten()
    # amplitudes = np.outer(k2_amp, rate_dist[:, 1]).flatten()
    #
    # c = np.empty((1, 2, rate_const.size), dtype=rate_const.dtype)
    # c[:, 0] = amplitudes
    # c[:, 1] = rate_const
    # print("c-shape")
    # print(c.shape)
    # return c


def gaussian2rates(
        means: typing.List[float],
        sigmas: typing.List[float],
        amplitudes: typing.List[float],
        tau0: float = 4.0,
        kappa2: float = 0.667,
        R0: float = 52.0,
        n_points: int = 64,
        m_sigma: float = 1.5,
        interleaved: bool = True
):
    """
    Calculate distribution of FRET-rates given a list of normal/Gaussian distributed
    distances.

    :param means:
    :param sigmas:
    :param amplitudes:
    :param tau0:
    :param kappa2:
    :param R0:
    :param n_points:
    :param m_sigma:
    :param interleaved:
    :return:

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


def rates2lifetimes_old(
        rates,
        donors,
        x_donly: float = 0.0
):
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
        ls[i] = rate_constant_to_lifetime(r, ld[i])
    ls = ls.ravel()
    ps = ps.ravel()

    gl = np.dstack((ps, ls)).ravel()
    if x_donly > 0.0:
        donor = np.dstack((pd * x_donly, ld)).ravel()
        return np.hstack([gl, donor])
    else:
        return gl


def rates2lifetimes_new(
        fret_rate_spectrum: np.array,
        donor_rate_spectrum: np.array,
        x_donly: float = 0.0
):
    """Converts an interleaved rate spectrum to an interleaved lifetime spectrum
    given an interleaved donor spectrum and the fraction of donor-only

    :param fret_rate_spectrum: numpy-array
    :param donor_rate_spectrum: numpy-array
    :param x_donly: float

    """
    rate_spectrum = chisurf.math.datatools.ere2(
        fret_rate_spectrum,
        donor_rate_spectrum
    )
    scaled_fret = chisurf.math.datatools.e1tn(
        rate_spectrum,
        1. - x_donly
    )
    scaled_donor = chisurf.math.datatools.e1tn(
        donor_rate_spectrum,
        x_donly
    )
    rs = np.append(
        scaled_fret,
        scaled_donor
    )
    return chisurf.math.datatools.invert_interleaved(rs)


rates2lifetimes = rates2lifetimes_new


@nb.jit
def calculate_fluorescence_decay(
        lifetime_spectrum: np.ndarray,
        time_axis: np.ndarray,
        normalize: bool = True
) -> typing.Tuple[np.ndarray, np.ndarray]:
    """Converts a interleaved lifetime spectrum into a intensity model_decay

    :param lifetime_spectrum: interleaved lifetime spectrum
    :param time_axis: time-axis
    :param normalize:
    :return:

    Examples
    --------

    >>> import chisurf.structure.structure
    >>> time_axis = np.linspace(0, 20, num=100)
    >>> structure = chisurf.structure.structure.Structure('./test/data/modelling/pdb_files/hGBP1_closed.pdb')
    >>> donor_description = {'residue_seq_number': 344, 'atom_name': 'CB'}
    >>> acceptor_description = {'residue_seq_number': 496, 'atom_name': 'CB'}
    >>> donor_lifetime_spectrum = np.array([1., 4.])
    >>> lifetime_spectrum = structure.av_lifetime_spectrum(donor_lifetime_spectrum, donor_description, acceptor_description)
    >>> time_axis, model_decay = calculate_fluorescence_decay(lifetime_spectrum, time_axis)
    """
    decay = np.zeros_like(time_axis)
    am = lifetime_spectrum[0::2]
    if normalize:
        am /= am.sum()
    ls = lifetime_spectrum[1::2]
    for amplitude, lifetime in zip(am, ls):
        if lifetime == 0:
            continue
        decay += np.exp(-time_axis / lifetime) * amplitude
    return time_axis, decay

