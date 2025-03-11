from chisurf import typing

import numba as nb
import numpy as np


def kappasq_dwt(
        sD2: float,
        sA2: float,
        fret_efficiency: float,
        n_samples: int = 10000,
        n_bins: int = 31,
        k2_min: float = 0.0,
        k2_max: float = 4.0
):
    """
    Diffusion with traps.

    This function simulates a kappa² distribution in the presence of donor and
    acceptor dye trapping. It generates random donor and acceptor orientations,
    computes the orientation factor kappa² for each pair, and returns a histogram
    over the generated kappa² values.

    Parameters
    ----------
    sD2 : float
        Second rank order parameter S² of the donor dye (can correspond to the
        fraction of trapped donor dye).
    sA2 : float
        Second rank order parameter S² of the acceptor dye (can correspond to the
        fraction of trapped acceptor dye).
    fret_efficiency : float
        FRET efficiency.
    n_samples : int, optional
        Number of random vector pairs to generate (default: 10000).
    n_bins : int, optional
        Number of bins in the generated kappa² histogram (default: 31).
    k2_min : float, optional
        Lower bound of kappa² values for the histogram (default: 0.0).
    k2_max : float, optional
        Upper bound of kappa² values for the histogram (default: 4.0).

    Returns
    -------
    tuple
        A tuple containing:
          - k2_scale (np.ndarray): The linear kappa² scale (bin edges).
          - k2hist (np.ndarray): The histogram counts of kappa².
          - k2s (np.ndarray): Array of computed kappa² values for each sample.

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(42)  # Set seed for reproducibility in this example
    >>> k2_scale, k2hist, k2s = kappasq_dwt(sD2=0.3, sA2=0.4, fret_efficiency=0.5, n_samples=100, n_bins=11)
    >>> k2_scale  # doctest: +SKIP
    array([0. , 0.4, 0.8, 1.2, 1.6, 2. , 2.4, 2.8, 3.2, 3.6, 4. ])
    >>> k2hist.sum()  # doctest: +SKIP
    100
    """
    # Binning of the kappa² distribution
    k2_step = (k2_max - k2_min) / (n_bins - 1)
    k2_scale = np.arange(k2_min, k2_max + 1e-14, k2_step, dtype=np.float64)

    # Generate random orientations for the TDM vectors
    donor_vec = np.random.randn(n_samples, 3)
    acceptor_vec = np.random.randn(n_samples, 3)

    # x = (R_DA/R_0)^6; relative DA distance from the FRET efficiency
    x = 1 / fret_efficiency - 1

    k2s = np.zeros(n_samples, dtype=np.float64)
    for i in range(n_samples):
        donor = donor_vec[i]
        acceptor = acceptor_vec[i]
        # Assumption: connecting vector R_DA is along the x-axis (R_DA = [1,0,0])
        delta = np.arccos(np.dot(donor, acceptor) /
                          (np.linalg.norm(donor) * np.linalg.norm(acceptor)))
        beta1 = np.arccos(donor[0] / np.linalg.norm(donor))
        beta2 = np.arccos(acceptor[0] / np.linalg.norm(acceptor))

        k2_trapped_free = kappasq(
            delta=delta,
            sD2=1,
            sA2=0,
            beta1=beta1,
            beta2=beta2
        )
        k2_free_trapped = kappasq(
            delta=delta,
            sD2=0,
            sA2=1,
            beta1=beta1,
            beta2=beta2
        )
        k2_trapped_trapped = kappasq(
            delta=delta,
            sD2=1,
            sA2=1,
            beta1=beta1,
            beta2=beta2
        )
        Ek2 = ((1 - sD2) * (1 - sA2) / (1 + x) +
               sD2 * sA2 / (1 + 2 / 3.0 / k2_trapped_trapped * x) +
               sD2 * (1 - sA2) / (1 + 2 / 3.0 / k2_trapped_free * x) +
               (1 - sD2) * sA2 / (1 + 2 / 3.0 / k2_free_trapped * x))
        k2 = 2 / 3.0 * x / (1 / Ek2 - 1)
        k2s[i] = k2

    # Create the histogram
    k2hist, _ = np.histogram(k2s, bins=k2_scale)
    return k2_scale, k2hist, k2s


def kappasq_all_delta_new(
        delta: float,
        sD2: float,
        sA2: float,
        step: float = 0.25,
        n_bins: int = 31,
        k2_min: float = 0.0,
        k2_max: float = 4.0
) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes a kappa² distribution for a given delta using a grid of beta angles.

    This function evaluates kappa² for a set of beta1 and beta2 angles (sampled
    in a specified step size) for a fixed delta and returns a weighted histogram
    over the computed kappa² values.

    Parameters
    ----------
    delta : float
        The fixed angle (in radians) between the symmetry axes of the dyes.
    sD2 : float
        Second rank order parameter S² of the donor dye.
    sA2 : float
        Second rank order parameter S² of the acceptor dye.
    step : float, optional
        Step size in degrees for sampling the beta angles (default: 0.25).
    n_bins : int, optional
        Number of bins in the resulting kappa² histogram (default: 31).
    k2_min : float, optional
        Lower bound for kappa² histogram (default: 0.0).
    k2_max : float, optional
        Upper bound for kappa² histogram (default: 4.0).

    Returns
    -------
    tuple
        A tuple containing:
          - k2scale (np.ndarray): Linear scale of kappa² values.
          - k2hist (np.ndarray): Histogram (weighted) of computed kappa² values.
          - ks (np.ndarray): Array of all computed kappa² values.

    Examples
    --------
    >>> import numpy as np
    >>> k2scale, k2hist, ks = kappasq_all_delta_new(delta=0.5, sD2=0.2, sA2=0.3, step=1.0, n_bins=11)
    >>> k2scale  # doctest: +SKIP
    array([0. , 0.4, 0.8, 1.2, 1.6, 2. , 2.4, 2.8, 3.2, 3.6, 4. ])
    """
    ks = list()
    weights = list()
    # Loop over beta1 values (in radians)
    for beta1 in np.arange(0, np.pi / 2, step / 180.0 * np.pi):
        weight_beta1 = np.sin(beta1)
        for beta2 in np.arange(
                start=abs(delta - beta1),
                stop=min(delta + beta1, np.pi / 2.0),
                step=step / 180.0 * np.pi
        ):
            weight_beta2 = np.sin(beta1)
            weights.append(weight_beta1 * weight_beta2)
            ks.append(
                kappasq(
                    sD2=sD2,
                    sA2=sA2,
                    delta=delta,
                    beta1=beta1,
                    beta2=beta2
                )
            )
    ks = np.array(ks)

    # Define histogram bin edges
    k2_step = (k2_max - k2_min) / (n_bins - 1)
    k2scale = np.arange(k2_min, k2_max + 1e-14, k2_step, dtype=np.float64)

    k2hist, _ = np.histogram(ks, bins=k2scale, weights=weights)
    return k2scale, k2hist, ks


@nb.jit(nopython=True)
def kappasq_all_delta(
        delta: float,
        sD2: float,
        sA2: float,
        step: float = 0.25,
        n_bins: int = 31,
        k2_min: float = 0.0,
        k2_max: float = 4.0
) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes an orientation factor distribution for a wobbling-in-a-cone model
    for a given delta and dye order parameters.

    The function calculates the distribution p(kappa²) by sampling over beta1 and
    phi angles for a fixed delta. The result is a weighted histogram of computed
    kappa² values.

    Parameters
    ----------
    delta : float
        Angle (in radians) between the symmetry axes of the dyes.
    sD2 : float
        Second rank order parameter S² of the donor dye.
    sA2 : float
        Second rank order parameter S² of the acceptor dye.
    step : float, optional
        Step size in degrees used for sampling angles (default: 0.25).
    n_bins : int, optional
        Number of bins in the resulting kappa² histogram (default: 31).
    k2_min : float, optional
        Lower bound for kappa² histogram (default: 0.0).
    k2_max : float, optional
        Upper bound for kappa² histogram (default: 4.0).

    Returns
    -------
    tuple
        A tuple containing:
          - k2scale (np.ndarray): Linear scale of kappa² values.
          - k2hist (np.ndarray): Histogram of kappa² values (weighted by sin(beta1)).
          - k2 (np.ndarray): 2D array of computed kappa² values for each (beta1, phi).

    Examples
    --------
    >>> import numpy as np
    >>> k2scale, k2hist, k2 = kappasq_all_delta(delta=0.2, sD2=0.15, sA2=0.25, step=2.0, n_bins=31)
    >>> np.allclose(k2hist, np.array([   0.        ,    0.        ,    0.        ,    0.        ,
    ...    3205.72877776, 1001.19048825,  611.44917432,  252.97166906,
    ...       0.        ,    0.        ,    0.        ,    0.        ,
    ...       0.        ,    0.        ,    0.        ,    0.        ,
    ...       0.        ,    0.        ,    0.        ,    0.        ,
    ...       0.        ,    0.        ,    0.        ,    0.        ,
    ...       0.        ,    0.        ,    0.        ,    0.        ,
    ...       0.        ,    0.        ]), rtol=0.3, atol=2.0)
    True

    Notes
    -----
    The beta1 angle is sampled in the range (0, π/2) and phi in the range (0, 2π).

    References
    ----------
    .. [1] Simon Sindbert, et al., "Accurate Distance Determination of Nucleic
           Acids via Foerster Resonance Energy Transfer: Implications of Dye Linker
           Length and Rigidity", J. Am. Chem. Soc., 2011.
    """
    # Define beta1 and phi arrays (in radians)
    beta1 = np.arange(0.001, np.pi / 2.0, step * np.pi / 180.0, dtype=np.float64)
    phi = np.arange(0.001, 2.0 * np.pi, step * np.pi / 180.0, dtype=np.float64)
    n = beta1.shape[0]
    m = phi.shape[0]
    rda_vec = np.array([1, 0, 0], dtype=np.float64)

    # Allocate arrays for kappa² and histogram
    k2 = np.zeros((n, m), dtype=np.float64)
    k2hist = np.zeros(n_bins - 1, dtype=np.float64)

    # Define histogram bin edges
    k2_step = (k2_max - k2_min) / (n_bins - 1)
    k2scale = np.arange(k2_min, k2_max + 1e-14, k2_step, dtype=np.float64)
    for i in range(n):
        d1 = np.array([np.cos(beta1[i]), 0, np.sin(beta1[i])])
        n1 = np.array([-np.sin(beta1[i]), 0, np.cos(beta1[i])])
        n2 = np.array([0, 1, 0])
        for j in range(m):
            d2 = (n1 * np.cos(phi[j]) + n2 * np.sin(phi[j])) * np.sin(delta) + d1 * np.cos(delta)
            beta2 = np.arccos(np.abs(np.dot(d2, rda_vec)))
            k2[i, j] = kappasq(
                delta=delta,
                sD2=sD2,
                sA2=sA2,
                beta1=beta1[i],
                beta2=beta2
            )
        y, _ = np.histogram(k2[i, :], bins=k2scale)
        k2hist += y * np.sin(beta1[i])
    return k2scale, k2hist, k2


@nb.jit(nopython=True)
def kappasq_all(
        sD2: float,
        sA2: float,
        n_bins: int = 81,
        k2_min: float = 0.0,
        k2_max: float = 4.0,
        n_samples: int = 10000
) -> typing.Tuple[np.array, np.array, np.array]:
    """
    Computes an orientation factor distribution for a wobbling-in-a-cone model
    based on random sampling of donor and acceptor orientations.

    This function generates random donor and acceptor vector pairs, computes
    kappa² for each pair, and returns a histogram of the kappa² values.

    Parameters
    ----------
    sD2 : float
        Second rank order parameter S² of the donor dye.
    sA2 : float
        Second rank order parameter S² of the acceptor dye.
    n_bins : int, optional
        Number of bins in the kappa² histogram (default: 81).
    k2_min : float, optional
        Lower bound for the kappa² histogram (default: 0.0).
    k2_max : float, optional
        Upper bound for the kappa² histogram (default: 4.0).
    n_samples : int, optional
        Number of random vector pairs to generate (default: 10000).

    Returns
    -------
    tuple
        A tuple containing:
          - k2scale (np.ndarray): Linear scale of kappa² values.
          - k2hist (np.ndarray): Histogram counts of kappa².
          - k2 (np.ndarray): Array of computed kappa² values.

    Examples
    --------
    >>> import numpy as np
    >>> k2_scale, k2_hist, k2 = kappasq_all(sD2=0.3, sA2=0.5, n_bins=31, n_samples=100000)
    >>> k2_scale  # doctest: +SKIP
    array([0.        , 0.13333333, 0.26666667, 0.4       , 0.53333333,
           0.66666667, 0.8       , 0.93333333, 1.06666667, 1.2       ,
           1.33333333, 1.46666667, 1.6       , 1.73333333, 1.86666667,
           2.        , 2.13333333, 2.26666667, 2.4       , 2.53333333,
           2.66666667, 2.8       , 2.93333333, 3.06666667, 3.2       ,
           3.33333333, 3.46666667, 3.6       , 3.73333333, 3.86666667,
           4.        ])
    >>> reference = np.array([0.0000e+00, 0.0000e+00, 0.0000e+00, 3.1920e+04, 4.3248e+04,
    ...    1.4842e+04, 5.8930e+03, 2.5190e+03, 1.0840e+03, 3.9700e+02,
    ...    9.4000e+01, 3.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
    ...    0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
    ...    0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
    ...    0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00])
    >>> np.allclose(reference, k2_hist, rtol=0.3, atol=2.0)
    True

    References
    ----------
    .. [1] Simon Sindbert, et al., "Accurate Distance Determination of Nucleic
           Acids via Foerster Resonance Energy Transfer: Implications of Dye Linker
           Length and Rigidity", J. Am. Chem. Soc., 2011.
    """
    k2 = np.zeros(n_samples, dtype=np.float64)
    step = (k2_max - k2_min) / (n_bins - 1)
    k2scale = np.arange(k2_min, k2_max + 1e-14, step, dtype=np.float64)
    k2hist = np.zeros(k2scale.shape[0] - 1, dtype=np.float64)
    for i in range(n_samples):
        d1 = np.random.random(3)
        d2 = np.random.random(3)
        n1 = np.linalg.norm(d1)
        n2 = np.linalg.norm(d2)
        # Assumption: connecting vector R_DA is along the x-axis (R_DA = [1,0,0])
        delta = np.arccos(np.dot(d1, d2) / (n1 * n2))
        beta1 = np.arccos(d1[0] / n1)
        beta2 = np.arccos(d2[0] / n2)
        k2[i] = kappasq(
            delta=delta,
            sD2=sD2,
            sA2=sA2,
            beta1=beta1,
            beta2=beta2
        )
    y, _ = np.histogram(k2, bins=k2scale)
    k2hist += y
    return k2scale, k2hist, k2


@nb.jit(nopython=True)
def kappa_distance(
        d1: np.array,
        d2: np.array,
        a1: np.array,
        a2: np.array
) -> typing.Tuple[float, float]:
    """
    Calculates the distance between the centers of two dipoles and the
    orientation factor kappa.

    Given the endpoints of the donor (d1 and d2) and acceptor (a1 and a2) dipoles,
    this function computes the center-to-center distance and the orientation
    factor kappa based on the dipole geometry.

    Parameters
    ----------
    d1 : np.array
        3D coordinates of the first point of the donor dipole.
    d2 : np.array
        3D coordinates of the second point of the donor dipole.
    a1 : np.array
        3D coordinates of the first point of the acceptor dipole.
    a2 : np.array
        3D coordinates of the second point of the acceptor dipole.

    Returns
    -------
    tuple
        A tuple (distance, kappa) where:
          - distance: The center-to-center distance between the dipoles.
          - kappa: The calculated orientation factor.

    Examples
    --------
    >>> import numpy as np
    >>> d1 = np.array([0.0, 0.0, 0.0])
    >>> d2 = np.array([1.0, 0.0, 0.0])
    >>> a1 = np.array([0.0, 0.5, 0.0])
    >>> a2 = np.array([0.0, 0.5, 1.0])
    >>> distance, k = kappa_distance(d1, d2, a1, a2)
    >>> round(distance, 5)
    0.86603
    >>> round(k, 5)
    1.00000
    """
    # Donor dipole endpoints
    d11 = d1[0]
    d12 = d1[1]
    d13 = d1[2]

    d21 = d2[0]
    d22 = d2[1]
    d23 = d2[2]

    # Distance between donor endpoints
    dD21 = np.sqrt((d11 - d21) ** 2 + (d12 - d22) ** 2 + (d13 - d23) ** 2)

    # Normalized donor dipole vector
    muD1 = (d21 - d11) / dD21
    muD2 = (d22 - d12) / dD21
    muD3 = (d23 - d13) / dD21

    # Midpoint of donor dipole
    dM1 = d11 + dD21 * muD1 / 2.0
    dM2 = d12 + dD21 * muD2 / 2.0
    dM3 = d13 + dD21 * muD3 / 2.0

    # Acceptor dipole endpoints
    a11 = a1[0]
    a12 = a1[1]
    a13 = a1[2]

    a21 = a2[0]
    a22 = a2[1]
    a23 = a2[2]

    # Distance between acceptor endpoints
    dA21 = np.sqrt((a11 - a21) ** 2 + (a12 - a22) ** 2 + (a13 - a23) ** 2)

    # Normalized acceptor dipole vector
    muA1 = (a21 - a11) / dA21
    muA2 = (a22 - a12) / dA21
    muA3 = (a23 - a13) / dA21

    # Midpoint of acceptor dipole
    aM1 = a11 + dA21 * muA1 / 2.0
    aM2 = a12 + dA21 * muA2 / 2.0
    aM3 = a13 + dA21 * muA3 / 2.0

    # Vector connecting dipole midpoints
    RDA1 = dM1 - aM1
    RDA2 = dM2 - aM2
    RDA3 = dM3 - aM3

    # Distance between midpoints
    dRDA = np.sqrt(RDA1 ** 2 + RDA2 ** 2 + RDA3 ** 2)

    # Normalized connecting vector
    nRDA1 = RDA1 / dRDA
    nRDA2 = RDA2 / dRDA
    nRDA3 = RDA3 / dRDA

    # Orientation factor kappa
    kappa = (muA1 * muD1 + muA2 * muD2 + muA3 * muD3 -
             3.0 * (muD1 * nRDA1 + muD2 * nRDA2 + muD3 * nRDA3) *
             (muA1 * nRDA1 + muA2 * nRDA2 + muA3 * nRDA3))
    return dRDA, kappa


def kappa(
        donor_dipole: np.ndarray,
        acceptor_dipole: np.ndarray
) -> typing.Tuple[float, float]:
    """
    Calculates the orientation factor kappa based on donor and acceptor dipoles.

    Parameters
    ----------
    donor_dipole : np.ndarray
        A 2x3 array representing the donor dipole endpoints.
    acceptor_dipole : np.ndarray
        A 2x3 array representing the acceptor dipole endpoints.

    Returns
    -------
    tuple
        A tuple (distance, kappa) where distance is the center-to-center distance
        and kappa is the orientation factor.

    Example
    -------
    >>> import numpy as np
    >>> donor_dipole = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
    >>> acceptor_dipole = np.array([[0.0, 0.5, 0.0], [0.0, 0.5, 1.0]], dtype=np.float64)
    >>> distance, k = kappa(donor_dipole, acceptor_dipole)
    >>> round(distance, 5)
    0.86603
    >>> round(k, 5)
    1.00000
    """
    return kappa_distance(
        donor_dipole[0], donor_dipole[1],
        acceptor_dipole[0], acceptor_dipole[1]
    )


def s2delta(
        s2_donor: float,
        s2_acceptor: float,
        r_inf_AD: float,
        r_0: float = 0.38
) -> typing.Tuple[float, float]:
    """
    Calculate s2delta from the residual anisotropies of the donor and acceptor.

    Parameters
    ----------
    s2_donor : float
        Second rank order parameter of the donor dye.
    s2_acceptor : float
        Second rank order parameter of the directly excited acceptor dye.
    r_inf_AD : float
        Residual anisotropy on the acceptor excited by the donor dye.
    r_0 : float, optional
        Fundamental anisotropy at time zero (default: 0.38).

    Returns
    -------
    tuple
        A tuple (s2delta, delta) where:
          - s2delta: A computed second rank order parameter.
          - delta: The angle (in radians) between the two dipole symmetry axes.

    Examples
    --------
    >>> from chisurf.fluorescence.anisotropy.kappa2 import s2delta
    >>> r0 = 0.38
    >>> s2donor = 0.2
    >>> s2acceptor = 0.3
    >>> r_inf_AD = 0.01
    >>> s2d, delta = s2delta(s2_donor=s2donor, s2_acceptor=s2acceptor, r_inf_AD=r_inf_AD, r_0=r0)
    >>> round(s2d, 5)
    0.06977
    >>> round(delta, 5)
    0.6583
    """
    s2_delta = r_inf_AD / (r_0 * s2_donor * s2_acceptor)
    delta = np.arccos(np.sqrt((2.0 * s2_delta + 1.0) / 3.0))
    return s2_delta, delta


def calculate_kappa_distance(
        xyz: np.array,
        aid1: int,
        aid2: int,
        aia1: int,
        aia2: int
) -> typing.Tuple[np.ndarray, np.ndarray]:
    """
    Calculates the dipole center distance and the orientation factor kappa
    over a trajectory.

    This function extracts the coordinates corresponding to the donor
    and acceptor dipole endpoints (specified by atom indices) from a trajectory
    and computes the center-to-center distance and orientation factor for each frame.

    Parameters
    ----------
    xyz : np.array
        A 3D array of shape (n_frames, n_atoms, 3) containing the coordinates.
    aid1 : int
        Atom index for the first point of the donor dipole.
    aid2 : int
        Atom index for the second point of the donor dipole.
    aia1 : int
        Atom index for the first point of the acceptor dipole.
    aia2 : int
        Atom index for the second point of the acceptor dipole.

    Returns
    -------
    tuple
        A tuple (ds, ks) where:
          - ds (np.ndarray): Array of distances between dipole centers for each frame.
          - ks (np.ndarray): Array of corresponding orientation factors kappa.

    Examples
    --------
    >>> import numpy as np
    >>> # Create a simple trajectory with one frame and four atoms
    >>> xyz = np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]], dtype=np.float64)
    >>> ds, ks = calculate_kappa_distance(xyz, 0, 1, 2, 3)
    >>> ds.shape, ks.shape  # doctest: +SKIP
    ((1,), (1,))
    """
    n_frames = xyz.shape[0]
    ks = np.empty(n_frames, dtype=np.float32)
    ds = np.empty(n_frames, dtype=np.float32)

    for i_frame in range(n_frames):
        try:
            d, k = kappa_distance(
                xyz[i_frame, aid1], xyz[i_frame, aid2],
                xyz[i_frame, aia1], xyz[i_frame, aia2]
            )
            ks[i_frame] = k
            ds[i_frame] = d
        except Exception:
            print("Frame ", i_frame, "skipped, calculation error")
    return ds, ks


@nb.jit(nopython=True)
def kappasq(
        delta: float,
        sD2: float,
        sA2: float,
        beta1: float,
        beta2: float
) -> float:
    """
    Calculates kappa² given a set of order parameters and angles.

    This function implements eq. 9 from [1]_, computing the orientation factor
    based on the dye order parameters and the angles between the dye symmetry axes
    and the donor–acceptor vector.

    Parameters
    ----------
    delta : float
        Angle (in radians) between the symmetry axes of the dyes.
    sD2 : float
        Second rank order parameter of the donor.
    sA2 : float
        Second rank order parameter of the acceptor.
    beta1 : float
        Angle (in radians) between the donor dye's symmetry axis and the
        donor–acceptor vector.
    beta2 : float
        Angle (in radians) between the acceptor dye's symmetry axis and the
        donor–acceptor vector.

    Returns
    -------
    float
        The computed kappa² value.

    Notes
    -----
    See eq. 9 in [1]_ for details.

    References
    ----------
    .. [1] Simon Sindbert, et al., "Accurate Distance Determination of Nucleic
           Acids via Foerster Resonance Energy Transfer: Implications of Dye Linker
           Length and Rigidity", J. Am. Chem. Soc., 2011.
    """
    s2delta = (3.0 * np.cos(delta) * np.cos(delta) - 1.0) / 2.0
    s2beta1 = (3.0 * np.cos(beta1) * np.cos(beta1) - 1.0) / 2.0
    s2beta2 = (3.0 * np.cos(beta2) * np.cos(beta2) - 1.0) / 2.0
    k2 = 2.0 / 3.0 * (
            1.0 +
            sD2 * s2beta1 +
            sA2 * s2beta2 +
            sD2 * sA2 * (
                    s2delta +
                    6 * s2beta1 * s2beta2 +
                    1 +
                    2 * s2beta1 +
                    2 * s2beta2 -
                    9 * np.cos(beta1) * np.cos(beta2) * np.cos(delta)
            )
    )
    return k2


def p_isotropic_orientation_factor(
        k2: np.ndarray,
        normalize: bool = True
) -> np.ndarray:
    """
    Calculates the probability distribution of kappa² for isotropically oriented dipoles.

    Given an array of kappa² values, this function computes the corresponding
    probability distribution assuming an isotropic orientation factor distribution.

    Parameters
    ----------
    k2 : np.ndarray
        Array of kappa² values.
    normalize : bool, optional
        If True (default), the output distribution is normalized to unity.

    Returns
    -------
    np.ndarray
        The probability distribution for the provided kappa² values.

    Example
    -------
    >>> import numpy as np
    >>> k2 = np.linspace(0.1, 4, 32)
    >>> p_k2 = p_isotropic_orientation_factor(k2=k2)
    >>> p_k2  # doctest: +SKIP
    array([0.17922824, 0.11927194, 0.09558154, 0.08202693, 0.07297372,
           0.06637936, 0.06130055, 0.05723353, 0.04075886, 0.03302977,
           0.0276794 , 0.02359627, 0.02032998, 0.01763876, 0.01537433,
           0.01343829, 0.01176177, 0.01029467, 0.00899941, 0.00784718,
           0.00681541, 0.00588615, 0.00504489, 0.0042798 , 0.0035811 ,
           0.00294063, 0.00235153, 0.001808  , 0.00130506, 0.00083845,
           0.0004045 , 0.        ])
    Notes
    -----
    For more details on isotropic kappa² distributions, see:
    http://www.fretresearch.org/kappasquaredchapter.pdf
    """
    ks = np.sqrt(k2)
    s3 = np.sqrt(3.)
    r = np.zeros_like(k2)
    for i, k in enumerate(ks):
        if 0 <= k <= 1:
            r[i] = 0.5 / (s3 * k) * np.log(2 + s3)
        elif 1 <= k <= 2:
            r[i] = 0.5 / (s3 * k) * np.log((2 + s3) / (k + np.sqrt(k ** 2 - 1.0)))
    if normalize:
        r /= max(1.0, r.sum())
    return r

