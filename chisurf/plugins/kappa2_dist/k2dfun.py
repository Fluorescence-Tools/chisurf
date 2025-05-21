import numba as nb
import numpy as np
import typing


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

    Parameters
    ----------
    sD2 : float
        Second rank order parameter S2 of the donor dye. This can correspond
        to the fraction of trapped donor dye.
    sA2 : float
        Second rank order parameter S2 of the acceptor dye. This can correspond
        to the fraction of trapped acceptor dye.
    fret_efficiency : float
        FRET efficiency
    n_bins : int
        The number of bins in the kappa2 distribution that is
        generated.
    k2_max : float
        Upper kappa2 bound in the generated histogram
    k2_min : float
        Lower kappa2 bound in the generate histogram
    n_samples : int
        The number random vector pairs that are drawn (default: 10000)

    Returns
    -------
    """

    # Binning of the kappa2 distribution
    k2_step = (k2_max - k2_min) / (n_bins - 1)
    k2_scale = np.arange(k2_min, k2_max + 1e-14, k2_step, dtype=np.float64)

    # Generate random orientations for the TDM vectors
    donor_vec = np.random.randn(n_samples, 3)
    acceptor_vec = np.random.randn(n_samples, 3)

    # x = (R_DA/R_0)^6; relative DA distance
    x = 1 / fret_efficiency - 1

    k2s = np.zeros(n_samples, dtype=np.float64)
    for i in range(n_samples):
        donor = donor_vec[i]
        acceptor = acceptor_vec[i]
        # Assumption here: connecting vector R_DA is along the x-axis (R_DA=[1,0,0])
        delta = np.arccos(np.dot(donor, acceptor) / (np.linalg.norm(donor) * np.linalg.norm(acceptor)))
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
        ##
        Ek2 = (1 - sD2) * (1 - sA2) / (1 + x) + sD2 * sA2 / (1 + 2 / 3. / k2_trapped_trapped * x) + sD2 * (1 - sA2) / (
                    1 + 2 / 3. / k2_trapped_free * x) + (1 - sD2) * sA2 / (1 + 2 / 3. / k2_free_trapped * x)
        k2 = 2 / 3. * x / (1 / Ek2 - 1)
        k2s[i] = k2

    # create the histogram
    k2hist, bins = np.histogram(k2s, k2_scale)
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
    ks = list()
    weights = list()
    for beta1 in np.arange(0, np.pi/2, step=step / 180. * np.pi):
        weight_beta1 = np.sin(beta1)
        for beta2 in np.arange(
                start=abs(delta - beta1),
                stop=min(delta + beta1, np.pi / 2.),
                step=step / 180. * np.pi
        ):
            weight_beta2 = np.sin(beta1)
            weights.append(
                weight_beta1 * weight_beta2
            )
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

    # histogram bin edges
    k2_step = (k2_max - k2_min) / (n_bins - 1)
    k2scale = np.arange(k2_min, k2_max + 1e-14, k2_step, dtype=np.float64)

    k2hist, x = np.histogram(ks, bins=k2scale, weights=weights)
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
    """Computes a orientation factor distribution for a wobbling in a cone model
    using parameters that can be estimated by experimental anisotropies.

    The function used second rank order parameter of the donor and acceptor
    and the angle delta between the symmetry axes of the dyes as input. These
    parameters can be estimated by the residual anisotropy the the dyes. The
    second rank order parameter of the donor and acceptor are estimated by the
    dye's residual anisotropies. The angle between the symmetry axes is estimated
    by the residual anisotropy of the FRET sensitized emission (see:
    `chisurf.fluorescence.anisotropy.kappa2.s2delta`).

    This function computes a orientation factor distribution, :math:`p(/kappa^2)`,
    for a wobbling in a cone model (WIC) for second rank structure factors of
    the donor and acceptor, and an angle :math:`delta`. The angle
    :math:`delta` is the angle between the symmetry axes of the dyes and can be
    estimated using experimental residual anisotropies [1]_.

    Parameters
    ----------
    delta : float
        The angle delta (in rad) for which the WIC orientation factor
        distribution is calculated.
    sD2 : float
        Second rank order parameter S2 of the donor dye. This can correspond
        to the fraction of trapped donor dye.
    sA2 : float
        Second rank order parameter S2 of the acceptor dye. This can correspond
        to the fraction of trapped acceptor dye.
    step : float
        The step size in degrees that is used to sample the
        angles.
    n_bins : int
        The number of bins in the kappa2 distribution that is
        generated.
    k2_max : float
        Upper kappa2 bound in the generated histogram
    k2_min : float
        Lower kappa2 bound in the generate histogram

    Returns
    -------
    k2scale : numpy-array
        A linear scale in the range of [0, 4] with *n_bins* elements
    k2hist : numpy-array
        The histogram of kappa2 values
    k2 : numpy-array
        A numpy-array containing all computed kappa2 values. The histogram
        corresponds to a histogram over all returned kappa2 values.

    Examples
    --------
    >>> from scikit_fluorescence.modeling.kappa2 import kappasq_all_delta
    >>> k2s, k2h, k2v = kappasq_all_delta(
    ...     delta=0.2,
    ...     sD2=0.15,
    ...     sA2=0.25,
    ...     step=2.0,
    ...     n_bins=31
    ... )
    >>> np.allclose(k2h, np.array([   0.        ,    0.        ,    0.        ,    0.        ,
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
    The angle :math:`/beta_1` is varied in the range (0,pi/2)
    The angle :math:`/phi` is varied in the range (0, 2 pi)

    References
    ----------
    .. [1] Simon Sindbert, Stanislav Kalinin, Hien Nguyen, Andrea Kienzler,
    Lilia Clima, Willi Bannwarth, Bettina Appel, Sabine Mueller, Claus A. M.
    Seidel, "Accurate Distance Determination of Nucleic Acids via Foerster
    Resonance Energy Transfer: Implications of Dye Linker Length and Rigidity"
    vol. 133, pp. 2463-2480, J. Am. Chem. Soc., 2011

    """
    # beta angles
    beta1 = np.arange(0.001, np.pi / 2.0, step * np.pi / 180.0, dtype=np.float64)
    phi = np.arange(0.001, 2.0 * np.pi, step * np.pi / 180.0, dtype=np.float64)
    n = beta1.shape[0]
    m = phi.shape[0]
    rda_vec = np.array([1, 0, 0], dtype=np.float64)

    # kappa-square values for allowed betas
    k2 = np.zeros((n, m), dtype=np.float64)
    k2hist = np.zeros(n_bins - 1, dtype=np.float64)

    # histogram bin edges
    k2_step = (k2_max - k2_min) / (n_bins - 1)
    k2scale = np.arange(k2_min, k2_max + 1e-14, k2_step, dtype=np.float64)
    for i in range(n):
        d1 = np.array([np.cos(beta1[i]),  0, np.sin(beta1[i])])
        n1 = np.array([-np.sin(beta1[i]), 0, np.cos(beta1[i])])
        n2 = np.array([0, 1, 0])
        for j in range(m):
            d2 = (n1*np.cos(phi[j])+n2*np.sin(phi[j]))*np.sin(delta)+d1*np.cos(delta)
            beta2 = np.arccos(np.abs(d2.dot(rda_vec)))
            k2[i, j] = kappasq(
                delta=delta,
                sD2=sD2,
                sA2=sA2,
                beta1=beta1[i],
                beta2=beta2
            )
        y, x = np.histogram(k2[i, :], bins=k2scale)
        k2hist += y*np.sin(beta1[i])
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
    """Computes a orientation factor distribution for a wobbling in a cone model
    using specific second rank structure factors of the donor and acceptor.

    This function computes a orientation factor distribution, :math:`p(/kappa^2)`,
    for a wobbling in a cone model (WIC) for second rank structure factors of
    the donor and acceptor estimated using experimental residual anisotropies [1]_.

    Parameters
    ----------
    sD2 : float
        Second rank order parameter S2 of the donor dye
    sA2 : float
        Second rank order parameter S2 of the acceptor dye
    n_bins : int
        The number of bins in the kappa2 histogram that is
        generated.
    k2_max : float
        Upper kappa2 bound in the generated histogram
    k2_min : float
        Lower kappa2 bound in the generate histogram
    n_samples : int
        The number random vector pairs that are drawn (default: 10000)

    Returns
    -------
    k2scale : numpy-array
        A linear scale in the range of [0, 4] with *n_bins* elements
    k2hist : numpy-array
        The histogram of kappa2 values
    k2 : numpy-array
        A numpy-array containing all computed kappa2 values. The histogram
        corresponds to a histogram over all returned kappa2 values.

    Examples
    --------
    >>> from scikit_fluorescence.modeling.kappa2 import kappasq_all_delta
    >>> k2_scale, k2_hist, k2 = kappasq_all(
    ...     sD2=0.3,
    ...     sA2=0.5,
    ...     n_bins=31,
    ...     n_samples=100000
    ... )
    >>> k2_scale
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
    .. [1] Simon Sindbert, Stanislav Kalinin, Hien Nguyen, Andrea Kienzler,
    Lilia Clima, Willi Bannwarth, Bettina Appel, Sabine Mueller, Claus A. M.
    Seidel, "Accurate Distance Determination of Nucleic Acids via Foerster
    Resonance Energy Transfer: Implications of Dye Linker Length and Rigidity"
    vol. 133, pp. 2463-2480, J. Am. Chem. Soc., 2011

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
        # Assumption here: connecting vector R_DA is along the x-axis (R_DA=[1,0,0])
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
    y, x = np.histogram(k2, bins=k2scale)
    k2hist += y
    return k2scale, k2hist, k2


@nb.jit(nopython=True)
def kappa_distance(
        d1: np.array,
        d2: np.array,
        a1: np.array,
        a2: np.array
) -> typing.Tuple[float, float]:
    """Calculates the distance between the center of two dipoles and the
    orientation-factor kappa of the dipoles

    Calculates for the vectors d1 and d2 pointing to the donors and the vectors
    a1 and a2 pointing to the ends of the acceptor dipole the orientation
    factor kappa.

    Parameters
    ----------
    d1 : numpy-array
        Vector pointing to the first point of the dipole D
    d2 : numpy-array
        Vector pointing to the second point of the dipole D
    a1 : numpy-array
        Vector pointing to the first point of the dipole A
    a2 : numpy-array
        Vector pointing to the second point of the dipole A

    Returns
    -------
    tuple
        distance between the center of the dipoles and the orientation factor
        for the two dipoles kappa

    Notes
    -----
    The four vectors defining the dipole of the donor :math:`\vec{r}_{D1}` and
    :math:`\vec{r}_{D2}` specified by the parameters `d1` and `d2` and
    :math:`\vec{r}_{A1}` and :math:`\vec{r}_{A2}` specified by the parameters
    `a1` and `a1` are used to compute orientation factor :math:`kappa^2`
    and the distance between the center of the two dipoles :math:`R_{DA}`.

    The distance :math:`R_{DA}` between the dipole centers and :math:`kappa`
    is calculated as follows:

    ..math::

        R_{D,21}=|\vec{r}_{D2} - \vec{r}_{D1}| \\
        R_{A,21}=|\vec{r}_{A2} - \vec{r}_{A1}| \\
        \hat{\mu}_{D}=1/R_{D,21} \cdot (\vec{r}_{D2}-\vec{r}_{D1}) \\
        \hat{\mu}_{A}=1/R_{A,21} \cdot (\vec{r}_{A2}-\vec{r}_{A1}) \\
        \vec{m}_{D}=\vec{r}_{D1}+1/2 \cdot \hat{\mu}_{D} \\
        \vec{m}_{A}=\vec{r}_{A1}+1/2 \cdot \hat{\mu}_{A} \\
        \vec{r}_{DA}=\vec{m}_{D}-\vec{m}_{A} \\
        R_{DA}=|\vec{m}_{D}-\vec{m}_{A}| \\
        \hat{\mu}_{DA}=\vec{r}_{DA} / R_{DA} \\
        \kappa=\langle\mu_A,\mu_D\rangle-3\cdot\langle\mu_D,\mu_{DA}\rangle \cdot \langle\mu_A,\mu_{DA}\rangle

    Examples
    --------
    >>> import scikit_fluorescence.modeling.kappa2
    >>> donor_dipole = np.array(
    ...      [
    ...          [0.0, 0.0, 0.0],
    ...          [1.0, 0.0, 0.0]
    ...      ], dtype=np.float64
    ... )
    >>> acceptor_dipole = np.array(
    ...     [
    ...         [0.0, 0.5, 0.0],
    ...         [0.0, 0.5, 1.0]
    ...     ], dtype=np.float64
    ... )
    >>> scikit_fluorescence.modeling.kappa2.kappa(
    ...     donor_dipole,
    ...     acceptor_dipole
    ... ) 
    (0.8660254037844386, 1.0000000000000002)

    """
    # coordinates of the dipole
    d11 = d1[0]
    d12 = d1[1]
    d13 = d1[2]

    d21 = d2[0]
    d22 = d2[1]
    d23 = d2[2]

    # distance between the two end points of the donor
    dD21 = np.sqrt(
        (d11 - d21) * (d11 - d21) +
        (d12 - d22) * (d12 - d22) +
        (d13 - d23) * (d13 - d23)
    )

    # normal vector of the donor-dipole
    muD1 = (d21 - d11) / dD21
    muD2 = (d22 - d12) / dD21
    muD3 = (d23 - d13) / dD21

    # vector to the middle of the donor-dipole
    dM1 = d11 + dD21 * muD1 / 2.0
    dM2 = d12 + dD21 * muD2 / 2.0
    dM3 = d13 + dD21 * muD3 / 2.0

    ### Acceptor ###
    # cartesian coordinates of the acceptor
    a11 = a1[0]
    a12 = a1[1]
    a13 = a1[2]

    a21 = a2[0]
    a22 = a2[1]
    a23 = a2[2]

    # distance between the two end points of the acceptor
    dA21 = np.sqrt(
        (a11 - a21) * (a11 - a21) +
        (a12 - a22) * (a12 - a22) +
        (a13 - a23) * (a13 - a23)
    )

    # normal vector of the acceptor-dipole
    muA1 = (a21 - a11) / dA21
    muA2 = (a22 - a12) / dA21
    muA3 = (a23 - a13) / dA21

    # vector to the middle of the acceptor-dipole
    aM1 = a11 + dA21 * muA1 / 2.0
    aM2 = a12 + dA21 * muA2 / 2.0
    aM3 = a13 + dA21 * muA3 / 2.0

    # vector connecting the middle of the dipoles
    RDA1 = dM1 - aM1
    RDA2 = dM2 - aM2
    RDA3 = dM3 - aM3

    # Length of the dipole-dipole vector (distance)
    dRDA = np.sqrt(RDA1 * RDA1 + RDA2 * RDA2 + RDA3 * RDA3)

    # Normalized dipole-diple vector
    nRDA1 = RDA1 / dRDA
    nRDA2 = RDA2 / dRDA
    nRDA3 = RDA3 / dRDA

    # Orientation factor kappa2
    kappa = muA1 * muD1 + \
            muA2 * muD2 + \
            muA3 * muD3 - \
            3.0 * (muD1 * nRDA1 + muD2 * nRDA2 + muD3 * nRDA3) * \
            (muA1 * nRDA1 + muA2 * nRDA2 + muA3 * nRDA3)
    return dRDA, kappa


def kappa(
        donor_dipole: np.ndarray,
        acceptor_dipole: np.ndarray
) -> typing.Tuple[float, float]:
    """Calculates the orientation-factor kappa

    :param donor_dipole: 2x3 vector of the donor-dipole
    :param acceptor_dipole: 2x3 vector of the acceptor-dipole
    :return: distance, kappa

    Example
    -------

    >>> import numpy as np
    >>> donor_dipole = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
    >>> acceptor_dipole = np.array([[0.0, 0.5, 0.0], [0.0, 0.5, 1.0]], dtype=np.float64)
    >>> kappa(donor_dipole, acceptor_dipole)
    (0.8660254037844386, 1.0000000000000002)
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
    """Calculate s2delta from the residual anisotropies of the donor and acceptor

    Parameters
    ----------
    r_0 : float
        Fundamental anisotropy, the anisotropy of the dyes at time zero (
        default value 0.4)
    s2_donor : float
        The second rank oder parameter of the donor dye. The second rank oder
        parameter can be computed using the dye's residual anisotropy (see
        Notes below)
    s2_acceptor : float
        The second rank oder parameter of the direct excited acceptor dye.
    r_inf_AD : float
        The residual anisotropy on the acceptor excited by the donor dye.

    Returns
    -------
    s2delta : float
         A second rank order parameter of the angle [1]_ eq. 10
    delta : float
        The angle between the two symmetry axes of the dipols in units of rad.

    Examples
    --------
    >>> from scikit_fluorescence.modeling.kappa2 import s2delta
    >>> r0 = 0.38
    >>> s2donor = 0.2
    >>> s2acceptor = 0.3
    >>> r_inf_AD = 0.01
    >>> s2delta(
    ...     r_0=r0,
    ...     s2_donor=s2donor,
    ...     s2_acceptor=s2acceptor,
    ...     r_inf_AD=r_inf_AD
    ... )
    (0.4385964912280701, 0.6583029208008411)

    Notes
    -----
    The parameters `s2_donor` and `s2_acceptor`, which correspond to :math:`S^{(2)}_D`
    and :math:`S^{(2)}_A` are calculated using the dye's residual anisotropy [1]_

    ..math::

        S^{(2)}_D = - /sqrt{/frac{r_{D,inf}}{r_{0,D}}} \\
        S^{(2)}_D = /sqrt{/frac{r_{A,inf}}{r_{0,A}}} \\

    References
    ----------

    .. [1] Simon Sindbert, Stanislav Kalinin, Hien Nguyen, Andrea Kienzler,
    Lilia Clima, Willi Bannwarth, Bettina Appel, Sabine Mueller, Claus A. M.
    Seidel, "Accurate Distance Determination of Nucleic Acids via Foerster
    Resonance Energy Transfer: Implications of Dye Linker Length and Rigidity"
    vol. 133, pp. 2463-2480, J. Am. Chem. Soc., 2011

    """
    s2_delta = r_inf_AD/(r_0 * s2_donor * s2_acceptor)
    delta = np.arccos(np.sqrt((2.0 * s2_delta + 1.0) / 3.0))
    return s2_delta, delta


def calculate_kappa_distance(
        xyz: np.array,
        aid1: int,
        aid2: int,
        aia1: int,
        aia2: int
) -> typing.Tuple[np.ndarray, np.ndarray]:
    """Calculates the orientation factor kappa2 and the distance of a
    trajectory given the atom-indices of the donor and the acceptor.

    :param xyz: numpy-array (frame, atom, xyz)
    :param aid1: int, atom-index of d-dipole 1
    :param aid2: int, atom-index of d-dipole 2
    :param aia1: int, atom-index of a-dipole 1
    :param aia2: int, atom-index of a-dipole 2

    :return: distances, kappa2
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
        except:
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
    """Calculates kappa2 given a set of oder parameters and angles

    Parameters
    ----------
    delta : float
        The angle between the symmetry axis of rotation of the dyes in units
        of rad.
    sD2 : float
        The second rank oder parameter of the donor
    sA2 : float
        The second rank oder parameter of the acceptor
    beta1 : float
        The angle between the symmetry axes of the rotation of the dye and
        the distance vector RDA between the two dipoles
    beta2
        The angle between the symmetry axes of the rotation of the dye and
        the distance vector RDA between the two dipoles

    Returns
    -------
    kappa2 : float
        The orientation factor that corresponds to the provided angles.

    Notes
    -----

    This function corresponds to eq. 9 in [1]_.

    References
    ----------
    .. [1] Simon Sindbert, Stanislav Kalinin, Hien Nguyen, Andrea Kienzler,
    Lilia Clima, Willi Bannwarth, Bettina Appel, Sabine Mueller, Claus A. M.
    Seidel, "Accurate Distance Determination of Nucleic Acids via Foerster
    Resonance Energy Transfer: Implications of Dye Linker Length and Rigidity"
    vol. 133, pp. 2463-2480, J. Am. Chem. Soc., 2011

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
    """Calculates an the probability of a given kappa2 according to
    an isotropic orientation factor distribution

    Parameters
    ----------
    k2 : numpy-array
        An array containing kappa squared values.
    normalize : bool
        If this parameter is set to True (default) the returned distribution is
        normalized to unity.

    Returns
    -------
    p_k2 : numpy-array
        The probability distribution of kappa2 for isotropic oriented dipoles


    Example
    -------
    >>> import scikit_fluorescence.modeling.kappa2
    >>> k2 = np.linspace(0.1, 4, 32)
    >>> p_k2 = scikit_fluorescence.modeling.kappa2.p_isotropic_orientation_factor(k2=k2)
    >>> p_k2
    array([0.17922824, 0.11927194, 0.09558154, 0.08202693, 0.07297372,
           0.06637936, 0.06130055, 0.05723353, 0.04075886, 0.03302977,
           0.0276794 , 0.02359627, 0.02032998, 0.01763876, 0.01537433,
           0.01343829, 0.01176177, 0.01029467, 0.00899941, 0.00784718,
           0.00681541, 0.00588615, 0.00504489, 0.0042798 , 0.0035811 ,
           0.00294063, 0.00235153, 0.001808  , 0.00130506, 0.00083845,
           0.0004045 , 0.        ])

    Notes
    -----
    http://www.fretresearch.org/kappasquaredchapter.pdf

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
