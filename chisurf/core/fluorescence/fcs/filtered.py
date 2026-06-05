import numpy as np


def calc_lifetime_filter(
        decays,
        experimental_decay,
        normalize_patterns: bool = True
) -> np.array:
    """
    Calculate lifetime filters for fluorescence lifetime correlation spectroscopy.

    This function computes filter coefficients for lifetime‐filtered correlations
    following the method described by Enderlein and co‐workers [1]. Given a list
    of fluorescence decay patterns (representing different decay components) and
    an experimental decay curve (assumed to be a linear combination of the patterns),
    the algorithm derives a filter matrix that can be used to extract the individual
    component contributions from the experimental decay.

    The procedure is as follows:

      1. Optionally normalize each decay pattern so that its sum equals one.
      2. Stack the (normalized) decay patterns into a matrix M.
      3. Form a diagonal matrix D with elements given by the inverse of the
         experimental decay.
      4. Compute the pseudo-inverse of the product M · D · Mᵀ.
      5. Multiply the pseudo-inverse with M and D to obtain the filter matrix R.
      6. If normalization is enabled, scale the filters so that the sum of the
         filter responses to the experimental decay equals one.

    Parameters
    ----------
    decays : list of np.array
        A list of 1D arrays representing fluorescence decay curves for different
        components.
    experimental_decay : np.array
        A 1D array representing the experimental fluorescence decay (a linear
        combination of the decay patterns).
    normalize_patterns : bool, optional
        If True (default), each decay curve is normalized such that its sum equals one
        before computing the filters.

    Returns
    -------
    np.array
        A 2D array of filter coefficients. Each row corresponds to a filter for one
        decay component.

    Examples
    --------
    >>> import numpy as np
    >>> lifetime_1 = 1.0
    >>> lifetime_2 = 3.0
    >>> times = np.linspace(0, 20, num=10)
    >>> d1 = np.exp(-times/lifetime_1)
    >>> d2 = np.exp(-times/lifetime_2)
    >>> decays = [d1, d2]
    >>> w1 = 0.8  # weight of the first component
    >>> experimental_decay = w1 * d1 + (1.0 - w1) * d2
    >>> filters = calc_lifetime_filter(decays, experimental_decay)
    >>> filters  # doctest: +SKIP
    array([[ 1.19397553, -0.42328685, -1.94651679, -2.57788423, -2.74922322,
            -2.78989942, -2.79923872, -2.80136643, -2.80185031, -2.80196031],
           [-0.19397553,  1.42328685,  2.94651679,  3.57788423,  3.74922322,
             3.78989942,  3.79923872,  3.80136643,  3.80185031,  3.80196031]])

    References
    ----------
    [1] Kapusta, P., Wahl, M., Benda, A., Hof, M., & Enderlein, J. (2007).
        Fluorescence Lifetime Correlation Spectroscopy. Journal of Fluorescence,
        17, 43-48.
    [2] Enderlein, J., & Erdmann, R. (1997). Fast fitting of multi-exponential decay
        curves. Optics Communications, 134, 371-378.
    [3] Bohmer, M., Wahl, M., Rahn, H.-J., Erdmann, R., & Enderlein, J. (2002).
        Time-resolved fluorescence correlation spectroscopy. Chemical Physics Letters,
        353, 439-445.
    """
    # Normalize the fluorescence decays serving as references.
    if normalize_patterns:
        decay_patterns = [decay / decay.sum() for decay in decays]
    else:
        decay_patterns = decays
    d = np.diag(1. / experimental_decay)
    m = np.stack(decay_patterns)
    iv = np.linalg.pinv(np.dot(m, np.dot(d, m.T)))
    r = np.dot(np.dot(iv, m), d)
    if normalize_patterns:
        w = np.array([np.dot(fi, experimental_decay) for fi in r]).sum()
        r /= w
    return r


def calc_ffcs_filters(
        experimental_decay,
        species_decays,
):
    """Compute fFCS-style lifetime filters and reconstruction.

    This helper mirrors the weighted least-squares scheme used in PAM's
    ``Calc_fFCS_Filters`` implementation for filtered FCS/FLCS.

    Parameters
    ----------
    experimental_decay : array_like
        1D array with the total fluorescence decay (one value per TAC bin).
    species_decays : sequence of array_like
        Sequence of 1D arrays with species- (or pattern-) specific decays.
        Each element must have the same length as ``experimental_decay``.

    Returns
    -------
    filters : np.ndarray
        2D array with shape ``(n_species, n_bins)`` containing the filters
        (one row per species / pattern).
    reconstruction : np.ndarray
        1D array with the reconstructed total decay, obtained from the
        normalized patterns and filter matrix.
    weighted_residuals : np.ndarray
        1D array with weighted residuals,

        ``(experimental_decay - reconstruction) / sqrt(experimental_decay_safe)``.
    """

    import numpy as np

    y = np.asarray(experimental_decay, dtype=float).copy()
    if y.ndim != 1:
        raise ValueError("experimental_decay must be a 1D array")

    D = np.column_stack([np.asarray(v, dtype=float) for v in species_decays])
    if D.ndim != 2:
        raise ValueError("species_decays must form a 2D matrix (bins × species)")

    # Normalize columns of D (Decay_par = Decay_par ./ sum(Decay_par,1))
    col_sums = D.sum(axis=0)
    col_sums[col_sums == 0.0] = 1.0
    D_norm = D / col_sums

    # Protect against zeros in the total decay before building the weights
    y_safe = y.copy()
    y_safe[y_safe == 0.0] = 1.0

    # Weight vector w = 1 / y
    w = 1.0 / y_safe

    # Compute G = D^T W D efficiently without forming the full diagonal W
    # DW = D^T * w  (each column j of D^T scaled by w_j)
    DW = (D_norm.T * w)
    G = DW @ D_norm  # shape: (n_species, n_species)

    # Invert G robustly; fall back to pseudo-inverse on failure
    try:
        G_inv = np.linalg.inv(G)
    except np.linalg.LinAlgError:
        G_inv = np.linalg.pinv(G)

    # Filters: F = G^{-1} D^T W = G_inv @ DW
    filters = G_inv @ DW  # shape: (n_species, n_bins)

    # Reconstruction and weighted residuals as in the PAM implementation
    # reconstruction = sum( (D^T W D)^{-1} D^T , 1)
    A = G_inv @ D_norm.T  # unweighted version for reconstruction
    reconstruction = A.sum(axis=0)  # shape: (n_bins,)

    # Weighted residuals = (total - reconstruction) ./ sqrt(total)
    denom = np.sqrt(y_safe)
    weighted_residuals = (y - reconstruction) / denom

    return filters, reconstruction, weighted_residuals
