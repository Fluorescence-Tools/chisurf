from __future__ import annotations

from typing import Optional, Tuple, Union

import numba as nb
import numpy as np

from .av import AccessibleVolume

N_DISTANCE_SAMPLES: int = 50000


@nb.njit
def _random_distances(
    p1: np.ndarray,
    p2: np.ndarray,
    n_samples: int,
    seed: int = 0,
) -> np.ndarray:
    """Draw random distance-weight pairs from two AV point clouds.

    Returns (n_samples, 2) array: col 0 = distance, col 1 = weight product.
    """
    np.random.seed(seed)
    n1 = p1.shape[0]
    n2 = p2.shape[0]
    result = np.empty((n_samples, 2), dtype=np.float64)
    for i in range(n_samples):
        i1 = np.random.randint(0, n1)
        i2 = np.random.randint(0, n2)
        dx = p1[i1, 0] - p2[i2, 0]
        dy = p1[i1, 1] - p2[i2, 1]
        dz = p1[i1, 2] - p2[i2, 2]
        result[i, 0] = np.sqrt(dx * dx + dy * dy + dz * dz)
        result[i, 1] = p1[i1, 3] * p2[i2, 3]
    return result


@nb.njit
def _sample_vectors(
    p1: np.ndarray,
    p2: np.ndarray,
    n_samples: int,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Draw random vector-weight pairs from two AV point clouds.

    Parameters
    ----------
    p1 : np.ndarray
        Points in the first accessible volume (N, 4).
    p2 : np.ndarray
        Points in the second accessible volume (M, 4).
    n_samples : int
        Number of random samples to draw.
    seed : int, optional
        Random seed. Default is 0.

    Returns
    -------
    v : np.ndarray
        Sampled vectors from av1 to av2 (n_samples, 3).
    w : np.ndarray
        Sampled weight products (n_samples,).
    """
    np.random.seed(seed)
    n1 = p1.shape[0]
    n2 = p2.shape[0]
    v = np.empty((n_samples, 3), dtype=np.float64)
    w = np.empty(n_samples, dtype=np.float64)
    for i in range(n_samples):
        i1 = np.random.randint(0, n1)
        i2 = np.random.randint(0, n2)
        v[i, 0] = p2[i2, 0] - p1[i1, 0]
        v[i, 1] = p2[i2, 1] - p1[i1, 1]
        v[i, 2] = p2[i2, 2] - p1[i1, 2]
        w[i] = p1[i1, 3] * p2[i2, 3]
    return v, w


def _sample_av_distance(
    av1: AccessibleVolume,
    av2: AccessibleVolume,
    n_samples: int = N_DISTANCE_SAMPLES,
) -> np.ndarray:
    """Sample distances between two AVs."""
    if av1.n_points == 0 or av2.n_points == 0:
        raise ValueError("Cannot sample distance: one or both AVs have no points")
    return _random_distances(av1.points, av2.points, n_samples)


def average_distance(
    av1: AccessibleVolume,
    av2: AccessibleVolume,
    n_samples: int = N_DISTANCE_SAMPLES,
) -> float:
    """Calculate the mean inter-dye distance <R_DA>.

    Uses weighted random sampling from both AV point clouds.
    """
    d = _sample_av_distance(av1, av2, n_samples)
    return float(np.dot(d[:, 0], d[:, 1]) / d[:, 1].sum())


def mean_fret_distance(
    av1: AccessibleVolume,
    av2: AccessibleVolume,
    forster_radius: float = 52.0,
    n_samples: int = N_DISTANCE_SAMPLES,
) -> float:
    """Calculate the FRET-averaged distance R_E.

    R_E = R0 * (1/<E> - 1)^(1/6)
    where <E> is the weighted mean FRET efficiency over the AV distribution.
    """
    d = _sample_av_distance(av1, av2, n_samples)
    r = d[:, 0]
    w = d[:, 1]
    e = 1.0 / (1.0 + (r / forster_radius) ** 6.0)
    mean_e = np.dot(w, e) / w.sum()
    return float(forster_radius * (1.0 / mean_e - 1.0) ** (1.0 / 6.0))


def distance_between_mean_positions(
    av1: AccessibleVolume,
    av2: AccessibleVolume,
) -> float:
    """Calculate the distance between the mean positions (Rmp)."""
    return float(np.sqrt(((av1.mean_position - av2.mean_position) ** 2).sum()))


def standard_deviation_of_distances(
    av1: AccessibleVolume,
    av2: AccessibleVolume,
    n_samples: int = N_DISTANCE_SAMPLES,
) -> float:
    """Calculate the standard deviation of the inter-AV distance distribution."""
    d = _sample_av_distance(av1, av2, n_samples)
    w = d[:, 1]
    mean_r = np.dot(d[:, 0], w) / w.sum()
    var_r = np.dot(d[:, 0] ** 2, w) / w.sum() - mean_r ** 2
    return float(np.sqrt(max(var_r, 0.0)))


def av_pair_statistics(
    av1: AccessibleVolume,
    av2: AccessibleVolume,
    forster_radius: float = 52.0,
    n_samples: int = N_DISTANCE_SAMPLES,
) -> Tuple[float, float, float, float]:
    """Calculate distance statistics between two accessible volumes.

    Parameters
    ----------
    av1 : AccessibleVolume
        The first accessible volume.
    av2 : AccessibleVolume
        The second accessible volume.
    forster_radius : float, optional
        Förster radius for FRET-averaged distance calculation. Default is 52.0.
    n_samples : int, optional
        Number of random distance samples. Default is 50000.

    Returns
    -------
    rmp : float
        Distance between the mean positions (Rmp).
    rda_mean : float
        Mean inter-dye distance <R_DA>.
    rda_mean_e : float
        FRET-averaged distance R_E.
    sigma_r : float
        Standard deviation of the inter-AV distance distribution.
    """
    rmp = distance_between_mean_positions(av1, av2)
    if av1.n_points == 0 or av2.n_points == 0:
        return rmp, rmp, rmp, 0.0

    d = _sample_av_distance(av1, av2, n_samples)
    r = d[:, 0]
    w = d[:, 1]
    w_sum = w.sum()
    if w_sum <= 0:
        return rmp, rmp, rmp, 0.0

    rda_mean = float(np.dot(r, w) / w_sum)

    # FRET-averaged distance R_E
    e = 1.0 / (1.0 + (r / forster_radius) ** 6.0)
    mean_e = np.dot(w, e) / w_sum
    if mean_e <= 0:
        rda_mean_e = rda_mean
    elif mean_e >= 1:
        rda_mean_e = 0.0
    else:
        rda_mean_e = float(forster_radius * (1.0 / mean_e - 1.0) ** (1.0 / 6.0))

    # Standard deviation of distance distribution
    var_r = np.dot(r ** 2, w) / w_sum - rda_mean ** 2
    sigma_r = float(np.sqrt(max(var_r, 0.0)))

    return rmp, rda_mean, rda_mean_e, sigma_r


def fit_transfer_polynomial(
    av1: AccessibleVolume,
    av2: AccessibleVolume,
    distance_type: str,
    forster_radius: float = 52.0,
    degree: int = 3,
    n_samples: int = 10000,
) -> np.ndarray:
    """Fit a polynomial relating Rmp to RDAMean or RDAMeanE by translating the AVs.

    Parameters
    ----------
    av1 : AccessibleVolume
        The first accessible volume.
    av2 : AccessibleVolume
        The second accessible volume.
    distance_type : str
        The target distance type ('RDAMean' or 'RDAMeanE').
    forster_radius : float, optional
        Förster radius for FRET-averaged distance calculation. Default is 52.0.
    degree : int, optional
        Degree of the polynomial. Default is 3.
    n_samples : int, optional
        Number of samples for fit. Default is 10000.

    Returns
    -------
    coeffs : np.ndarray
        Polynomial coefficients [c_0, c_1, c_2, ...].
    """
    rmp_initial = distance_between_mean_positions(av1, av2)
    if av1.n_points == 0 or av2.n_points == 0 or rmp_initial <= 1e-6:
        # Return identity polynomial: y = x
        coeffs = np.zeros(degree + 1)
        coeffs[-2] = 1.0  # slope = 1.0
        return coeffs

    # Draw point cloud relative vectors
    v, w = _sample_vectors(av1.points, av2.points, n_samples)
    w_sum = w.sum()
    if w_sum <= 0:
        coeffs = np.zeros(degree + 1)
        coeffs[-2] = 1.0
        return coeffs

    # Unit vector along the line connecting mean positions
    u = (av2.mean_position - av1.mean_position) / rmp_initial

    # Generate test offsets
    t_min = -min(rmp_initial - 5.0, 15.0)
    t_max = 20.0
    t_values = np.linspace(t_min, t_max, 7)

    rmp_values = rmp_initial + t_values
    r_eff_values = np.empty(len(t_values))

    # Pre-calculate dot product for each vector in sample with u
    v_dot_u = v @ u
    # Pre-calculate squared norm of each vector
    v_sq = np.sum(v ** 2, axis=1)

    for i, t in enumerate(t_values):
        # New distance for each pair is: sqrt(v_sq + 2*t*(v_dot_u) + t^2)
        d_new = np.sqrt(np.maximum(v_sq + 2.0 * t * v_dot_u + t**2, 1e-10))
        if distance_type == "RDAMeanE":
            e = 1.0 / (1.0 + (d_new / forster_radius) ** 6.0)
            mean_e = np.dot(w, e) / w_sum
            if mean_e <= 0:
                r_eff = float(np.dot(d_new, w) / w_sum)
            elif mean_e >= 1:
                r_eff = 0.0
            else:
                r_eff = float(forster_radius * (1.0 / mean_e - 1.0) ** (1.0 / 6.0))
        else:  # RDAMean
            r_eff = float(np.dot(d_new, w) / w_sum)
        r_eff_values[i] = r_eff

    # Fit polynomial
    coeffs = np.polyfit(rmp_values, r_eff_values, degree)
    return coeffs


def polynomial_transfer(
    rmp: Union[float, np.ndarray], coeffs: np.ndarray
) -> Union[float, np.ndarray]:
    """Evaluate polynomial transfer function using Horner's method.

    Parameters
    ----------
    rmp : float or np.ndarray
        Distance(s) to evaluate.
    coeffs : np.ndarray
        Polynomial coefficients (highest power first, i.e., from np.polyfit).

    Returns
    -------
    float or np.ndarray
        Evaluated value(s).
    """
    res = 0.0
    for c in coeffs:
        res = res * rmp + c
    return res


def gaussian_rmp_to_rda_mean(
    rmp: Union[float, np.ndarray], sigma: float
) -> Union[float, np.ndarray]:
    """Apply Gaussian width correction to convert Rmp to RDAMean.

    RDAMean ≈ Rmp + (sigma^2 / (2 * rmp))

    Parameters
    ----------
    rmp : float or np.ndarray
        Mean position distance(s).
    sigma : float
        Standard deviation of the inter-dye distance distribution.

    Returns
    -------
    float or np.ndarray
        Corrected distance(s).
    """
    return rmp + (sigma ** 2) / (2.0 * np.maximum(rmp, 1e-10))


def histogram_rda(
    av1: AccessibleVolume,
    av2: AccessibleVolume,
    rda_axis: Optional[np.ndarray] = None,
    rda_min: float = 1.0,
    rda_max: float = 200.0,
    n_rda_bins: int = 100,
    use_log: bool = False,
    n_samples: int = N_DISTANCE_SAMPLES,
    normalize: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute a histogram of the inter-AV distance distribution.

    Returns (histogram, bin_edges).
    """
    if rda_axis is None:
        if use_log:
            rda_axis = np.logspace(
                np.log10(rda_min), np.log10(rda_max), n_rda_bins, dtype=np.float64
            )
        else:
            rda_axis = np.linspace(rda_min, rda_max, n_rda_bins, dtype=np.float64)

    d = _sample_av_distance(av1, av2, n_samples)
    p = np.histogram(d[:, 0], bins=rda_axis, weights=d[:, 1])[0]
    if normalize:
        total = p.sum()
        if total > 0:
            p = p / total
    return p, rda_axis


def model_distance(
    av1: AccessibleVolume,
    av2: AccessibleVolume,
    distance_type: str,
    forster_radius: float = 52.0,
    n_samples: int = N_DISTANCE_SAMPLES,
) -> float:
    """Compute a model distance for the given distance type.

    Supported types: ``Rmp``, ``RDAMean``, ``RDAMeanE``.
    """
    if distance_type == "Rmp":
        return distance_between_mean_positions(av1, av2)
    elif distance_type == "RDAMean":
        return average_distance(av1, av2, n_samples=n_samples)
    elif distance_type == "RDAMeanE":
        return mean_fret_distance(av1, av2, forster_radius=forster_radius, n_samples=n_samples)
    else:
        raise ValueError(f"Unknown distance type: {distance_type}")


def chi2_score(
    model_distance: float,
    experimental_distance: float,
    error_neg: float,
    error_pos: float,
) -> float:
    """Asymmetric chi-squared contribution for one distance restraint.

    ``chi2 = (d_m - d_e)^2 / error^2`` with asymmetric errors.
    """
    delta = model_distance - experimental_distance
    err = error_neg if delta < 0 else error_pos
    if err <= 0:
        return 0.0
    return (delta / err) ** 2


def fret_efficiency(distance: float, forster_radius: float = 52.0) -> float:
    """Single-pair FRET efficiency."""
    return 1.0 / (1.0 + (distance / forster_radius) ** 6.0)


def distance_from_fret_efficiency(
    efficiency: float, forster_radius: float = 52.0
) -> float:
    """Convert FRET efficiency back to distance."""
    return forster_radius * (1.0 / efficiency - 1.0) ** (1.0 / 6.0)
