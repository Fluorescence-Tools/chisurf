"""Olga-style greedy informative FRET pair selection (Python port).

This module is a headless implementation of the "informative pair selection" / "experiment planning"
algorithm used by Olga.

Original Olga sources (ChiSurf repo copy):

- `playground/Olga/src/best_dist.h`
  - `greedySelection(...)`
  - `bestPair(...)`
  - `precisionDecay(...)`
  - `rmsdMeanMean(...)`, `rmsdMeanMeanAdd(...)`
  - `chiSquared(...)`

- `playground/Olga/src/chisqdist.hpp`
  - `chisqRTcdf(...)` and gamma-function helpers

- Call flow in GUI:
  `playground/Olga/src/gui/GetInformativePairsDialog.cpp`
  - prepares the efficiency matrix and RMSD matrix
  - performs NaN thresholding/filling
  - calls `greedySelection(...)` then `precisionDecay(...)`

Notes / differences vs Olga:

- This module assumes `effs` contains finite values already (Olga has explicit NaN handling).
- The weighting uses the chi-squared *right-tail* CDF (`chisqRTcdf`) exactly as in Olga.
- Numeric types are kept mostly as `float32` for speed, like the C++/Eigen code path.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numba as nb
import numpy as np


@nb.njit(cache=True)
def _finite_gamma_q_int(a: int, x: float) -> float:
    """Finite-series Q(a, x) for integer a.

    Port of the integer branch used by Olga's `chisqdist.hpp` (via `chisqRTcdf`).
    """
    e = math.exp(-x)
    s = e
    if s != 0.0:
        term = s
        for n in range(1, a):
            term /= n
            term *= x
            s += term
    return s


@nb.njit(cache=True)
def _finite_half_gamma_q(a: float, x: float) -> float:
    """Finite-series Q(a, x) for half-integer a.

    Port of the half-integer branch used by Olga's `chisqdist.hpp` (via `chisqRTcdf`).
    """
    e = math.erfc(math.sqrt(x))
    if e != 0.0 and a > 1.0 and x != 0.0:
        term = math.exp(-x) / math.sqrt(math.pi * x)
        term *= x
        term /= 0.5
        s = term
        a_int = int(a)
        for n in range(2, a_int):
            term /= (n - 0.5)
            term *= x
            s += term
        e += s
    return e


@nb.njit(cache=True)
def _gamma_q(a: float, x: float) -> float:
    """Regularized upper incomplete gamma function Q(a, x).

    This matches the strategy in Olga's `chisqdist.hpp`:

    - normal approximation for large a
    - closed-form integer / half-integer expansions otherwise
    """
    if a > 100.0:
        return 0.5 * math.erfc((x - a) / math.sqrt(2.0 * a))

    # a = ndof/2 -> integer if ndof even, half-integer if ndof odd
    twice_a = int(2.0 * a)
    if (twice_a % 2) > 0:
        return _finite_half_gamma_q(a, x)

    return _finite_gamma_q_int(int(a), x)


@nb.njit(cache=True)
def _chisq_rt_cdf(chisq: float, ndof: int) -> float:
    """Chi-squared right-tail CDF used as weight.

    Port of Olga's `chisqRTcdf(chisq, ndof)` from `chisqdist.hpp`.
    """
    return _gamma_q(0.5 * ndof, 0.5 * chisq)


@nb.njit(cache=True)
def _rmsd_col_mean(rmsd_col: np.ndarray, chi2_col: np.ndarray, ndof: int, diag_weight: float) -> float:
    """Column-wise expected RMSD under chi-squared tail weights.

    Corresponds to the inner loop of Olga's `rmsdMeanMean(...)` in `best_dist.h`.

    The `diag_weight` implements Olga's diagonal correction (default ~0.99).
    """
    s = 0.0
    prod = 0.0
    for i in range(rmsd_col.shape[0]):
        w = _chisq_rt_cdf(chi2_col[i], ndof)
        s += w
        prod += w * rmsd_col[i]
    return prod / (s - 1.0 + diag_weight)


@nb.njit(cache=True)
def _rmsd_mean_mean(rmsds: np.ndarray, chi2: np.ndarray, ndof: int, diag_weight: float) -> float:
    """Mean of column-wise expected RMSD.

    Port of Olga's `rmsdMeanMean(rmsds, chi2, ndof, diagWeight)`.
    """
    n = rmsds.shape[0]
    ave = 0.0
    for col in range(n):
        ave += _rmsd_col_mean(rmsds[:, col], chi2[:, col], ndof, diag_weight)
    return ave / n


@nb.njit(cache=True)
def _rmsd_mean_mean_add(rmsds: np.ndarray, chi2: np.ndarray, e_add: np.ndarray,
                       inv_err_sq: float, ndof: int, diag_weight: float) -> float:
    """Expected mean RMSD after *adding* one candidate pair.

    Port of Olga's `rmsdMeanMeanAdd(...)` in `best_dist.h`.

    `e_add` is the per-frame efficiency vector of the candidate pair.
    `inv_err_sq = 1/err^2` matches Olga's chi2 update: Δχ² = (ΔE)^2 / err^2.
    """
    n = rmsds.shape[0]
    ave = 0.0
    for col in range(n):
        sum_w = 0.0
        prod = 0.0
        e_col = e_add[col]
        for row in range(n):
            d = e_add[row] - e_col
            chi2_new = chi2[row, col] + (d * d) * inv_err_sq
            w = _chisq_rt_cdf(chi2_new, ndof)
            sum_w += w
            prod += w * rmsds[row, col]
        ave += prod / (sum_w - 1.0 + diag_weight)
    return ave / n


@nb.njit(cache=True)
def _add_pair_to_chi2(chi2: np.ndarray, e_pair: np.ndarray, inv_err_sq: float) -> None:
    """Accumulate the contribution of one selected pair into χ².

    Matches Olga's `chiSquared(...)` accumulation logic used by `greedySelection(...)`.
    """
    n = chi2.shape[0]
    for col in range(n):
        e_col = e_pair[col]
        for row in range(n):
            d = e_pair[row] - e_col
            chi2[row, col] += (d * d) * inv_err_sq


@nb.njit(parallel=True, cache=True)
def _best_pair_all_candidates(effs: np.ndarray, rmsds: np.ndarray, chi2: np.ndarray,
                             inv_err_sq: float, ndof: int, diag_weight: float,
                             selected_mask: np.ndarray, unique_only: bool) -> np.ndarray:
    """Score all candidate pairs (parallel) and return their expected mean RMSD.

    This corresponds to Olga's `bestPair(...)` helper used inside `greedySelection(...)`.
    """
    m = effs.shape[1]
    out = np.empty(m, dtype=np.float32)
    for i in nb.prange(m):
        if unique_only and selected_mask[i] != 0:
            out[i] = np.float32(3.4028235e38)  # max float32
        else:
            out[i] = np.float32(_rmsd_mean_mean_add(rmsds, chi2, effs[:, i], inv_err_sq, ndof, diag_weight))
    return out


@nb.njit(cache=True)
def _precision_decay(selected_pairs: np.ndarray, effs: np.ndarray, rmsds: np.ndarray,
                    inv_err_sq: float, diag_weight: float) -> np.ndarray:
    """Compute the precision decay curve after greedy selection.

    Port of Olga's `precisionDecay(...)` in `best_dist.h`.
    """
    n_steps = selected_pairs.shape[0]
    n = rmsds.shape[0]
    chi2 = np.zeros((n, n), dtype=np.float32)
    decay = np.empty(n_steps, dtype=np.float32)
    for i in range(n_steps):
        _add_pair_to_chi2(chi2, effs[:, selected_pairs[i]], inv_err_sq)
        decay[i] = np.float32(_rmsd_mean_mean(rmsds, chi2, i + 1, diag_weight))
    return decay


def select_informative_pairs(
    effs: np.ndarray,
    rmsds: np.ndarray,
    err: float,
    max_pairs: int,
    unique_only: bool = True,
    diag_weight: float = 0.99,
) -> Tuple[np.ndarray, np.ndarray]:
    """Olga-style greedy informative pair selection.

    Mapping to Olga:

    - This function implements the control flow of `greedySelection(...)` and
      `precisionDecay(...)` from `playground/Olga/src/best_dist.h`.
    - The chi-squared right-tail weights come from `chisqRTcdf(...)` in
      `playground/Olga/src/chisqdist.hpp`.

    Notes:

    - Olga's GUI performs NaN filtering/filling before calling the selector. This
      function expects finite `effs` values.
    - `ndof` is increased with each added pair; for the very first step we clamp
      to 1 to avoid invalid degrees of freedom.

    Parameters
    ----------
    effs:
        Array of shape (n_frames, n_pairs) with FRET efficiencies per frame.
    rmsds:
        Array of shape (n_frames, n_frames) with pairwise RMSDs (Angstrom).
    err:
        Expected absolute error in FRET efficiency.
    max_pairs:
        Number of pairs to select (will be capped to n_pairs).
    unique_only:
        If True, each candidate pair can be selected at most once.
    diag_weight:
        Weight used in Olga for the diagonal correction (default 0.99).

    Returns
    -------
    selected_pair_indices:
        Indices into the *pair* dimension of `effs`.
    precision_decay:
        Vector of length n_selected with the expected mean RMSD after adding each pair.
    """
    effs_f = np.ascontiguousarray(effs, dtype=np.float32)
    rmsds_f = np.ascontiguousarray(rmsds, dtype=np.float32)

    if effs_f.ndim != 2:
        raise ValueError("effs must be 2D (n_frames, n_pairs)")
    if rmsds_f.ndim != 2 or rmsds_f.shape[0] != rmsds_f.shape[1]:
        raise ValueError("rmsds must be 2D square (n_frames, n_frames)")
    if rmsds_f.shape[0] != effs_f.shape[0]:
        raise ValueError("rmsds size must match effs number of frames")

    n_pairs = effs_f.shape[1]
    max_pairs = int(min(max_pairs, n_pairs))
    if max_pairs <= 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32)

    inv_err_sq = float(1.0 / (err * err))

    n = rmsds_f.shape[0]
    chi2 = np.zeros((n, n), dtype=np.float32)
    selected_mask = np.zeros(n_pairs, dtype=np.uint8)
    selected = np.empty(max_pairs, dtype=np.int64)

    for step in range(max_pairs):
        ndof = max(step - 1, 1)
        scores = _best_pair_all_candidates(
            effs_f, rmsds_f, chi2,
            inv_err_sq, ndof, float(diag_weight),
            selected_mask, bool(unique_only),
        )
        best = int(np.argmin(scores))
        selected[step] = best
        selected_mask[best] = 1
        _add_pair_to_chi2(chi2, effs_f[:, best], inv_err_sq)

    decay = _precision_decay(selected, effs_f, rmsds_f, inv_err_sq, float(diag_weight))
    return selected, decay
