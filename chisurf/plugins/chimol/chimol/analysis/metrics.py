from __future__ import annotations

from typing import Tuple

import numpy as np


def compute_rmsd(mobile: np.ndarray, target: np.ndarray) -> float:
    """Compute RMSD between two (N, 3) coordinate arrays.

    Raises ValueError if shapes are incompatible or not 2D xyz.
    """

    mob = np.asarray(mobile, dtype=float)
    tgt = np.asarray(target, dtype=float)
    if mob.shape != tgt.shape or mob.ndim != 2 or mob.shape[1] != 3:
        raise ValueError(
            "Selections must contain the same number of 3D coordinates"
        )

    diff = mob - tgt
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))


def compute_kabsch(mobile: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Return optimal rotation, translation, and RMSD aligning mobile onto target.

    Both inputs must be shape (N, 3). Returns (R, t, rmsd) with R (3x3) and t (3,).
    """

    mob = np.asarray(mobile, dtype=float)
    tgt = np.asarray(target, dtype=float)
    if mob.shape != tgt.shape or mob.ndim != 2 or mob.shape[1] != 3:
        raise ValueError(
            "Selections must contain the same number of 3D coordinates"
        )

    mob_center = mob.mean(axis=0)
    tgt_center = tgt.mean(axis=0)

    mob_c = mob - mob_center
    tgt_c = tgt - tgt_center

    cov = mob_c.T @ tgt_c
    try:
        V, _, Wt = np.linalg.svd(cov)
    except Exception as exc:
        raise ValueError(f"Alignment failed during SVD: {exc}") from exc

    d = np.linalg.det(V @ Wt)
    D = np.eye(3)
    if d < 0:
        D[-1, -1] = -1.0

    R = V @ D @ Wt
    trans = tgt_center - mob_center @ R

    aligned = (mob @ R) + trans
    rmsd = compute_rmsd(aligned, tgt)

    return R, trans, rmsd


__all__ = ["compute_rmsd", "compute_kabsch"]
