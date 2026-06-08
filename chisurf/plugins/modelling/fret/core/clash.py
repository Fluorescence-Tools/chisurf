from __future__ import annotations

from typing import Tuple

import numba as nb
import numpy as np

# vdW radii (Angstrom) — adapted from FPS data/vdW.txt
# fmt: off
VDW_RADII = np.array([
    0.00,  # 0  dummy
    1.20,  # 1  H
    1.40,  # 2  He
    1.82,  # 3  Li
    1.53,  # 4  Be
    1.92,  # 5  B
    1.70,  # 6  C
    1.55,  # 7  N
    1.52,  # 8  O
    1.47,  # 9  F
    1.54,  # 10 Ne
    2.27,  # 11 Na
    1.73,  # 12 Mg
    1.84,  # 13 Al
    2.10,  # 14 Si
    1.80,  # 15 P
    1.80,  # 16 S
    1.75,  # 17 Cl
    1.88,  # 18 Ar
    2.27,  # 19 K
    1.97,  # 20 Ca
    1.87,  # 21 Sc
    1.56,  # 22 Ti  (different sources vary)
    1.56,  # 23 V
    1.56,  # 24 Cr
    1.56,  # 25 Mn
    1.56,  # 26 Fe
    1.56,  # 27 Co
    1.56,  # 28 Ni
    1.39,  # 29 Cu
    1.39,  # 30 Zn
], dtype=np.float64)

_DEFAULT_VDW = 1.70


@nb.njit
def _vdw_radius(element: int) -> float:
    if 0 <= element < len(VDW_RADII):
        return VDW_RADII[element]
    return _DEFAULT_VDW


@nb.njit
def _clash_score_single(
    x1: float, y1: float, z1: float, r1: float,
    x2: float, y2: float, z2: float, r2: float,
    k_clash: float,
) -> Tuple[float, float, float, float]:
    """Clash energy and force magnitude for one atom pair.

    Returns (E, fx, fy, fz) where f points *from* atom1 *to* atom2.
    """
    r_sum = r1 + r2
    dx = x2 - x1
    if abs(dx) >= r_sum:
        return 0.0, 0.0, 0.0, 0.0
    dy = y2 - y1
    if abs(dy) >= r_sum:
        return 0.0, 0.0, 0.0, 0.0
    dz = z2 - z1
    if abs(dz) >= r_sum:
        return 0.0, 0.0, 0.0, 0.0

    d_sq = dx * dx + dy * dy + dz * dz
    if d_sq >= r_sum * r_sum:
        return 0.0, 0.0, 0.0, 0.0

    d = np.sqrt(d_sq) if d_sq > 1e-12 else 1e-12
    overlap = r_sum - d
    E = 0.5 * k_clash * overlap * overlap
    f_mag = k_clash * overlap / d
    return E, f_mag * dx, f_mag * dy, f_mag * dz


@nb.njit
def body_clash_energy(
    xyzr1: np.ndarray,
    xyzr2: np.ndarray,
    k_clash: float = 10.0,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Compute clash energy and forces between two rigid bodies.

    Parameters
    ----------
    xyzr1 : (N1, 4) float64 — xyz + vdw radius
    xyzr2 : (N2, 4) float64
    k_clash : float
        Harmonic spring constant for overlap.

    Returns
    -------
    E : total clash energy
    forces1 : (N1, 3) force on each atom in body 1
    forces2 : (N2, 3) force on each atom in body 2
    """
    n1 = xyzr1.shape[0]
    n2 = xyzr2.shape[0]
    forces1 = np.zeros((n1, 3), dtype=np.float64)
    forces2 = np.zeros((n2, 3), dtype=np.float64)
    E = 0.0

    if n1 == 0 or n2 == 0:
        return 0.0, forces1, forces2

    # Quick AABB check
    min_x1 = xyzr1[0, 0] - xyzr1[0, 3]
    max_x1 = xyzr1[0, 0] + xyzr1[0, 3]
    min_y1 = xyzr1[0, 1] - xyzr1[0, 3]
    max_y1 = xyzr1[0, 1] + xyzr1[0, 3]
    min_z1 = xyzr1[0, 2] - xyzr1[0, 3]
    max_z1 = xyzr1[0, 2] + xyzr1[0, 3]

    for i in range(1, n1):
        x, y, z, r = xyzr1[i]
        if x - r < min_x1: min_x1 = x - r
        if x + r > max_x1: max_x1 = x + r
        if y - r < min_y1: min_y1 = y - r
        if y + r > max_y1: max_y1 = y + r
        if z - r < min_z1: min_z1 = z - r
        if z + r > max_z1: max_z1 = z + r

    min_x2 = xyzr2[0, 0] - xyzr2[0, 3]
    max_x2 = xyzr2[0, 0] + xyzr2[0, 3]
    min_y2 = xyzr2[0, 1] - xyzr2[0, 3]
    max_y2 = xyzr2[0, 1] + xyzr2[0, 3]
    min_z2 = xyzr2[0, 2] - xyzr2[0, 3]
    max_z2 = xyzr2[0, 2] + xyzr2[0, 3]

    for j in range(1, n2):
        x, y, z, r = xyzr2[j]
        if x - r < min_x2: min_x2 = x - r
        if x + r > max_x2: max_x2 = x + r
        if y - r < min_y2: min_y2 = y - r
        if y + r > max_y2: max_y2 = y + r
        if z - r < min_z2: min_z2 = z - r
        if z + r > max_z2: max_z2 = z + r

    if (min_x1 >= max_x2 or min_x2 >= max_x1 or
        min_y1 >= max_y2 or min_y2 >= max_y1 or
        min_z1 >= max_z2 or min_z2 >= max_z1):
        return 0.0, forces1, forces2

    # Find list of atoms in body 1 that overlap with the bounding box of body 2
    indices1 = np.empty(n1, dtype=np.int32)
    idx1_count = 0
    for i in range(n1):
        x, y, z, r = xyzr1[i]
        if (x + r >= min_x2 and x - r <= max_x2 and
            y + r >= min_y2 and y - r <= max_y2 and
            z + r >= min_z2 and z - r <= max_z2):
            indices1[idx1_count] = i
            idx1_count += 1

    # Find list of atoms in body 2 that overlap with the bounding box of body 1
    indices2 = np.empty(n2, dtype=np.int32)
    idx2_count = 0
    for j in range(n2):
        x, y, z, r = xyzr2[j]
        if (x + r >= min_x1 and x - r <= max_x1 and
            y + r >= min_y1 and y - r <= max_y1 and
            z + r >= min_z1 and z - r <= max_z1):
            indices2[idx2_count] = j
            idx2_count += 1

    for idx1 in range(idx1_count):
        i = indices1[idx1]
        x1, y1, z1, r1 = xyzr1[i]
        for idx2 in range(idx2_count):
            j = indices2[idx2]
            x2, y2, z2, r2 = xyzr2[j]
            e, fx, fy, fz = _clash_score_single(
                x1, y1, z1, r1, x2, y2, z2, r2, k_clash
            )
            if e > 0:
                E += e
                forces1[i, 0] -= fx
                forces1[i, 1] -= fy
                forces1[i, 2] -= fz
                forces2[j, 0] += fx
                forces2[j, 1] += fy
                forces2[j, 2] += fz

    return E, forces1, forces2


@nb.njit
def total_clash_energy(
    body_xyzr: np.ndarray,
    body_indices: np.ndarray,
    k_clash: float = 10.0,
) -> float:
    """Total clash energy across all bodies.

    Parameters
    ----------
    body_xyzr : (N, 4) all atoms from all bodies (concatenated)
    body_indices : (N,) int — which body each atom belongs to
    k_clash : float
    """
    n = body_xyzr.shape[0]
    E = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            if body_indices[i] == body_indices[j]:
                continue
            e, _, _, _ = _clash_score_single(
                body_xyzr[i, 0], body_xyzr[i, 1], body_xyzr[i, 2], body_xyzr[i, 3],
                body_xyzr[j, 0], body_xyzr[j, 1], body_xyzr[j, 2], body_xyzr[j, 3],
                k_clash,
            )
            E += e
    return E


def has_clash(
    xyzr1: np.ndarray,
    xyzr2: np.ndarray,
    tolerance: float = 0.0,
) -> bool:
    """Check whether any pair of atoms between two bodies overlaps beyond tolerance."""
    n1 = xyzr1.shape[0]
    n2 = xyzr2.shape[0]
    for i in range(n1):
        x1, y1, z1, r1 = xyzr1[i]
        for j in range(n2):
            x2, y2, z2, r2 = xyzr2[j]
            dx = x1 - x2
            dy = y1 - y2
            dz = z1 - z2
            d = np.sqrt(dx * dx + dy * dy + dz * dz)
            overlap = r1 + r2 - d
            if overlap > tolerance:
                return True
    return False
