from __future__ import annotations

from typing import Optional

import numpy as np

try:  # Optional acceleration via numba
    import numba as nb  # type: ignore
    _HAVE_NUMBA = True
except Exception:  # pragma: no cover - run-time availability
    nb = None  # type: ignore
    _HAVE_NUMBA = False


if _HAVE_NUMBA and nb is not None:

    @nb.jit(nopython=True, nogil=True, cache=True)  # type: ignore[misc]
    def _build_bond_pairs_nb(pts: np.ndarray, max_length: float) -> np.ndarray:
        n = pts.shape[0]
        r2 = max_length * max_length
        if n < 2 or r2 <= 0.0:
            return np.zeros((0, 2), dtype=np.int64)

        count = 0
        for i in range(n - 1):
            x0 = pts[i, 0]
            y0 = pts[i, 1]
            z0 = pts[i, 2]
            for j in range(i + 1, n):
                dx = pts[j, 0] - x0
                dy = pts[j, 1] - y0
                dz = pts[j, 2] - z0
                if dx * dx + dy * dy + dz * dz <= r2:
                    count += 1

        if count == 0:
            return np.zeros((0, 2), dtype=np.int64)

        out = np.empty((count, 2), dtype=np.int64)
        k = 0
        for i in range(n - 1):
            x0 = pts[i, 0]
            y0 = pts[i, 1]
            z0 = pts[i, 2]
            for j in range(i + 1, n):
                dx = pts[j, 0] - x0
                dy = pts[j, 1] - y0
                dz = pts[j, 2] - z0
                if dx * dx + dy * dy + dz * dz <= r2:
                    out[k, 0] = i
                    out[k, 1] = j
                    k += 1

        return out


def _build_bond_pairs(coords: np.ndarray, max_length: float) -> np.ndarray:
    """Return an array of (i, j) index pairs for simple covalent bonds.

    Bonds are inferred purely from distance using a cutoff ``max_length`` in
    the *raw* coordinate frame. A simple grid-based neighbor search is used
    so the cost grows roughly linearly with the number of atoms. This is a
    lightweight approximation similar in spirit to pyball's stick geometry.
    """

    pts = np.asarray(coords, dtype=float)
    if pts.ndim != 2 or pts.shape[0] < 2:
        return np.zeros((0, 2), dtype=int)

    r = float(max_length)
    if not np.isfinite(r) or r <= 0.0:
        return np.zeros((0, 2), dtype=int)

    n = pts.shape[0]
    if _HAVE_NUMBA and nb is not None and n > 1:
        try:
            return _build_bond_pairs_nb(pts, r)  # type: ignore[name-defined]
        except Exception:
            pass

    cell = r
    inv_cell = 1.0 / cell

    centered = pts - pts.mean(axis=0)
    ijk = np.floor(centered * inv_cell).astype(np.int32)

    grid: dict[tuple[int, int, int], list[int]] = {}
    for idx, key in enumerate(map(tuple, ijk)):
        grid.setdefault(key, []).append(idx)

    neighbor_offsets = [
        (dx, dy, dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
    ]

    r2 = r * r
    bonds: list[tuple[int, int]] = []

    for i, key in enumerate(map(tuple, ijk)):
        ix, iy, iz = key
        cand_idx: list[int] = []
        for dx, dy, dz in neighbor_offsets:
            cand_idx.extend(grid.get((ix + dx, iy + dy, iz + dz), []))

        if not cand_idx:
            continue

        pi = pts[i]
        for j in cand_idx:
            if j <= i:
                continue
            d = pts[j] - pi
            if float(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]) <= r2:
                bonds.append((i, j))

    if not bonds:
        return np.zeros((0, 2), dtype=int)

    return np.asarray(bonds, dtype=int)


__all__ = ["_build_bond_pairs"]

