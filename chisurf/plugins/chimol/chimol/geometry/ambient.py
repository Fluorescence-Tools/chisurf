from __future__ import annotations

from typing import Optional

import math

import numpy as np

try:  # Optional acceleration via numba
    import numba as nb  # type: ignore
    _HAVE_NUMBA = True
except Exception:  # pragma: no cover - run-time availability
    nb = None  # type: ignore
    _HAVE_NUMBA = False


if _HAVE_NUMBA and nb is not None:

    @nb.jit(nopython=True, nogil=True, cache=True)  # type: ignore[misc]
    def _estimate_ambient_occlusion_nb(
        pts: np.ndarray,
        radius: float,
        max_neighbors: int,
    ) -> np.ndarray:
        n = pts.shape[0]
        occ = np.zeros(n, dtype=np.float64)
        r2 = radius * radius
        if n <= 1 or r2 <= 0.0:
            return occ

        for i in range(n):
            count = 0
            x0 = pts[i, 0]
            y0 = pts[i, 1]
            z0 = pts[i, 2]
            for j in range(n):
                if i == j:
                    continue
                dx = pts[j, 0] - x0
                dy = pts[j, 1] - y0
                dz = pts[j, 2] - z0
                if dx * dx + dy * dy + dz * dz < r2:
                    count += 1
                    if max_neighbors > 0 and count >= max_neighbors:
                        break

            if max_neighbors > 0:
                if count > max_neighbors:
                    count = max_neighbors
                occ[i] = float(count) / float(max_neighbors)

        return occ


def _estimate_ambient_occlusion(
    points: np.ndarray,
    radius: float = 4.0,
    max_neighbors: int = 32,
) -> Optional[np.ndarray]:
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[0] == 0:
        return None

    n = pts.shape[0]
    if n == 1:
        return np.zeros(1, dtype=float)

    r = float(radius)
    if not np.isfinite(r) or r <= 0.0:
        return None

    if _HAVE_NUMBA and nb is not None:
        try:
            occ_nb = _estimate_ambient_occlusion_nb(pts, r, int(max_neighbors))  # type: ignore[name-defined]
            return np.clip(occ_nb, 0.0, 1.0)
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

    occ = np.zeros(n, dtype=float)
    r2 = r * r

    for idx, key in enumerate(map(tuple, ijk)):
        ix, iy, iz = key
        cand_idx: list[int] = []
        for dx, dy, dz in neighbor_offsets:
            cand_idx.extend(grid.get((ix + dx, iy + dy, iz + dz), []))

        if not cand_idx:
            continue

        if len(cand_idx) > max_neighbors * 4:
            cand_idx = cand_idx[: max_neighbors * 4]

        cand = np.asarray(cand_idx, dtype=int)
        diffs = pts[cand] - pts[idx]
        dist2 = np.einsum("ij,ij->i", diffs, diffs)
        within = dist2 < r2
        count = int(np.count_nonzero(within))
        if count <= 0:
            continue

        if count > max_neighbors:
            count = max_neighbors
        occ[idx] = float(count)

    if max_neighbors > 0:
        occ /= float(max_neighbors)

    return np.clip(occ, 0.0, 1.0)


__all__ = ["_estimate_ambient_occlusion"]

