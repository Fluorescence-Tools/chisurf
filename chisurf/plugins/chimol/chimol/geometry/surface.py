from __future__ import annotations

from typing import Optional, Tuple

import math

import numpy as np

try:  # Optional acceleration via numba
    import numba as nb  # type: ignore
    _HAVE_NUMBA = True
except Exception:  # pragma: no cover - run-time availability
    nb = None  # type: ignore
    _HAVE_NUMBA = False

try:  # Optional marching cubes implementation
    from skimage import measure as _sk_measure  # type: ignore
    _HAVE_SKIMAGE = True
except Exception:  # pragma: no cover - optional dependency
    _sk_measure = None  # type: ignore
    _HAVE_SKIMAGE = False


if _HAVE_NUMBA and nb is not None:

    @nb.jit(nopython=True, nogil=True)  # type: ignore[misc]
    def _accumulate_gaussians_nb(
        pts: np.ndarray,
        sigmas: np.ndarray,
        grid: np.ndarray,
        origin: np.ndarray,
        spacing: float,
        cutoff_factor: float,
    ) -> None:
        nx, ny, nz = grid.shape
        for i in range(pts.shape[0]):
            sigma = sigmas[i]
            if sigma <= 0.0 or not np.isfinite(sigma):
                continue
            cutoff = cutoff_factor * sigma
            px = pts[i, 0]
            py = pts[i, 1]
            pz = pts[i, 2]
            inv_two_sigma2 = 1.0 / (2.0 * sigma * sigma)

            min_ix = max(int(math.floor((px - origin[0] - cutoff) / spacing)), 0)
            min_iy = max(int(math.floor((py - origin[1] - cutoff) / spacing)), 0)
            min_iz = max(int(math.floor((pz - origin[2] - cutoff) / spacing)), 0)
            max_ix = min(int(math.ceil((px - origin[0] + cutoff) / spacing)), nx - 1)
            max_iy = min(int(math.ceil((py - origin[1] + cutoff) / spacing)), ny - 1)
            max_iz = min(int(math.ceil((pz - origin[2] + cutoff) / spacing)), nz - 1)

            for ix in range(min_ix, max_ix + 1):
                dx = origin[0] + ix * spacing - px
                dx2 = dx * dx
                for iy in range(min_iy, max_iy + 1):
                    dy = origin[1] + iy * spacing - py
                    dy2 = dy * dy
                    for iz in range(min_iz, max_iz + 1):
                        dz = origin[2] + iz * spacing - pz
                        dist2 = dx2 + dy2 + dz * dz
                        grid[ix, iy, iz] += math.exp(-dist2 * inv_two_sigma2)
else:

    def _accumulate_gaussians_nb(
        pts: np.ndarray,
        sigmas: np.ndarray,
        grid: np.ndarray,
        origin: np.ndarray,
        spacing: float,
        cutoff_factor: float,
    ) -> None:
        nx, ny, nz = grid.shape
        for i in range(pts.shape[0]):
            sigma = sigmas[i]
            if sigma <= 0.0 or not np.isfinite(sigma):
                continue
            cutoff = cutoff_factor * sigma
            px, py, pz = pts[i]
            inv_two_sigma2 = 1.0 / (2.0 * sigma * sigma)

            min_ix = max(int(math.floor((px - origin[0] - cutoff) / spacing)), 0)
            min_iy = max(int(math.floor((py - origin[1] - cutoff) / spacing)), 0)
            min_iz = max(int(math.floor((pz - origin[2] - cutoff) / spacing)), 0)
            max_ix = min(int(math.ceil((px - origin[0] + cutoff) / spacing)), nx - 1)
            max_iy = min(int(math.ceil((py - origin[1] + cutoff) / spacing)), ny - 1)
            max_iz = min(int(math.ceil((pz - origin[2] + cutoff) / spacing)), nz - 1)

            for ix in range(min_ix, max_ix + 1):
                dx = origin[0] + ix * spacing - px
                dx2 = dx * dx
                for iy in range(min_iy, max_iy + 1):
                    dy = origin[1] + iy * spacing - py
                    dy2 = dy * dy
                    for iz in range(min_iz, max_iz + 1):
                        dz = origin[2] + iz * spacing - pz
                        dist2 = dx2 + dy2 + dz * dz
                        grid[ix, iy, iz] += math.exp(-dist2 * inv_two_sigma2)


def _build_gaussian_density_grid(
    pts: np.ndarray,
    sigmas: np.ndarray,
    grid_spacing: float = 1.0,
    padding: float = 2.5,
    cutoff_factor: float = 2.5,
    max_dim: int = 96,
) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
    pts_arr = np.asarray(pts, dtype=float)
    if pts_arr.ndim != 2 or pts_arr.shape[0] < 4:
        return None
    sig_arr = np.asarray(sigmas, dtype=float)
    if sig_arr.shape[0] != pts_arr.shape[0]:
        return None

    spacing = max(float(grid_spacing), 0.2)
    max_sigma = float(np.max(sig_arr))
    pad = max(float(padding), max_sigma * cutoff_factor + spacing * 2.0)
    max_dim = max(int(max_dim), 16)

    mins = pts_arr.min(axis=0) - pad
    maxs = pts_arr.max(axis=0) + pad
    extent = maxs - mins
    dims_float = np.ceil(extent / spacing).astype(int) + 1
    max_axis = int(np.max(dims_float))
    if max_axis > max_dim:
        scale = max_axis / float(max_dim)
        spacing *= scale
        dims_float = np.ceil(extent / spacing).astype(int) + 1

    dims = np.maximum(dims_float, 3)
    grid = np.zeros(tuple(int(x) for x in dims), dtype=np.float32)

    sig_arr = np.clip(sig_arr, spacing * 0.25, spacing * 5.0)
    origin = mins.astype(np.float32)

    _accumulate_gaussians_nb(
        pts_arr.astype(np.float32),
        sig_arr.astype(np.float32),
        grid,
        origin,
        float(spacing),
        float(cutoff_factor),
    )

    if not np.isfinite(grid.max()) or grid.max() <= 0.0:
        return None

    return grid, origin, float(spacing)


def _generate_surface_mesh_from_gaussians(
    pts: np.ndarray,
    sigmas: np.ndarray,
    *,
    grid_spacing: float = 1.0,
    padding: float = 2.5,
    cutoff_factor: float = 2.5,
    iso_value: float = 0.2,
    max_dim: int = 96,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    if not _HAVE_SKIMAGE:
        return None

    density_data = _build_gaussian_density_grid(
        pts,
        sigmas,
        grid_spacing=grid_spacing,
        padding=padding,
        cutoff_factor=cutoff_factor,
        max_dim=max_dim,
    )
    if density_data is None:
        return None

    grid, origin, spacing = density_data
    level = float(iso_value)
    if not np.isfinite(level) or level <= 0.0:
        level = 0.2 * float(grid.max())
    level = min(level, float(grid.max()) * 0.9)
    if level <= 0.0:
        return None

    try:
        verts, faces, norms, _ = _sk_measure.marching_cubes(  # type: ignore[call-arg]
            grid,
            level=level,
            spacing=(spacing, spacing, spacing),
        )
    except Exception:
        return None

    verts = np.asarray(verts, dtype=np.float32)
    verts += origin
    faces = np.asarray(faces, dtype=np.int32)
    # skimage marching_cubes normals point towards higher values (inward). We want outward.
    norms = -np.asarray(norms, dtype=np.float32)
    # Swap winding order from CW to CCW (when viewed from outside) so front faces aren't culled
    faces = faces[:, [0, 2, 1]]
    return verts, faces, norms


__all__ = [
    "_build_gaussian_density_grid",
    "_generate_surface_mesh_from_gaussians",
]

