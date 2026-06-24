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

try:  # Optional scipy distance transform
    from scipy.ndimage import (  # type: ignore
        binary_dilation,
        distance_transform_edt,
        gaussian_filter,
    )
    _HAVE_SCIPY_EDT = True
except ImportError:  # pragma: no cover
    binary_dilation = None  # type: ignore
    distance_transform_edt = None  # type: ignore
    gaussian_filter = None  # type: ignore
    _HAVE_SCIPY_EDT = False


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

    @nb.jit(nopython=True, nogil=True)  # type: ignore[misc]
    def _accumulate_wyvill_nb(
        pts: np.ndarray,
        sigmas: np.ndarray,
        grid: np.ndarray,
        origin: np.ndarray,
        spacing: float,
    ) -> None:
        nx, ny, nz = grid.shape
        for i in range(pts.shape[0]):
            r_max = sigmas[i]
            if r_max <= 0.0 or not np.isfinite(r_max):
                continue
            px = pts[i, 0]
            py = pts[i, 1]
            pz = pts[i, 2]
            r_max2 = r_max * r_max
            inv_r_max2 = 1.0 / r_max2

            min_ix = max(int(math.floor((px - origin[0] - r_max) / spacing)), 0)
            min_iy = max(int(math.floor((py - origin[1] - r_max) / spacing)), 0)
            min_iz = max(int(math.floor((pz - origin[2] - r_max) / spacing)), 0)
            max_ix = min(int(math.ceil((px - origin[0] + r_max) / spacing)), nx - 1)
            max_iy = min(int(math.ceil((py - origin[1] + r_max) / spacing)), ny - 1)
            max_iz = min(int(math.ceil((pz - origin[2] + r_max) / spacing)), nz - 1)

            for ix in range(min_ix, max_ix + 1):
                dx = origin[0] + ix * spacing - px
                dx2 = dx * dx
                for iy in range(min_iy, max_iy + 1):
                    dy = origin[1] + iy * spacing - py
                    dy2 = dy * dy
                    for iz in range(min_iz, max_iz + 1):
                        dz = origin[2] + iz * spacing - pz
                        dist2 = dx2 + dy2 + dz * dz
                        if dist2 < r_max2:
                            u = dist2 * inv_r_max2
                            u2 = u * u
                            val = (9.0 - 22.0 * u + 17.0 * u2 - 4.0 * u2 * u) / 9.0
                            grid[ix, iy, iz] += val

    @nb.jit(nopython=True, nogil=True, parallel=True)  # type: ignore[misc]
    def _compute_distance_grid_nb(
        pts: np.ndarray,
        radii: np.ndarray,
        grid: np.ndarray,
        origin: np.ndarray,
        spacing: float,
    ) -> None:
        nx, ny, nz = grid.shape
        for ix in nb.prange(nx):
            x = origin[0] + ix * spacing
            for iy in range(ny):
                y = origin[1] + iy * spacing
                for iz in range(nz):
                    z = origin[2] + iz * spacing
                    min_dist = 999999.0
                    for i in range(pts.shape[0]):
                        dx = x - pts[i, 0]
                        dy = y - pts[i, 1]
                        dz = z - pts[i, 2]
                        d = math.sqrt(dx*dx + dy*dy + dz*dz) - radii[i]
                        if d < min_dist:
                            min_dist = d
                    grid[ix, iy, iz] = min_dist
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

    def _accumulate_wyvill_nb(
        pts: np.ndarray,
        sigmas: np.ndarray,
        grid: np.ndarray,
        origin: np.ndarray,
        spacing: float,
    ) -> None:
        nx, ny, nz = grid.shape
        for i in range(pts.shape[0]):
            r_max = sigmas[i]
            if r_max <= 0.0 or not np.isfinite(r_max):
                continue
            px, py, pz = pts[i]
            r_max2 = r_max * r_max
            inv_r_max2 = 1.0 / r_max2

            min_ix = max(int(math.floor((px - origin[0] - r_max) / spacing)), 0)
            min_iy = max(int(math.floor((py - origin[1] - r_max) / spacing)), 0)
            min_iz = max(int(math.floor((pz - origin[2] - r_max) / spacing)), 0)
            max_ix = min(int(math.ceil((px - origin[0] + r_max) / spacing)), nx - 1)
            max_iy = min(int(math.ceil((py - origin[1] + r_max) / spacing)), ny - 1)
            max_iz = min(int(math.ceil((pz - origin[2] + r_max) / spacing)), nz - 1)

            for ix in range(min_ix, max_ix + 1):
                dx = origin[0] + ix * spacing - px
                dx2 = dx * dx
                for iy in range(min_iy, max_iy + 1):
                    dy = origin[1] + iy * spacing - py
                    dy2 = dy * dy
                    for iz in range(min_iz, max_iz + 1):
                        dz = origin[2] + iz * spacing - pz
                        dist2 = dx2 + dy2 + dz * dz
                        if dist2 < r_max2:
                            u = dist2 * inv_r_max2
                            u2 = u * u
                            val = (9.0 - 22.0 * u + 17.0 * u2 - 4.0 * u2 * u) / 9.0
                            grid[ix, iy, iz] += val

    def _compute_distance_grid_nb(
        pts: np.ndarray,
        radii: np.ndarray,
        grid: np.ndarray,
        origin: np.ndarray,
        spacing: float,
    ) -> None:
        nx, ny, nz = grid.shape
        for ix in range(nx):
            x = origin[0] + ix * spacing
            for iy in range(ny):
                y = origin[1] + iy * spacing
                for iz in range(nz):
                    z = origin[2] + iz * spacing
                    min_dist = 999999.0
                    for i in range(pts.shape[0]):
                        dx = x - pts[i, 0]
                        dy = y - pts[i, 1]
                        dz = z - pts[i, 2]
                        d = math.sqrt(dx*dx + dy*dy + dz*dz) - radii[i]
                        if d < min_dist:
                            min_dist = d
                    grid[ix, iy, iz] = min_dist


def _build_density_grid(
    pts: np.ndarray,
    sigmas: np.ndarray,
    *,
    field_function: str = "gaussian",
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
    
    field_function = field_function.lower()
    if field_function == "wyvill":
        pad = max(float(padding), max_sigma + spacing * 2.0)
    else:
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

    sig_arr = np.maximum(sig_arr, spacing * 0.1)
    origin = mins.astype(np.float32)

    if field_function == "wyvill":
        _accumulate_wyvill_nb(
            pts_arr.astype(np.float32),
            sig_arr.astype(np.float32),
            grid,
            origin,
            float(spacing),
        )
    else:
        sig_arr = np.clip(sig_arr, spacing * 0.25, spacing * 5.0)
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


def _build_gaussian_density_grid(
    pts: np.ndarray,
    sigmas: np.ndarray,
    grid_spacing: float = 1.0,
    padding: float = 2.5,
    cutoff_factor: float = 2.5,
    max_dim: int = 96,
) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
    return _build_density_grid(
        pts,
        sigmas,
        field_function="gaussian",
        grid_spacing=grid_spacing,
        padding=padding,
        cutoff_factor=cutoff_factor,
        max_dim=max_dim,
    )


def _generate_surface_mesh_from_density(
    pts: np.ndarray,
    sigmas: np.ndarray,
    *,
    field_function: str = "gaussian",
    grid_spacing: float = 1.0,
    padding: float = 2.5,
    cutoff_factor: float = 2.5,
    iso_value: float = 0.2,
    max_dim: int = 96,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    if not _HAVE_SKIMAGE:
        return None

    density_data = _build_density_grid(
        pts,
        sigmas,
        field_function=field_function,
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
    norms = -np.asarray(norms, dtype=np.float32)
    faces = faces[:, [0, 2, 1]]
    return verts, faces, norms


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
    return _generate_surface_mesh_from_density(
        pts,
        sigmas,
        field_function="gaussian",
        grid_spacing=grid_spacing,
        padding=padding,
        cutoff_factor=cutoff_factor,
        iso_value=iso_value,
        max_dim=max_dim,
    )


def _generate_surface_mesh_from_points(
    pts: np.ndarray,
    *,
    grid_spacing: float = 1.0,
    padding: float = 1.5,
    smoothing_sigma: float = 0.75,
    dilation_iterations: int = 1,
    max_dim: int = 96,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Generate a surface mesh around occupied point-cloud voxels.

    Parameters
    ----------
    pts : numpy.ndarray
        Point coordinates with shape ``(N, 3)``.
    grid_spacing : float, optional
        Target voxel spacing in the same coordinate units as ``pts``.
    padding : float, optional
        Empty border around the point cloud.
    smoothing_sigma : float, optional
        Gaussian smoothing sigma in voxel units. Set to ``0`` to disable
        smoothing.
    dilation_iterations : int, optional
        Number of binary dilation passes before smoothing.
    max_dim : int, optional
        Maximum grid dimension. The actual spacing is increased if needed.

    Returns
    -------
    tuple of numpy.ndarray or None
        ``(vertices, faces, normals)`` if marching cubes succeeds, otherwise
        ``None``.
    """
    if not _HAVE_SKIMAGE:
        return None

    points = np.asarray(pts, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] < 4:
        return None
    finite_mask = np.isfinite(points).all(axis=1)
    points = points[finite_mask]
    if points.shape[0] < 4:
        return None

    spacing = max(float(grid_spacing), 1e-3)
    pad = max(float(padding), spacing)
    max_grid_dim = max(int(max_dim), 8)

    xyz_min = points.min(axis=0) - pad
    xyz_max = points.max(axis=0) + pad
    extent = np.maximum(xyz_max - xyz_min, spacing)
    shape = np.ceil(extent / spacing).astype(int) + 1

    largest = int(shape.max())
    if largest > max_grid_dim:
        spacing *= largest / float(max_grid_dim)
        shape = np.ceil(extent / spacing).astype(int) + 1

    shape = np.maximum(shape, 4)
    grid = np.zeros(tuple(int(v) for v in shape), dtype=np.float32)
    indices = np.rint((points - xyz_min) / spacing).astype(int)
    for axis in range(3):
        indices[:, axis] = np.clip(indices[:, axis], 0, shape[axis] - 1)
    grid[indices[:, 0], indices[:, 1], indices[:, 2]] = 1.0

    if binary_dilation is not None and dilation_iterations > 0:
        grid = binary_dilation(
            grid > 0.0,
            iterations=int(dilation_iterations),
        ).astype(np.float32)

    if gaussian_filter is not None and smoothing_sigma > 0.0:
        grid = gaussian_filter(grid, sigma=float(smoothing_sigma)).astype(np.float32)

    level = 0.5
    grid_max = float(grid.max())
    if grid_max <= 0.0:
        return None
    if level >= grid_max:
        level = grid_max * 0.5

    try:
        verts, faces, norms, _ = _sk_measure.marching_cubes(  # type: ignore[call-arg]
            grid,
            level=level,
            spacing=(spacing, spacing, spacing),
        )
    except Exception:
        return None

    verts = np.asarray(verts, dtype=np.float32)
    verts += xyz_min
    faces = np.asarray(faces, dtype=np.int32)
    norms = -np.asarray(norms, dtype=np.float32)
    faces = faces[:, [0, 2, 1]]
    return verts, faces, norms


def _generate_surface_mesh_edt(
    pts: np.ndarray,
    radii: np.ndarray,
    *,
    method: str = "sas",
    probe_radius: float = 1.4,
    grid_spacing: float = 0.8,
    padding: float = 3.0,
    max_dim: int = 96,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Generate a SAS or SES surface mesh using Euclidean Distance Transform (EDT).

    Parameters
    ----------
    pts : np.ndarray
        Coordinates of the atoms, shape (N, 3).
    radii : np.ndarray
        Radii of the atoms, shape (N,).
    method : str, optional
        Either "sas" (Solvent Accessible Surface) or "ses" (Solvent Excluded Surface).
        Default is "sas".
    probe_radius : float, optional
        Radius of the rolling solvent probe. Default is 1.4.
    grid_spacing : float, optional
        Target grid spacing/resolution in Angstroms. Default is 0.8.
    padding : float, optional
        Padding around the bounding box of coordinates in Angstroms. Default is 3.0.
    max_dim : int, optional
        Maximum dimension of the grid to prevent excessive memory/CPU usage.
        Default is 96.

    Returns
    -------
    Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]
        A tuple of (vertices, faces, normals) or None if the calculation failed.
    """
    if not _HAVE_SKIMAGE:
        return None

    method = method.lower()
    if method == "ses" and not _HAVE_SCIPY_EDT:
        return None

    pts_arr = np.asarray(pts, dtype=float)
    if pts_arr.ndim != 2 or pts_arr.shape[0] == 0:
        return None
    radii_arr = np.asarray(radii, dtype=float)
    if radii_arr.shape[0] != pts_arr.shape[0]:
        return None

    spacing = max(float(grid_spacing), 0.1)
    max_radius = float(np.max(radii_arr))
    pad = max(float(padding), max_radius + probe_radius + spacing * 2.0)
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
    origin = mins.astype(np.float32)

    _compute_distance_grid_nb(
        pts_arr.astype(np.float32),
        radii_arr.astype(np.float32),
        grid,
        origin,
        float(spacing),
    )

    if method == "sas":
        # For SAS: we run on -grid (higher inside) at level -probe_radius.
        grid_to_mesh = -grid
        level = -float(probe_radius)
    elif method == "ses":
        # For SES: B_forbidden = (D < r_p)
        # 1 inside forbidden region, 0 inside allowed region
        forbidden = (grid < float(probe_radius)).astype(np.uint8)
        # distance_transform_edt computes distance to nearest 0 (allowed region).
        # We multiply by spacing to get physical distance.
        grid_to_mesh = distance_transform_edt(forbidden).astype(np.float32) * float(spacing)
        level = float(probe_radius)
    else:
        return None

    if level <= grid_to_mesh.min() or level >= grid_to_mesh.max():
        return None

    try:
        verts, faces, norms, _ = _sk_measure.marching_cubes(  # type: ignore[call-arg]
            grid_to_mesh,
            level=level,
            spacing=(spacing, spacing, spacing),
        )
    except Exception:
        return None

    verts = np.asarray(verts, dtype=np.float32)
    verts += origin
    faces = np.asarray(faces, dtype=np.int32)
    # Negate normals to point outward
    norms = -np.asarray(norms, dtype=np.float32)
    # Swap winding order to CCW
    faces = faces[:, [0, 2, 1]]
    return verts, faces, norms


def _get_surface_atom_mask(
    pts: np.ndarray,
    *,
    radius: float = 5.0,
    max_neighbors: int = 20,
) -> np.ndarray:
    """Return a boolean mask selecting surface-exposed atoms.

    An atom is considered *buried* (not surface) when it has more than
    ``max_neighbors`` other atoms within ``radius`` Angstroms.  This is a
    fast O(N log N) heuristic that avoids a full SASA calculation while
    still removing the vast majority of interior atoms from the density
    field, which dramatically speeds up metaball rendering for large
    structures.

    Parameters
    ----------
    pts : (N, 3) ndarray
        Atom coordinates.
    radius : float
        Search radius for neighbor counting (Angstroms).
    max_neighbors : int
        Atoms with more neighbors than this threshold are classified as
        buried and excluded from the returned mask.

    Returns
    -------
    mask : (N,) ndarray of bool
        ``True`` for surface-exposed atoms.
    """
    arr = np.asarray(pts, dtype=float)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return np.zeros(0, dtype=bool)
    if arr.shape[0] <= max_neighbors:
        return np.ones(arr.shape[0], dtype=bool)

    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(arr)
        counts = tree.query_ball_point(arr, r=float(radius), return_length=True)
        return np.asarray(counts, dtype=int) <= int(max_neighbors)
    except Exception:
        return np.ones(arr.shape[0], dtype=bool)


__all__ = [
    "_build_gaussian_density_grid",
    "_generate_surface_mesh_from_gaussians",
    "_generate_surface_mesh_from_density",
    "_generate_surface_mesh_edt",
    "_get_surface_atom_mask",
]
