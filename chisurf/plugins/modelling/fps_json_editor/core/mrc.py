"""MRC export helpers for FPS accessible volumes."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _mrc_path(path: str | Path) -> Path:
    """Return a path with an MRC-compatible suffix."""
    out_path = Path(path)
    if out_path.suffix.lower() not in {".mrc", ".map", ".ccp4"}:
        out_path = out_path.with_suffix(".mrc")
    return out_path


def _voxelize_points(
    points: np.ndarray,
    grid_step: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert AV point coordinates into a dense voxel grid.

    Parameters
    ----------
    points : numpy.ndarray
        ``(N, 3)`` xyz coordinates or ``(N, 4)`` xyz plus weight.
    grid_step : float
        Voxel spacing in Angstrom.

    Returns
    -------
    tuple
        ``(density, origin)`` where density is ordered as ``(nx, ny, nz)``.
    """
    coords = np.asarray(points, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[0] == 0 or coords.shape[1] < 3:
        raise ValueError("AV MRC export requires a non-empty (N, 3/4) point array")

    spacing = float(grid_step)
    if not np.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("AV MRC export requires a positive finite grid step")

    xyz = coords[:, :3]
    weights = coords[:, 3] if coords.shape[1] >= 4 else np.ones(coords.shape[0])
    weights = np.asarray(weights, dtype=np.float32)

    origin = xyz.min(axis=0)
    ijk = np.rint((xyz - origin) / spacing).astype(np.int64)
    shape = tuple(int(v) + 1 for v in ijk.max(axis=0))
    density = np.zeros(shape, dtype=np.float32)
    np.add.at(density, (ijk[:, 0], ijk[:, 1], ijk[:, 2]), weights)
    return density, origin.astype(np.float64)


def save_av_mrc(
    path: str | Path,
    points: np.ndarray,
    grid_step: float,
) -> Path:
    """Save AV points as an MRC density map using IMP.em.

    Parameters
    ----------
    path : str or pathlib.Path
        Output path. ``.mrc`` is appended when no MRC-compatible suffix is
        present.
    points : numpy.ndarray
        ``(N, 3)`` xyz coordinates or ``(N, 4)`` xyz plus density weights.
    grid_step : float
        Voxel spacing in Angstrom.

    Returns
    -------
    pathlib.Path
        Path of the written MRC file.
    """
    import IMP.algebra
    import IMP.em

    density, origin = _voxelize_points(points, grid_step)
    spacing = float(grid_step)
    lower = IMP.algebra.Vector3D(*(origin - spacing / 2.0))
    upper = IMP.algebra.Vector3D(*(origin + np.asarray(density.shape) * spacing - spacing / 2.0))
    density_map = IMP.em.create_density_map(
        IMP.algebra.BoundingBox3D(lower, upper),
        spacing,
    )

    for index_tuple in np.argwhere(density > 0.0):
        map_index = density_map.xyz_ind2voxel(
            int(index_tuple[0]),
            int(index_tuple[1]),
            int(index_tuple[2]),
        )
        density_map.set_value(int(map_index), float(density[tuple(index_tuple)]))

    out_path = _mrc_path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    IMP.em.write_map(density_map, str(out_path), IMP.em.MRCReaderWriter())
    return out_path


__all__ = ["save_av_mrc"]
