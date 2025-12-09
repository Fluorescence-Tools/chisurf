from __future__ import annotations

from pathlib import Path
from typing import Tuple

import gzip
import struct

import numpy as np


def load_mrc_as_points(path: Path, max_points: int = 250_000) -> Tuple[np.ndarray, dict]:
    raw = Path(path).read_bytes()
    if not raw:
        raise ValueError(f"Empty MRC file: {path}")

    suffix = Path(path).suffix.lower()
    if suffix.endswith(".gz"):
        try:
            raw = gzip.decompress(raw)
        except OSError:
            pass

    if len(raw) < 1024:
        raise ValueError(f"MRC file too short to contain a header: {path}")

    header = raw[:1024]

    def _i32(offset: int) -> int:
        return int(struct.unpack_from("<i", header, offset)[0])

    def _f32(offset: int) -> float:
        return float(struct.unpack_from("<f", header, offset)[0])

    nx = _i32(0)
    ny = _i32(4)
    nz = _i32(8)
    mode = _i32(12)
    mx = _i32(28)
    my = _i32(32)
    mz = _i32(36)
    xlen = _f32(40)
    ylen = _f32(44)
    zlen = _f32(48)
    nsymbt = _i32(92)

    if nx <= 0 or ny <= 0 or nz <= 0:
        raise ValueError(
            f"Invalid MRC grid dimensions in {path}: nx={nx}, ny={ny}, nz={nz}"
        )

    dtype_map = {
        0: np.int8,
        1: np.int16,
        2: np.float32,
        6: np.uint16,
    }
    dtype = dtype_map.get(mode)
    if dtype is None:
        raise ValueError(
            f"Unsupported MRC mode {mode} in {path} (supported: 0, 1, 2, 6)"
        )

    if nsymbt < 0 or nsymbt > len(raw) - 1024:
        nsymbt = 0
    data_offset = 1024 + nsymbt

    n_voxels = int(nx) * int(ny) * int(nz)
    itemsize = np.dtype(dtype).itemsize
    expected_bytes = data_offset + n_voxels * itemsize
    if expected_bytes > len(raw):
        raise ValueError(
            f"MRC file {path} is truncated (expected at least {expected_bytes} bytes, got {len(raw)})"
        )

    data = np.frombuffer(raw, dtype=dtype, count=n_voxels, offset=data_offset).astype(
        np.float32
    )
    data = data.reshape((nz, ny, nx))

    mx_eff = mx if mx > 0 else nx
    my_eff = my if my > 0 else ny
    mz_eff = mz if mz > 0 else nz

    def _spacing(cell_len: float, n_int: int, n_grid: int) -> float:
        if cell_len > 0.0 and n_int > 0:
            return float(cell_len) / float(n_int)
        if cell_len > 0.0 and n_grid > 0:
            return float(cell_len) / float(n_grid)
        return 1.0

    vx = _spacing(xlen, mx_eff, nx)
    vy = _spacing(ylen, my_eff, ny)
    vz = _spacing(zlen, mz_eff, nz)

    finite_mask = np.isfinite(data)
    if not finite_mask.any():
        raise ValueError(f"MRC map in {path} contains no finite density values")

    finite_vals = data[finite_mask]
    mean = float(finite_vals.mean())
    std = float(finite_vals.std())
    if not np.isfinite(std) or std <= 0.0:
        std = 0.0
    level = mean + 0.5 * std if std > 0.0 else mean
    if not np.isfinite(level):
        level = mean

    mask = data >= level
    if not mask.any():
        mask = finite_mask

    z_idx, y_idx, x_idx = np.nonzero(mask)
    if x_idx.size == 0:
        raise ValueError(f"MRC map in {path} did not yield any voxels above threshold")

    positions = np.column_stack(
        (
            x_idx.astype(np.float32) * float(vx),
            y_idx.astype(np.float32) * float(vy),
            z_idx.astype(np.float32) * float(vz),
        )
    ).astype(np.float32, copy=False)

    n_points = positions.shape[0]
    if max_points > 0 and n_points > max_points:
        step = max(1, n_points // max_points)
        positions = positions[::step]
        n_points = positions.shape[0]

    meta = {
        "shape": (int(nz), int(ny), int(nx)),
        "voxel_size": (float(vx), float(vy), float(vz)),
        "n_voxels": int(n_voxels),
        "threshold": float(level),
        "mean": float(mean),
        "std": float(std),
        "n_points": int(n_points),
    }

    return positions, meta


__all__ = ["load_mrc_as_points"]
