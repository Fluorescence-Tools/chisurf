from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np


class RmfNotAvailableError(RuntimeError):
    pass


def load_rmf_frames(path: Path, frame_indices: Optional[Sequence[int]] = None) -> np.ndarray:
    try:
        import IMP  # type: ignore[import]
        import IMP.core  # type: ignore[import]
        import IMP.atom  # type: ignore[import]
        import IMP.rmf  # type: ignore[import]
        import RMF  # type: ignore[import]
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RmfNotAvailableError(
            "RMF and IMP.rmf are required to load RMF files. "
            "Install IMP with RMF support (e.g. via conda-forge) to enable this feature."
        ) from exc

    p = Path(path)
    rh = RMF.open_rmf_file_read_only(str(p))
    model = IMP.Model()
    hierarchies = IMP.rmf.create_hierarchies(rh, model)
    if not hierarchies:
        raise RuntimeError(f"RMF file {p!s} does not contain any hierarchies")

    hierarchy = hierarchies[0]
    leaves = IMP.atom.get_leaves(hierarchy)
    if not leaves:
        raise RuntimeError(f"RMF file {p!s} does not contain any atom leaves")

    n_frames = int(rh.get_number_of_frames())
    if n_frames <= 0:
        raise RuntimeError(f"RMF file {p!s} does not contain any frames")

    if frame_indices is None:
        frame_list = list(range(n_frames))
    else:
        frame_list = []
        for idx in frame_indices:
            try:
                i = int(idx)
            except Exception:
                continue
            if 0 <= i < n_frames:
                frame_list.append(i)
        if not frame_list:
            raise ValueError("No valid frame indices for RMF file")

    frames = []
    for fi in frame_list:
        IMP.rmf.load_frame(rh, RMF.FrameID(int(fi)))
        coords = []
        for particle in leaves:
            xyzr = IMP.core.XYZR(particle)
            c = xyzr.get_coordinates()
            coords.append([float(c[0]), float(c[1]), float(c[2])])
        arr = np.asarray(coords, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 3 or arr.shape[0] == 0:
            raise RuntimeError(
                f"Invalid coordinate array from RMF frame {fi} in {p!s}: shape={arr.shape!r}"
            )
        frames.append(arr)

    if not frames:
        raise RuntimeError(f"No usable frames found in RMF file {p!s}")

    first_shape = frames[0].shape
    for arr in frames[1:]:
        if arr.shape != first_shape:
            raise RuntimeError(
                f"RMF file {p!s} contains frames with inconsistent shape: "
                f"{first_shape!r} vs {arr.shape!r}"
            )

    return np.stack(frames, axis=0)


__all__ = ["load_rmf_frames", "RmfNotAvailableError"]
