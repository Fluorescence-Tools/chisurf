from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple

import numpy as np
from qtpy import QtWidgets

StructureFactory = Optional[Callable[[str], object]]
FileDialogCallable = Optional[Callable[..., Sequence[str]]]

_DEFAULT_FILTER = "Structure files (*.pdb *.ent *.gro *.cif *.mmcif);;All files (*.*)"


class MdtrajNotAvailableError(RuntimeError):
    """Raised when a trajectory requires MDTraj but it is not installed."""

    pass


def _fallback_open_files(parent: Optional[QtWidgets.QWidget] = None) -> list[str]:
    files, _ = QtWidgets.QFileDialog.getOpenFileNames(
        parent,
        "Open structure file",
        "",
        _DEFAULT_FILTER,
    )
    return [str(f) for f in files]


def open_structure_files(
    parent: Optional[QtWidgets.QWidget] = None,
    *,
    opener: FileDialogCallable = None,
    description: str = "Open structure file",
    file_type: str = _DEFAULT_FILTER,
) -> list[str]:
    if opener is not None:
        try:
            result = opener(description=description, file_type=file_type)
            return [str(f) for f in result]
        except Exception:
            pass
    return _fallback_open_files(parent)


def _simple_load_pdb_coords(path: str) -> list[Tuple[float, float, float]]:
    coords: list[Tuple[float, float, float]] = []
    with open(path, "rt", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue
            coords.append((x, y, z))
    if not coords:
        raise ValueError(f"No atom coordinates found in {path!r}")
    return coords


def load_trajectory_frames(path: Path) -> np.ndarray:
    """Load a trajectory or multi-frame structure using MDTraj.

    The returned array has shape ``(T, N, 3)`` with coordinates in Angstroms.
    This helper is only used for formats that are not handled by the IMP-based
    readers in :mod:`chisurf.core.fio.structure.coordinates`.
    """

    try:  # Lazy import so Moview does not hard-depend on mdtraj
        import mdtraj as md  # type: ignore[import]
    except Exception as exc:  # pragma: no cover - environment dependent
        raise MdtrajNotAvailableError(
            "MDTraj is required to load this file type (e.g. GRO/HDF5 trajectory). "
            "Install it with 'conda install -c conda-forge mdtraj' or 'pip install mdtraj'."
        ) from exc

    suffix = path.suffix.lower()
    # For now we restrict to formats where MDTraj can infer topology from the
    # file itself without a separate topology argument.
    if suffix not in {".gro", ".g96", ".h5", ".hdf5"}:
        raise ValueError(f"File type '{suffix}' is not recognised as an MDTraj trajectory/structure")

    try:
        traj = md.load(str(path))
    except Exception as exc:
        raise RuntimeError(f"mdtraj failed to load '{path}': {exc}") from exc

    xyz = getattr(traj, "xyz", None)
    if xyz is None:
        raise RuntimeError(f"mdtraj did not return coordinates for '{path}'")

    arr = np.asarray(xyz, dtype=float)
    if arr.ndim != 3 or arr.shape[2] != 3 or arr.shape[0] == 0 or arr.shape[1] == 0:
        raise RuntimeError(f"mdtraj returned invalid xyz array for '{path}' with shape {arr.shape!r}")

    # MDTraj uses nanometers; convert to Angstrom to be consistent with IMP
    # based loaders used elsewhere in ChiSurf/Moview.
    arr *= 10.0
    return arr


def load_structure_payload(
    path: Path,
    *,
    structure_factory: StructureFactory = None,
) -> tuple[Optional[object], Optional[np.ndarray]]:
    structure = None
    if structure_factory is not None:
        try:
            structure = structure_factory(str(path))
        except Exception:
            structure = None

    # Reject empty/invalid structures so that callers can fall back to
    # alternative loaders (e.g. MDTraj for trajectories or GRO files).
    if structure is not None:
        try:
            n_atoms = getattr(structure, "n_atoms", None)
        except Exception:
            n_atoms = None
        if isinstance(n_atoms, int) and n_atoms <= 0:
            structure = None

    if structure is not None:
        return structure, None

    coords = np.asarray(_simple_load_pdb_coords(str(path)), dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3 or coords.shape[0] == 0:
        raise ValueError(f"No valid coordinates in {path}")
    return None, coords


__all__ = [
    "open_structure_files",
    "load_structure_payload",
    "load_trajectory_frames",
    "MdtrajNotAvailableError",
]
