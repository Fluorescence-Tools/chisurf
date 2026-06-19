from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, Optional, Tuple

import numpy as np

from . import io

# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------

_HAS_LABELLIB = False
_LABELLIB_BACKEND = False  # True = labellib, False = IMP.bff
try:
    import LabelLib as _ll

    _HAS_LABELLIB = True
except ImportError:
    _ll = None

_HAS_IMP_BFF = False
try:
    import IMP
    import IMP.algebra
    import IMP.atom
    import IMP.bff
    import IMP.core
    import IMP.em

    _HAS_IMP_BFF = bool(
        hasattr(IMP.bff, "AV")
        and hasattr(IMP.bff, "PM_TILE_ACCESSIBLE_DENSITY")
    )
except ImportError:
    pass

def _auto_uses_labellib() -> bool:
    """Return whether automatic backend selection should use LabelLib.

    Returns
    -------
    bool
        ``True`` for LabelLib, ``False`` for IMP.bff. Windows keeps LabelLib
        as the preferred backend; other platforms prefer IMP.bff when it is
        available and use LabelLib as the fallback.
    """
    if sys.platform.startswith("win") and _HAS_LABELLIB:
        return True
    if _HAS_IMP_BFF:
        return False
    if _HAS_LABELLIB:
        return True
    return True  # will fail at runtime


_LABELLIB_BACKEND = _auto_uses_labellib()


def _active_backend_name() -> str:
    """Return the active backend name for diagnostics.

    Returns
    -------
    str
        ``"labellib"`` when LabelLib is active, otherwise ``"imp-bff"``.
    """
    return "labellib" if _LABELLIB_BACKEND else "imp-bff"


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class AccessibleVolume:
    """Represents the 3D positional distribution of a dye label."""

    points: np.ndarray  # (N, 4) float64 — xyz + weight
    density: np.ndarray  # (nx, ny, nz) float32 — 3D voxel density
    grid_origin: np.ndarray  # (3,) float64
    grid_step: float
    grid_shape: Tuple[int, int, int]
    attachment_point: np.ndarray  # (3,) float64
    position_name: str = ""
    params: Dict = field(default_factory=dict)

    @property
    def n_points(self) -> int:
        return self.points.shape[0] if self.points.ndim == 2 else 0

    @property
    def mean_position(self) -> np.ndarray:
        if self.n_points == 0:
            return self.attachment_point.copy()
        w = self.points[:, 3]
        if w.sum() == 0:
            return self.attachment_point.copy()
        return np.average(self.points[:, :3], axis=0, weights=w)

    @property
    def has_volume(self) -> bool:
        return self.n_points > 0


# ---------------------------------------------------------------------------
# Labellib backend
# ---------------------------------------------------------------------------

def _av_labellib(
    atoms: np.ndarray,
    source_xyz: np.ndarray,
    linker_length: float,
    linker_width: float,
    radii: Tuple[float, float, float],
    disc_step: float,
) -> AccessibleVolume:
    """Compute AV using LabelLib.

    Parameters
    ----------
    atoms : (N, 4) float32 — xyzr (r = vdW radius)
    source_xyz : (3,) float64 — attachment point
    """
    # LabelLib expects F-order contiguous float32 arrays
    at = np.asfortranarray(atoms.T.astype(np.float32))  # (4, N)
    src = source_xyz.astype(np.float32)  # (3,)
    r1, r2, r3 = radii

    if r2 > 0 and r3 > 0:
        dye_radii = np.array([r1, r2, r3], dtype=np.float32)  # (3,)
        grid = _ll.dyeDensityAV3(at, src, linker_length, linker_width, dye_radii, disc_step)
    else:
        grid = _ll.dyeDensityAV1(at, src, linker_length, linker_width, r1, disc_step)

    pts = grid.points()  # (4, N) float32
    gshape = tuple(grid.shape)
    origin = np.array(grid.originXYZ, dtype=np.float64)
    step = float(grid.discStep)

    if pts.shape[1] == 0:
        points = np.zeros((0, 4), dtype=np.float64)
    else:
        points = pts.T.astype(np.float64)  # (N, 4) — x, y, z, weight

    density = np.zeros(gshape, dtype=np.float32)

    return AccessibleVolume(
        points=points,
        density=density,
        grid_origin=origin,
        grid_step=step,
        grid_shape=gshape,
        attachment_point=source_xyz.copy(),
    )


# ---------------------------------------------------------------------------
# IMP.bff backend
# ---------------------------------------------------------------------------

def _av_imp_bff(
    pdb_path: str,
    source_info: Dict,
    linker_length: float,
    linker_width: float,
    radii: Tuple[float, float, float],
    disc_step: float,
    allowed_sphere_radius: float = 1.5,
) -> AccessibleVolume:
    """Compute AV using IMP.bff."""
    if not _HAS_IMP_BFF:
        raise RuntimeError("IMP.bff is not available")

    model = IMP.Model()
    hierarchy = IMP.atom.read_pdb(pdb_path, model, IMP.atom.NonWaterPDBSelector())

    chain = source_info.get("chain_identifier", "")
    resseq = source_info.get("residue_seq_number", 0)
    aname = source_info.get("atom_name", "CA")

    sel = IMP.atom.Selection(hierarchy)
    if chain:
        sel.set_chain_id(chain)
    sel.set_residue_index(int(resseq))
    sel.set_atom_type(IMP.atom.AtomType(str(aname)))
    particles = sel.get_selected_particles()
    if not particles:
        raise ValueError(f"Attachment site {chain}:{resseq}:{aname} not found")
    attachment_particle = particles[0]

    av_particle = IMP.Particle(model)
    r1, r2, r3 = radii
    radii_list = [r1, r2, r3]
    if r2 <= 0:
        radii_list = [r1, r2, r3]
    IMP.bff.AV.do_setup_particle(
        model,
        av_particle,
        attachment_particle,
        linker_length=linker_length,
        linker_width=linker_width,
        radii=radii_list,
        allowed_sphere_radius=allowed_sphere_radius,
        contact_volume_thickness=0.0,
        contact_volume_trapped_fraction=-1,
        simulation_grid_resolution=disc_step,
    )
    av = IMP.bff.AV(model, av_particle)
    av.resample()

    path_map = av.get_map()
    header = path_map.get_header()
    nx, ny, nz = header.get_nx(), header.get_ny(), header.get_nz()
    tile_values = path_map.get_tile_values(
        IMP.bff.PM_TILE_ACCESSIBLE_DENSITY, (0.0, path_map.get_path_map_header().get_max_path_length())
    )
    density = np.asarray(tile_values, dtype=np.float32).reshape((nx, ny, nz), order="C")

    xyz_density = path_map.get_xyz_density()
    if xyz_density:
        points = np.asarray([[v[0], v[1], v[2], 1.0] for v in xyz_density], dtype=np.float64)
    else:
        points = np.zeros((0, 4), dtype=np.float64)

    origin = np.array(header.get_origin(), dtype=np.float64)
    step = header.get_spacing()
    att_xyz = np.array(av.get_source_coordinates(), dtype=np.float64)

    return AccessibleVolume(
        points=points,
        density=density,
        grid_origin=origin,
        grid_step=float(step),
        grid_shape=(nx, ny, nz),
        attachment_point=att_xyz,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def select_backend(name: str) -> None:
    """Set the active AV backend.

    Parameters
    ----------
    name : str
        One of ``'auto'``, ``'labellib'``, ``'imp-bff'``.

    Raises
    ------
    ValueError
        If *name* is not a recognised backend.
    RuntimeError
        If the requested backend is not available on this system.
    """
    global _LABELLIB_BACKEND
    if name == "auto":
        if not _HAS_LABELLIB and not _HAS_IMP_BFF:
            raise RuntimeError("No AV backend is available on this system.")
        _LABELLIB_BACKEND = _auto_uses_labellib()
    elif name == "labellib":
        if not _HAS_LABELLIB:
            raise RuntimeError("LabelLib backend is requested but not available.")
        _LABELLIB_BACKEND = True
    elif name == "imp-bff":
        if not _HAS_IMP_BFF:
            raise RuntimeError("IMP.bff backend is requested but not available.")
        _LABELLIB_BACKEND = False
    else:
        raise ValueError(f"Unknown backend name: '{name}'")


def compute_av(
    atoms: np.ndarray,
    source_xyz: np.ndarray,
    linker_length: float,
    linker_width: float,
    radii: Tuple[float, float, float],
    disc_step: float = 1.5,
    pdb_path: Optional[str] = None,
    source_info: Optional[Dict] = None,
) -> AccessibleVolume:
    """Compute an accessible volume.

    On Windows uses LabelLib; on macOS/Linux uses IMP.bff.

    Parameters
    ----------
    atoms : (N, 4) float64
        Columns: x, y, z, vdw_radius.
    source_xyz : (3,) float64
        Attachment point coordinates.
    """
    if _LABELLIB_BACKEND:
        if not _HAS_LABELLIB:
            raise RuntimeError("LabelLib is not available")
        return _av_labellib(
            atoms[:, :4].astype(np.float64),
            source_xyz.astype(np.float64),
            linker_length,
            linker_width,
            radii,
            disc_step,
        )
    if not _HAS_IMP_BFF:
        if _HAS_LABELLIB:
            return _av_labellib(
                atoms[:, :4].astype(np.float64),
                source_xyz.astype(np.float64),
                linker_length,
                linker_width,
                radii,
                disc_step,
            )
        raise RuntimeError("IMP.bff is not available")
    if pdb_path is None or source_info is None:
        raise ValueError("IMP.bff backend requires pdb_path and source_info")
    try:
        return _av_imp_bff(
            pdb_path,
            source_info,
            linker_length,
            linker_width,
            radii,
            disc_step,
        )
    except Exception:
        if not _HAS_LABELLIB:
            raise
        return _av_labellib(
            atoms[:, :4].astype(np.float64),
            source_xyz.astype(np.float64),
            linker_length,
            linker_width,
            radii,
            disc_step,
        )


def compute_avs_for_structure(
    atoms: np.ndarray,
    positions: Dict,
    pdb_path: str | list[str] | None = None,
    disc_step: Optional[float] = None,
) -> Dict[str, AccessibleVolume]:
    """Compute AVs for all positions in an fps.json ``Positions`` dict.

    Parameters
    ----------
    atoms : (N, 4) float64
        xyzr from :func:`load_structure_with_vdw`. Used as fallback if pdb_path is not given.
    positions : dict
        fps.json Positions section.
    """
    if isinstance(pdb_path, (list, tuple)):
        pdb_paths = list(pdb_path)
    elif isinstance(pdb_path, str) and "," in pdb_path:
        pdb_paths = [p.strip() for p in pdb_path.split(",")]
    elif isinstance(pdb_path, str):
        pdb_paths = [pdb_path]
    else:
        pdb_paths = []

    avs: Dict[str, AccessibleVolume] = {}
    for pname, pdef in positions.items():
        bi = int(pdef.get("body_id", 0))
        curr_pdb = pdb_paths[bi] if bi < len(pdb_paths) else (pdb_paths[0] if pdb_paths else None)

        if curr_pdb is not None:
            curr_atoms = load_structure_with_vdw(curr_pdb)
        else:
            curr_atoms = atoms

        ll = float(pdef.get("linker_length", 20.0))
        lw = float(pdef.get("linker_width", 1.0))
        r1 = float(pdef.get("radius1", 3.5))
        r2 = float(pdef.get("radius2", 0.0))
        r3 = float(pdef.get("radius3", 0.0))
        ds = float(disc_step or pdef.get("simulation_grid_resolution", 1.5))

        chain = pdef.get("chain_identifier", "")
        resseq = pdef.get("residue_seq_number", 0)
        aname = pdef.get("atom_name", "CA")

        source_xyz = _find_attachment_point(curr_atoms, chain, resseq, aname, pdb_path=curr_pdb)
        if source_xyz is None:
            avs[pname] = AccessibleVolume(
                points=np.zeros((0, 4), dtype=np.float64),
                density=np.zeros((1, 1, 1), dtype=np.float32),
                grid_origin=np.zeros(3),
                grid_step=ds,
                grid_shape=(1, 1, 1),
                attachment_point=np.zeros(3),
                position_name=pname,
            )
            continue

        clean_atoms = _strip_residue_atoms(curr_atoms, chain, resseq, pdb_path=curr_pdb)

        av = compute_av(
            atoms=clean_atoms,
            source_xyz=source_xyz,
            linker_length=ll,
            linker_width=lw,
            radii=(r1, r2, r3),
            disc_step=ds,
            pdb_path=curr_pdb,
            source_info=pdef,
        )
        av.position_name = pname
        av.params = pdef
        avs[pname] = av
    return avs


# ---------------------------------------------------------------------------
# Helper: vdW radii
# ---------------------------------------------------------------------------

# From FPS data/vdW.txt (selected common elements)
VDW_RADII = {
    1: 1.20, 2: 1.40, 3: 1.82, 4: 1.53, 5: 1.92, 6: 1.70, 7: 1.55,
    8: 1.52, 9: 1.47, 12: 1.73, 14: 2.10, 15: 1.80, 16: 1.80,
    17: 1.75, 19: 2.27, 20: 1.97, 26: 1.56, 30: 1.39,
}
_DEFAULT_VDW = 1.70
_ELEMENT_NUMBERS = {
    "H": 1,
    "HE": 2,
    "LI": 3,
    "BE": 4,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "MG": 12,
    "SI": 14,
    "P": 15,
    "S": 16,
    "CL": 17,
    "K": 19,
    "CA": 20,
    "FE": 26,
    "ZN": 30,
}


def _element_symbol_from_pdb_line(line: str) -> str:
    """Return an element symbol parsed from a PDB ATOM/HETATM line.

    Parameters
    ----------
    line : str
        PDB ATOM or HETATM record.

    Returns
    -------
    str
        Uppercase element symbol, or an empty string when it cannot be parsed.
    """
    symbol = line[76:78].strip().upper() if len(line) >= 78 else ""
    if symbol:
        return symbol

    atom_name = line[12:16].strip().upper()
    letters = "".join(ch for ch in atom_name if ch.isalpha())
    if not letters:
        return ""
    if len(letters) >= 2 and letters[:2] in _ELEMENT_NUMBERS:
        return letters[:2]
    return letters[:1]


def _pdb_cache_token(pdb_path: str) -> tuple[str, int, int]:
    """Return a cache token that changes when a PDB file changes.

    Parameters
    ----------
    pdb_path : str
        Path to a PDB file.

    Returns
    -------
    tuple
        Absolute path, modification time in ns, and file size.
    """
    path = os.path.abspath(pdb_path)
    stat = os.stat(path)
    return path, stat.st_mtime_ns, stat.st_size


@lru_cache(maxsize=32)
def _load_pdb_records_cached(
    pdb_path: str,
    mtime_ns: int,
    size: int,
) -> tuple[tuple[str, int, str, float, float, float, float], ...]:
    """Load ATOM/HETATM records from a PDB file.

    Parameters
    ----------
    pdb_path : str
        Absolute path to a PDB file.
    mtime_ns : int
        File modification timestamp used as part of the cache key.
    size : int
        File size used as part of the cache key.

    Returns
    -------
    tuple
        Records containing chain, residue number, atom name, xyz, and vdW radius.
    """
    del mtime_ns, size
    rows = []
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            try:
                xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
                resseq = int(line[22:26].strip())
            except ValueError:
                continue
            chain = line[21].strip()
            atom_name = line[12:16].strip()
            element = _element_symbol_from_pdb_line(line)
            atomic_number = _ELEMENT_NUMBERS.get(element, 0)
            rows.append((
                chain,
                resseq,
                atom_name,
                xyz[0],
                xyz[1],
                xyz[2],
                VDW_RADII.get(atomic_number, _DEFAULT_VDW),
            ))
    if not rows:
        raise ValueError(f"No ATOM/HETATM coordinates found in '{pdb_path}'")
    return tuple(rows)


def _cached_pdb_records(pdb_path: str) -> tuple[tuple[str, int, str, float, float, float, float], ...]:
    """Return cached PDB records for a path.

    Parameters
    ----------
    pdb_path : str
        Path to a PDB file.

    Returns
    -------
    tuple
        Cached ATOM/HETATM records.
    """
    return _load_pdb_records_cached(*_pdb_cache_token(pdb_path))


def _load_pdb_xyzr_direct(pdb_path: str) -> np.ndarray:
    """Load PDB ATOM/HETATM coordinates and vdW radii without IMP.

    Parameters
    ----------
    pdb_path : str
        Path to a PDB file.

    Returns
    -------
    numpy.ndarray
        ``(N, 4)`` array with ``x, y, z, vdw_radius`` columns.
    """
    records = _cached_pdb_records(pdb_path)
    return np.asarray(
        [(x, y, z, radius) for _, _, _, x, y, z, radius in records],
        dtype=np.float64,
    )


def load_structure_with_vdw(pdb_path: str) -> np.ndarray:
    """Load a PDB and return (N, 4) array: x, y, z, vdw_radius.

    Parses PDB records directly to avoid IMP/CHARMM warnings for unsupported
    HETATM residues, then falls back to IMP.atom if direct parsing fails.
    """
    try:
        return _load_pdb_xyzr_direct(pdb_path)
    except Exception:
        pass

    coords, particles, _model, _hier = io.load_structure_with_particles(pdb_path)
    vdw = np.full(coords.shape[0], _DEFAULT_VDW, dtype=np.float64)
    for i, p in enumerate(particles):
        try:
            at = IMP.atom.Atom(p)
            elem = at.get_element()
            vdw[i] = VDW_RADII.get(elem, _DEFAULT_VDW)
        except Exception:
            pass
    return np.column_stack([coords, vdw])


def _find_attachment_point(
    atoms: np.ndarray,
    chain: str,
    resseq: int,
    atom_name: str,
    pdb_path: Optional[str] = None,
) -> Optional[np.ndarray]:
    """Find the coordinates of an attachment atom.

    If pdb_path is provided, parses the PDB file directly to locate the exact
    atom matching chain, residue number, and atom name. Otherwise, falls back
    to using residue sequence number as a proxy index.

    Parameters
    ----------
    atoms : (N, 4) ndarray
        The atoms array (x, y, z, vdw_radius).
    chain : str
        The chain identifier.
    resseq : int
        The residue sequence number.
    atom_name : str
        The attachment atom name (e.g., 'CA', 'CB').
    pdb_path : str, optional
        Path to the PDB file for exact matching.

    Returns
    -------
    ndarray or None
        The (3,) coordinates of the attachment atom, or None if not found.
    """
    if pdb_path and os.path.exists(pdb_path):
        try:
            for line_chain, line_resseq, line_atom_name, x, y, z, _ in _cached_pdb_records(pdb_path):
                if line_resseq == resseq and line_atom_name == atom_name:
                    if not chain or line_chain == chain:
                        return np.array([x, y, z], dtype=np.float64)
        except Exception:
            pass

    return atoms[resseq - 1, :3] if resseq > 0 and resseq <= atoms.shape[0] else None


def _strip_residue_atoms(
    atoms: np.ndarray,
    chain: str,
    resseq: int,
    pdb_path: Optional[str] = None,
) -> np.ndarray:
    """Remove the attachment residue's atoms from the coordinate array.

    Parameters
    ----------
    atoms : (N, 4) ndarray
        The atoms array (x, y, z, vdw_radius).
    chain : str
        The chain identifier of the residue to exclude.
    resseq : int
        The residue sequence number of the residue to exclude.
    pdb_path : str, optional
        Path to the PDB file for exact matching of residue atoms.

    Returns
    -------
    clean_atoms : (M, 4) ndarray
        The coordinate array with residue atoms removed.
    """
    if not pdb_path or not os.path.exists(pdb_path):
        return atoms

    try:
        records = _cached_pdb_records(pdb_path)
    except Exception:
        return atoms

    if len(records) == atoms.shape[0]:
        keep_mask = np.asarray(
            [
                not (line_resseq == resseq and (not chain or line_chain == chain))
                for line_chain, line_resseq, _, _, _, _, _ in records
            ],
            dtype=bool,
        )
        return atoms[keep_mask]

    coords_to_exclude = [
        [x, y, z]
        for line_chain, line_resseq, _, x, y, z, _ in records
        if line_resseq == resseq and (not chain or line_chain == chain)
    ]
    if not coords_to_exclude:
        return atoms

    coords_to_exclude_arr = np.array(coords_to_exclude, dtype=np.float64)
    distances = np.linalg.norm(
        atoms[:, None, :3] - coords_to_exclude_arr[None, :, :],
        axis=2,
    )
    return atoms[~np.any(distances < 0.01, axis=1)]
