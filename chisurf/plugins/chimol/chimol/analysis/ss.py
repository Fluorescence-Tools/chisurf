"""Secondary-structure (SS) assignment helpers for Chimol.

This module implements a simplified DSSP-style algorithm to assign
C3-type secondary structure codes ("H", "E", "C") from protein
backbone coordinates.

The implementation is inspired by the PyDSSP project
(Shintaro Minami, https://github.com/ShintaroMinami/PyDSSP) and the
original DSSP algorithm by Kabsch & Sander (1983):

    Kabsch, W. & Sander, C. (1983)
    "Dictionary of protein secondary structure: pattern recognition of
    hydrogen-bonded and geometrical features", Biopolymers 22, 25772637.

For Chimol we only need an approximate classification for
visualization, so several aspects are deliberately simplified:

- Only backbone atoms (N, CA, C, O) are used; H positions are modeled
  geometrically.
- Hydrogen bonds are detected using the classic KabschSander
  electrostatic energy formula with a single threshold.
- C3 codes are assigned using simple helix (i,i+4) and strand (long-range
  H-bond) patterns.

This keeps the dependency surface small (NumPy + Biopython only) and
avoids depending on external DSSP binaries or mdtraj.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional

import numpy as np

try:  # Optional speed-up for large systems
    import numba as nb  # type: ignore[import]
    _HAVE_NUMBA = True
except Exception:  # pragma: no cover - numba not available
    nb = None  # type: ignore[assignment]
    _HAVE_NUMBA = False


CONST_Q1Q2 = 0.084
CONST_F = 332.0
DEFAULT_CUTOFF = -0.5
DEFAULT_MARGIN = 1.0


def _build_backbone_from_atoms(atoms: np.ndarray) -> Optional[np.ndarray]:
    """Return backbone coordinates array of shape (N, 4, 3) or ``None``.

    Axes are: residue index, atom index, xyz. Atom order is (N, CA, C, O).
    Built directly from a ChiSurf-style ``atoms`` structured array.
    Residues missing any of these atoms are skipped.
    """

    if not isinstance(atoms, np.ndarray):
        return None
    fields = set(atoms.dtype.fields or {})
    if "atom_name" not in fields or "xyz" not in fields:
        return None

    names = atoms["atom_name"]
    try:
        text = np.char.strip(names.astype(str))
    except Exception:
        text = np.array([str(n).strip() for n in names])

    xyz = np.asarray(atoms["xyz"], dtype=float)

    if "res_id" in fields:
        res_ids = np.asarray(atoms["res_id"])
    else:
        res_ids = np.arange(len(atoms))

    try:
        chains = atoms["chain"] if "chain" in fields else np.zeros(len(atoms), dtype="U1")
    except Exception:
        chains = np.zeros(len(atoms), dtype="U1")

    res_ids_arr = np.asarray(res_ids)
    chains_arr = np.asarray(chains).astype(str)

    ca_mask = text == "CA"
    if not ca_mask.any():
        return None
    ca_indices = np.nonzero(ca_mask)[0]

    bb_list: list[np.ndarray] = []
    for idx in ca_indices:
        rid = res_ids_arr[idx]
        chain = chains_arr[idx]
        same_res = res_ids_arr == rid
        if "chain" in fields:
            same_res &= chains_arr == chain

        n_idx = np.nonzero(same_res & (text == "N"))[0]
        c_idx = np.nonzero(same_res & (text == "C"))[0]
        o_idx = np.nonzero(same_res & (text == "O"))[0]
        if n_idx.size == 0 or c_idx.size == 0 or o_idx.size == 0:
            continue

        n_coord = xyz[n_idx[0]]
        ca_coord = xyz[idx]
        c_coord = xyz[c_idx[0]]
        o_coord = xyz[o_idx[0]]
        bb = np.stack([n_coord, ca_coord, c_coord, o_coord], axis=0).astype(float)
        bb_list.append(bb)

    if not bb_list:
        return None

    return np.stack(bb_list, axis=0)


def _pydssp_check_input(coord: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
    """Ensure a batch dimension on ``coord`` (adapted from PyDSSP).

    Accepts either ``(L, A, 3)`` or ``(B, L, A, 3)`` and always returns a
    4D array plus the original shape.
    """

    org_shape = coord.shape
    if coord.ndim == 3:
        coord_b = coord[None, ...]
    elif coord.ndim == 4:
        coord_b = coord
    else:
        raise ValueError(
            "Backbone coord must have shape (L, A, 3) or (B, L, A, 3)"
        )
    return coord_b, org_shape


def _pydssp_get_hydrogen_atom_position(coord_b: np.ndarray) -> np.ndarray:
    """Return modeled backbone H positions (PyDSSP geometry).

    Parameters
    ----------
    coord_b:
        Array of shape ``(B, L, A, 3)`` with atoms ordered as ``(N, CA, C, O)
        or (N, CA, C, O, H)``.
    """

    # coord_b[:, 1:, 0] -> N_i   for i = 1..L-1
    # coord_b[:, :-1, 2] -> C_{i-1}
    vec_cn = coord_b[:, 1:, 0] - coord_b[:, :-1, 2]
    vec_cn /= np.linalg.norm(vec_cn, axis=-1, keepdims=True)

    vec_can = coord_b[:, 1:, 0] - coord_b[:, 1:, 1]
    vec_can /= np.linalg.norm(vec_can, axis=-1, keepdims=True)

    vec_nh = vec_cn + vec_can
    vec_nh /= np.linalg.norm(vec_nh, axis=-1, keepdims=True)
    return coord_b[:, 1:, 0] + 1.01 * vec_nh


def _pydssp_get_hbond_map(
    coord: np.ndarray,
    donor_mask: Optional[np.ndarray] = None,
    cutoff: float = DEFAULT_CUTOFF,
    margin: float = DEFAULT_MARGIN,
    return_e: bool = False,
) -> np.ndarray:
    """Compute continuous H-bond map as in PyDSSP (NumPy port).

    Parameters
    ----------
    coord:
        Backbone coordinates of shape ``(L, 4, 3)`` or ``(B, L, 4, 3)`` with
        atom order (N, CA, C, O).
    donor_mask:
        Optional length-``L`` mask (1=can donate, 0=cannot), e.g. for Proline.
    cutoff, margin:
        Same meaning as in PyDSSP: electrostatic cutoff and smoothing margin.
    """

    coord_b, org_shape = _pydssp_check_input(coord)
    b, l, a, _ = coord_b.shape
    if a not in (4, 5):
        raise ValueError("coord must have 4 or 5 backbone atoms (N,CA,C,O[,H])")

    # Add pseudo-H atom positions if not available
    if a == 5:
        h = coord_b[:, 1:, 4]
    else:
        h = _pydssp_get_hydrogen_atom_position(coord_b)

    # Donor (N, H) for residues 1..L-1, acceptor (C,O) for residues 0..L-2
    N = coord_b[:, 1:, 0]  # (B, L-1, 3)
    C = coord_b[:, :-1, 2]  # (B, L-1, 3)
    O = coord_b[:, :-1, 3]  # (B, L-1, 3)

    N_i = N[:, :, None, :]  # (B, L-1, 1, 3)
    H_i = h[:, :, None, :]  # (B, L-1, 1, 3)
    C_j = C[:, None, :, :]  # (B, 1, L-1, 3)
    O_j = O[:, None, :, :]  # (B, 1, L-1, 3)

    def _dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        d = a - b
        return np.linalg.norm(d, axis=-1)

    d_on = _dist(O_j, N_i)
    d_ch = _dist(C_j, H_i)
    d_oh = _dist(O_j, H_i)
    d_cn = _dist(C_j, N_i)

    # Electrostatic interaction energy
    e = CONST_Q1Q2 * CONST_F * (
        1.0 / d_on + 1.0 / d_ch - 1.0 / d_oh - 1.0 / d_cn
    )
    # Pad to full (L, L) as in PyDSSP
    e = np.pad(e, ((0, 0), (1, 0), (0, 1)))  # (B, L, L)

    if return_e:
        return e if coord.ndim == 4 else e[0]

    # Local pair mask (i,i), (i,i+1), (i,i+2)
    local_mask = ~np.eye(l, dtype=bool)
    if l > 1:
        local_mask &= ~np.eye(l, k=-1, dtype=bool)
    if l > 2:
        local_mask &= ~np.eye(l, k=-2, dtype=bool)

    # Donor mask (e.g. Proline); default: all can donate
    if donor_mask is not None:
        dmask = np.asarray(donor_mask, dtype=float).reshape(l)
    else:
        dmask = np.ones(l, dtype=float)
    donor_2d = dmask[:, None] * np.ones((1, l), dtype=float)

    # Continuous H-bond map
    hbond_map = np.clip(cutoff - margin - e, a_min=-margin, a_max=margin)
    hbond_map = (np.sin(hbond_map / margin * (np.pi / 2.0)) + 1.0) / 2.0
    hbond_map *= local_mask[None, :, :]
    hbond_map *= donor_2d[None, :, :]

    return hbond_map if coord.ndim == 4 else hbond_map[0]


def _pydssp_unfold(a: np.ndarray, window: int, axis: int) -> np.ndarray:
    """Sliding-window view along an axis (PyDSSP-style)."""

    # Follow the original PyDSSP numpy implementation closely: respect
    # negative ``axis`` values as-is so that ``axis-1`` in ``moveaxis`` has
    # the same semantics. Normalizing ``axis`` to a positive index changes
    # this behavior and breaks the expected output shape.

    idx = (
        np.arange(window)[:, None]
        + np.arange(a.shape[axis] - window + 1)[None, :]
    )
    unfolded = np.take(a, idx, axis=axis)
    return np.moveaxis(unfolded, axis - 1, -1)


def _pydssp_assign_onehot(
    coord: np.ndarray,
    donor_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Return one-hot C3 labels (loop, helix, strand) as in PyDSSP.

    Parameters
    ----------
    coord:
        Backbone coordinates ``(L, 4, 3)`` or ``(B, L, 4, 3)``.
    donor_mask:
        Optional donor mask of length L.
    """

    coord_b, org_shape = _pydssp_check_input(coord)

    # Hydrogen-bond map: shape (B, L, L)
    hbmap = _pydssp_get_hbond_map(coord_b, donor_mask=donor_mask)
    # Convert into "i:C=O, j:N-H" form
    hbmap = hbmap.transpose(0, 2, 1)

    # Identify turn 3, 4, 5
    turn3 = np.diagonal(hbmap, axis1=-2, axis2=-1, offset=3) > 0.0
    turn4 = np.diagonal(hbmap, axis1=-2, axis2=-1, offset=4) > 0.0
    turn5 = np.diagonal(hbmap, axis1=-2, axis2=-1, offset=5) > 0.0

    # Assignment of helical SS
    h3 = np.pad(turn3[:, :-1] * turn3[:, 1:], ((0, 0), (1, 3)))
    h4 = np.pad(turn4[:, :-1] * turn4[:, 1:], ((0, 0), (1, 4)))
    h5 = np.pad(turn5[:, :-1] * turn5[:, 1:], ((0, 0), (1, 5)))

    helix4 = h4 + np.roll(h4, 1, 1) + np.roll(h4, 2, 1) + np.roll(h4, 3, 1)
    h3 = h3 * ~np.roll(helix4, -1, 1) * ~helix4
    h5 = h5 * ~np.roll(helix4, -1, 1) * ~helix4
    helix3 = h3 + np.roll(h3, 1, 1) + np.roll(h3, 2, 1)
    helix5 = (
        h5
        + np.roll(h5, 1, 1)
        + np.roll(h5, 2, 1)
        + np.roll(h5, 3, 1)
        + np.roll(h5, 4, 1)
    )

    # Identify bridges and ladders
    unfoldmap = _pydssp_unfold(_pydssp_unfold(hbmap, 3, -2), 3, -2) > 0.0
    unfoldmap_rev = np.swapaxes(unfoldmap, 1, 2)

    p_bridge = (
        unfoldmap[:, :, :, 0, 1] * unfoldmap_rev[:, :, :, 1, 2]
        + unfoldmap_rev[:, :, :, 0, 1] * unfoldmap[:, :, :, 1, 2]
    )
    p_bridge = np.pad(p_bridge, ((0, 0), (1, 1), (1, 1)))

    a_bridge = (
        unfoldmap[:, :, :, 1, 1] * unfoldmap_rev[:, :, :, 1, 1]
        + unfoldmap[:, :, :, 0, 2] * unfoldmap_rev[:, :, :, 0, 2]
    )
    a_bridge = np.pad(a_bridge, ((0, 0), (1, 1), (1, 1)))

    ladder = (p_bridge + a_bridge).sum(-1) > 0

    # Final C3 one-hot labels
    helix = (helix3 + helix4 + helix5) > 0
    strand = ladder
    loop = (~helix) & (~strand)

    onehot = np.stack([loop, helix, strand], axis=-1)
    if len(org_shape) == 3:
        onehot = onehot[0]
    return onehot


def _model_hydrogen(n: np.ndarray, ca: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Model backbone H position from N, CA, C.

    Very simple geometry: place H at 1.0  from N in the direction of the
    bisector between N->CA and N->C. This is sufficient for approximate
    hydrogen-bond energy evaluation.
    """

    v1 = ca - n
    v2 = c - n
    v = v1 + v2
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    # Fallback direction if degenerate
    v[norm[:, 0] == 0.0] = np.array([1.0, 0.0, 0.0])
    norm[norm == 0.0] = 1.0
    v /= norm
    return n + 1.0 * v


def _compute_hbond_energy_matrix(bb: np.ndarray) -> np.ndarray:
    """Compute NxN hydrogen-bond energy matrix using DSSP-like formula.

    Parameters
    ----------
    bb:
        Backbone coordinates of shape (N, 4, 3) with atoms (N, CA, C, O).

    Returns
    -------
    E:
        Array of shape (N, N) with electrostatic energies (kcal/mol). Larger
        negative values indicate stronger hydrogen bonds.
    """

    n = bb[:, 0, :]
    ca = bb[:, 1, :]
    c = bb[:, 2, :]
    o = bb[:, 3, :]

    h = _model_hydrogen(n, ca, c)

    # Broadcast to pairwise distances: i = donor (N-H-C), j = acceptor (C=O)
    N_i = n[:, None, :]
    H_i = h[:, None, :]
    C_j = c[None, :, :]
    O_j = o[None, :, :]

    # Distances
    def _dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        d = a - b
        return np.linalg.norm(d, axis=-1)

    r_ON = _dist(N_i, O_j)
    r_CH = _dist(C_j, H_i)
    r_OH = _dist(O_j, H_i)
    r_CN = _dist(C_j, N_i)

    # Avoid division by zero
    eps = 1e-6
    r_ON = np.maximum(r_ON, eps)
    r_CH = np.maximum(r_CH, eps)
    r_OH = np.maximum(r_OH, eps)
    r_CN = np.maximum(r_CN, eps)

    # Kabsch–Sander electrostatic energy (approximate constants), as in
    # DSSP and PyDSSP: E = q1*q2*F*(1/r_ON + 1/r_CH - 1/r_OH - 1/r_CN)
    E = CONST_Q1Q2 * CONST_F * (
        1.0 / r_ON
        + 1.0 / r_CH
        - 1.0 / r_OH
        - 1.0 / r_CN
    )

    return E


if _HAVE_NUMBA and nb is not None:

    @nb.njit(nopython=True, nogil=True, cache=True)  # type: ignore[misc]
    def _model_hydrogen_nb(n: np.ndarray, ca: np.ndarray, c: np.ndarray) -> np.ndarray:
        v1 = ca - n
        v2 = c - n
        v = v1 + v2
        norm = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
        if norm == 0.0:
            return n + np.array([1.0, 0.0, 0.0], dtype=np.float64)
        inv_norm = 1.0 / norm
        return n + v * inv_norm

    @nb.njit(nopython=True, nogil=True, cache=True)  # type: ignore[misc]
    def _compute_hbond_energy_matrix_nb(bb: np.ndarray) -> np.ndarray:
        n_res = bb.shape[0]
        E = np.zeros((n_res, n_res), dtype=np.float64)
        eps = 1e-6
        for i in range(n_res):
            n_i = bb[i, 0]
            ca_i = bb[i, 1]
            c_i = bb[i, 2]
            h_i = _model_hydrogen_nb(n_i, ca_i, c_i)
            for j in range(n_res):
                c_j = bb[j, 2]
                o_j = bb[j, 3]

                dx_on = n_i[0] - o_j[0]
                dy_on = n_i[1] - o_j[1]
                dz_on = n_i[2] - o_j[2]
                r_on = math.sqrt(dx_on * dx_on + dy_on * dy_on + dz_on * dz_on)

                dx_ch = c_j[0] - h_i[0]
                dy_ch = c_j[1] - h_i[1]
                dz_ch = c_j[2] - h_i[2]
                r_ch = math.sqrt(dx_ch * dx_ch + dy_ch * dy_ch + dz_ch * dz_ch)

                dx_oh = o_j[0] - h_i[0]
                dy_oh = o_j[1] - h_i[1]
                dz_oh = o_j[2] - h_i[2]
                r_oh = math.sqrt(dx_oh * dx_oh + dy_oh * dy_oh + dz_oh * dz_oh)

                dx_cn = c_j[0] - n_i[0]
                dy_cn = c_j[1] - n_i[1]
                dz_cn = c_j[2] - n_i[2]
                r_cn = math.sqrt(dx_cn * dx_cn + dy_cn * dy_cn + dz_cn * dz_cn)

                if r_on < eps:
                    r_on = eps
                if r_ch < eps:
                    r_ch = eps
                if r_oh < eps:
                    r_oh = eps
                if r_cn < eps:
                    r_cn = eps

                E[i, j] = CONST_Q1Q2 * CONST_F * (
                    1.0 / r_on + 1.0 / r_ch - 1.0 / r_oh - 1.0 / r_cn
                )
        return E


def _compute_hbond_energy_matrix_fast(bb: np.ndarray) -> np.ndarray:
    """Fast hydrogen-bond energy matrix with optional numba acceleration."""

    if _HAVE_NUMBA and nb is not None:
        try:
            return _compute_hbond_energy_matrix_nb(bb)  # type: ignore[name-defined]
        except Exception:
            pass
    return _compute_hbond_energy_matrix(bb)


def _energy_to_hbond_map(
    E: np.ndarray,
    cutoff: float = DEFAULT_CUTOFF,
    margin: float = DEFAULT_MARGIN,
) -> np.ndarray:
    """Convert electrostatic energy matrix to continuous H-bond map.

    This mirrors the definition used in PyDSSP:

        Hbond(i,j) = (1 + sin((cutoff - margin - E(i,j))/margin * pi/2)) / 2

    Values lie in [0, 1]. We also mask out local pairs (i,i), (i,i+1),
    (i,i+2) which are not considered backbone hydrogen bonds.
    """

    # Continuous extension of the original DSSP thresholding
    h = np.clip(cutoff - margin - E, a_min=-margin, a_max=margin)
    h = (np.sin(h / margin * (np.pi / 2.0)) + 1.0) / 2.0

    L = E.shape[0]
    local_mask = ~np.eye(L, dtype=bool)
    if L > 1:
        local_mask &= ~np.eye(L, k=-1, dtype=bool)
    if L > 2:
        local_mask &= ~np.eye(L, k=-2, dtype=bool)
    return h * local_mask


def _assign_c3_from_hbond(E: np.ndarray, energy_threshold: float = -0.5) -> List[str]:
    """Assign C3 secondary structure (H/E/C) from H-bond energy matrix.

    This follows a very simplified DSSP-like scheme:

    - A hydrogen bond exists if E(i,j) < energy_threshold.
    - Helix (H): residues participating in i->i+4 or i+4->i hydrogen bonds
      are marked as helix (including the span between them).
    - Strand (E): residues in long-range H-bonds (|i-j| > 4) are marked as
      strand if they are not already helix.
    - Coil (C): everything else.
    """

    n_res = E.shape[0]
    # Convert electrostatic energies into continuous H-bond strengths and
    # threshold at 0.0 (PyDSSP-style) to obtain a boolean H-bond map.
    # With the chosen mapping, hb_map > 0.0 is equivalent to E below the
    # cutoff, but preserves more marginal bonds than a hb_map > 0.5 test.
    hb_map = _energy_to_hbond_map(E, cutoff=energy_threshold)
    hb = hb_map > 0.0

    ss = np.array(["C"] * n_res, dtype="U1")

    # Helices: i <-> i+4 pattern
    for i in range(n_res - 4):
        if hb[i, i + 4] or hb[i + 4, i]:
            j = i + 4
            ss[i : j + 1] = "H"

    # Strands: long-range H-bonds, prefer not to overwrite helices
    for i in range(n_res):
        for j in range(i + 3, n_res):
            if hb[i, j] or hb[j, i]:
                if ss[i] != "H":
                    ss[i] = "E"
                if ss[j] != "H":
                    ss[j] = "E"

    return ss.tolist()


def assign_ss_c3_from_atoms(
    atoms: np.ndarray,
    n_res: int,
    verbose: bool = True,
) -> Optional[List[str]]:
    """Assign C3 codes (H/E/C) from a ChiSurf-style atoms array.

    If ``verbose`` is True (default), prints a short summary of the
    assignment (backbone length, requested length, and H/E/C counts).
    """

    if n_res <= 0:
        return None

    bb = _build_backbone_from_atoms(atoms)
    if bb is None or bb.shape[0] == 0:
        if verbose:
            print("Chimol SS: no valid backbone could be built from atoms")
        return None

    # Use full PyDSSP-style assignment on backbone coordinates to obtain
    # one-hot (loop, helix, strand) labels.
    try:
        onehot = _pydssp_assign_onehot(bb)
    except Exception as e:
        if verbose:
            print(f"Chimol SS: PyDSSP onehot assignment failed: {e!r}")
        return None

    if onehot is None or onehot.size == 0 or onehot.ndim != 2 or onehot.shape[1] != 3:
        if verbose:
            shape = None if onehot is None else onehot.shape
            print(f"Chimol SS: invalid onehot output shape={shape}")
        return None

    helix = np.asarray(onehot[:, 1], dtype=bool)
    strand = np.asarray(onehot[:, 2], dtype=bool)

    n_bb = onehot.shape[0]
    ss_arr = np.full(n_bb, "C", dtype="U1")
    ss_arr[helix] = "H"
    ss_arr[strand & ~helix] = "E"

    ss_codes = ss_arr.tolist()

    # Align with n_res
    if len(ss_codes) < n_res:
        ss_codes.extend(["C"] * (n_res - len(ss_codes)))
    elif len(ss_codes) > n_res:
        ss_codes = ss_codes[:n_res]

    if verbose:
        n_total = len(ss_codes)
        n_H = sum(c == "H" for c in ss_codes)
        n_E = sum(c == "E" for c in ss_codes)
        n_C = sum(c == "C" for c in ss_codes)
        print(
            "Chimol SS: n_backbone=%d, n_res_requested=%d, "
            "assigned=%d (H=%d, E=%d, C=%d)" % (n_bb, n_res, n_total, n_H, n_E, n_C)
        )

    return ss_codes


def assign_ss_c3_from_file(
    filename: str,
    n_res: int,
    verbose: bool = True,
) -> Optional[List[str]]:
    """Assign C3 secondary-structure codes (H/E/C) from a structure file.

    Thin wrapper that loads a ChiSurf-style atoms array using the
    IMP-based coordinate reader and then delegates to
    :func:`assign_ss_c3_from_atoms`.
    """

    if n_res <= 0:
        return None

    try:  # Lazy import to avoid hard-wiring Chimol to IMP at import time
        from chisurf.core.fio.structure import coordinates as _coords  # type: ignore[import]
    except Exception:
        return None

    try:
        atoms = _coords.read_coordinates(str(filename))
    except Exception as e:
        if verbose:
            print(f"Chimol SS: failed to read coordinates from {filename!r}: {e!r}")
        return None

    if atoms is None or getattr(atoms, "size", 0) == 0:
        if verbose:
            print(f"Chimol SS: empty atoms array from {filename!r}")
        return None

    return assign_ss_c3_from_atoms(atoms, n_res, verbose=verbose)


if _HAVE_NUMBA:
    # Optional numba acceleration for large systems. We wrap the vectorized
    # implementation instead of rewriting it in explicit loops to keep the
    # code compact and close to the reference NumPy formulation.
    try:  # pragma: no cover - best-effort acceleration
        _compute_hbond_energy_matrix  # JIT disabled: keep pure-NumPy path
    except Exception:
        pass
