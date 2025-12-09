from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from ..config import _DISPLAY_CONFIG


def _extract_ca_coordinates_from_atoms(atoms: np.ndarray) -> Optional[np.ndarray]:
    """Return CA atom coordinates if available, else ``None``.

    The ``atoms`` array is expected to expose ``'atom_name'`` and ``'xyz'``
    fields compatible with :mod:`chisurf.structure`.
    """

    coords, _, _, _ = _extract_ca_trace(atoms)
    return coords


def _extract_ca_trace(
    atoms: np.ndarray,
) -> Tuple[
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
]:
    """Return (coords, res_ids, res_names, chain_ids) for backbone atoms.

    The ``atoms`` array is expected to expose at least ``'atom_name'`` and
    ``'xyz'`` fields; residue ids/names/chain identifiers are optional.
    """

    if not isinstance(atoms, np.ndarray):
        return None, None, None, None

    fields = set(atoms.dtype.fields or {})
    if "atom_name" not in fields or "xyz" not in fields:
        return None, None, None, None

    names = atoms["atom_name"]
    try:
        text = np.char.strip(names.astype(str))
    except Exception:
        text = np.array([str(n).strip() for n in names])

    cfg = _DISPLAY_CONFIG.get("backbone_trace", {})
    prot_atoms = cfg.get("protein_atoms", ["CA"])
    nuc_atoms = cfg.get("nucleic_atoms", ["P", "C4'", "C1'"])

    candidates: list[str] = []
    for seq in (prot_atoms, nuc_atoms):
        if not seq:
            continue
        for val in seq:
            s = str(val).strip()
            if s:
                candidates.append(s.upper())

    if candidates:
        try:
            text_u = np.char.upper(text.astype(str))
        except Exception:
            text_u = np.array([str(t).upper() for t in text])
        mask = np.isin(text_u, np.asarray(candidates, dtype=text_u.dtype))
    else:
        mask = text == "CA"

    if not mask.any():
        mask = text == "CA"
        if not mask.any():
            return None, None, None, None

    coords = np.asarray(atoms["xyz"][mask], dtype=float)

    res_ids = None
    res_names = None
    chain_ids = None
    if "res_id" in fields:
        res_ids = np.asarray(atoms["res_id"][mask])
    if "res_name" in fields:
        try:
            res_names = np.char.strip(atoms["res_name"][mask].astype(str))
        except Exception:
            res_names = np.array(
                [str(x).strip() for x in atoms["res_name"][mask]]
            )
    if "chain" in fields:
        try:
            chain_ids = np.char.strip(atoms["chain"][mask].astype(str))
        except Exception:
            chain_ids = np.array(
                [str(x).strip() for x in atoms["chain"][mask]]
            )
    return coords, res_ids, res_names, chain_ids


__all__ = ["_extract_ca_trace", "_extract_ca_coordinates_from_atoms"]

