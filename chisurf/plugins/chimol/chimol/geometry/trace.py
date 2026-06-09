from __future__ import annotations

import numpy as np

from ..config import _DISPLAY_CONFIG


def _extract_ca_coordinates_from_atoms(atoms: np.ndarray) -> np.ndarray | None:
    """Return CA atom coordinates if available, else ``None``.

    The ``atoms`` array is expected to expose ``'atom_name'`` and ``'xyz'``
    fields compatible with :mod:`chisurf.core.structure`.
    """
    coords, _, _, _ = _extract_ca_trace(atoms)
    return coords


def _extract_ca_trace(
    atoms: np.ndarray,
) -> tuple[
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
]:
    """Return (coords, res_ids, res_names, chain_ids) for backbone atoms.

    The ``atoms`` array is expected to expose at least ``'atom_name'`` and
    ``'xyz'`` fields; residue ids/names/chain identifiers are optional.

    For nucleic acids, this function selects only ONE atom per residue to ensure
    a smooth trace. For proteins, it selects CA atoms.
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

    # PyMOL-style nucleic acid trace atoms: P is primary (mode 4 default),
    # sugar atoms are fallbacks for residues missing P.
    nuc_priority = ["P", "O5'", "C5'", "C4'", "C3'", "O3'", "C1'", "C1*"]

    if "res_id" in fields and "chain" in fields and "res_name" in fields:
        res_ids_all = np.asarray(atoms["res_id"])
        chain_ids_all = np.array([str(c).strip() for c in atoms["chain"]])
        res_names_all = np.array([str(r).strip().upper() for r in atoms["res_name"]])

        try:
            text_u = np.char.upper(text.astype(str))
        except Exception:
            text_u = np.array([str(t).upper() for t in text])

        prot_atoms_set = set(p.upper() for p in prot_atoms)
        nuc_priority_upper = [p.upper() for p in nuc_priority]
        priority_index = {a: idx for idx, a in enumerate(nuc_priority_upper)}
        nuc_set = set(nuc_priority_upper)

        nucleic_names = {
            "DA", "DC", "DG", "DT", "A", "C", "G", "T", "U",
            "2DA", "2DC", "2DG", "2DT",
            "RA", "RC", "RG", "RU", "I",
            "5MC", "5HC", "OMC", "H2U", "PSU", "M2G", "1MA", "7MG",
            "D2A", "D2C", "D2G", "D2T", "R2A", "R2C", "R2G", "R2U",
        }

        best_by_key = {}
        for i in range(len(text_u)):
            r_name = res_names_all[i]
            r_id = res_ids_all[i]
            ch = chain_ids_all[i]
            key = (r_id, ch)
            atom_name_u = text_u[i]

            if r_name in nucleic_names:
                if atom_name_u in nuc_set:
                    pi = priority_index.get(atom_name_u, 999)
                    if key not in best_by_key or pi < best_by_key[key][1]:
                        best_by_key[key] = (i, pi)
            else:
                if atom_name_u in prot_atoms_set:
                    if key not in best_by_key:
                        best_by_key[key] = (i, 0)

        if best_by_key:
            ordered = [idx for idx, _ in sorted(best_by_key.values())]
            mask = np.zeros(len(atoms), dtype=bool)
            mask[ordered] = True
        else:
            mask = np.zeros(len(atoms), dtype=bool)
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
