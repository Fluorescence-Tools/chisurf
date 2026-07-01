"""Model precision from repeated docking (FPS-style positional uncertainty).

Given the best-scoring structures of several independent docking runs, superpose
them on the fixed (reference) body and measure how much each atom of the mobile
body wanders — the per-atom RMSF. This is the FPS "precision of the model":
a mean structure whose B-factor column carries the positional uncertainty, plus
a per-atom CSV.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence

import numpy as np


def _read_pdb_atoms(path: str):
    """Return ``(lines, xyz)`` for ATOM/HETATM records of a PDB file."""
    lines: List[str] = []
    xyz: List[tuple] = []
    chains: List[str] = []
    with open(path) as fh:
        for line in fh:
            if line.startswith(("ATOM", "HETATM")):
                lines.append(line.rstrip("\n"))
                xyz.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
                chains.append(line[21])
    return lines, np.asarray(xyz, dtype=float), np.asarray(chains)


def _kabsch(mobile: np.ndarray, target: np.ndarray):
    """Rigid transform (R, t) minimising ``||R·mobile + t - target||``."""
    mc, tc = mobile.mean(0), target.mean(0)
    h = (mobile - mc).T @ (target - tc)
    u, _s, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return r, tc - r @ mc


def estimate_position_uncertainty(
    pdb_paths: Sequence[str],
    fixed_chains: Sequence[str],
    out_pdb: Optional[str] = None,
    out_csv: Optional[str] = None,
) -> Dict:
    """Superpose docked models on the fixed body and report per-atom RMSF.

    Parameters
    ----------
    pdb_paths : sequence of str
        Best-scoring PDB per docking run (same atom ordering — they are the same
        structure docked from different starts).
    fixed_chains : sequence of str
        Chain identifiers of the fixed/reference body used for superposition.
    out_pdb : str, optional
        Write the mean structure with B-factor = per-atom RMSF here.
    out_csv : str, optional
        Write a per-atom ``index,chain,rmsf`` table here.

    Returns
    -------
    dict
        ``{n_models, rmsf_mean, rmsf_max, mobile_rmsf_mean, uncertainty_pdb,
        uncertainty_csv}``.
    """
    paths = [p for p in pdb_paths if p and os.path.exists(p)]
    if len(paths) < 2:
        return {"n_models": len(paths), "rmsf_mean": float("nan"),
                "rmsf_max": float("nan"), "mobile_rmsf_mean": float("nan"),
                "uncertainty_pdb": None, "uncertainty_csv": None}

    lines0, xyz0, chains = _read_pdb_atoms(paths[0])
    fixed_mask = np.isin(chains, list(fixed_chains))
    if not fixed_mask.any():
        fixed_mask = np.ones(len(chains), dtype=bool)  # no fixed body: align on all

    aligned = [xyz0]
    for p in paths[1:]:
        _lines, xyz, _ch = _read_pdb_atoms(p)
        if xyz.shape != xyz0.shape:
            continue  # skip models with a different atom count
        r, t = _kabsch(xyz[fixed_mask], xyz0[fixed_mask])
        aligned.append((r @ xyz.T).T + t)

    stack = np.stack(aligned)                  # (n_models, n_atoms, 3)
    mean = stack.mean(0)
    rmsf = np.sqrt(((stack - mean) ** 2).sum(-1).mean(0))  # per-atom

    if out_pdb:
        _write_bfactor_pdb(lines0, mean, rmsf, out_pdb)
    if out_csv:
        import csv
        with open(out_csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["atom_index", "chain", "rmsf"])
            for i, (c, v) in enumerate(zip(chains, rmsf)):
                w.writerow([i, c, round(float(v), 3)])

    mobile = rmsf[~fixed_mask] if (~fixed_mask).any() else rmsf
    return {
        "n_models": len(aligned),
        "rmsf_mean": float(rmsf.mean()),
        "rmsf_max": float(rmsf.max()),
        "mobile_rmsf_mean": float(mobile.mean()),
        "uncertainty_pdb": out_pdb if out_pdb else None,
        "uncertainty_csv": out_csv if out_csv else None,
    }


def _write_bfactor_pdb(lines, coords, bfactors, out_pdb) -> None:
    """Rewrite ATOM lines with mean coordinates and RMSF in the B-factor column."""
    with open(out_pdb, "w") as fh:
        for line, (x, y, z), b in zip(lines, coords, bfactors):
            b = min(999.99, float(b))
            fh.write(f"{line[:30]}{x:8.3f}{y:8.3f}{z:8.3f}{line[54:60]}{b:6.2f}{line[66:]}\n")
        fh.write("END\n")


__all__ = ["estimate_position_uncertainty"]
