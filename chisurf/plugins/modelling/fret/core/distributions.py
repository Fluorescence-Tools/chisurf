"""Full FRET distance distributions P(R_DA) for a structure (FPS-style).

For each experimental pair, compute the accessible volumes of the two dyes on a
(docked) structure and the distribution of donor-acceptor distances between the
two AV point clouds — the P(R_DA) that FPS reports, richer than the single mean
distance used for fast docking.
"""

from __future__ import annotations

import csv
from typing import Dict, Optional

import numpy as np

from . import av as _av
from . import distance as _dist


def compute_distance_distributions(
    pdb_path: str,
    positions: Dict,
    distances: Dict,
    out_csv: Optional[str] = None,
    *,
    rda_min: float = 1.0,
    rda_max: float = 200.0,
    n_bins: int = 100,
) -> Dict:
    """Return per-pair ``P(R_DA)`` distributions for the structure at ``pdb_path``.

    Parameters
    ----------
    pdb_path : str
        Structure (e.g. a docked PDB) on which to compute the AVs.
    positions, distances : dict
        The ``Positions`` / ``Distances`` sections of the fps.json.
    out_csv : str, optional
        Write a table with an ``R_DA`` column plus one probability column per
        pair.

    Returns
    -------
    dict
        ``{"rda_axis": [...], "pairs": {name: {"p": [...], "mean": float}},
        "distributions_csv": path}``.
    """
    atoms = _av.load_structure_with_vdw(pdb_path)
    avs = _av.compute_avs_for_structure(atoms, positions, pdb_path=pdb_path)

    rda_axis = None
    pairs: Dict[str, Dict] = {}
    for name, d in distances.items():
        av1 = avs.get(d.get("position1_name"))
        av2 = avs.get(d.get("position2_name"))
        if av1 is None or av2 is None or not av1.has_volume or not av2.has_volume:
            continue
        # histogram_rda returns (histogram, bin_edges)
        p, edges = _dist.histogram_rda(
            av1, av2, rda_min=rda_min, rda_max=rda_max, n_rda_bins=n_bins,
            normalize=True)
        centers = 0.5 * (np.asarray(edges[:-1]) + np.asarray(edges[1:]))
        rda_axis = centers
        p = np.asarray(p, dtype=float)
        total = float(p.sum())
        mean = float(np.sum(centers * p) / total) if total > 0 else float("nan")
        pairs[name] = {"p": p.tolist(), "mean": mean}

    if out_csv and pairs and rda_axis is not None:
        names = list(pairs)
        with open(out_csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["R_DA"] + names)
            for i, r in enumerate(rda_axis):
                w.writerow([round(float(r), 2)]
                           + [round(pairs[n]["p"][i], 6) for n in names])

    return {
        "rda_axis": rda_axis.tolist() if rda_axis is not None else [],
        "pairs": pairs,
        "distributions_csv": out_csv if (out_csv and pairs) else None,
    }


__all__ = ["compute_distance_distributions"]
