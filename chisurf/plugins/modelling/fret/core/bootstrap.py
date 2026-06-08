"""Parametric bootstrap error estimation for FRET docking."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .docking import run_docking
from .engine import SpringParameters


@dataclass
class BootstrapResult:
    """Result of a bootstrap error estimation run.

    Parameters
    ----------
    n_bootstrap : int
        Number of bootstrap iterations.
    translations : list of list of ndarray
        translations[iteration][body_idx]
    rotations : list of list of ndarray
        rotations[iteration][body_idx]
    energies : list of float
        Final docking energies of each bootstrap iteration.
    model_distances_mean : dict of str -> float
        Mean model distance for each restraint across bootstrap trials.
    model_distances_std : dict of str -> float
        Standard deviation of model distance for each restraint across bootstrap trials.
    """
    n_bootstrap: int
    translations: List[List[np.ndarray]]
    rotations: List[List[np.ndarray]]
    energies: List[float]
    model_distances_mean: Dict[str, float]
    model_distances_std: Dict[str, float]


def run_bootstrap(
    pdb_path: str,
    positions: Dict,
    distances: Dict,
    params: SpringParameters,
    n_bootstrap: int = 100,
    disc_step: Optional[float] = None,
) -> BootstrapResult:
    """FPS error estimation via parametric bootstrap.

    1. Dock once to obtain reference model distances.
    2. For each bootstrap iteration:
       a. For each active distance, draw a perturbed distance:
            d_perturbed = d_model + randn * error_pos  (if randn > 0)
            d_perturbed = d_model + randn * error_neg  (if randn < 0)
       b. Re-dock with perturbed distances.
    3. Collect body translations and rotations across all iterations.

    Parameters
    ----------
    pdb_path : str
        Path to PDB file.
    positions : dict
        fps.json Positions section.
    distances : dict
        fps.json Distances section.
    params : SpringParameters
        Spring parameters for docking.
    n_bootstrap : int
        Number of bootstrap trials.
    disc_step : float, optional
        AV grid step size.

    Returns
    -------
    result : BootstrapResult
        The accumulated bootstrap results.
    """
    # 1. Reference dock
    ref_results, ref_avs, ref_bodies = run_docking(
        pdb_path,
        positions,
        distances,
        params=params,
        n_trials=1,
        disc_step=disc_step,
    )
    if not ref_results:
        raise RuntimeError("Reference docking failed to produce any results.")

    ref_model_distances = ref_results[0].model_distances

    translations: List[List[np.ndarray]] = []
    rotations: List[List[np.ndarray]] = []
    energies: List[float] = []
    iter_distances: Dict[str, List[float]] = {dname: [] for dname in distances.keys()}

    # 2. Bootstrap loop
    for _ in range(n_bootstrap):
        # Perturb distances
        perturbed_distances = {}
        for dname, ddef in distances.items():
            if dname not in ref_model_distances:
                continue
            d_model = ref_model_distances[dname]
            randn = np.random.randn()
            if randn > 0:
                d_perturbed = d_model + randn * float(ddef.get("error_pos", 5.0))
            else:
                d_perturbed = d_model + randn * float(ddef.get("error_neg", 5.0))

            p_ddef = ddef.copy()
            p_ddef["distance"] = max(0.1, d_perturbed)
            perturbed_distances[dname] = p_ddef

        # Dock with perturbed distances
        iter_results, _, _ = run_docking(
            pdb_path,
            positions,
            perturbed_distances,
            params=params,
            n_trials=1,
            disc_step=disc_step,
        )

        if iter_results:
            sr = iter_results[0]
            translations.append(sr.translations)
            rotations.append(sr.rotations)
            energies.append(sr.energy)
            for dname in perturbed_distances:
                if dname in sr.model_distances:
                    iter_distances[dname].append(sr.model_distances[dname])

    # Compute stats
    model_distances_mean = {}
    model_distances_std = {}
    for dname, vals in iter_distances.items():
        if vals:
            model_distances_mean[dname] = float(np.mean(vals))
            model_distances_std[dname] = float(np.std(vals))
        else:
            model_distances_mean[dname] = 0.0
            model_distances_std[dname] = 0.0

    return BootstrapResult(
        n_bootstrap=len(energies),
        translations=translations,
        rotations=rotations,
        energies=energies,
        model_distances_mean=model_distances_mean,
        model_distances_std=model_distances_std,
    )
