from __future__ import annotations

import glob
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from . import av as _av
from . import distance as _dist
from . import io as _io
from .results import ScreeningResult


def compute_screening_transfer_functions(
    pdb_path: str,
    positions: Dict,
    distances: Dict,
    disc_step: Optional[float] = None,
    tf_type: str = "Polynomial",
) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """Compute transfer function coefficients and sigmas on a reference structure.

    Parameters
    ----------
    pdb_path : str
        Reference PDB structure path.
    positions : dict
        Positions dictionary.
    distances : dict
        Distances dictionary.
    disc_step : float, optional
        Grid resolution.
    tf_type : str, optional
        Transfer function type. Default is 'Polynomial'.

    Returns
    -------
    convfuns : dict of str -> np.ndarray
        Polynomial coefficients per distance key.
    sigmas : dict of str -> float
        AV distance distribution width (standard deviation) per distance key.
    """
    convfuns = {}
    sigmas = {}
    try:
        atoms_xyzr = _av.load_structure_with_vdw(pdb_path)
        avs = _av.compute_avs_for_structure(
            atoms_xyzr, positions, pdb_path=pdb_path, disc_step=disc_step
        )
        for dkey, ddef in distances.items():
            p1 = ddef.get("position1_name", "")
            p2 = ddef.get("position2_name", "")
            if p1 in avs and p2 in avs and avs[p1].has_volume and avs[p2].has_volume:
                dtype = ddef.get("distance_type", "RDAMean")
                R0 = float(ddef.get("Forster_radius", 52.0))
                rmp, rda, rda_e, sigma = _dist.av_pair_statistics(
                    avs[p1], avs[p2], forster_radius=R0
                )
                sigmas[dkey] = sigma
                if tf_type == "Polynomial" and dtype != "Rmp":
                    convfuns[dkey] = _dist.fit_transfer_polynomial(
                        avs[p1], avs[p2], distance_type=dtype, forster_radius=R0
                    )
    except Exception as e:
        print(f"Warning: could not pre-compute screening transfer functions: {e}")
    return convfuns, sigmas


def score_single_structure(
    pdb_path: str,
    positions: Dict,
    distances: Dict,
    disc_step: Optional[float] = None,
    n_samples: int = 50000,
    score_set: Optional[List[str]] = None,
    convfuns: Optional[Dict[str, np.ndarray]] = None,
    sigmas: Optional[Dict[str, float]] = None,
    tf_type: str = "Polynomial",
) -> ScreeningResult:
    """Score one structure against FRET distance restraints.

    Parameters
    ----------
    pdb_path : str
        Path to PDB file.
    positions : dict
        fps.json Positions.
    distances : dict
        fps.json Distances.
    disc_step : float, optional
        AV grid resolution.
    n_samples : int, optional
        Number of samples for full AV distance calculations. Default is 50000.
    score_set : list of str, optional
        If given, only score distance keys in this list.
    convfuns : dict, optional
        Cached polynomial coefficients from reference structure.
    sigmas : dict, optional
        Cached standard deviations from reference structure.
    tf_type : str, optional
        Type of transfer function to apply. Default is 'Polynomial'.

    Returns
    -------
    ScreeningResult
    """
    try:
        atoms_xyzr = _av.load_structure_with_vdw(pdb_path)
    except Exception as e:
        sr = ScreeningResult(filename=os.path.basename(pdb_path))
        sr.n_nan = len(distances)
        return sr

    avs = _av.compute_avs_for_structure(
        atoms_xyzr, positions, pdb_path=pdb_path, disc_step=disc_step
    )

    result = ScreeningResult(filename=os.path.basename(pdb_path))
    result.n_distances = len(distances)

    distance_keys = list(distances.keys())
    if score_set is not None:
        distance_keys = [k for k in distance_keys if k in score_set]

    chi2_total = 0.0
    n_valid = 0
    n_viol_1 = 0
    n_viol_2 = 0
    n_viol_3 = 0
    n_nan = 0

    for dkey in distance_keys:
        ddef = distances[dkey]
        p1 = ddef.get("position1_name", "")
        p2 = ddef.get("position2_name", "")

        if p1 not in avs or p2 not in avs:
            result.model_distances[dkey] = -1.0
            result.chi2_contributions[dkey] = float("nan")
            n_nan += 1
            continue

        av1 = avs[p1]
        av2 = avs[p2]
        if not av1.has_volume or not av2.has_volume:
            result.model_distances[dkey] = -1.0
            result.chi2_contributions[dkey] = float("nan")
            n_nan += 1
            continue

        dtype = ddef.get("distance_type", "RDAMean")
        R0 = float(ddef.get("Forster_radius", 52.0))
        dexp = float(ddef.get("distance", 0.0))
        epos = float(ddef.get("error_pos", 5.0))
        eneg = float(ddef.get("error_neg", 5.0))

        try:
            if dtype == "Rmp" or tf_type == "None":
                dmod = _dist.distance_between_mean_positions(av1, av2)
            elif tf_type == "Gaussian" and sigmas and dkey in sigmas:
                rmp = _dist.distance_between_mean_positions(av1, av2)
                dmod = float(_dist.gaussian_rmp_to_rda_mean(rmp, sigmas[dkey]))
            elif tf_type == "Polynomial" and convfuns and dkey in convfuns:
                rmp = _dist.distance_between_mean_positions(av1, av2)
                dmod = float(_dist.polynomial_transfer(rmp, convfuns[dkey]))
            else:
                dmod = _dist.model_distance(av1, av2, dtype, R0, n_samples=n_samples)
        except Exception:
            dmod = -1.0

        result.model_distances[dkey] = dmod
        if dmod < 0:
            result.chi2_contributions[dkey] = float("nan")
            n_nan += 1
            continue

        chi2_contrib = _dist.chi2_score(dmod, dexp, eneg, epos)
        result.chi2_contributions[dkey] = chi2_contrib
        chi2_total += chi2_contrib
        n_valid += 1

        # Count violations by sigma
        delta = dmod - dexp
        sigma = eneg if delta < 0 else epos
        if sigma > 0:
            n_sigma = abs(delta) / sigma
            if n_sigma >= 3:
                n_viol_3 += 1
            elif n_sigma >= 2:
                n_viol_2 += 1
            elif n_sigma >= 1:
                n_viol_1 += 1

    result.chi2 = chi2_total
    result.reduced_chi2 = chi2_total / max(n_valid - 1, 1)
    result.n_valid = n_valid
    result.n_nan = n_nan
    result.n_violations_1sigma = n_viol_1
    result.n_violations_2sigma = n_viol_2
    result.n_violations_3sigma = n_viol_3
    return result


def screen_structure_library(
    pdb_dir: str,
    positions: Dict,
    distances: Dict,
    pattern: str = "*.pdb",
    n_threads: int = 1,
    disc_step: Optional[float] = None,
    n_samples: int = 50000,
    score_set: Optional[List[str]] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> List[ScreeningResult]:
    """Screen a directory of PDB structures against FRET restraints.

    Results are sorted by chi-squared (ascending).
    """
    pdb_files = sorted(glob.glob(os.path.join(pdb_dir, pattern)))
    if not pdb_files:
        raise FileNotFoundError(f"No PDB files matching '{pattern}' in {pdb_dir}")

    import chisurf
    fret_settings = getattr(chisurf.core.settings, "fret", {})
    tf_type = fret_settings.get("transfer_function", "Polynomial")

    convfuns = {}
    sigmas = {}
    if tf_type != "None":
        convfuns, sigmas = compute_screening_transfer_functions(
            pdb_files[0], positions, distances, disc_step=disc_step, tf_type=tf_type
        )

    results: List[ScreeningResult] = []
    if n_threads <= 1:
        for i, f in enumerate(pdb_files):
            sr = score_single_structure(
                f, positions, distances, disc_step, n_samples, score_set,
                convfuns=convfuns, sigmas=sigmas, tf_type=tf_type
            )
            results.append(sr)
            if progress_callback:
                progress_callback(i + 1, len(pdb_files))
    else:
        with ProcessPoolExecutor(max_workers=n_threads) as ex:
            futures = {
                ex.submit(
                    score_single_structure,
                    f,
                    positions,
                    distances,
                    disc_step,
                    n_samples,
                    score_set,
                    convfuns,
                    sigmas,
                    tf_type
                ): f
                for f in pdb_files
            }
            for i, future in enumerate(as_completed(futures)):
                results.append(future.result())
                if progress_callback:
                    progress_callback(i + 1, len(pdb_files))

    results.sort(key=lambda sr: sr.chi2)
    return results
