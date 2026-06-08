"""OLGA FRET pair selection functionality."""

from __future__ import annotations

import glob
import os
from typing import Dict, List, Optional, Tuple

import mdtraj as md
import numpy as np

from .olga_greedy import select_informative_pairs
from . import av as _av
from . import io as _io


def preprocess_efficiency_matrix(
    effs: np.ndarray,
    rmsds: np.ndarray,
    max_nan_fraction: float = 0.20,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """NaN preprocessing matching OLGA's GetInformativePairsDialog.cpp.

    Steps (in order):
    1. Drop columns where NaN fraction > max_nan_fraction.
    2. For remaining NaN cells, copy efficiency from the frame with
       the smallest RMSD to the NaN frame that has a valid value.

    Parameters
    ----------
    effs : (n_frames, n_pairs) float32 — may contain NaN
    rmsds : (n_frames, n_frames) float32 — all-vs-all RMSD matrix
    max_nan_fraction : float
        Columns with more NaN than this fraction are dropped.

    Returns
    -------
    effs_clean : (n_frames, n_valid_pairs) — no NaN
    rmsds : (n_frames, n_frames) — unchanged
    valid_pair_indices : list of int — original column indices surviving filter
    """
    n_frames, n_pairs = effs.shape
    nan_counts = np.isnan(effs).sum(axis=0)
    nan_fractions = nan_counts / n_frames

    # Drop columns where NaN fraction > max_nan_fraction
    valid_pair_indices = [i for i in range(n_pairs) if nan_fractions[i] <= max_nan_fraction]
    effs_clean = effs[:, valid_pair_indices].copy()

    # For remaining NaN cells, fill from the nearest frame (by RMSD) that has a valid value
    n_valid_pairs = len(valid_pair_indices)
    for p_idx in range(n_valid_pairs):
        nan_mask = np.isnan(effs_clean[:, p_idx])
        if not np.any(nan_mask):
            continue
        valid_indices = np.where(~nan_mask)[0]
        if len(valid_indices) == 0:
            continue
        nan_indices = np.where(nan_mask)[0]
        for f_idx in nan_indices:
            # Find closest valid frame by RMSD
            dists = rmsds[f_idx, valid_indices]
            best_valid_idx = valid_indices[np.argmin(dists)]
            effs_clean[f_idx, p_idx] = effs_clean[best_valid_idx, p_idx]

    return effs_clean, rmsds, valid_pair_indices


def compute_rmsd_matrix_from_pdb_dir(
    pdb_dir: str,
    selection: str = "name CA",
    pattern: str = "*.pdb",
) -> Tuple[np.ndarray, List[str]]:
    """Build all-vs-all RMSD matrix from a directory of PDB files.

    Uses superposition (Kabsch) before RMSD computation.

    Parameters
    ----------
    pdb_dir : str
        Directory containing PDB files.
    selection : str
        Atom selection syntax.
    pattern : str
        Glob pattern.

    Returns
    -------
    rmsds : (n, n) float32 — symmetric RMSD matrix
    filenames : list of str — PDB basenames, in matrix order
    """
    paths = sorted(glob.glob(os.path.join(pdb_dir, pattern)))
    if not paths:
        raise FileNotFoundError(f"No structures found in {pdb_dir} matching {pattern}")

    trajs = [md.load(p) for p in paths]
    combined = md.join(trajs)

    atom_indices = combined.topology.select(selection)
    if len(atom_indices) == 0:
        raise ValueError(f"Selection '{selection}' matches zero atoms in topology")

    n = len(paths)
    rmsds = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        rmsds[i, :] = md.rmsd(combined, combined, frame=i, atom_indices=atom_indices, precentered=False)

    filenames = [os.path.basename(p) for p in paths]
    return rmsds, filenames


def compute_efficiency_matrix_from_evaluators(
    pdb_dir: str,
    positions: Dict,
    distances: Dict,
    forster_radii: Optional[Dict[str, float]] = None,
    pattern: str = "*.pdb",
    n_threads: int = 1,
) -> Tuple[np.ndarray, List[str]]:
    """Compute (n_frames, n_pairs) FRET efficiency matrix.

    Uses FretEfficiencyEvaluator for each distance pair.
    NaN is recorded for structures where AV computation failed.

    Parameters
    ----------
    pdb_dir : str
        Directory of PDB files.
    positions : dict
        fps.json Positions section.
    distances : dict
        fps.json Distances section.
    forster_radii : dict, optional
        Map of pair_name -> Forster radius (overrides fps.json settings).
    pattern : str
        Glob pattern.
    n_threads : int
        Number of threads.

    Returns
    -------
    effs : (n_frames, n_pairs) float32 — may contain NaN
    pair_names : list of str — distance keys in column order
    """
    from .evaluate import evaluate_directory
    from ..evaluators import FretEfficiencyEvaluator

    evaluators = []
    pair_names = sorted(distances.keys())
    for k in pair_names:
        ddef = distances[k]
        pos1 = ddef["position1_name"]
        pos2 = ddef["position2_name"]

        R0 = 52.0
        if forster_radii and k in forster_radii:
            R0 = forster_radii[k]
        else:
            R0 = float(ddef.get("Forster_radius", ddef.get("forster_radius", 52.0)))

        evaluators.append(FretEfficiencyEvaluator(k, pos1, pos2, R0))

    storage = evaluate_directory(pdb_dir, positions, evaluators, pattern=pattern, n_threads=n_threads)

    n_frames = len(storage.filenames)
    n_pairs = len(evaluators)
    effs = np.empty((n_frames, n_pairs), dtype=np.float32)
    for p_idx, ev in enumerate(evaluators):
        effs[:, p_idx] = storage.results[ev.name]

    return effs, pair_names


def write_pair_selection_report(
    selected_pair_names: List[str],
    precision_decay: np.ndarray,
    initial_rmsd: float,
    output_path: str,
) -> None:
    """Write OLGA-format precision decay report (tab-separated).

    Format::

        #\tPair_added\t<<RMSD>>/A
        0\t--\t<initial_rmsd>
        1\t<pair_name>\t<rmsd>
        ...

    Parameters
    ----------
    selected_pair_names : list of str
        Pair names in selection order.
    precision_decay : (n_selected,) ndarray
        Expected mean RMSD after adding each pair.
    initial_rmsd : float
        Expected mean RMSD before any pair is added.
    output_path : str
        Destination file path.
    """
    lines = ["#\tPair_added\t<<RMSD>>/A"]
    lines.append(f"0\t--\t{initial_rmsd:.4f}")
    for idx, pname in enumerate(selected_pair_names):
        lines.append(f"{idx + 1}\t{pname}\t{precision_decay[idx]:.4f}")
    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def compute_rmsd_matrix_from_trajectory(
    top_path: str,
    traj_path: str,
    selection: str = "name CA",
) -> Tuple[np.ndarray, List[str]]:
    """Build all-vs-all RMSD matrix from a trajectory file.

    Uses superposition (Kabsch) before RMSD computation.

    Parameters
    ----------
    top_path : str
        Path to topology PDB file.
    traj_path : str
        Path to trajectory file (e.g. XTC, DCD).
    selection : str
        Atom selection syntax.

    Returns
    -------
    rmsds : (n, n) float32 — symmetric RMSD matrix
    filenames : list of str — frame names, in matrix order
    """
    t = md.load(traj_path, top=top_path)
    atom_indices = t.topology.select(selection)
    if len(atom_indices) == 0:
        raise ValueError(f"Selection '{selection}' matches zero atoms in topology")

    n = t.n_frames
    rmsds = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        rmsds[i, :] = md.rmsd(t, t, frame=i, atom_indices=atom_indices, precentered=False)

    filenames = [f"frame_{i}" for i in range(n)]
    return rmsds, filenames


def compute_efficiency_matrix_from_evaluators_trajectory(
    top_path: str,
    traj_path: str,
    positions: Dict,
    distances: Dict,
    forster_radii: Optional[Dict[str, float]] = None,
) -> Tuple[np.ndarray, List[str]]:
    """Compute (n_frames, n_pairs) FRET efficiency matrix from a trajectory.

    Uses FretEfficiencyEvaluator for each distance pair.

    Parameters
    ----------
    top_path : str
        Path to topology PDB file.
    traj_path : str
        Path to trajectory file (e.g. XTC, DCD).
    positions : dict
        fps.json Positions section.
    distances : dict
        fps.json Distances section.
    forster_radii : dict, optional
        Map of pair_name -> Forster radius (overrides fps.json settings).

    Returns
    -------
    effs : (n_frames, n_pairs) float32 — may contain NaN
    pair_names : list of str — distance keys in column order
    """
    from .evaluate import evaluate_trajectory
    from ..evaluators import FretEfficiencyEvaluator

    evaluators = []
    pair_names = sorted(distances.keys())
    for k in pair_names:
        ddef = distances[k]
        pos1 = ddef["position1_name"]
        pos2 = ddef["position2_name"]

        R0 = 52.0
        if forster_radii and k in forster_radii:
            R0 = forster_radii[k]
        else:
            R0 = float(ddef.get("Forster_radius", ddef.get("forster_radius", 52.0)))

        evaluators.append(FretEfficiencyEvaluator(k, pos1, pos2, R0))

    storage = evaluate_trajectory(top_path, traj_path, positions, evaluators)

    n_frames = len(storage.filenames)
    n_pairs = len(evaluators)
    effs = np.empty((n_frames, n_pairs), dtype=np.float32)
    for p_idx, ev in enumerate(evaluators):
        effs[:, p_idx] = storage.results[ev.name]

    return effs, pair_names

