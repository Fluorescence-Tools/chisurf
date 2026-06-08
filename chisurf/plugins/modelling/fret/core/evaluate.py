"""Evaluation functions for FRET metrics over structure structures, directories, and trajectories."""

from __future__ import annotations

import concurrent.futures
import glob
import os
import tempfile
from typing import Dict, List, Optional

import numpy as np

from . import av as _av
from . import io as _io
from .engine import RigidBody
from ..evaluators import Evaluator, EvaluatorResult, EvaluationStorage


def make_bodies_for_structure(
    atoms_xyzr: np.ndarray,
    positions: Dict,
) -> List[RigidBody]:
    """Helper to construct basic RigidBody objects for evaluation."""
    body_map = {pname: int(pdef.get("body_id", 0)) for pname, pdef in positions.items()}
    n_bodies = max(body_map.values()) + 1 if body_map else 1
    bodies = []
    for bi in range(n_bodies):
        com = atoms_xyzr[:, :3].mean(axis=0)
        local_coords = atoms_xyzr[:, :3] - com
        local_xyzr = np.column_stack([local_coords, atoms_xyzr[:, 3]])
        rb = RigidBody(
            name=f"body_{bi}",
            atoms_local=local_xyzr,
            com=com.copy(),
            rotation=np.eye(3),
            translation=com.copy(),
            mass=float(atoms_xyzr.shape[0]),
            inertia=np.eye(3) * 1000.0,
        )
        bodies.append(rb)
    return bodies


def evaluate_structure(
    pdb_path: str,
    positions: Dict,
    evaluators: List[Evaluator],
    disc_step: Optional[float] = None,
) -> Dict[str, EvaluatorResult]:
    """Run all evaluators on a single PDB structure.

    Parameters
    ----------
    pdb_path : str
        Path to PDB file.
    positions : dict
        fps.json Positions section.
    evaluators : list of Evaluator
        Evaluators to run.
    disc_step : float, optional
        AV grid step size.

    Returns
    -------
    results : dict of str -> EvaluatorResult
        Evaluated metrics.
    """
    atoms_xyzr = _av.load_structure_with_vdw(pdb_path)
    avs = _av.compute_avs_for_structure(atoms_xyzr, positions, pdb_path=pdb_path, disc_step=disc_step)
    bodies = make_bodies_for_structure(atoms_xyzr, positions)

    results = {}
    for ev in evaluators:
        results[ev.name] = ev.evaluate(avs, bodies)
    return results


def evaluate_directory(
    pdb_dir: str,
    positions: Dict,
    evaluators: List[Evaluator],
    pattern: str = "*.pdb",
    n_threads: int = 1,
    disc_step: Optional[float] = None,
) -> EvaluationStorage:
    """Run all evaluators on all PDB files matching a pattern in a directory.

    Parameters
    ----------
    pdb_dir : str
        Directory containing PDB files.
    positions : dict
        fps.json Positions section.
    evaluators : list of Evaluator
        Evaluators to run.
    pattern : str
        Glob pattern.
    n_threads : int
        Number of threads to run in parallel.
    disc_step : float, optional
        AV grid step size.

    Returns
    -------
    storage : EvaluationStorage
        Accumulated results.
    """
    storage = EvaluationStorage()
    paths = sorted(glob.glob(os.path.join(pdb_dir, pattern)))

    def _eval_one(path: str):
        basename = os.path.basename(path)
        try:
            res = evaluate_structure(path, positions, evaluators, disc_step=disc_step)
            return basename, res
        except Exception as e:
            # Return empty/default result on error
            return basename, {ev.name: EvaluatorResult(ev.name, np.nan) for ev in evaluators}

    if n_threads > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as executor:
            fut_to_path = {executor.submit(_eval_one, p): p for p in paths}
            for fut in concurrent.futures.as_completed(fut_to_path):
                basename, res = fut.result()
                storage.add_frame(basename, res)
    else:
        for p in paths:
            basename, res = _eval_one(p)
            storage.add_frame(basename, res)

    return storage


def evaluate_trajectory(
    top_path: str,
    traj_path: str,
    positions: Dict,
    evaluators: List[Evaluator],
    disc_step: Optional[float] = None,
) -> EvaluationStorage:
    """Run all evaluators on each frame of a trajectory using MDTraj.

    Parameters
    ----------
    top_path : str
        Path to topology PDB file.
    traj_path : str
        Path to trajectory file (e.g. XTC, DCD).
    positions : dict
        fps.json Positions section.
    evaluators : list of Evaluator
        Evaluators to run.
    disc_step : float, optional
        AV grid step size.

    Returns
    -------
    storage : EvaluationStorage
        Accumulated results.
    """
    import mdtraj as md
    t = md.load(traj_path, top=top_path)
    storage = EvaluationStorage()

    # Get vdW radii map
    vdw_radii = []
    for atom in t.topology.atoms:
        elem = atom.element.atomic_number if atom.element is not None else 6
        vdw_radii.append(_av.VDW_RADII.get(elem, _av._DEFAULT_VDW))
    vdw_radii = np.array(vdw_radii, dtype=np.float64)

    # Convert mdtraj coordinates from nanometers to Angstroms
    xyz_angstrom = t.xyz * 10.0

    for f_idx in range(t.n_frames):
        coords = xyz_angstrom[f_idx]
        atoms_xyzr = np.column_stack([coords, vdw_radii])
        bodies = make_bodies_for_structure(atoms_xyzr, positions)

        # Write frame to a temp PDB file for IMP.bff if needed, or if not LabelLib
        with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False, mode="w") as tmp_f:
            pdb_path = tmp_f.name
        try:
            t[f_idx].save_pdb(pdb_path)
            avs = _av.compute_avs_for_structure(atoms_xyzr, positions, pdb_path=pdb_path, disc_step=disc_step)
        finally:
            try:
                os.unlink(pdb_path)
            except Exception:
                pass

        results = {}
        for ev in evaluators:
            results[ev.name] = ev.evaluate(avs, bodies)

        storage.add_frame(f"frame_{f_idx}", results)

    return storage
