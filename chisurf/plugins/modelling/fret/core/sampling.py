from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from . import av as _av
from . import clash as _clash
from . import distance as _dist
from . import engine as _eng
from .engine import DistanceRestraint, RigidBody, SpringParameters
from .results import SimulationResult


def run_metropolis(
    bodies: List[RigidBody],
    restraints: List[DistanceRestraint],
    positions: Dict,
    avs: Dict[str, _av.AccessibleVolume],
    atoms_xyzr: np.ndarray,
    n_samples: int = 1000,
    n_burnin: int = 100,
    step_translation: float = 0.5,
    step_rotation: float = 0.05,
    temperature: float = 1.0,
    params: Optional[SpringParameters] = None,
) -> List[SimulationResult]:
    """Metropolis Monte Carlo sampling with FRET restraints.

    Parameters
    ----------
    bodies : list of RigidBody
        Initial body states.
    restraints : list of DistanceRestraint
        FRET distance restraints.
    positions : dict
        fps.json Positions (needed for AV mean position tracking).
    avs : dict of AccessibleVolume
        Pre-computed AVs (for distance computation).
    atoms_xyzr : (N, 4) ndarray
        xyzr coordinates of the whole system (for clash detection).
    n_samples : int
        Number of MC steps after burn-in.
    n_burnin : int
        Number of burn-in steps.
    step_translation : float
        RMS translation step size (Angstrom).
    step_rotation : float
        RMS rotation step size (radians).
    temperature : float
        MC temperature (kt in kT).
    params : SpringParameters, optional
        Clash parameters.

    Returns
    -------
    results : list of SimulationResult
        One entry per MC step (only post-burnin).
    """
    if params is None:
        params = SpringParameters()

    n_body = len(bodies)
    all_results: List[SimulationResult] = []

    def _compute_energy() -> float:
        """Compute current total energy = restraint chi2 + clash."""
        e = 0.0
        # Restraint energy
        for rst in restraints:
            if not rst.active:
                continue
            pa = rst.global_position_a(bodies)
            pb = rst.global_position_b(bodies)
            d = np.linalg.norm(pb - pa)
            delta = d - rst.distance_exp
            k = (2.0 / (rst.error_pos ** 2)) if delta > 0 else (2.0 / (rst.error_neg ** 2))
            e += 0.5 * k * delta * delta
        # Clash energy
        for i in range(n_body):
            for j in range(i + 1, n_body):
                es, _, _ = _clash.body_clash_energy(
                    bodies[i].global_xyzr(),
                    bodies[j].global_xyzr(),
                    params.k_clash,
                )
                e += es
        return e

    def _random_move(idx: int):
        """Apply random perturbation to body *idx*."""
        t = np.random.randn(3) * step_translation
        bodies[idx].com += t
        angle = np.random.randn() * step_rotation
        axis = np.random.randn(3)
        axis /= np.linalg.norm(axis) + 1e-12
        rot = _eng._rotation_matrix(axis, angle)
        bodies[idx].rotation = bodies[idx].rotation @ rot

    # Burn-in
    for _step in range(n_burnin):
        for bi in range(n_body):
            old_com = bodies[bi].com.copy()
            old_rot = bodies[bi].rotation.copy()
            e_old = _compute_energy()
            _random_move(bi)
            e_new = _compute_energy()
            delta_e = e_new - e_old
            if delta_e > 0 and np.random.random() > np.exp(-delta_e / temperature):
                # Reject
                bodies[bi].com = old_com
                bodies[bi].rotation = old_rot

    # Production
    for step in range(n_samples):
        for bi in range(n_body):
            old_com = bodies[bi].com.copy()
            old_rot = bodies[bi].rotation.copy()
            e_old = _compute_energy()
            _random_move(bi)
            e_new = _compute_energy()
            delta_e = e_new - e_old
            if delta_e > 0 and np.random.random() > np.exp(-delta_e / temperature):
                bodies[bi].com = old_com
                bodies[bi].rotation = old_rot

        sr = SimulationResult(
            converged=True,
            iterations=step + 1,
            energy=_compute_energy(),
            translations=[b.com.copy() for b in bodies],
            rotations=[b.rotation.copy() for b in bodies],
        )
        all_results.append(sr)

    return all_results
