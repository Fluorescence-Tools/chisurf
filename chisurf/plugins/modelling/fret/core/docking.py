from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from . import av as _av
from . import distance as _dist
from . import engine as _eng
from . import results as _res
from .engine import DistanceRestraint, RigidBody, SpringParameters
from .results import SimulationResult


def run_docking(
    pdb_path: str,
    positions: Dict,
    distances: Dict,
    params: Optional[SpringParameters] = None,
    max_refinement_cycles: int = 0,
    n_trials: int = 1,
    disc_step: Optional[float] = None,
) -> Tuple[List[SimulationResult], Dict[str, _av.AccessibleVolume], List[RigidBody]]:
    """Run FRET-restrained rigid-body docking.

    Parameters
    ----------
    pdb_path : str
        Path to input PDB file.
    positions : dict
        fps.json ``Positions`` section.
    distances : dict
        fps.json ``Distances`` section.
    params : SpringParameters, optional
    max_refinement_cycles : int
        Number of AV-recalculation cycles after initial docking.
    n_trials : int
        Number of independent docking trials.
    disc_step : float, optional
        AV simulation grid step (overrides per-position setting).

    Returns
    -------
    results : list of SimulationResult
    avs : dict of AccessibleVolume
    bodies : list of RigidBody
    """
    if params is None:
        params = SpringParameters()

    if isinstance(pdb_path, (list, tuple)):
        pdb_paths = list(pdb_path)
    elif isinstance(pdb_path, str) and "," in pdb_path:
        pdb_paths = [p.strip() for p in pdb_path.split(",")]
    elif isinstance(pdb_path, str):
        pdb_paths = [pdb_path]
    else:
        pdb_paths = []

    # Load structures for all paths
    atoms_xyzr_list = [_av.load_structure_with_vdw(p) for p in pdb_paths]
    atoms_xyzr = atoms_xyzr_list[0] if atoms_xyzr_list else np.zeros((0, 4))

    # Build AVs for all positions
    avs = _av.compute_avs_for_structure(
        atoms_xyzr, positions, pdb_path=pdb_path, disc_step=disc_step
    )

    # Map position names to body indices (single body by default)
    # For simplicity, all positions are on body 0 unless positions have
    # a ``body_id`` field.
    body_map: Dict[str, int] = {}
    for pname, pdef in positions.items():
        body_map[pname] = int(pdef.get("body_id", 0))

    n_bodies = max(max(body_map.values()) + 1 if body_map else 1, len(pdb_paths))

    # Build rigid bodies
    bodies = []
    for bi in range(n_bodies):
        b_atoms_xyzr = atoms_xyzr_list[bi] if bi < len(atoms_xyzr_list) else atoms_xyzr
        b_xyz = b_atoms_xyzr[:, :3]
        com = b_xyz.mean(axis=0) if b_xyz.shape[0] > 0 else np.zeros(3)
        local_coords = b_xyz - com
        local_xyzr = np.column_stack([local_coords, b_atoms_xyzr[:, 3]])
        rb = RigidBody(
            name=f"body_{bi}",
            atoms_local=local_xyzr,
            com=com.copy(),
            rotation=np.eye(3),
            translation=com.copy(),
            mass=float(b_xyz.shape[0]) if b_xyz.shape[0] > 0 else 1.0,
            inertia=np.eye(3) * 1000.0,  # approximate
        )
        bodies.append(rb)

    # Build distance restraints
    import chisurf
    fret_settings = getattr(chisurf.core.settings, "fret", {})
    tf_type = fret_settings.get("transfer_function", "Polynomial")

    restraints = []
    for dname, ddef in distances.items():
        p1_name = ddef["position1_name"]
        p2_name = ddef["position2_name"]
        if p1_name not in avs or p2_name not in avs:
            continue
        if not avs[p1_name].has_volume or not avs[p2_name].has_volume:
            continue
        b1 = body_map.get(p1_name, 0)
        b2 = body_map.get(p2_name, 0)
        offset_a = bodies[b1].rotation.T @ (avs[p1_name].mean_position - bodies[b1].com)
        offset_b = bodies[b2].rotation.T @ (avs[p2_name].mean_position - bodies[b2].com)
        
        distance_type = str(ddef.get("distance_type", "RDAMean"))
        forster_radius = float(ddef.get("Forster_radius", 52.0))
        
        # Compute AV statistics & fit transfer function
        rmp, rda_mean, rda_mean_e, sigma_r = _dist.av_pair_statistics(
            avs[p1_name], avs[p2_name], forster_radius=forster_radius
        )
        
        convfun = None
        if tf_type == "Polynomial" and distance_type != "Rmp":
            convfun = _dist.fit_transfer_polynomial(
                avs[p1_name], avs[p2_name], distance_type=distance_type, forster_radius=forster_radius
            )

        rst = DistanceRestraint(
            name=dname,
            body_a=b1,
            offset_a=offset_a,
            body_b=b2,
            offset_b=offset_b,
            distance_exp=float(ddef.get("distance", 0.0)),
            error_neg=float(ddef.get("error_neg", 5.0)),
            error_pos=float(ddef.get("error_pos", 5.0)),
            distance_type=distance_type,
            forster_radius=forster_radius,
            active=True,
            position_name_a=p1_name,
            position_name_b=p2_name,
            sigma_rda=sigma_r,
            convfun=convfun,
            transfer_function_type=tf_type,
        )
        restraints.append(rst)

    # Run trials
    all_results: List[SimulationResult] = []
    for trial in range(n_trials):
        # Re-initialize body positions
        for bi, body in enumerate(bodies):
            if bi > 0:
                # Random shake for non-primary bodies
                body.random_shake(translation_scale=5.0, rotation_scale=0.5)

        for cycle in range(max_refinement_cycles + 1):
            engine = _eng.SpringEngine(
                [b for b in bodies],  # copy references
                [r for r in restraints],
                params,
            )
            converged = engine.simulate()

            if cycle < max_refinement_cycles:
                # Recompute AVs at new positions
                for pname, av_obj in avs.items():
                    bi = body_map.get(pname, 0)
                    body_atoms_xyz = bodies[bi].global_coords()
                    body_vdw = bodies[bi].atoms_local[:, 3]
                    body_xyzr = np.column_stack([body_atoms_xyz, body_vdw])
                    pdef = positions[pname]
                    curr_pdb = pdb_paths[bi] if bi < len(pdb_paths) else (pdb_paths[0] if pdb_paths else None)
                    source_info = _av._find_attachment_point(
                        body_xyzr,
                        pdef.get("chain_identifier", ""),
                        pdef.get("residue_seq_number", 0),
                        pdef.get("atom_name", "CA"),
                        pdb_path=curr_pdb,
                    )
                    if source_info is not None:
                        # Recompute AV
                        ra = float(pdef.get("radius1", 3.5))
                        rb = float(pdef.get("radius2", 0.0))
                        rc = float(pdef.get("radius3", 0.0))
                        ds = float(disc_step or pdef.get("simulation_grid_resolution", 1.5))
                        clean_body_xyzr = _av._strip_residue_atoms(
                            body_xyzr,
                            pdef.get("chain_identifier", ""),
                            pdef.get("residue_seq_number", 0),
                            pdb_path=curr_pdb,
                        )
                        new_av = _av.compute_av(
                            clean_body_xyzr,
                            source_info,
                            linker_length=float(pdef.get("linker_length", 20.0)),
                            linker_width=float(pdef.get("linker_width", 1.0)),
                            radii=(ra, rb, rc),
                            disc_step=ds,
                            pdb_path=curr_pdb,
                            source_info=pdef,
                        )
                        new_av.position_name = pname
                        avs[pname] = new_av

                for rst in restraints:
                    ba = bodies[rst.body_a]
                    bb = bodies[rst.body_b]
                    rst.offset_a = ba.rotation.T @ (avs[rst.position_name_a].mean_position - ba.com)
                    rst.offset_b = bb.rotation.T @ (avs[rst.position_name_b].mean_position - bb.com)
                    
                    # Update AV statistics & fit transfer function
                    rmp, rda_mean, rda_mean_e, sigma_r = _dist.av_pair_statistics(
                        avs[rst.position_name_a], avs[rst.position_name_b], forster_radius=rst.forster_radius
                    )
                    rst.sigma_rda = sigma_r
                    if rst.transfer_function_type == "Polynomial" and rst.distance_type != "Rmp":
                        rst.convfun = _dist.fit_transfer_polynomial(
                            avs[rst.position_name_a], avs[rst.position_name_b],
                            distance_type=rst.distance_type, forster_radius=rst.forster_radius
                        )
                continue

        model_dists: Dict[str, float] = {}
        for rst in restraints:
            pa = rst.global_position_a(bodies)
            pb = rst.global_position_b(bodies)
            d = float(np.linalg.norm(pb - pa))
            model_dists[rst.name] = d

        sr = SimulationResult(
            converged=converged,
            iterations=engine.iteration,
            energy=engine.get_energy(),
            clash_energy=engine.get_clash_energy(),
            restraint_energy=engine.get_restraint_energy(),
            translations=[b.com.copy() for b in bodies],
            rotations=[b.rotation.copy() for b in bodies],
            model_distances=model_dists,
            force_norms=engine._force_norms,
            torque_norms=engine._torque_norms,
        )
        all_results.append(sr)

    return all_results, avs, bodies
