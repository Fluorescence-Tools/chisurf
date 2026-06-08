"""Iterative AV-recompute + re-docking refinement."""

from __future__ import annotations

import os
import tempfile
from typing import Dict, List, Optional

import numpy as np

from . import av as _av
from . import io as _io
from . import engine as _eng
from .engine import DistanceRestraint, RigidBody, SpringParameters


def run_refinement(
    bodies: List[RigidBody],
    restraints: List[DistanceRestraint],
    positions: Dict,
    atoms_xyzr: np.ndarray,
    params: SpringParameters,
    n_cycles: int = 3,
    disc_step: Optional[float] = None,
) -> List[RigidBody]:
    """Iterative AV-recompute + re-docking refinement.

    Port of FPS Refinement.cs. At each cycle, AVs are recomputed on the
    current body coordinates, restraint offsets are updated, and the
    spring engine is run again.

    Parameters
    ----------
    bodies : list of RigidBody
        Body states to refine (modified in place).
    restraints : list of DistanceRestraint
        Active restraints (offsets updated each cycle).
    positions : dict
        fps.json Positions section.
    atoms_xyzr : (N, 4) ndarray
        System-wide atom xyzr array.
    params : SpringParameters
        Engine parameters for each docking cycle.
    n_cycles : int
        Number of refinement cycles.
    disc_step : float, optional
        AV grid step override.

    Returns
    -------
    bodies : list of RigidBody
        Refined body states.
    """
    body_map: Dict[str, int] = {}
    for pname, pdef in positions.items():
        body_map[pname] = int(pdef.get("body_id", 0))

    for cycle in range(n_cycles):
        avs = {}
        temp_files: List[str] = []
        try:
            for pname, pdef in positions.items():
                bi = body_map.get(pname, 0)
                if bi >= len(bodies):
                    continue
                body = bodies[bi]
                body_xyzr = body.global_xyzr()

                ds = float(disc_step or pdef.get("simulation_grid_resolution", 1.5))
                source_xyz = _av._find_attachment_point(
                    body_xyzr,
                    pdef.get("chain_identifier", ""),
                    pdef.get("residue_seq_number", 0),
                    pdef.get("atom_name", "CA"),
                )

                pdb_path = None
                if not _av._LABELLIB_BACKEND:
                    # Write current coordinates to a temp file for IMP.bff
                    with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False, mode="w") as tmp_f:
                        pdb_path = tmp_f.name
                    _io.write_pdb(body_xyzr[:, :3], pdb_path)
                    temp_files.append(pdb_path)

                av_obj = _av.compute_av(
                    atoms=body_xyzr,
                    source_xyz=source_xyz,
                    linker_length=float(pdef.get("linker_length", 20.0)),
                    linker_width=float(pdef.get("linker_width", 1.0)),
                    radii=(
                        float(pdef.get("radius1", 3.5)),
                        float(pdef.get("radius2", 0.0)),
                        float(pdef.get("radius3", 0.0)),
                    ),
                    disc_step=ds,
                    pdb_path=pdb_path,
                    source_info=pdef,
                )
                av_obj.position_name = pname
                av_obj.params = pdef
                avs[pname] = av_obj

            for rst in restraints:
                parts = rst.name.split("_")
                if len(parts) >= 2:
                    pos_a, pos_b = parts[0], parts[1]
                    if pos_a in avs and pos_b in avs:
                        rst.offset_a = avs[pos_a].mean_position - bodies[rst.body_a].com
                        rst.offset_b = avs[pos_b].mean_position - bodies[rst.body_b].com

        finally:
            for fpath in temp_files:
                try:
                    os.unlink(fpath)
                except Exception:
                    pass

        # Simulate
        engine = _eng.SpringEngine(bodies, restraints, params)
        engine.simulate()

    return bodies
