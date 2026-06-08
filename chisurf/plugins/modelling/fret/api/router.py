"""FastAPI router for FRET modeling actions.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException

from ..core import av, docking, engine, io, results, sampling, screening, refine, bootstrap, evaluate, pair_selection


router = APIRouter(prefix="/fret", tags=["fret"])


class BackendInfoResponse(BaseModel):
    """Response model for AV backends information."""
    has_labellib: bool
    has_imp_bff: bool
    active_backend: str


class InfoRequest(BaseModel):
    """Request model for retrieving info from an fps.json file."""
    fps_path: str


class InfoResponse(BaseModel):
    """Response model for fps.json summary."""
    positions_count: int
    distances_count: int
    positions: List[Dict[str, Any]]
    distances: List[Dict[str, Any]]


class DockRequest(BaseModel):
    """Request model for running FRET-restrained docking."""
    fps_path: str
    pdb_path: str
    output_dir: str
    n_trials: int = 3
    max_iterations: int = 50000
    max_force: float = 100.0
    k_clash: float = 10.0
    f_tol: float = 0.1
    t_tol: float = 0.01
    av_backend: str = "auto"


class RefineRequest(BaseModel):
    """Request model for running FRET refinement."""
    fps_path: str
    pdb_path: str
    output_dir: str
    n_cycles: int = 3
    max_iterations: int = 10000
    k_clash: float = 10.0
    av_backend: str = "auto"


class BootstrapRequest(BaseModel):
    """Request model for running parametric bootstrap error estimation."""
    fps_path: str
    pdb_path: str
    output_dir: str
    n_bootstrap: int = 100
    max_iterations: int = 10000
    k_clash: float = 10.0
    av_backend: str = "auto"


class SampleRequest(BaseModel):
    """Request model for running Metropolis MC sampling."""
    fps_path: str
    pdb_path: str
    output_dir: str
    n_samples: int = 1000
    step_size: float = 0.5
    av_backend: str = "auto"


class ScreenRequest(BaseModel):
    """Request model for screening structure library."""
    fps_path: str
    pdb_dir: str
    output_csv: str
    n_threads: int = 4
    av_backend: str = "auto"


class EvaluateRequest(BaseModel):
    """Request model for running structure evaluation."""
    fps_path: str
    input_type: str = Field(..., description="One of 'Single PDB File', 'PDB Directory', 'MDTraj Trajectory'")
    input_path: str
    traj_path: Optional[str] = None
    output_csv: str
    av_backend: str = "auto"


class PairSelectRequest(BaseModel):
    """Request model for running informative pair selection."""
    fps_path: str
    pdb_dir: str
    output_report: str
    max_pairs: int = 3
    err: float = 5.0
    av_backend: str = "auto"


@router.get("/info-backends", response_model=BackendInfoResponse)
def get_info_backends() -> BackendInfoResponse:
    """Get information about available and active AV backends.

    Returns
    -------
    BackendInfoResponse
    """
    active = "labellib" if av._LABELLIB_BACKEND else "imp-bff"
    return BackendInfoResponse(
        has_labellib=bool(av._HAS_LABELLIB),
        has_imp_bff=bool(av._HAS_IMP_BFF),
        active_backend=active
    )


@router.post("/info", response_model=InfoResponse)
def get_info(req: InfoRequest) -> InfoResponse:
    """Parse and return a summary of an fps.json file.

    Parameters
    ----------
    req : InfoRequest

    Returns
    -------
    InfoResponse
    """
    if not os.path.exists(req.fps_path):
        raise HTTPException(status_code=404, detail=f"File not found: {req.fps_path}")
    try:
        positions, distances, score_sets, extra = io.read_fps_json(req.fps_path)
        pos_list = []
        for name, d in sorted(positions.items()):
            pos_list.append({
                "name": name,
                "chain": d.get("chain_identifier", ""),
                "residue": d.get("residue_seq_number", 0),
                "atom": d.get("atom_name", "CA"),
                "linker_length": d.get("linker_length", 20.0),
                "radius": d.get("radius1", 3.5),
            })
        dist_list = []
        for name, d in sorted(distances.items()):
            dist_list.append({
                "name": name,
                "position1": d.get("position1_name"),
                "position2": d.get("position2_name"),
                "distance": d.get("distance", 0.0),
                "error_neg": d.get("error_neg", 5.0),
                "error_pos": d.get("error_pos", 5.0),
                "type": d.get("distance_type", "RDAMean")
            })
        return InfoResponse(
            positions_count=len(positions),
            distances_count=len(distances),
            positions=pos_list,
            distances=dist_list
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/dock")
def run_dock(req: DockRequest) -> Dict[str, Any]:
    """Run FRET-restrained docking.

    Parameters
    ----------
    req : DockRequest

    Returns
    -------
    dict
    """
    av.select_backend(req.av_backend)
    
    if "," in req.pdb_path:
        pdb_paths = [p.strip() for p in req.pdb_path.split(",")]
    else:
        pdb_paths = [req.pdb_path]

    for p in pdb_paths:
        if not os.path.exists(p):
            raise HTTPException(status_code=404, detail=f"PDB file not found: {p}")
    if not os.path.exists(req.fps_path):
        raise HTTPException(status_code=404, detail=f"labeling.fps.json not found: {req.fps_path}")

    try:
        positions, distances, _score_sets, _extra = io.read_fps_json(req.fps_path, pdb_paths=pdb_paths)
        params = engine.SpringParameters(
            max_iterations=req.max_iterations,
            max_force=req.max_force,
            k_clash=req.k_clash,
            F_tolerance=req.f_tol,
            T_tolerance=req.t_tol,
        )
        res_list, avs, bodies = docking.run_docking(
            pdb_paths,
            positions,
            distances,
            params=params,
            n_trials=req.n_trials,
        )
        os.makedirs(req.output_dir, exist_ok=True)
        atoms_per_body = [b.atoms_local for b in bodies]
        results.write_docking_results_pdb(res_list, atoms_per_body, req.output_dir)
        
        summary_path = os.path.join(req.output_dir, "summary.json")
        summary = [
            {
                "trial": i,
                "converged": sr.converged,
                "iterations": sr.iterations,
                "energy": sr.energy,
                "clash_energy": sr.clash_energy,
                "restraint_energy": sr.restraint_energy,
            }
            for i, sr in enumerate(res_list)
        ]
        import json
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        return {
            "status": "success",
            "trials_run": len(res_list),
            "converged_count": sum(1 for sr in res_list if sr.converged),
            "output_dir": req.output_dir
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/refine")
def run_refine(req: RefineRequest) -> Dict[str, Any]:
    """Run iterative refinement.

    Parameters
    ----------
    req : RefineRequest

    Returns
    -------
    dict
    """
    av.select_backend(req.av_backend)
    
    if "," in req.pdb_path:
        pdb_paths = [p.strip() for p in req.pdb_path.split(",")]
    else:
        pdb_paths = [req.pdb_path]

    for p in pdb_paths:
        if not os.path.exists(p):
            raise HTTPException(status_code=404, detail=f"PDB file not found: {p}")
    if not os.path.exists(req.fps_path):
        raise HTTPException(status_code=404, detail=f"labeling.fps.json not found: {req.fps_path}")

    try:
        positions, distances, _score_sets, _extra = io.read_fps_json(req.fps_path, pdb_paths=pdb_paths)
        import numpy as np
        atoms_xyzr_list = [av.load_structure_with_vdw(p) for p in pdb_paths]
        atoms_xyzr = atoms_xyzr_list[0] if atoms_xyzr_list else np.zeros((0, 4))
        avs = av.compute_avs_for_structure(atoms_xyzr, positions, pdb_path=pdb_paths)

        body_map = {pname: int(pdef.get("body_id", 0)) for pname, pdef in positions.items()}
        n_bodies = max(max(body_map.values()) + 1 if body_map else 1, len(pdb_paths))

        bodies = []
        for bi in range(n_bodies):
            b_atoms_xyzr = atoms_xyzr_list[bi] if bi < len(atoms_xyzr_list) else atoms_xyzr
            b_xyz = b_atoms_xyzr[:, :3]
            com = b_xyz.mean(axis=0) if b_xyz.shape[0] > 0 else np.zeros(3)
            local_coords = b_xyz - com
            local_xyzr = np.column_stack([local_coords, b_atoms_xyzr[:, 3]])
            rb = engine.RigidBody(
                name=f"body_{bi}",
                atoms_local=local_xyzr,
                com=com.copy(),
                rotation=np.eye(3),
                translation=com.copy(),
                mass=float(b_xyz.shape[0]) if b_xyz.shape[0] > 0 else 1.0,
                inertia=np.eye(3) * 1000.0,
            )
            bodies.append(rb)

        restraints = []
        for dname, ddef in distances.items():
            p1 = ddef["position1_name"]
            p2 = ddef["position2_name"]
            if p1 not in avs or p2 not in avs:
                continue
            if not avs[p1].has_volume or not avs[p2].has_volume:
                continue
            b1 = body_map.get(p1, 0)
            b2 = body_map.get(p2, 0)
            offset_a = avs[p1].mean_position - bodies[b1].com
            offset_b = avs[p2].mean_position - bodies[b2].com
            rst = engine.DistanceRestraint(
                name=dname,
                body_a=b1,
                offset_a=offset_a,
                body_b=b2,
                offset_b=offset_b,
                distance_exp=float(ddef.get("distance", 0.0)),
                error_neg=float(ddef.get("error_neg", 5.0)),
                error_pos=float(ddef.get("error_pos", 5.0)),
                distance_type=str(ddef.get("distance_type", "RDAMean")),
                forster_radius=float(ddef.get("Forster_radius", 52.0)),
                active=True,
            )
            restraints.append(rst)

        params = engine.SpringParameters(
            max_iterations=req.max_iterations,
            k_clash=req.k_clash,
        )

        final_bodies = refine.run_refinement(
            bodies,
            restraints,
            positions,
            atoms_xyzr,
            params=params,
            n_cycles=req.n_cycles,
        )

        os.makedirs(req.output_dir, exist_ok=True)
        # Save refined body coordinates as PDB
        for bi, body in enumerate(final_bodies):
            out_pdb = os.path.join(req.output_dir, f"refined_body_{bi}.pdb")
            io.write_pdb(body.global_coords(), out_pdb)

        return {
            "status": "success",
            "bodies_refined": len(final_bodies),
            "output_dir": req.output_dir
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/bootstrap")
def run_boot(req: BootstrapRequest) -> Dict[str, Any]:
    """Run parametric bootstrap error estimation.

    Parameters
    ----------
    req : BootstrapRequest

    Returns
    -------
    dict
    """
    av.select_backend(req.av_backend)
    if not os.path.exists(req.pdb_path):
        raise HTTPException(status_code=404, detail=f"PDB file not found: {req.pdb_path}")
    if not os.path.exists(req.fps_path):
        raise HTTPException(status_code=404, detail=f"labeling.fps.json not found: {req.fps_path}")

    try:
        positions, distances, _score_sets, _extra = io.read_fps_json(req.fps_path)
        params = engine.SpringParameters(
            max_iterations=req.max_iterations,
            k_clash=req.k_clash,
        )
        boot_res = bootstrap.run_bootstrap(
            req.pdb_path,
            positions,
            distances,
            params=params,
            n_bootstrap=req.n_bootstrap,
        )

        os.makedirs(req.output_dir, exist_ok=True)
        results.write_bootstrap_results(boot_res, req.output_dir)

        return {
            "status": "success",
            "n_bootstrap": req.n_bootstrap,
            "output_dir": req.output_dir
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sample")
def run_sample(req: SampleRequest) -> Dict[str, Any]:
    """Run Metropolis Monte Carlo sampling.

    Parameters
    ----------
    req : SampleRequest

    Returns
    -------
    dict
    """
    av.select_backend(req.av_backend)
    if not os.path.exists(req.pdb_path):
        raise HTTPException(status_code=404, detail=f"PDB file not found: {req.pdb_path}")
    if not os.path.exists(req.fps_path):
        raise HTTPException(status_code=404, detail=f"labeling.fps.json not found: {req.fps_path}")

    try:
        positions, distances, _score_sets, _extra = io.read_fps_json(req.fps_path)
        params = engine.SpringParameters()
        
        # Build bodies
        atoms_xyzr = av.load_structure_with_vdw(req.pdb_path)
        com = atoms_xyzr[:, :3].mean(axis=0)
        local_xyzr = atoms_xyzr.copy()
        local_xyzr[:, :3] -= com
        body = engine.RigidBody(
            name="body_0",
            atoms_local=local_xyzr,
            com=com.copy(),
            rotation=np.eye(3),
            translation=com.copy(),
            mass=float(atoms_xyzr.shape[0]) if atoms_xyzr.shape[0] > 0 else 1.0,
            inertia=np.eye(3) * 1000.0,
        )
        
        avs = av.compute_avs_for_structure(atoms_xyzr, positions, pdb_path=req.pdb_path)
        restraints = []
        for dname, ddef in distances.items():
            p1 = ddef["position1_name"]
            p2 = ddef["position2_name"]
            if p1 not in avs or p2 not in avs:
                continue
            offset_a = avs[p1].mean_position - body.com
            offset_b = avs[p2].mean_position - body.com
            rst = engine.DistanceRestraint(
                name=dname,
                body_a=0,
                offset_a=offset_a,
                body_b=0,
                offset_b=offset_b,
                distance_exp=float(ddef.get("distance", 0.0)),
                error_neg=float(ddef.get("error_neg", 5.0)),
                error_pos=float(ddef.get("error_pos", 5.0)),
                distance_type=str(ddef.get("distance_type", "RDAMean")),
                forster_radius=float(ddef.get("Forster_radius", 52.0)),
                active=True,
            )
            restraints.append(rst)

        samples = sampling.run_metropolis_sampling(
            [body],
            restraints,
            n_samples=req.n_samples,
            step_size=req.step_size,
        )

        os.makedirs(req.output_dir, exist_ok=True)
        # Write coordinates to PDB file
        out_pdb = os.path.join(req.output_dir, "sampled_trajectory.pdb")
        # Extract translations from samples
        coords_list = []
        for s in samples:
            # Reconstruct global coords of body
            b_global = s.rotations[0] @ body.atoms_local[:, :3].T + s.translations[0][:, np.newaxis]
            coords_list.append(b_global.T)
        
        # Write as multi-model PDB
        with open(out_pdb, "w") as f:
            for idx, c in enumerate(coords_list):
                f.write(f"MODEL     {idx + 1:4d}\n")
                for atom_idx, xyz in enumerate(c):
                    f.write(f"ATOM  {atom_idx+1:5d}  CA  ALA A   1    {xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00  0.00\n")
                f.write("ENDMDL\n")

        return {
            "status": "success",
            "samples_generated": len(samples),
            "output_file": out_pdb
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/screen")
def run_screen(req: ScreenRequest) -> Dict[str, Any]:
    """Screen a structure library directory.

    Parameters
    ----------
    req : ScreenRequest

    Returns
    -------
    dict
    """
    av.select_backend(req.av_backend)
    if not os.path.exists(req.pdb_dir):
        raise HTTPException(status_code=404, detail=f"PDB directory not found: {req.pdb_dir}")
    if not os.path.exists(req.fps_path):
        raise HTTPException(status_code=404, detail=f"labeling.fps.json not found: {req.fps_path}")

    try:
        positions, distances, _score_sets, _extra = io.read_fps_json(req.fps_path)
        scr_results = screening.screen_structure_library(
            req.pdb_dir,
            positions,
            distances,
            n_threads=req.n_threads,
        )
        
        # Save screening CSV
        results.write_screening_results_csv(scr_results, req.output_csv)

        return {
            "status": "success",
            "structures_screened": len(scr_results),
            "output_csv": req.output_csv
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/evaluate")
def run_eval(req: EvaluateRequest) -> Dict[str, Any]:
    """Run OLGA-style structure evaluations.

    Parameters
    ----------
    req : EvaluateRequest

    Returns
    -------
    dict
    """
    av.select_backend(req.av_backend)
    if not os.path.exists(req.input_path):
        raise HTTPException(status_code=404, detail=f"Input path not found: {req.input_path}")
    if not os.path.exists(req.fps_path):
        raise HTTPException(status_code=404, detail=f"labeling.fps.json not found: {req.fps_path}")

    try:
        positions, distances, _, _ = io.read_fps_json(req.fps_path)
        evaluators = io.read_evaluators_json(req.fps_path)
        if not evaluators:
            # Fallback: construct DistanceEvaluators from distances
            from ..evaluators import DistanceEvaluator
            evaluators = [
                DistanceEvaluator(name, d["position1_name"], d["position2_name"], distance_type=d.get("distance_type", "RDAMean"))
                for name, d in distances.items()
            ]

        if req.input_type == "Single PDB File":
            res = evaluate.evaluate_structure(req.input_path, positions, evaluators)
            storage = evaluate.EvaluationStorage()
            storage.add_frame(os.path.basename(req.input_path), res)
            storage.to_csv(req.output_csv)
            count = 1
        elif req.input_type == "PDB Directory":
            storage = evaluate.evaluate_directory(req.input_path, positions, evaluators)
            storage.to_csv(req.output_csv)
            count = len(storage.frames)
        elif req.input_type == "MDTraj Trajectory":
            if not req.traj_path:
                raise HTTPException(status_code=400, detail="traj_path is required for MDTraj Trajectory evaluation.")
            if not os.path.exists(req.traj_path):
                raise HTTPException(status_code=404, detail=f"Trajectory file not found: {req.traj_path}")
            storage = evaluate.evaluate_trajectory(req.input_path, req.traj_path, positions, evaluators)
            storage.to_csv(req.output_csv)
            count = len(storage.frames)
        else:
            raise HTTPException(status_code=400, detail=f"Invalid input_type: {req.input_type}")

        return {
            "status": "success",
            "evaluations_run": count,
            "output_csv": req.output_csv
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/select-pairs")
def run_pair_select(req: PairSelectRequest) -> Dict[str, Any]:
    """Run informative pair selection.

    Parameters
    ----------
    req : PairSelectRequest

    Returns
    -------
    dict
    """
    av.select_backend(req.av_backend)
    if not os.path.exists(req.pdb_dir):
        raise HTTPException(status_code=404, detail=f"PDB directory not found: {req.pdb_dir}")
    if not os.path.exists(req.fps_path):
        raise HTTPException(status_code=404, detail=f"labeling.fps.json not found: {req.fps_path}")

    try:
        positions, distances, _, _ = io.read_fps_json(req.fps_path)
        
        # 1. Compute RMSD matrix
        rmsds, filenames = pair_selection.compute_rmsd_matrix_from_pdb_dir(req.pdb_dir)

        # 2. Compute FRET efficiencies
        effs, pair_names = pair_selection.compute_efficiency_matrix_from_evaluators(
            req.pdb_dir, positions, distances
        )

        # 3. Pre-process NaN values
        effs_clean, rmsds_clean, valid_indices = pair_selection.preprocess_efficiency_matrix(
            effs, rmsds
        )
        clean_pair_names = [pair_names[i] for i in valid_indices]

        # 4. Run greedy selection
        selected_indices, decay = pair_selection.select_informative_pairs(
            effs_clean, rmsds_clean, err=req.err, max_pairs=req.max_pairs
        )
        selected_pair_names = [clean_pair_names[i] for i in selected_indices]

        # 5. Write decay report
        pair_selection.write_pair_selection_report(
            selected_pair_names, decay, req.output_report, rmsds.mean()
        )

        return {
            "status": "success",
            "selected_pairs": selected_pair_names,
            "output_report": req.output_report
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


from fastapi import FastAPI
app = FastAPI(title="FRET Modeling API")
app.include_router(router)
