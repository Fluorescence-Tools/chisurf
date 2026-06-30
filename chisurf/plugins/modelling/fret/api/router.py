"""FastAPI router for FRET modeling actions.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException

from ..core import av, io, results, screening, evaluate, pair_selection


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
