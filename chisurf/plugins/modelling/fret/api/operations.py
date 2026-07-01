"""Transport-agnostic FRET-docking operations.

This is the single seam the CLI, in-process RPC services, and FastAPI router
call. Each function accepts plain JSON-compatible parameters (or the
dataclasses in :mod:`.models`) and returns a JSON-safe dict, delegating the
IMP work to :mod:`...core.imp_engine`.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Union

from ..core import imp_engine
from .models import (
    DockRequest,
    ErrorRequest,
    OperationResult,
    RefineRequest,
    ScoreRequest,
    ScreenRequest,
)
from .project import load_docking_project


def backend_info() -> Dict:
    """Return availability of the IMP/IMP.bff backend."""
    return OperationResult(
        status="ok",
        operation="backend_info",
        data={"has_imp_bff": imp_engine.has_imp()},
    ).to_dict()


def _as(req_cls, req: Union[dict, object]):
    """Coerce a dict into the given request dataclass (pass dataclasses through)."""
    if isinstance(req, dict):
        return req_cls(**req)
    return req


def dock(req: Union[DockRequest, dict], stop_check=None) -> Dict:
    """Run FRET-restrained rigid-body docking (minimisation or Monte-Carlo).

    ``stop_check`` is an optional zero-arg callable polled during minimisation;
    when it returns True the run aborts and the partially-docked pose is returned.
    """
    r = _as(DockRequest, req)
    params = imp_engine.DockingParameters(
        n_frames=r.n_frames,
        mc_steps=r.mc_steps,
        mc_temperature=r.mc_temperature,
        max_translation=r.max_translation,
        max_rotation=r.max_rotation,
        simulated_annealing=r.simulated_annealing,
        n_best=r.n_best,
        ev_weight=r.ev_weight,
        shuffle_max_translation=r.shuffle_max_translation,
        sigma_da=r.sigma_da,
        score_set=r.score_set,
        fixed_body=r.fixed_body,
        refine_av_cycles=r.refine_av_cycles,
        save_distributions=r.save_distributions,
        av_backend=r.av_backend,
    )
    if r.method == "minimize":
        # A single minimisation is a deterministic, local refinement of the input
        # pose — don't shuffle it into a random (often worse) basin. Independent
        # random starts belong to repeated docking (estimate_errors).
        params.shuffle_max_translation = 0.0
        res = imp_engine.dock_minimize(
            r.pdb_paths, r.fps_json, r.output_dir, params, stop_check=stop_check)
    else:
        res = imp_engine.dock(r.pdb_paths, r.fps_json, r.output_dir, params)
    return OperationResult(status="ok", operation="dock", data=res.to_dict()).to_dict()


def dock_project(project_path: str, overrides: Optional[dict] = None) -> Dict:
    """Load a docking project file and run its ``dock`` operation.

    Parameters
    ----------
    project_path : str
        Path to a ``*.json`` written by :func:`...api.project.save_docking_project`.
    overrides : dict, optional
        Request keys (e.g. ``output_dir``, ``n_frames``) that take precedence
        over the project's stored values — handy for a quick CLI smoke run.

    Returns
    -------
    dict
        The same envelope as :func:`dock`.
    """
    proj = load_docking_project(project_path)
    req = proj.to_dock_request()
    if overrides:
        req.update({k: v for k, v in overrides.items() if v is not None})
    return dock(req)


def refine(req: Union[RefineRequest, dict]) -> Dict:
    """Locally refine a pose with conjugate gradients."""
    r = _as(RefineRequest, req)
    res = imp_engine.refine(
        r.pdb_paths, r.fps_json, r.output_dir,
        score_set=r.score_set, steps=r.steps, ev_weight=r.ev_weight,
    )
    return OperationResult(status="ok", operation="refine", data=res.to_dict()).to_dict()


def score(req: Union[ScoreRequest, dict]) -> Dict:
    """Score a single structure against the FRET restraints."""
    r = _as(ScoreRequest, req)
    res = imp_engine.score(
        r.pdb_paths, r.fps_json,
        score_set=r.score_set, output_csv=r.output_csv,
        mean_position_restraint=r.mean_position_restraint,
        sigma_da=r.sigma_da,
    )
    return OperationResult(status="ok", operation="score", data=res.to_dict()).to_dict()


def screen(req: Union[ScreenRequest, dict]) -> Dict:
    """Score and rank a structure library."""
    r = _as(ScreenRequest, req)
    ranked = imp_engine.screen(
        r.pdb_inputs, r.fps_json, score_set=r.score_set, output_csv=r.output_csv,
    )
    return OperationResult(
        status="ok", operation="screen",
        data={"ranked": [{"pdb": p, "score": s} for p, s in ranked],
              "output_csv": r.output_csv},
    ).to_dict()


def estimate_errors(req: Union[ErrorRequest, dict], stop_check=None) -> Dict:
    """Repeat docking from random starts and report the score spread."""
    r = _as(ErrorRequest, req)
    params = imp_engine.DockingParameters(
        n_frames=r.n_frames, mc_steps=r.mc_steps, n_best=r.n_best,
        fixed_body=r.fixed_body, sigma_da=r.sigma_da,
        simulated_annealing=r.simulated_annealing, score_set=r.score_set,
        refine_av_cycles=r.refine_av_cycles, ev_weight=r.ev_weight,
        save_distributions=r.save_distributions, av_backend=r.av_backend,
    )
    data = imp_engine.estimate_errors(
        r.pdb_paths, r.fps_json, r.output_dir, n_trials=r.n_trials, params=params,
        method=r.method, n_workers=r.n_workers, stop_check=stop_check,
    )
    return OperationResult(status="ok", operation="estimate_errors", data=data).to_dict()


__all__ = [
    "backend_info", "dock", "dock_project", "refine", "score", "screen",
    "estimate_errors",
]
