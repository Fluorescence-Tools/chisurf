"""JSON-safe request/result dataclasses for the FRET docking API.

These are transport-agnostic: the CLI, the in-process RPC services, and the
FastAPI router all speak in these plain dicts/dataclasses, while
:mod:`...core.imp_engine` does the IMP work.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional


@dataclass
class DockRequest:
    """Parameters for a docking run (see :class:`...core.imp_engine.DockingParameters`)."""

    pdb_paths: List[str]
    fps_json: str
    output_dir: str
    n_frames: int = 500
    mc_steps: int = 10
    mc_temperature: float = 1.0
    max_translation: float = 4.0
    max_rotation: float = 0.1
    simulated_annealing: bool = False
    n_best: int = 20
    ev_weight: float = 1.0
    shuffle_max_translation: float = 10.0
    sigma_da: float = 6.0
    score_set: str = ""
    fixed_body: int = 0
    #: "minimize" = fast IMP conjugate-gradient docking (default); "mc" = the
    #: replica-exchange Monte-Carlo sampler.
    method: str = "minimize"
    #: FPS-style refinement cycles (minimize only): re-sample AVs on the docked
    #: structure (real AV calc) and re-minimise. 0 = no refinement.
    refine_av_cycles: int = 0
    #: Export full P(R_DA) distance distributions after docking.
    save_distributions: bool = False
    #: AV backend for distributions/screening: "auto" | "labellib" | "imp-bff".
    av_backend: str = "auto"


@dataclass
class RefineRequest:
    """Parameters for conjugate-gradient refinement."""

    pdb_paths: List[str]
    fps_json: str
    output_dir: str
    steps: int = 500
    score_set: str = ""
    ev_weight: float = 1.0


@dataclass
class ScoreRequest:
    """Parameters for scoring a single structure."""

    pdb_paths: List[str]
    fps_json: str
    score_set: str = ""
    output_csv: Optional[str] = None
    mean_position_restraint: bool = False
    sigma_da: float = 6.0


@dataclass
class ScreenRequest:
    """Parameters for screening a structure library."""

    pdb_inputs: List[str]
    fps_json: str
    score_set: str = ""
    output_csv: Optional[str] = None


@dataclass
class ErrorRequest:
    """Parameters for repeated-trial error estimation (repeated docking)."""

    pdb_paths: List[str]
    fps_json: str
    output_dir: str
    n_trials: int = 10
    n_frames: int = 200
    mc_steps: int = 10
    n_best: int = 20
    fixed_body: int = 0
    sigma_da: float = 6.0
    simulated_annealing: bool = False
    score_set: str = ""
    method: str = "minimize"
    refine_av_cycles: int = 0
    ev_weight: float = 1.0
    save_distributions: bool = False
    av_backend: str = "auto"
    #: Parallel worker processes for the trials (None -> CPU count, 1 -> serial).
    n_workers: Optional[int] = None


@dataclass
class OperationResult:
    """Uniform JSON-safe result envelope returned by api.operations."""

    status: str
    operation: str
    data: Dict = field(default_factory=dict)
    message: str = ""

    def to_dict(self) -> Dict:
        return asdict(self)
