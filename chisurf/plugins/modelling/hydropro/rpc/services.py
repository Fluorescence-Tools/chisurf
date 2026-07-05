"""ServiceDispatcher-compatible RPC handlers for HydroPro (Qt-free)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from ..core import HydroProSettings, run_hydro
from ..core.runner import parse_diffusion_coefficient

METHOD_RUN = "hydropro.run"
METHOD_PARSE_RES = "hydropro.parse_res"


def run_handler(
    structures: list,
    exe_path: str,
    settings: Dict[str, Any] | None = None,
    work_dir: str | None = None,
) -> Dict[str, Any]:
    """Run HYDRO over ``structures`` and return per-file diffusion coefficients."""
    s = HydroProSettings.from_dict(settings or {})
    s.validate()
    results = run_hydro(
        [Path(p) for p in structures], s, Path(exe_path),
        Path(work_dir) if work_dir else None,
    )
    return {
        "results": [
            {"file": r.struct_file, "diffusion_coefficient": r.diffusion_coefficient}
            for r in results
        ]
    }


def parse_res_handler(res_path: str) -> Dict[str, Any]:
    """Parse a HYDRO ``*.res`` report for the translational diffusion coefficient."""
    return {"diffusion_coefficient": parse_diffusion_coefficient(Path(res_path))}


def register_services(dispatcher: Any) -> None:
    """Register HydroPro RPC handlers with a ServiceDispatcher."""
    dispatcher.register(METHOD_RUN, lambda params: run_handler(**params))
    dispatcher.register(METHOD_PARSE_RES, lambda params: parse_res_handler(**params))


__all__ = ["register_services", "run_handler", "parse_res_handler",
           "METHOD_RUN", "METHOD_PARSE_RES"]
