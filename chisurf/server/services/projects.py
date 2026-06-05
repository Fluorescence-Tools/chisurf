from __future__ import annotations

import pathlib
from typing import Any, Dict, Optional

from chisurf.server.services import (
    ServiceResult,
    service_error,
    NOT_FOUND,
    OPERATION_FAILED,
)
from chisurf.server.session import SessionState


def get_project_info(state: SessionState) -> ServiceResult:
    """Return a summary of the current session as project info.

    Parameters
    ----------
    state : SessionState
        Server-side session state.

    """
    return {
        "ok": True,
        "project_path": None,
        "fit_count": len(state.fits),
        "dataset_count": len(state.datasets),
    }


def save_project(
    state: SessionState,
    target_path: str,
    project_name: Optional[str] = None,
) -> ServiceResult:
    """Save the current session as a project to disk.

    Parameters
    ----------
    state : SessionState
        Server-side session state.
    target_path : str
        Directory path for the project.
    project_name : str, optional
        Project display name (defaults to directory basename).

    """
    from chisurf.core.project import Project
    import chisurf

    path = pathlib.Path(target_path)
    path.mkdir(parents=True, exist_ok=True)

    project = Project(
        name=str(project_name or path.name),
        chisurf_version=chisurf.__version__,
    )

    for ds in state.datasets:
        uid = getattr(ds, "unique_identifier", None)
        if uid:
            project.datasets[uid] = _safe_to_dict(ds)

    if state.fits:
        try:
            from chisurf.macros.core_fit import _build_fitgroup_payload
            fit_datasets: Dict[str, Any] = {}
            def _register_ds(ds: Any) -> str:
                """Register a dataset in the project payload."""
                uid = str(getattr(ds, "unique_identifier", "")) or f"ds_{len(fit_datasets)}"
                fit_datasets[uid] = _safe_to_dict(ds)
                return uid
            class _Log:
                def warning(self, msg: str) -> None:
                    """Suppress warnings during project export."""
                    pass
            log = _Log()
            for i, fg in enumerate(state.fits):
                key, payload = _build_fitgroup_payload(fg, _register_ds, log, i)
                if payload:
                    project.fits.append(payload)
            project.datasets.update(fit_datasets)
        except Exception:
            pass

    try:
        out = project.save(target_path)
        return {"ok": True, "path": str(out)}
    except Exception as e:
        return service_error(str(e), error_code=OPERATION_FAILED, exception=e)


def load_project(
    state: SessionState,
    project_path: str,
) -> ServiceResult:
    """Load a project from disk and return its metadata (does not restore session).

    Parameters
    ----------
    state : SessionState
        Server-side session state.
    project_path : str
        Path to the project directory.

    """
    from chisurf.core.project import load_project as project_loader

    path = pathlib.Path(project_path)
    project_file = path / "project.json"
    if not project_file.is_file():
        return service_error(f"project.json not found at {project_path}", error_code=NOT_FOUND)

    try:
        project = project_loader(project_path)
        return {
            "ok": True,
            "project_name": project.name,
            "dataset_count": len(project.datasets),
            "fit_count": len(project.fits),
        }
    except Exception as e:
        return service_error(str(e), error_code=OPERATION_FAILED, exception=e)


def _safe_to_dict(obj: Any) -> Dict[str, Any]:
    """Convert *obj* to a dict via ``to_dict()``, falling back to name/uid.

    Parameters
    ----------
    obj : object
        Object to convert.

    """
    try:
        if hasattr(obj, "to_dict"):
            d = obj.to_dict()
            return _convert_numpy(d)
        return {"name": str(getattr(obj, "name", "")), "uid": str(getattr(obj, "unique_identifier", ""))}
    except Exception:
        return {}


def _convert_numpy(value: Any) -> Any:
    """Recursively convert numpy arrays to lists for JSON serialization."""
    import numpy as np
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: _convert_numpy(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_convert_numpy(v) for v in value]
    return value
