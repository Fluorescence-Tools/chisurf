from __future__ import annotations

import json
import pathlib
from typing import Any, Dict, Optional

import chisurf as cs

from chisurf.core.project import Project
from chisurf.core.project.archive import PROJECT_ARCHIVE_SUFFIX, PROJECT_JSON, ProjectArchive
from chisurf.server.services import (
    NOT_FOUND,
    OPERATION_FAILED,
    ServiceResult,
    service_error,
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
    """Save the current session as a ``.csp`` project archive.

    Parameters
    ----------
    state : SessionState
        Server-side session state.
    target_path : str
        Destination ``.csp`` path, or a directory for ``project.csp``.
    project_name : str, optional
        Project display name (defaults to destination stem).

    """
    try:
        project_path, name = _project_archive_path(target_path, project_name)
        project = Project(
            name=name,
            chisurf_version=cs.__version__,
        )

        for ds in state.datasets:
            uid = getattr(ds, "unique_identifier", None)
            if uid:
                project.datasets[str(uid)] = _safe_to_dict(ds)

        if state.fits:
            try:
                from chisurf.macros.core_fit import _build_fitgroup_payload

                fit_datasets: Dict[str, Any] = {}

                def _register_ds(ds: Any) -> str:
                    uid = str(getattr(ds, "unique_identifier", "")) or f"ds_{len(fit_datasets)}"
                    fit_datasets[uid] = _safe_to_dict(ds)
                    return uid

                class _Log:
                    def warning(self, msg: str) -> None:
                        """Suppress warnings during project export."""

                log = _Log()
                for i, fg in enumerate(state.fits):
                    key, payload = _build_fitgroup_payload(fg, _register_ds, log, i)
                    if payload:
                        project.fits.append(payload)
                project.datasets.update(fit_datasets)
            except Exception:
                pass

        archive = ProjectArchive()
        archive.write_text(PROJECT_JSON, json.dumps(project.to_dict(), indent=2, sort_keys=True))
        archive.save(project_path)
        return {"ok": True, "path": str(project_path)}
    except Exception as e:
        return service_error(str(e), error_code=OPERATION_FAILED, exception=e)


def load_project(
    state: SessionState,
    project_path: str,
) -> ServiceResult:
    """Load a ``.csp`` project from disk and return metadata.

    Parameters
    ----------
    state : SessionState
        Server-side session state.
    project_path : str
        Path to the project archive.

    """
    try:
        archive_path = _project_archive_input_path(project_path)
        if not archive_path.is_file():
            return service_error(
                f"project archive not found at {archive_path}", error_code=NOT_FOUND
            )
        archive = ProjectArchive.open(archive_path)
        project = Project.from_dict(json.loads(archive.read_text(PROJECT_JSON)))
        archive.close()
        return {
            "ok": True,
            "project_name": project.name,
            "dataset_count": len(project.datasets),
            "fit_count": len(project.fits),
        }
    except FileNotFoundError:
        return service_error(f"project archive not found at {project_path}", error_code=NOT_FOUND)
    except Exception as e:
        return service_error(str(e), error_code=OPERATION_FAILED, exception=e)


def _project_archive_path(target_path: str, project_name: Optional[str]) -> tuple[pathlib.Path, str]:
    path = pathlib.Path(target_path)
    if path.suffix.lower() == PROJECT_ARCHIVE_SUFFIX:
        return path, path.stem or (project_name or "chisurf_project")
    name = project_name or path.name or "chisurf_project"
    return path / f"{name}{PROJECT_ARCHIVE_SUFFIX}", name


def _project_archive_input_path(project_path: str) -> pathlib.Path:
    path = pathlib.Path(project_path)
    if path.suffix.lower() == PROJECT_ARCHIVE_SUFFIX:
        return path
    if path.is_dir():
        return path / f"project{PROJECT_ARCHIVE_SUFFIX}"
    return pathlib.Path(f"{path}{PROJECT_ARCHIVE_SUFFIX}")


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
