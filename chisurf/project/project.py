from __future__ import annotations

import datetime
import json
import pathlib
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union

PathLike = Union[str, pathlib.Path]


@dataclass
class Project:
    """Minimal, GUI-independent representation of a ChiSurf project.

    This class is designed to be extended incrementally. For now it only
    captures a small, generic subset of possible project state and
    provides JSON-based save/load to a project *directory*.
    """

    name: str = "untitled"
    description: str = ""
    chisurf_version: Optional[str] = None
    # Schema / on-disk format version. Version 2 stores per-fit folders.
    project_format_version: int = 2
    # Creation timestamp (ISO 8601). Mainly for user information.
    created: str = field(default_factory=lambda: datetime.datetime.now().isoformat())

    # Generic containers for logical state. These will later be replaced or
    # complemented by more structured experiment/model-specific state.
    datasets: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    experiments: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    fits: list[Dict[str, Any]] = field(default_factory=list)
    ui_state: Dict[str, Any] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert this project into a JSON-serializable dictionary.

        Only uses basic Python types (dict, list, str, int, float, bool,
        None) so it is safe to store as JSON. Any non-serializable values
        should be converted by callers before placing them into the
        `datasets` / `experiments` / `fits` / `ui_state` / `extra`
        structures.
        """

        return {
            "project_format_version": self.project_format_version,
            "name": self.name,
            "description": self.description,
            "chisurf_version": self.chisurf_version,
            "created": self.created,
            "datasets": self.datasets,
            "experiments": self.experiments,
            "fits": self.fits,
            "ui_state": self.ui_state,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Project":
        """Reconstruct a :class:`Project` from a dictionary.

        Missing fields are filled with sensible defaults so that we can
        evolve the schema without breaking older project files.
        """

        return cls(
            name=data.get("name", "untitled"),
            description=data.get("description", ""),
            chisurf_version=data.get("chisurf_version"),
            project_format_version=int(data.get("project_format_version", 1)),
            created=data.get("created") or datetime.datetime.now().isoformat(),
            datasets=data.get("datasets") or {},
            experiments=data.get("experiments") or {},
            fits=data.get("fits") or [],
            ui_state=data.get("ui_state") or {},
            extra=data.get("extra") or {},
        )

    def save(self, project_dir: PathLike) -> pathlib.Path:
        """Save this project into a directory as ``project.json``.

        Parameters
        ----------
        project_dir:
            Directory where the project should be stored. It will be
            created if it does not exist.

        Returns
        -------
        pathlib.Path
            The full path to the written ``project.json`` file.
        """

        path = pathlib.Path(project_dir)
        path.mkdir(parents=True, exist_ok=True)
        project_file = path / "project.json"

        with project_file.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, sort_keys=True)

        return project_file

    @classmethod
    def load(cls, project_dir: PathLike) -> "Project":
        """Load a project from a directory containing ``project.json``.

        Parameters
        ----------
        project_dir:
            Directory that holds the ``project.json`` file.

        Raises
        ------
        FileNotFoundError
            If the expected ``project.json`` file is missing.
        """

        path = pathlib.Path(project_dir)
        project_file = path / "project.json"

        if not project_file.is_file():
            raise FileNotFoundError(f"Project JSON not found: {project_file}")

        with project_file.open("r", encoding="utf-8") as f:
            data = json.load(f)

        return cls.from_dict(data)


def save_project(project: Project, target_path: PathLike) -> pathlib.Path:
    """Convenience wrapper to save a :class:`Project`.

    ``target_path`` is treated as a *directory*; it will be created if it
    does not already exist. The function returns the path to the
    resulting ``project.json`` file.
    """

    return project.save(target_path)


def load_project(target_path: PathLike) -> Project:
    """Convenience wrapper to load a :class:`Project` from a directory."""

    return Project.load(target_path)
