from __future__ import annotations

import datetime
import json
import pathlib
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from .archive import PROJECT_JSON, SESSION_FILENAME, PROJECT_ARCHIVE_SUFFIX, ProjectArchive

PathLike = Union[str, pathlib.Path]


@dataclass
class Project:
    """Minimal, GUI-independent representation of a ChiSurf project.

    This class provides JSON-based save/load to a single ``.csp`` project
    archive. The archive stores ``project.json`` plus optional supporting files
    such as history, chinet session data, and embedded external data files.
    """

    name: str = "untitled"
    description: str = ""
    chisurf_version: Optional[str] = None
    project_format_version: int = 4
    # Creation timestamp (ISO 8601). Mainly for user information.
    created: str = field(default_factory=lambda: datetime.datetime.now().isoformat())

    # Core state sections keyed by UID or strict lists:
    datasets: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    experiments: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    fits: List[Dict[str, Any]] = field(default_factory=list)
    ui_state: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert this project into a deterministic JSON-serializable dictionary."""
        sorted_datasets = {k: self.datasets[k] for k in sorted(self.datasets.keys())}
        sorted_experiments = {k: self.experiments[k] for k in sorted(self.experiments.keys())}

        return {
            "project_format_version": self.project_format_version,
            "meta": {
                "name": self.name,
                "description": self.description,
                "chisurf_version": self.chisurf_version,
                "created": self.created,
                **self.metadata,
            },
            "datasets": sorted_datasets,
            "experiments": sorted_experiments,
            "fits": self.fits,
            "ui": self.ui_state,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Project":
        """Reconstruct a :class:`Project` from a dictionary. V4 format only."""
        version = int(data.get("project_format_version", 1))
        if version < 4:
            raise ValueError(
                f"Unsupported project format version: {version}. "
                "This build requires v4 (UID-keyed). v3 and earlier projects are not supported."
            )

        meta = data.get("meta", {})

        # Extract explicit metadata keys not part of the root
        core_meta_keys = {"name", "description", "chisurf_version", "created"}
        metadata = {k: v for k, v in meta.items() if k not in core_meta_keys}

        return cls(
            name=meta.get("name", "untitled"),
            description=meta.get("description", ""),
            chisurf_version=meta.get("chisurf_version"),
            project_format_version=4,
            created=meta.get("created") or datetime.datetime.now().isoformat(),
            datasets=data.get("datasets") or {},
            experiments=data.get("experiments") or {},
            fits=data.get("fits") or [],
            ui_state=data.get("ui") or {},
            metadata=metadata,
            extra=data.get("extra") or {},
        )

    def get_dataset(self, uid: str) -> Optional[Dict[str, Any]]:
        """Return a dataset payload by UID."""
        return self.datasets.get(uid)

    def get_fit(self, uid: str) -> Optional[Dict[str, Any]]:
        """Return a fit payload by UID."""
        for fit in self.fits:
            if fit.get("uid") == uid:
                return fit
        return None

    def list_dataset_uids(self) -> List[str]:
        """Return sorted dataset UIDs."""
        return sorted(list(self.datasets.keys()))

    def list_fit_uids(self) -> List[str]:
        """Return sorted fit UIDs."""
        uids = [fit.get("uid") for fit in self.fits if fit.get("uid")]
        return sorted(uids)

    def save_to_archive(self, archive: ProjectArchive) -> None:
        """Write ``project.json`` to an existing project archive.

        Parameters
        ----------
        archive : ProjectArchive
            Archive to write into.
        """
        archive.write_text(PROJECT_JSON, json.dumps(self.to_dict(), indent=2, sort_keys=True))

    def save(self, target_path: PathLike) -> pathlib.Path:
        """Save this project as a ``.csp`` archive.

        Parameters
        ----------
        target_path : str or pathlib.Path
            Destination archive path. If a directory is provided, the project is
            saved as ``project.csp`` inside that directory.

        Returns
        -------
        pathlib.Path
            Path to the saved ``.csp`` archive.
        """
        archive_path = _archive_output_path(target_path)
        archive = ProjectArchive()
        self.save_to_archive(archive)

        try:
            import chinet

            with tempfile.TemporaryDirectory() as tmpdir:
                session_path = pathlib.Path(tmpdir) / SESSION_FILENAME
                chinet.session.save(str(session_path))
                archive.write_bytes(SESSION_FILENAME, session_path.read_bytes())
        except (ImportError, AttributeError):
            pass

        return archive.save(archive_path)

    @classmethod
    def load(cls, target_path: PathLike) -> "Project":
        """Load a project from a ``.csp`` archive.

        Parameters
        ----------
        target_path : str or pathlib.Path
            Archive path, or a directory containing ``project.csp``.

        Returns
        -------
        Project
            Loaded project instance.
        """
        archive_path = _archive_input_path(target_path)
        archive = ProjectArchive.open(archive_path)
        data = json.loads(archive.read_text(PROJECT_JSON))
        project = cls.from_dict(data)
        project._archive = archive
        project._archive_path = archive_path

        try:
            import chinet

            with tempfile.TemporaryDirectory() as tmpdir:
                session_path = archive.extract_entry_to(SESSION_FILENAME, tmpdir)
                chinet.session.load(str(session_path))
        except (ImportError, AttributeError, KeyError):
            pass

        return project


def save_project(project: Project, target_path: PathLike) -> pathlib.Path:
    """Convenience wrapper to save a :class:`Project`."""
    return project.save(target_path)


def load_project(target_path: PathLike) -> Project:
    """Convenience wrapper to load a :class:`Project` from a ``.csp`` archive."""
    return Project.load(target_path)


def _archive_output_path(target_path: PathLike) -> pathlib.Path:
    path = pathlib.Path(target_path)
    if path.suffix.lower() == PROJECT_ARCHIVE_SUFFIX:
        return path
    if path.exists() and path.is_dir():
        return path / f"project{PROJECT_ARCHIVE_SUFFIX}"
    if path.suffix:
        return path.with_suffix(PROJECT_ARCHIVE_SUFFIX)
    return pathlib.Path(f"{path}{PROJECT_ARCHIVE_SUFFIX}")


def _archive_input_path(target_path: PathLike) -> pathlib.Path:
    path = pathlib.Path(target_path)
    if path.suffix.lower() == PROJECT_ARCHIVE_SUFFIX:
        return path
    if path.is_dir():
        return path / f"project{PROJECT_ARCHIVE_SUFFIX}"
    if path.suffix:
        return path
    return pathlib.Path(f"{path}{PROJECT_ARCHIVE_SUFFIX}")
