from __future__ import annotations

import datetime
import json
import pathlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

PathLike = Union[str, pathlib.Path]


@dataclass
class Project:
    """Minimal, GUI-independent representation of a ChiSurf project.

    This class provides JSON-based save/load to a project *directory*.
    Version 3 supports fully deterministic save/load with UIDs.
    """

    name: str = "untitled"
    description: str = ""
    chisurf_version: Optional[str] = None
    project_format_version: int = 3
    # Creation timestamp (ISO 8601). Mainly for user information.
    created: str = field(default_factory=lambda: datetime.datetime.now().isoformat())

    # Core state sections keyed by UID or strict lists:
    datasets: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    experiments: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    fits: List[Dict[str, Any]] = field(default_factory=list)
    links: List[Dict[str, Any]] = field(default_factory=list)
    ui_state: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert this project into a deterministic JSON-serializable dictionary."""
        # Ensure v3 format is enforced
        self.project_format_version = 3
        
        # Sort dictionaries by key for deterministic output
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
            "links": self.links,
            "ui": self.ui_state,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Project":
        """Reconstruct a :class:`Project` from a dictionary. V3 format only."""
        version = int(data.get("project_format_version", 1))
        if version != 3:
            raise ValueError(f"Unsupported project format version: {version}. Only v3 is supported.")

        meta = data.get("meta", {})
        
        # Extract explicit metadata keys not part of the root
        core_meta_keys = {"name", "description", "chisurf_version", "created"}
        metadata = {k: v for k, v in meta.items() if k not in core_meta_keys}

        return cls(
            name=meta.get("name", "untitled"),
            description=meta.get("description", ""),
            chisurf_version=meta.get("chisurf_version"),
            project_format_version=3,
            created=meta.get("created") or datetime.datetime.now().isoformat(),
            datasets=data.get("datasets") or {},
            experiments=data.get("experiments") or {},
            fits=data.get("fits") or [],
            links=data.get("links") or [],
            ui_state=data.get("ui") or {},
            metadata=metadata,
        )

    def get_dataset(self, uid: str) -> Optional[Dict[str, Any]]:
        return self.datasets.get(uid)

    def get_fit(self, uid: str) -> Optional[Dict[str, Any]]:
        for f in self.fits:
            if f.get("uid") == uid:
                return f
        return None

    def list_dataset_uids(self) -> List[str]:
        return sorted(list(self.datasets.keys()))

    def list_fit_uids(self) -> List[str]:
        uids = [f.get("uid") for f in self.fits if f.get("uid")]
        return sorted(uids)

    def save(self, project_dir: PathLike) -> pathlib.Path:
        """Save this project into a directory as ``project.json``."""
        path = pathlib.Path(project_dir)
        path.mkdir(parents=True, exist_ok=True)
        project_file = path / "project.json"

        with project_file.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, sort_keys=True)

        return project_file

    @classmethod
    def load(cls, project_dir: PathLike) -> "Project":
        """Load a project from a directory containing ``project.json``."""
        path = pathlib.Path(project_dir)
        project_file = path / "project.json"

        if not project_file.is_file():
            raise FileNotFoundError(f"Project JSON not found: {project_file}")

        with project_file.open("r", encoding="utf-8") as f:
            data = json.load(f)

        return cls.from_dict(data)


def save_project(project: Project, target_path: PathLike) -> pathlib.Path:
    """Convenience wrapper to save a :class:`Project`."""
    return project.save(target_path)


def load_project(target_path: PathLike) -> Project:
    """Convenience wrapper to load a :class:`Project` from a directory."""
    return Project.load(target_path)

