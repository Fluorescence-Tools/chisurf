"""Load/save self-contained FRET-docking project files.

A docking *project* bundles the inputs (PDB rigid bodies + ``fps.json``
restraints), the chosen operation and its sampling parameters into a single
JSON file so a run can be reproduced from the CLI, the RPC services, or the
AutoForm GUI without re-entering paths.

Paths inside the file are stored relative to the project file's directory when
possible (so an example folder stays portable) and resolved to absolute paths
on load.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List

PROJECT_VERSION = 1

#: Keys under ``params`` understood by the docking operation.
_PARAM_KEYS = (
    "n_frames", "mc_steps", "n_best", "fixed_body", "sigma_da",
    "simulated_annealing", "score_set",
)


def _resolve(base_dir: str, p: str) -> str:
    """Resolve ``p`` against ``base_dir`` unless it is already absolute."""
    if not p:
        return p
    return p if os.path.isabs(p) else os.path.normpath(os.path.join(base_dir, p))


def _relativize(base_dir: str, p: str) -> str:
    """Store ``p`` relative to ``base_dir`` when it lives under it."""
    if not p:
        return p
    ap = os.path.abspath(p)
    try:
        rel = os.path.relpath(ap, base_dir)
    except ValueError:  # e.g. different drive on Windows
        return ap
    return rel if not rel.startswith(os.pardir + os.sep) and rel != os.pardir else ap


@dataclass
class DockingProject:
    """In-memory view of a docking project with absolute paths.

    Attributes
    ----------
    pdb_paths : list of str
        One PDB per rigid body (absolute paths).
    fps_json : str
        Path to the ``fps.json`` restraints file (absolute).
    output_dir : str
        Directory for RMF / PDB / CSV output (absolute).
    operation : str
        ``dock`` / ``refine`` / ``screen`` / ``score``.
    score_set : str
        Named chi2 score set (empty = all distances).
    params : dict
        Sampling controls passed through to the docking operation.
    name, description : str
        Free-text metadata.
    path : str
        Path the project was loaded from / last saved to (empty if unsaved).
    """

    pdb_paths: List[str] = field(default_factory=list)
    fps_json: str = ""
    output_dir: str = ""
    operation: str = "dock"
    method: str = "minimize"
    score_set: str = ""
    params: Dict = field(default_factory=dict)
    name: str = ""
    description: str = ""
    path: str = ""

    # -- requests ----------------------------------------------------------
    def to_dock_request(self) -> Dict:
        """Return a dict accepted by :func:`...api.operations.dock`."""
        req = {
            "pdb_paths": list(self.pdb_paths),
            "fps_json": self.fps_json,
            "output_dir": self.output_dir,
            "score_set": self.score_set,
            "method": self.method,
        }
        for k in _PARAM_KEYS:
            if k == "score_set":
                continue
            if k in self.params:
                req[k] = self.params[k]
        return req


def load_docking_project(path: str) -> DockingProject:
    """Read a docking project file and resolve its paths to absolute ones.

    Parameters
    ----------
    path : str
        Path to the project JSON file.

    Returns
    -------
    DockingProject
    """
    path = os.path.abspath(path)
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    base = os.path.dirname(path)

    raw_pdbs = data.get("pdb_paths") or []
    if isinstance(raw_pdbs, str):
        raw_pdbs = [p.strip() for p in raw_pdbs.split(",") if p.strip()]

    params = dict(data.get("params") or {})
    return DockingProject(
        pdb_paths=[_resolve(base, p) for p in raw_pdbs],
        fps_json=_resolve(base, data.get("fps_json", "")),
        output_dir=_resolve(base, data.get("output_dir", "")),
        operation=data.get("operation", "dock"),
        method=data.get("method", "minimize"),
        score_set=data.get("score_set", params.get("score_set", "")),
        params=params,
        name=data.get("name", ""),
        description=data.get("description", ""),
        path=path,
    )


def save_docking_project(
    path: str,
    *,
    pdb_paths: List[str],
    fps_json: str,
    output_dir: str,
    operation: str = "dock",
    method: str = "minimize",
    score_set: str = "",
    params: Dict | None = None,
    name: str = "",
    description: str = "",
) -> str:
    """Write a docking project file, storing paths relative to it when possible.

    Returns
    -------
    str
        The absolute path written.
    """
    path = os.path.abspath(path)
    base = os.path.dirname(path)
    os.makedirs(base, exist_ok=True)
    payload = {
        "fret_docking_project_version": PROJECT_VERSION,
        "name": name,
        "description": description,
        "operation": operation,
        "method": method,
        "pdb_paths": [_relativize(base, p) for p in pdb_paths],
        "fps_json": _relativize(base, fps_json),
        "output_dir": _relativize(base, output_dir),
        "score_set": score_set,
        "params": dict(params or {}),
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return path


__all__ = ["DockingProject", "load_docking_project", "save_docking_project", "PROJECT_VERSION"]
